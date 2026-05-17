"""
Can3Tok Training — MAIN NEW IDEA: decoder_zs_cross_attn
=========================================================
KEY ADDITION:
  --decoder_zs_cross_attn
    z_g [B, 496, 32] is the ONLY decoder input.
    z_s [B, 16, 32] conditions every decoder transformer layer via cross-attention.
    GS_decoder input: 496×384 = 190,464 dims (was 512×384 = 196,608).
    L_cross_recon / L_ortho still supported (now optional reinforcement).

  The cross-recon forward pass uses decode() with the full Z [B,512,32] and
  swapped z_s embedded inside it — the decode() method splits internally.

ALSO:
  Scene-level z_s InfoNCE (--z_s_infonce_weight > 0)
  Per-Gaussian InfoNCE kept for ablation (--semantic_mode hidden/geometric/dist)
  PCA visualisation for both: per-Gaussian + z_s space PLY
  All loss components printed every epoch
"""

import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse
from pathlib import Path
import math

from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
import torch.utils.data as Data

from semantic_losses import (compute_semantic_loss, compute_scene_infonce_loss,
                             compute_zs_token_infonce_loss,
                             compute_zs_layout_infonce_loss)
from distribution_loss import compute_distribution_loss
from pca_feature_visualization import visualize_semantic_features, visualize_z_s_space
try:
    from pca_feature_visualization import visualize_zs_tokens
except ImportError:
    print("[WARNING] visualize_zs_tokens not found in pca_feature_visualization.py — "
          "please copy the updated file to your working directory. "
          "z_s token PCA visualization will be disabled but training will continue.")
    def visualize_zs_tokens(*args, **kwargs):
        return None
from gs_ply_reconstructor import save_reconstructed_gaussians

from accelerate import Accelerator, DistributedDataParallelKwargs

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

# ============================================================================
# PARAMETER INDICES
# ============================================================================
PARAM_SLICES = {
    'position': slice(0, 3), 'color': slice(3, 6),
    'opacity':  slice(6, 7), 'scale': slice(7, 10), 'rotation': slice(10, 14),
}
GEOMETRIC_INDICES = (list(range(4, 7)) + list(range(7, 10)) + [10]
                     + list(range(11, 14)) + list(range(14, 18)))
GEO_ONLY_SLICES = {
    'position': slice(0, 3), 'opacity': slice(6, 7),
    'scale': slice(7, 10),   'rotation': slice(10, 14),
}

# ============================================================================
# LOSS HELPERS
# ============================================================================
def compute_reconstruction_loss(prediction, target, batch_size, color_weight=1.0):
    if color_weight == 1.0:
        return torch.norm(prediction - target, p=2) / batch_size
    return (torch.norm(prediction[:,:,0:3] - target[:,:,0:3], p=2)
          + torch.norm(prediction[:,:,3:6] - target[:,:,3:6], p=2) * color_weight
          + torch.norm(prediction[:,:,6:]  - target[:,:,6:],  p=2)) / batch_size

def compute_individual_losses(prediction, target):
    return {k: torch.norm(prediction[:,:,sl] - target[:,:,sl], p=2).item()
            for k, sl in PARAM_SLICES.items()}

def scene_semantic_kl_loss(p_hat, p_s, eps=1e-8):
    return (p_s * (torch.log(p_s + eps) - torch.log(p_hat.clamp(min=eps)))).sum(-1).mean()

def compute_cross_recon_loss(pred_cross_3d, target, batch_size):
    loss = torch.tensor(0.0, device=pred_cross_3d.device)
    for sl in GEO_ONLY_SLICES.values():
        loss = loss + torch.norm(pred_cross_3d[:,:,sl] - target[:,:,sl], p=2) / batch_size
    return loss

def compute_orthogonality_loss(mu_s, mu_g, proj_dim=64):
    B = mu_s.shape[0]
    if B < 2: return torch.tensor(0.0, device=mu_s.device)
    with torch.no_grad():
        p = min(proj_dim, B - 1, mu_s.shape[1], mu_g.shape[1])
        is_ = torch.randperm(mu_s.shape[1], device=mu_s.device)[:p]
        ig  = torch.randperm(mu_g.shape[1], device=mu_g.device)[:p]
    ps = F.normalize(mu_s[:,is_] - mu_s[:,is_].mean(0,True), p=2, dim=0)
    pg = F.normalize(mu_g[:,ig]  - mu_g[:,ig].mean(0,True),  p=2, dim=0)
    return ((ps.T @ pg) ** 2).mean()

def compute_layout_loss(pred_c, gt_c, gt_valid):
    return ((((pred_c - gt_c)**2).mean(-1)) * gt_valid).sum() / (gt_valid.sum() + 1e-8)

def compute_scale_penalty(pred_3d, threshold=0.5):
    return (torch.clamp(pred_3d[:,:,7:10] - threshold, min=0.0)**2).mean()

def compute_seg_pred_loss(seg_logits, segment_labels):
    B, N, C = seg_logits.shape
    fl = seg_logits.reshape(B*N, C); ll = segment_labels.reshape(B*N).long()
    valid = ll >= 0
    if valid.sum() == 0: return torch.tensor(0.0, device=seg_logits.device)
    return F.cross_entropy(fl[valid], ll[valid])

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(description='Can3Tok Training')
parser.add_argument('--batch_size',           type=int,   default=64)
parser.add_argument('--num_epochs',           type=int,   default=1000)
parser.add_argument('--lr',                   type=float, default=1e-4)
parser.add_argument('--kl_weight',            type=float, default=1e-5)
parser.add_argument('--weight_decay',         type=float, default=1e-2)
parser.add_argument('--warmup_steps',         type=int,   default=100)
parser.add_argument('--lr_min_ratio',         type=float, default=0.1)
parser.add_argument('--eval_every',           type=int,   default=20)
parser.add_argument('--failure_threshold',    type=float, default=100.0)
parser.add_argument('--train_scenes',         type=int,   default=None)
parser.add_argument('--val_scenes',           type=int,   default=None)
parser.add_argument('--sampling_method',      type=str,   default='opacity',
                    choices=['random','opacity','hybrid'])
# ── MAIN NEW IDEA ─────────────────────────────────────────────────────────────
parser.add_argument('--decoder_zs_cross_attn', action='store_true', default=False,
    help='NEW MAIN IDEA: exclude z_s from decoder input sequence; '
         'condition every decoder transformer layer via cross-attention instead. '
         'z_g [B,496,32] → decoder. z_s [B,16,32] → cross-attn K/V per layer. '
         'Requires latent_disentangle=True.')
# ── Per-Gaussian InfoNCE ──────────────────────────────────────────────────────
parser.add_argument('--semantic_mode',        type=str,   default='none',
                    choices=['none','hidden','geometric','dist'])
parser.add_argument('--segment_loss_weight',  type=float, default=0.0)
parser.add_argument('--instance_loss_weight', type=float, default=0.0)
parser.add_argument('--semantic_temperature', type=float, default=0.07)
parser.add_argument('--semantic_subsample',   type=int,   default=2000)
parser.add_argument('--sampling_strategy',    type=str,   default='balanced',
                    choices=['random','balanced'])
# ── Scene z_s InfoNCE ─────────────────────────────────────────────────────────
parser.add_argument('--z_s_infonce_weight',      type=float, default=0.0)
parser.add_argument('--z_s_infonce_temperature', type=float, default=0.07)
parser.add_argument('--z_s_infonce_delta',       type=float, default=0.4)
# ── Strategy B flags (new) ────────────────────────────────────────────────────────
parser.add_argument('--decoder_layout_cross_attn', action='store_true', default=False,
    help='Strategy B1: 512 geometry tokens in decoder + 16 layout tokens as '
         'cross-attention K/V per transformer layer. Layout tokens from shape_embed '
         'via Layout16Projector (SEPARATE from Z). Works with latent_disentangle=False.')
parser.add_argument('--decoder_layout_additive', action='store_true', default=False,
    help='Strategy B2: 512 geometry tokens in decoder + 16 layout tokens projected '
         'to additive broadcast bias before the transformer (once, not per-layer). '
         'Simpler than B1. Can be combined with B1 for additive+cross-attn.')
parser.add_argument('--structured_layout_tokens', action='store_true', default=False,
    help='Structured token split: no gradient interference between semantic and layout heads. '
         'WITHOUT: SceneSemanticHead and SceneLayoutHead both receive tokens 1-15 flattened '
         '[B, 480] — same floats, gradients interfere. '
         'WITH: tokens 1-8 [B,256] → SceneSemanticHead only; '
         'tokens 9-15 [B,224] → SceneLayoutHead only — exclusive, no cross-contamination. '
         'Token 0 → MeanColorHead (unchanged). '
         'Works for Strategy A (semantic_token_heads=True) and Strategy B. '
         'Requires scene_semantic_head=True AND scene_layout_head=True to have any effect.')
parser.add_argument('--zs_layout_infonce_weight',      type=float, default=0.0,
    help='Weight for z_layout InfoNCE. Requires decoder_layout_cross_attn or '
         'decoder_layout_additive. Prototype mechanism: same as per-Gaussian InfoNCE '
         'but at scene level. Recommended start: 0.1')
parser.add_argument('--zs_layout_infonce_temperature', type=float, default=0.07,
    help='Temperature for z_layout InfoNCE (default 0.07).')
# ── z_s Token InfoNCE (same mechanism as per-Gaussian, on the 16 z_s tokens) ─────
# ── z_s pool InfoNCE (new — mirrors decoder hidden InfoNCE) ───────────────────
parser.add_argument('--zs_pool_infonce_weight',      type=float, default=0.0,
    help='Weight for z_s/z_layout pool InfoNCE. '
         'mean_pool(tokens [B,16,32])->linear->[B,1024]->MLP->[B,128]->NCE. '
         'Same bottleneck and mechanism as decoder hidden InfoNCE. '
         'Works for Strategy A (latent_disentangle) and Strategy B. '
         'Recommended start: 0.1')
parser.add_argument('--zs_pool_infonce_temperature', type=float, default=0.07,
    help='Temperature for z_s pool InfoNCE (default 0.07).')
parser.add_argument('--zs_token_infonce_weight',      type=float, default=0.0,
    help='Weight for z_s token InfoNCE. Same cross-batch prototype mechanism as '
         'per-Gaussian InfoNCE but on the 16 z_s tokens directly. '
         'Each token labelled by scene dominant ScanNet72 category. '
         'Requires latent_disentangle=True. Recommended start: 0.1')
parser.add_argument('--zs_token_infonce_temperature', type=float, default=0.07,
    help='Temperature for z_s token InfoNCE (default 0.07, same as per-Gaussian).')
# ── Core ─────────────────────────────────────────────────────────────────────
parser.add_argument('--color_residual',       action='store_true', default=False)
parser.add_argument('--mean_color_weight',    type=float, default=1.0)
parser.add_argument('--scene_semantic_head',  action='store_true', default=False)
parser.add_argument('--scene_semantic_weight',type=float, default=0.3)
parser.add_argument('--position_scaffold',    action='store_true', default=False)
parser.add_argument('--anchor_loss_weight',   type=float, default=1.0)
parser.add_argument('--latent_disentangle',   action='store_true', default=False)
parser.add_argument('--semantic_dims',        type=int,   default=512)
parser.add_argument('--cross_recon_weight',   type=float, default=0.3)
parser.add_argument('--ortho_weight',         type=float, default=0.1)
parser.add_argument('--scene_layout_head',    action='store_true', default=False)
parser.add_argument('--layout_loss_weight',   type=float, default=0.3)
parser.add_argument('--position_layout_residual', action='store_true', default=False)
parser.add_argument('--decoder_pos_enc',      action='store_true', default=False)
parser.add_argument('--predict_seg_labels',   action='store_true', default=False)
parser.add_argument('--seg_pred_weight',      type=float, default=0.3)
parser.add_argument('--token_cond',           action='store_true', default=False)
parser.add_argument('--token_cond_approach',  type=str,   default='B',
                    choices=['A','B','AB'])
parser.add_argument('--decoder_fourier_pe',   action='store_true', default=False)
parser.add_argument('--token_cond_adaln',     action='store_true', default=False)
parser.add_argument('--semantic_token_heads', action='store_true', default=False)
# Legacy flags kept for compat
parser.add_argument('--jepa_idea1',           action='store_true', default=False)
parser.add_argument('--jepa_idea1_weight',    type=float, default=1.0)
parser.add_argument('--query_decoder',        action='store_true', default=False)
parser.add_argument('--label_input',          action='store_true', default=False)
parser.add_argument('--no_label_input',       dest='label_input', action='store_false')
parser.add_argument('--scale_norm_mode',      type=str,   default='linear',
                    choices=['log','linear'])
parser.add_argument('--color_loss_weight',    type=float, default=1.0)
parser.add_argument('--scale_penalty_weight', type=float, default=0.0)
parser.add_argument('--scale_penalty_threshold', type=float, default=0.5)
norm_grp = parser.add_mutually_exclusive_group()
norm_grp.add_argument('--use_canonical_norm', dest='use_canonical_norm',
                      action='store_true', default=True)
norm_grp.add_argument('--no_canonical_norm',  dest='use_canonical_norm',
                      action='store_false')
color_norm_grp = parser.add_mutually_exclusive_group()
color_norm_grp.add_argument('--normalize_colors',    dest='normalize_colors',
                            action='store_true', default=True)
color_norm_grp.add_argument('--no_normalize_colors', dest='normalize_colors',
                            action='store_false')
parser.add_argument('--pca_vis_freq',         type=int,   default=50)
parser.add_argument('--pca_brightness',       type=float, default=1.25)
parser.add_argument('--pca_num_scenes',       type=int,   default=3)
parser.add_argument('--recon_ply_freq',       type=int,   default=50)
parser.add_argument('--recon_ply_num_scenes', type=int,   default=3)
parser.add_argument('--recon_ply_max_sh',     type=int,   default=3)
parser.add_argument('--use_wandb',            action='store_true', default=False)
parser.add_argument('--wandb_project',        type=str,   default='Can3Tok-SceenSplat-7K')
parser.add_argument('--wandb_entity',         type=str,   default='3D-SSC')
parser.add_argument('--resume_checkpoint',    type=str,   default=None)
parser.add_argument('--resume_epoch',         type=int,   default=None)

args = parser.parse_args()

# ── Validation ───────────────────────────────────────────────────────────────
if args.decoder_zs_cross_attn and not args.latent_disentangle:
    raise ValueError("--decoder_zs_cross_attn requires --latent_disentangle")
if args.cross_recon_weight > 0 and not args.latent_disentangle:
    args.cross_recon_weight = 0.0
if args.ortho_weight > 0 and not args.latent_disentangle:
    args.ortho_weight = 0.0
if args.z_s_infonce_weight > 0 and not args.latent_disentangle:
    args.z_s_infonce_weight = 0.0
if args.zs_token_infonce_weight > 0 and not args.latent_disentangle:
    print("[WARNING] zs_token_infonce_weight > 0 requires latent_disentangle. Setting to 0.")
    args.zs_token_infonce_weight = 0.0
_any_B = args.decoder_layout_cross_attn or args.decoder_layout_additive
if args.zs_layout_infonce_weight > 0 and not _any_B:
    if args.latent_disentangle:
        # Strategy A: z_s tokens act as layout tokens.
        # last_z_layout_proj is routed from last_z_s_infonce_proj in model forward().
        # z_s_infonce_head must exist — it is created when latent_disentangle=True.
        print("[INFO] zs_layout_infonce_weight > 0 with Strategy A: "
              "routing z_s tokens as layout tokens (last_z_s_infonce_proj -> last_z_layout_proj)")
    else:
        print("[WARNING] zs_layout_infonce_weight > 0 requires decoder_layout_cross_attn, "
              "decoder_layout_additive, OR latent_disentangle. Setting to 0.")
        args.zs_layout_infonce_weight = 0.0
if _any_B and args.latent_disentangle:
    print("[INFO] decoder_layout_cross/additive=True with latent_disentangle=True: "
          "z_layout from shape_embed is separate from Z (which has z_s in first 16 pos).")
if args.semantic_dims % 32 != 0:
    raise ValueError("--semantic_dims must be divisible by 32")
if args.semantic_token_heads and not args.latent_disentangle:
    raise ValueError("--semantic_token_heads requires --latent_disentangle")
if args.position_layout_residual and not args.scene_layout_head:
    args.scene_layout_head = True
if args.token_cond and 'B' in args.token_cond_approach.upper() and not args.scene_layout_head:
    args.scene_layout_head = True

need_scaffold_data = args.position_scaffold
semantic_requested    = (args.semantic_mode != 'none')
semantic_loss_enabled = (args.segment_loss_weight > 0 or args.instance_loss_weight > 0)
enable_semantic       = semantic_requested and semantic_loss_enabled
effective_semantic_mode = args.semantic_mode if enable_semantic else 'none'
need_segment_labels = (enable_semantic or args.scene_semantic_head or args.predict_seg_labels)

# ============================================================================
# ACCELERATE
# ============================================================================
_ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=True)
accelerator = Accelerator(kwargs_handlers=[_ddp_kwargs])

# ============================================================================
# W&B
# ============================================================================
wandb_enabled = False
if args.use_wandb and accelerator.is_main_process:
    try:
        import wandb
        job_id   = os.environ.get('SLURM_JOB_ID', 'local')
        run_name = f"can3tok_{job_id}"
        flags = [
            (args.color_residual,             "_colorresidual"),
            (args.latent_disentangle,         f"_disent{args.semantic_dims}"),
            (args.decoder_zs_cross_attn,      "_zsCA"),
            (args.decoder_fourier_pe,         "_fourierpe"),
            (args.scene_layout_head,          "_layout"),
            (args.semantic_token_heads,       "_semTok"),
            (args.z_s_infonce_weight > 0,     "_zsNCE"),
    (args.zs_token_infonce_weight > 0,  "_zsTokNCE"),
    (args.decoder_layout_cross_attn,    "_layCA"),
    (args.decoder_layout_additive,      "_layAdd"),
    (args.zs_layout_infonce_weight > 0,   "_layNCE"),
    (args.zs_pool_infonce_weight > 0,      "_poolNCE"),
            (enable_semantic,                 f"_pgNCE{args.segment_loss_weight}"),
        ]
        for flag, label in flags:
            if flag: run_name += label
        run_name += "_inferencefixed"
        wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                               name=run_name, config=vars(args))
        wandb_enabled = True
        print("W&B enabled")
    except Exception as e:
        print(f"W&B failed: {e}")

# ============================================================================
# DEVICE + PATHS
# ============================================================================
device    = accelerator.device
data_path = "/home/yli7/scratch/datasets/gaussian_world/preprocessed/interior_gs"

job_id = os.environ.get('SLURM_JOB_ID', None)
tag    = (f"RGB_job_{job_id}_{effective_semantic_mode}" if job_id
          else f"RGB_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
flags = [
    (args.color_residual,             "_colorresidual"),
    (args.latent_disentangle,         f"_disent{args.semantic_dims}"),
    (args.decoder_zs_cross_attn,      "_zsCA"),
    (args.decoder_fourier_pe,         "_fourierpe"),
    (args.scene_layout_head,          "_layout"),
    (args.semantic_token_heads,       "_semTok"),
    (args.z_s_infonce_weight > 0,     "_zsNCE"),
    (args.zs_token_infonce_weight > 0,  "_zsTokNCE"),
    (args.decoder_layout_cross_attn,    "_layCA"),
    (args.decoder_layout_additive,      "_layAdd"),
    (args.zs_layout_infonce_weight > 0,   "_layNCE"),
    (args.zs_pool_infonce_weight > 0,      "_poolNCE"),
    (enable_semantic,                 f"_pgNCE"),
]
for flag, label in flags:
    if flag: tag += label
tag += "_inferencefixed"

save_path = f"/home/yli11/scratch-project/Hafeez_thesis/Can3Tok/checkpoints_stage1/{tag}/"
os.makedirs(save_path, exist_ok=True)

# ============================================================================
# STARTUP SUMMARY
# ============================================================================
if accelerator.is_main_process:
    print(f"\n{'='*70}")
    print(f"CAN3TOK — MAIN NEW IDEA: decoder_zs_cross_attn={args.decoder_zs_cross_attn}")
    if args.decoder_zs_cross_attn:
        n_s = args.semantic_dims // 32  # embed_dim=32
        n_g = 512 - n_s
        print(f"  z_s: {n_s} tokens → cross-attn K/V (NOT decoder sequence input)")
        print(f"  z_g: {n_g} tokens → decoder sequence input")
        print(f"  GS_decoder input: {n_g}×384 = {n_g*384}")
    else:
        print(f"  LEGACY: all 512 tokens → decoder")
    print(f"  color_residual={args.color_residual}")
    print(f"  latent_disentangle={args.latent_disentangle} semantic_dims={args.semantic_dims}")
    print(f"  scene_layout_head={args.scene_layout_head}")
    print(f"  decoder_fourier_pe={args.decoder_fourier_pe}")
    print(f"  token_cond={args.token_cond} adaln={args.token_cond_adaln}")
    print(f"  semantic_token_heads={args.semantic_token_heads}")
    print(f"  z_s InfoNCE weight={args.z_s_infonce_weight} temp={args.z_s_infonce_temperature} delta={args.z_s_infonce_delta}")
    print(f"  z_s TOKEN InfoNCE weight={args.zs_token_infonce_weight} temp={args.zs_token_infonce_temperature}")
    print(f"  ── Strategy B (layout conditioning) ────────────────────────────")
    print(f"  decoder_layout_cross_attn={args.decoder_layout_cross_attn}  (B1: 512 geom + cross-attn per layer)")
    print(f"  decoder_layout_additive  ={args.decoder_layout_additive}  (B2: 512 geom + additive bias)")
    if args.decoder_layout_cross_attn and args.decoder_layout_additive:
        print(f"  Strategy B3: additive + cross-attn both active")
    print(f"  zs_layout_infonce_weight ={args.zs_layout_infonce_weight}  temp={args.zs_layout_infonce_temperature}")
    print(f"  zs_pool_infonce_weight   ={args.zs_pool_infonce_weight}  temp={args.zs_pool_infonce_temperature}")
    print(f"  (pool: mean_pool([B,16,32])->Linear->[B,1024]->MLP->[B,128]->NCE, mirrors decoder)")
    print(f"  (z_s token InfoNCE: same prototype mechanism as per-Gaussian, direct gradient to z_s)")
    print(f"  per-Gaussian InfoNCE mode={effective_semantic_mode} weight={args.segment_loss_weight}")
    print(f"  cross_recon={args.cross_recon_weight} ortho={args.ortho_weight}")
    print(f"  Save: {save_path}")
    print(f"{'='*70}\n")

# ============================================================================
# MODEL
# ============================================================================
print("Loading model config...")
config_path  = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
model_config = get_config_from_file(config_path).model
p = model_config.params.shape_module_cfg.params
p.semantic_mode           = effective_semantic_mode
p.color_residual          = args.color_residual
p.scene_semantic_head     = args.scene_semantic_head
p.position_scaffold       = args.position_scaffold
p.latent_disentangle      = args.latent_disentangle
p.semantic_dims           = args.semantic_dims
p.scene_layout_head       = args.scene_layout_head
p.jepa_idea1              = args.jepa_idea1
p.decoder_pos_enc         = args.decoder_pos_enc
p.predict_seg_labels      = args.predict_seg_labels
p.token_cond              = args.token_cond
p.token_cond_approach     = args.token_cond_approach
p.query_decoder           = args.query_decoder
p.decoder_fourier_pe      = args.decoder_fourier_pe
p.token_cond_adaln        = args.token_cond_adaln
p.semantic_token_heads    = args.semantic_token_heads
p.decoder_zs_cross_attn       = args.decoder_zs_cross_attn  # Strategy D
p.decoder_layout_cross_attn   = args.decoder_layout_cross_attn  # Strategy B1 NEW
p.decoder_layout_additive     = args.decoder_layout_additive     # Strategy B2 NEW
p.structured_layout_tokens    = args.structured_layout_tokens     # token split, no interference
p.position_layout_residual    = args.position_layout_residual

cfg_point_feats = p.point_feats
expected_feats  = 12 if args.label_input else 11
if cfg_point_feats != expected_feats:
    raise ValueError(f"point_feats mismatch: yaml={cfg_point_feats}, expected {expected_feats}.")
print(f"  point_feats={cfg_point_feats} OK")

gs_autoencoder = instantiate_from_config(model_config)
gs_autoencoder.to(device)
optimizer = torch.optim.AdamW(
    gs_autoencoder.parameters(), lr=args.lr, betas=[0.9,0.999], weight_decay=args.weight_decay)

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================
start_epoch   = 0
best_val_loss = float('inf')
best_epoch    = 0

if args.resume_checkpoint:
    print(f"\nResuming from: {args.resume_checkpoint}")
    ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
    # Hard-fail on structural mismatches
    for flag_name, current_val, default_val in [
        ('color_residual',             args.color_residual,          False),
        ('label_input',                args.label_input,             False),
        ('latent_disentangle',         args.latent_disentangle,      False),
        ('semantic_dims',              args.semantic_dims,           512),
        ('position_layout_residual',   args.position_layout_residual, False),
    ]:
        saved = ckpt.get(flag_name, default_val)
        if saved != current_val:
            raise ValueError(f"{flag_name} mismatch: ckpt={saved}, current={current_val}.")

    # For decoder_zs_cross_attn: strict=False allows adding new GS_decoder_new etc.
    strict = all([
        ckpt.get('scene_semantic_head',   False) == args.scene_semantic_head,
        ckpt.get('semantic_mode', 'none') == effective_semantic_mode,
        ckpt.get('scene_layout_head',     False) == args.scene_layout_head,
        ckpt.get('decoder_fourier_pe',    False) == args.decoder_fourier_pe,
        ckpt.get('token_cond',            False) == args.token_cond,
        ckpt.get('token_cond_adaln',      False) == args.token_cond_adaln,
        ckpt.get('semantic_token_heads',  False) == args.semantic_token_heads,
        ckpt.get('decoder_zs_cross_attn', False) == args.decoder_zs_cross_attn,
    ])
    if not strict:
        print(f"  Architecture changed — loading strict=False (new components init fresh)")
    gs_autoencoder.load_state_dict(ckpt['model_state_dict'], strict=strict)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch   = ckpt.get('epoch', 0) + 1
    if args.resume_epoch is not None: start_epoch = args.resume_epoch
    best_val_loss = ckpt.get('val_l2_error', ckpt.get('best_val_l2', float('inf')))
    best_epoch    = ckpt.get('epoch', 0)
    print(f"  Resumed epoch {start_epoch} (val L2: {best_val_loss:.4f})")

# ============================================================================
# LR SCHEDULER
# ============================================================================
def build_lr_lambda(warmup_steps, total_steps, lr_min_ratio):
    cosine_steps = max(total_steps - warmup_steps, 1)
    def f(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(warmup_steps)
        t = step - warmup_steps
        return lr_min_ratio + (1-lr_min_ratio) * 0.5*(1 + math.cos(math.pi*t/cosine_steps))
    return f

_bpe          = max(1, (args.train_scenes or 300) // (args.batch_size * accelerator.num_processes))
_total_steps  = _bpe * args.num_epochs
_elapsed      = _bpe * start_epoch
scheduler     = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=build_lr_lambda(
    warmup_steps=max(0, args.warmup_steps - _elapsed),
    total_steps=_total_steps - _elapsed,
    lr_min_ratio=args.lr_min_ratio))
print(f"\n  LR: peak={args.lr:.2e} | floor={args.lr*args.lr_min_ratio:.2e}")

# ============================================================================
# DATASETS
# ============================================================================
from gs_dataset_scenesplat import gs_dataset

print(f"\n--- Training Dataset ---")
gs_dataset_train = gs_dataset(
    root=os.path.join(data_path, "train_grid1.0cm_chunk8x8_stride6x6"),
    resol=200, random_permute=True, train=True,
    sampling_method=args.sampling_method, max_scenes=args.train_scenes,
    normalize=args.use_canonical_norm, normalize_colors=args.normalize_colors,
    target_radius=10.0, scale_norm_mode=args.scale_norm_mode,
    label_input=args.label_input, color_residual=args.color_residual,
    position_scaffold=args.position_scaffold,
    scene_layout_head=args.scene_layout_head,
    jepa_idea1=args.jepa_idea1,
    position_layout_residual=args.position_layout_residual)
trainDataLoader = Data.DataLoader(
    dataset=gs_dataset_train, batch_size=args.batch_size,
    shuffle=False, num_workers=9, pin_memory=True, persistent_workers=False)

print(f"\n--- Validation Dataset ---")
gs_dataset_val = gs_dataset(
    root=os.path.join(data_path, "train_grid1.0cm_chunk8x8_stride6x6"),
    resol=200, random_permute=False, train=True,
    sampling_method=args.sampling_method, max_scenes=args.val_scenes,
    normalize=args.use_canonical_norm, normalize_colors=args.normalize_colors,
    target_radius=10.0, scale_norm_mode=args.scale_norm_mode,
    label_input=args.label_input, color_residual=args.color_residual,
    position_scaffold=args.position_scaffold,
    scene_layout_head=args.scene_layout_head,
    jepa_idea1=args.jepa_idea1,
    position_layout_residual=args.position_layout_residual)
valDataLoader = Data.DataLoader(
    dataset=gs_dataset_val, batch_size=args.batch_size,
    shuffle=False, num_workers=9, pin_memory=True, persistent_workers=False)

if accelerator.is_main_process:
    print(f"\n  Train: {len(gs_dataset_train)} scenes | Val: {len(gs_dataset_val)} scenes")

gs_autoencoder, optimizer, trainDataLoader, valDataLoader, scheduler = accelerator.prepare(
    gs_autoencoder, optimizer, trainDataLoader, valDataLoader, scheduler)
raw_model = accelerator.unwrap_model(gs_autoencoder)

# ============================================================================
# CHECKPOINT METADATA
# ============================================================================
_ckpt_meta = {
    'semantic_mode':              effective_semantic_mode,
    'enable_semantic':            enable_semantic,
    'label_input':                args.label_input,
    'color_residual':             args.color_residual,
    'scene_semantic_head':        args.scene_semantic_head,
    'position_scaffold':          args.position_scaffold,
    'latent_disentangle':         args.latent_disentangle,
    'semantic_dims':              args.semantic_dims,
    'scene_layout_head':          args.scene_layout_head,
    'decoder_fourier_pe':         args.decoder_fourier_pe,
    'token_cond':                 args.token_cond,
    'token_cond_approach':        args.token_cond_approach,
    'token_cond_adaln':           args.token_cond_adaln,
    'semantic_token_heads':       args.semantic_token_heads,
    'decoder_zs_cross_attn':      args.decoder_zs_cross_attn,  # MAIN NEW IDEA
    'z_s_infonce_weight':         args.z_s_infonce_weight,
    'z_s_infonce_temperature':    args.z_s_infonce_temperature,
    'z_s_infonce_delta':          args.z_s_infonce_delta,
    'zs_token_infonce_weight':    args.zs_token_infonce_weight,
    'zs_token_infonce_temperature': args.zs_token_infonce_temperature,
    'decoder_layout_cross_attn':  args.decoder_layout_cross_attn,
    'decoder_layout_additive':    args.decoder_layout_additive,
    'zs_layout_infonce_weight':   args.zs_layout_infonce_weight,
    'zs_pool_infonce_weight':      args.zs_pool_infonce_weight,
    'zs_pool_infonce_temperature': args.zs_pool_infonce_temperature,
    'structured_layout_tokens':   args.structured_layout_tokens,
    'zs_layout_infonce_temperature': args.zs_layout_infonce_temperature,
    'inference_fixed':            True,
    'position_layout_residual':   args.position_layout_residual,
    'mean_color_weight':          args.mean_color_weight,
    'scene_semantic_weight':      args.scene_semantic_weight,
    'anchor_loss_weight':         args.anchor_loss_weight,
    'cross_recon_weight':         args.cross_recon_weight,
    'ortho_weight':               args.ortho_weight,
    'layout_loss_weight':         args.layout_loss_weight,
    'color_loss_weight':          args.color_loss_weight,
    'scale_penalty_weight':       args.scale_penalty_weight,
    'scale_penalty_threshold':    args.scale_penalty_threshold,
    'use_canonical_norm':         args.use_canonical_norm,
    'scale_norm_mode':            args.scale_norm_mode,
}

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model(model, raw_model, dataloader, device, accelerator, epoch=None):
    model.eval()
    total_l2 = total_kl = total_color = total_scene_sem = 0.0
    total_anchor = total_layout = total_seg = total_z_s_nce = total_zs_tok_nce = total_zs_lay_nce = 0.0
    per_param    = {k: 0.0 for k in PARAM_SLICES}
    n_scenes     = 0

    recon_preds  = []; recon_means  = []
    pca_input    = []; pca_recon    = []
    pca_sem_feat = []
    z_s_proj_acc = []; label_dist_acc = []
    zs_tokens_acc  = []    # for z_s token InfoNCE visualization
    zs_layout_acc  = []    # for z_layout visualization (Strategy B)
    zs_pool_acc    = []    # for pool InfoNCE visualization (Strategy A + B)

    do_recon   = (epoch is not None and epoch % args.recon_ply_freq  == 0)
    do_pca     = (epoch is not None and epoch % args.pca_vis_freq    == 0)
    do_sem_pca = (do_pca and enable_semantic)
    do_z_s_vis     = (do_pca and raw_model.shape_model.z_s_infonce_head is not None)
    do_zs_tok_vis  = (do_pca and args.zs_token_infonce_weight > 0
                      and args.latent_disentangle)
    _any_B         = args.decoder_layout_cross_attn or args.decoder_layout_additive
    do_zs_lay_vis  = (do_pca and _any_B)
    do_zs_pool_vis = (do_pca and args.zs_pool_infonce_weight > 0)

    _pos_abs_min = _pos_abs_max = _pos_gt_range = 0.0

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating", leave=False):
            UV_gs_batch   = batch_data['features'].float().to(device)
            mean_color_gt = batch_data['mean_color'].float().to(device)
            label_dist_v  = batch_data['label_dist'].float().to(device)
            B = UV_gs_batch.shape[0]

            sa_gpu  = (batch_data['scaffold_anchors'].float().to(device)
                       if need_scaffold_data else None)
            sti_gpu = (batch_data['scaffold_token_ids'].long().to(device)
                       if args.position_scaffold else None)

            _rsf = True if do_sem_pca else None
            (shape_embed, mu, log_var, z,
             UV_gs_recover, pg_feats) = model(
                UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:,:,:3],
                scaffold_anchors=sa_gpu, scaffold_token_ids=sti_gpu,
                return_semantic_features=_rsf)

            mcp  = raw_model.shape_model.last_mean_color_pred
            ssp  = raw_model.shape_model.last_scene_semantic_pred
            anch = raw_model.shape_model.last_predicted_anchors_from_tokens
            slp  = raw_model.shape_model.last_scene_layout_pred
            sgp  = raw_model.shape_model.last_seg_pred
            zsp  = raw_model.shape_model.last_z_s_infonce_proj

            target_abs = UV_gs_batch[:, :, GEOMETRIC_INDICES]
            if args.position_scaffold:
                pos_off = batch_data['position_offsets'].float().to(device)
                target  = target_abs.clone(); target[:,:,0:3] = pos_off
                pred_3d = UV_gs_recover.reshape(B,-1,14).clone()
                if anch is not None and sti_gpu is not None:
                    idx_3d = sti_gpu.unsqueeze(-1).expand(-1,-1,3)
                    pred_3d[:,:,0:3] -= torch.gather(anch, 1, idx_3d)
            elif args.position_layout_residual:
                pos_res = batch_data['position_residuals'].float().to(device)
                target  = target_abs.clone(); target[:,:,0:3] = pos_res
                pred_3d = UV_gs_recover.reshape(B,-1,14)
            else:
                target  = target_abs
                pred_3d = UV_gs_recover.reshape(B,-1,14)

            pred_abs = UV_gs_recover.reshape(B,-1,14)

            recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight)
            kl_loss    = -0.5*torch.sum(1+log_var - mu.pow(2) - log_var.exp(), dim=1)

            if mcp is not None and args.color_residual:
                total_color += F.mse_loss(mcp, mean_color_gt).item() * B
            if ssp is not None and args.scene_semantic_head:
                p_s = batch_data['label_dist'].float().to(device)
                total_scene_sem += scene_semantic_kl_loss(ssp, p_s).item() * B
            if anch is not None and args.position_scaffold:
                total_anchor += F.mse_loss(anch, sa_gpu).item() * B
            if slp is not None and args.scene_layout_head:
                gt_c = batch_data['category_centroids'].float().to(device)
                gt_v = batch_data['category_valid'].float().to(device)
                total_layout += compute_layout_loss(slp, gt_c, gt_v).item() * B
            if args.predict_seg_labels and sgp is not None:
                total_seg += compute_seg_pred_loss(sgp, batch_data['segment_labels'].long().to(device)).item() * B
            # z_s token InfoNCE validation loss + collect tokens for visualization
            z_s_tokens_eval = None
            if args.latent_disentangle and args.semantic_dims > 0:
                _n_tok = args.semantic_dims // 32   # embed_dim=32, so 512//32=16
                z_s_tokens_eval = z.reshape(B, -1, 32)[:, :_n_tok, :].detach()  # [B,16,32]
            if args.zs_token_infonce_weight > 0 and z_s_tokens_eval is not None:
                zl_tok, _ = compute_zs_token_infonce_loss(
                    z_s_tokens_eval, label_dist_v, args.zs_token_infonce_temperature)
                total_zs_tok_nce += zl_tok.item() * B

            # z_layout InfoNCE validation loss (Strategy B)
            z_lay_proj_eval = raw_model.shape_model.last_z_layout_proj
            if args.zs_layout_infonce_weight > 0 and z_lay_proj_eval is not None:
                zl_lay, _ = compute_zs_layout_infonce_loss(
                    z_lay_proj_eval, label_dist_v, args.zs_layout_infonce_temperature)
                total_zs_lay_nce += zl_lay.item() * B

            if args.z_s_infonce_weight > 0 and zsp is not None:
                zl, _ = compute_scene_infonce_loss(zsp, label_dist_v,
                                                   args.z_s_infonce_temperature,
                                                   args.z_s_infonce_delta)
                total_z_s_nce += zl.item() * B

            total_l2 += recon_loss.item()
            total_kl += kl_loss.sum().item()
            n_scenes  += B

            if n_scenes <= B:
                _pos_abs_min  = pred_abs[:,:,0:3].cpu().min().item()
                _pos_abs_max  = pred_abs[:,:,0:3].cpu().max().item()
                _pos_gt_range = (UV_gs_batch[:,:,4:7].cpu().max()-UV_gs_batch[:,:,4:7].cpu().min()).item()/2

            ind = compute_individual_losses(pred_3d, target)
            for k in per_param: per_param[k] += ind[k]

            if do_recon and len(recon_preds) < args.recon_ply_num_scenes:
                pnp = pred_abs.cpu().numpy(); mnp = mean_color_gt.cpu().numpy()
                for si in range(B):
                    if len(recon_preds) >= args.recon_ply_num_scenes: break
                    recon_preds.append(pnp[si]); recon_means.append(mnp[si])

            if do_pca and len(pca_input) < args.pca_num_scenes:
                for si in range(B):
                    if len(pca_input) >= args.pca_num_scenes: break
                    pca_input.append(UV_gs_batch.cpu().numpy()[si])
                    pca_recon.append(pred_abs.cpu().numpy()[si])
                    if do_sem_pca and pg_feats is not None:
                        pca_sem_feat.append(pg_feats.cpu().numpy()[si])

            if do_z_s_vis and zsp is not None:
                z_s_proj_acc.append(zsp.detach().cpu().numpy())
                label_dist_acc.append(label_dist_v.cpu().numpy())
            if do_zs_tok_vis and z_s_tokens_eval is not None:
                zs_tokens_acc.append(z_s_tokens_eval.cpu().numpy())
                if not do_z_s_vis:
                    label_dist_acc.append(label_dist_v.cpu().numpy())

            # Collect z_layout tokens for visualization (Strategy B)
            z_lay_raw_eval = raw_model.shape_model.last_z_layout
            if do_zs_lay_vis and z_lay_raw_eval is not None:
                zs_layout_acc.append(z_lay_raw_eval.detach().cpu().numpy())
                if not do_z_s_vis and not do_zs_tok_vis:
                    label_dist_acc.append(label_dist_v.cpu().numpy())

            # Collect pool hidden [B,1024] for PCA vis (one point per scene)
            if do_zs_pool_vis:
                _ph = getattr(raw_model.shape_model, 'last_zs_pool_hidden', None)
                if _ph is None:
                    _ph = getattr(raw_model.shape_model,
                                  'last_z_layout_pool_hidden', None)
                if _ph is not None:
                    zs_pool_acc.append(_ph.detach().cpu().numpy())  # [B, 1024]
                    if not label_dist_acc:
                        label_dist_acc.append(label_dist_v.cpu().numpy())

    # PLY save
    if do_recon and recon_preds and accelerator.is_main_process:
        try:
            all_preds = np.stack(recon_preds, 0)
            if args.color_residual:
                for si in range(len(all_preds)):
                    all_preds[si,:,3:6] = np.clip(all_preds[si,:,3:6] + recon_means[si], 0, 1)
            recon_dir = Path(save_path)/"reconstructed_gaussians"/f"epoch_{epoch:03d}"
            save_reconstructed_gaussians(predictions=all_preds, output_dir=recon_dir, epoch=epoch,
                num_scenes=len(all_preds), max_sh_degree=args.recon_ply_max_sh, color_mode="1")
        except Exception as e: print(f"  PLY error: {e}")

    # PCA
    if do_pca and pca_input and accelerator.is_main_process:
        try:
            pca_dir = Path(save_path)/"pca_visualisations"/f"epoch_{epoch:03d}"
            pca_dir.mkdir(parents=True, exist_ok=True)
            for si in range(len(pca_input)):
                coords_in = pca_input[si][:,4:7]
                visualize_semantic_features(coords=coords_in, features=pca_input[si],
                    output_path=str(pca_dir/f"scene{si:02d}_input.ply"),
                    brightness=args.pca_brightness, verbose=False)
                visualize_semantic_features(coords=pca_recon[si][:,0:3], features=pca_recon[si],
                    output_path=str(pca_dir/f"scene{si:02d}_recon.ply"),
                    brightness=args.pca_brightness, verbose=False)
                if si < len(pca_sem_feat):
                    visualize_semantic_features(coords=coords_in, features=pca_sem_feat[si],
                        output_path=str(pca_dir/f"scene{si:02d}_semantic_infonce.ply"),
                        brightness=args.pca_brightness, verbose=False)
            print(f"  PCA PLYs: {pca_dir}")
        except Exception as e: print(f"  PCA error: {e}")

    # z_s space PLY
    if do_z_s_vis and z_s_proj_acc and accelerator.is_main_process:
        try:
            all_z_s = np.concatenate(z_s_proj_acc, 0)
            all_ld  = np.concatenate(label_dist_acc, 0)
            vis_dir = Path(save_path)/"pca_visualisations"
            vis_dir.mkdir(parents=True, exist_ok=True)
            out = visualize_z_s_space(all_z_s, all_ld,
                str(vis_dir/f"z_s_space_epoch_{epoch:03d}.ply"), verbose=True)
            if out: print(f"  z_s space PLY: {out}  ({len(all_z_s)} scenes)")
        except Exception as e: print(f"  z_s vis error: {e}")

    # z_s token PLY (NEW — analogous to per-Gaussian semantic_infonce.ply)
    if do_zs_tok_vis and zs_tokens_acc and accelerator.is_main_process:
        try:
            all_toks = np.concatenate(zs_tokens_acc, axis=0)   # [N_scenes, 16, 32]
            all_ld   = np.concatenate(label_dist_acc, axis=0)     # [N_scenes, 72]
            vis_dir  = Path(save_path) / "pca_visualisations"
            vis_dir.mkdir(parents=True, exist_ok=True)
            out_tok = visualize_zs_tokens(
                zs_tokens=all_toks,
                label_dists=all_ld,
                output_path=str(vis_dir / f"zs_tokens_epoch_{epoch:03d}.ply"),
                verbose=True)
            if out_tok:
                print(f"  z_s token PLY: {out_tok}  ({len(all_toks)} scenes × 16 tokens)")
        except Exception as e:
            print(f"  z_s token vis error: {e}")

    # z_layout token PLY (Strategy B — visualize 16 layout tokens per scene)
    if do_zs_lay_vis and zs_layout_acc and accelerator.is_main_process:
        try:
            all_lay = np.concatenate(zs_layout_acc, axis=0)   # [N, 16, 32]
            all_ld  = np.concatenate(label_dist_acc, axis=0) if label_dist_acc else None
            if all_ld is not None:
                vis_dir = Path(save_path) / "pca_visualisations"
                vis_dir.mkdir(parents=True, exist_ok=True)
                out_lay = visualize_zs_tokens(
                    zs_tokens=all_lay, label_dists=all_ld,
                    output_path=str(vis_dir / f"zs_layout_epoch_{epoch:03d}.ply"),
                    verbose=True)
                if out_lay:
                    print(f"  z_layout PLY: {out_lay}  ({len(all_lay)} scenes × 16 tokens)")
        except Exception as e:
            print(f"  z_layout vis error: {e}")

    # z_s pool PLY — same style as per-Gaussian PCA, one point per scene
    # Colors: dominant ScanNet72 category. Position: PCA of [B,1024] pool hidden states.
    # Compare with scene{i}_semantic_infonce.ply (per-Gaussian, 40k points).
    if do_zs_pool_vis and zs_pool_acc and accelerator.is_main_process:
        try:
            all_pool = np.concatenate(zs_pool_acc,  axis=0)   # [N, 128]
            all_ld   = np.concatenate(label_dist_acc, axis=0) if label_dist_acc else None
            if all_ld is not None:
                vis_dir = Path(save_path) / 'pca_visualisations'
                vis_dir.mkdir(parents=True, exist_ok=True)
                out_pool = visualize_z_s_space(
                    z_s_proj=all_pool, label_dists=all_ld,
                    output_path=str(vis_dir / f'zs_pool_epoch_{epoch:03d}.ply'),
                    verbose=True)
                if out_pool:
                    print(f'  z_s pool PLY: {out_pool}  ({len(all_pool)} scenes)')
        except Exception as e:
            print(f'  z_s pool vis error: {e}')

    model.train()
    n = max(n_scenes, 1)
    return {
        'avg_l2_error':       total_l2,
        'avg_kl':             total_kl / n,
        'color_pred_loss':    total_color / n,
        'scene_semantic_kl':  total_scene_sem / n,
        'anchor_loss':        total_anchor / n,
        'layout_loss':        total_layout / n,
        'seg_pred_loss':      total_seg / n,
        'z_s_infonce_loss':   total_z_s_nce / n,
        'zs_tok_infonce_loss':  total_zs_tok_nce / n,
        'zs_lay_infonce_loss':  total_zs_lay_nce / n,
        'zs_pool_infonce_loss': 0.0,   # computed in training loop, not eval
        'pos_abs_range':      _pos_abs_max - _pos_abs_min,
        'pos_abs_min':        _pos_abs_min,
        'pos_abs_max':        _pos_abs_max,
        'pos_gt_range':       _pos_gt_range,
        **{f'{k}_loss': v/n for k, v in per_param.items()},
    }

# ============================================================================
# TRAINING LOOP
# ============================================================================
print(f"\n{'='*70}\nSTARTING TRAINING  (epoch {start_epoch} -> {args.num_epochs-1})\n{'='*70}\n")

global_step = 0

for epoch in tqdm(range(start_epoch, args.num_epochs), desc="Training",
                  disable=not accelerator.is_main_process):
    gs_autoencoder.train()

    e = {k: 0.0 for k in [
        'loss','recon','kl','sem','color_pred','scene_sem','anchor',
        'layout','cross_recon','ortho','seg_pred','scale_pen',
        'z_s_nce','z_s_npos',
        'zs_tok_nce','zs_tok_ncats',
        'zs_lay_nce','zs_lay_ncats',
        'zs_pool_nce','zs_pool_ncats',
        'pos','col','opa','scl','rot']}

    for i_batch, batch_data in enumerate(trainDataLoader):
        UV_gs_batch   = batch_data['features'].float().to(device)
        mean_color_gt = batch_data['mean_color'].float().to(device)
        label_dist_v  = batch_data['label_dist'].float().to(device)
        B = UV_gs_batch.shape[0]

        seg_labels = inst_labels = None
        if need_segment_labels:
            seg_labels  = batch_data['segment_labels'].long().to(device)
            if enable_semantic:
                inst_labels = batch_data['instance_labels'].long().to(device)

        sa_gpu  = (batch_data['scaffold_anchors'].float().to(device) if need_scaffold_data else None)
        sti_gpu = (batch_data['scaffold_token_ids'].long().to(device) if args.position_scaffold else None)

        optimizer.zero_grad()

        (shape_embed, mu, log_var, z,
         UV_gs_recover, pg_features) = gs_autoencoder(
            UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:,:,:3],
            scaffold_anchors=sa_gpu, scaffold_token_ids=sti_gpu)

        mcp   = raw_model.shape_model.last_mean_color_pred
        ssp   = raw_model.shape_model.last_scene_semantic_pred
        anch  = raw_model.shape_model.last_predicted_anchors_from_tokens
        slp   = raw_model.shape_model.last_scene_layout_pred
        sgp   = raw_model.shape_model.last_seg_pred
        zsp   = raw_model.shape_model.last_z_s_infonce_proj
        _mu_s = raw_model.shape_model._mu_s_cache
        _mu_g = raw_model.shape_model._mu_g_cache

        # Target
        target_abs = UV_gs_batch[:, :, GEOMETRIC_INDICES]
        if args.position_scaffold:
            pos_off = batch_data['position_offsets'].float().to(device)
            target  = target_abs.clone(); target[:,:,0:3] = pos_off
            pred_3d = UV_gs_recover.reshape(B,-1,14).clone()
            if anch is not None:
                idx_3d = sti_gpu.unsqueeze(-1).expand(-1,-1,3)
                pred_3d[:,:,0:3] -= torch.gather(anch, 1, idx_3d)
        elif args.position_layout_residual:
            pos_res = batch_data['position_residuals'].float().to(device)
            target  = target_abs.clone(); target[:,:,0:3] = pos_res
            pred_3d = UV_gs_recover.reshape(B,-1,14)
        else:
            target  = target_abs
            pred_3d = UV_gs_recover.reshape(B,-1,14)

        # Losses
        recon_loss  = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight)
        KL_loss     = -0.5*torch.sum(1+log_var-mu.pow(2)-log_var.exp(), dim=1).mean()

        color_pred_loss = torch.tensor(0., device=device)
        if mcp is not None and args.color_residual:
            color_pred_loss = F.mse_loss(mcp, mean_color_gt)

        scene_sem_loss = torch.tensor(0., device=device)
        if ssp is not None and args.scene_semantic_head:
            p_s = batch_data['label_dist'].float().to(device)
            scene_sem_loss = scene_semantic_kl_loss(ssp, p_s)

        anchor_loss = torch.tensor(0., device=device)
        if anch is not None and args.position_scaffold and sa_gpu is not None:
            anchor_loss = F.mse_loss(anch, sa_gpu)

        layout_loss = torch.tensor(0., device=device)
        if slp is not None and args.scene_layout_head:
            gt_c = batch_data['category_centroids'].float().to(device)
            gt_v = batch_data['category_valid'].float().to(device)
            layout_loss = compute_layout_loss(slp, gt_c, gt_v)

        seg_pred_loss = torch.tensor(0., device=device)
        if args.predict_seg_labels and sgp is not None and seg_labels is not None:
            seg_pred_loss = compute_seg_pred_loss(sgp, seg_labels)

        # Per-Gaussian InfoNCE
        semantic_loss    = torch.tensor(0., device=device)
        semantic_metrics = {}
        if enable_semantic and seg_labels is not None and pg_features is not None:
            if args.semantic_mode == 'dist':
                semantic_loss, semantic_metrics = compute_distribution_loss(
                    dist_logits=pg_features, segment_labels=seg_labels,
                    weight=args.segment_loss_weight)
            else:
                semantic_loss, semantic_metrics = compute_semantic_loss(
                    embeddings=pg_features, segment_labels=seg_labels,
                    instance_labels=inst_labels, batch_size=B,
                    segment_weight=args.segment_loss_weight,
                    instance_weight=args.instance_loss_weight,
                    temperature=args.semantic_temperature,
                    subsample=args.semantic_subsample,
                    sampling_strategy=args.sampling_strategy)

        # Scene z_s InfoNCE
        z_s_nce_loss    = torch.tensor(0., device=device)
        z_s_nce_metrics = {'z_s_infonce_loss': 0., 'z_s_num_positives': 0., 'z_s_frac_anchors': 0.}
        if args.z_s_infonce_weight > 0 and zsp is not None:
            z_s_nce_loss, z_s_nce_metrics = compute_scene_infonce_loss(
                zsp, label_dist_v, args.z_s_infonce_temperature, args.z_s_infonce_delta)

        # z_s token InfoNCE (NEW — same mechanism as per-Gaussian)
        zs_tok_nce_loss    = torch.tensor(0., device=device)
        zs_tok_nce_metrics = {'zs_tok_infonce_loss': 0., 'zs_tok_num_categories': 0}
        if args.zs_token_infonce_weight > 0 and args.latent_disentangle:
            _n_tok        = args.semantic_dims // 32          # 16 for semantic_dims=512
            z_s_tokens    = z[:, :args.semantic_dims].reshape(B, _n_tok, 32)  # [B,16,32]
            zs_tok_nce_loss, zs_tok_nce_metrics = compute_zs_token_infonce_loss(
                z_s_tokens, label_dist_v, args.zs_token_infonce_temperature)

        # z_layout InfoNCE (Strategy B — same prototype mechanism as per-Gaussian)
        zs_lay_nce_loss    = torch.tensor(0., device=device)
        zs_lay_nce_metrics = {'zs_layout_infonce_loss': 0., 'zs_layout_num_cats': 0}
        z_lay_proj = raw_model.shape_model.last_z_layout_proj
        if args.zs_layout_infonce_weight > 0 and z_lay_proj is not None:
            zs_lay_nce_loss, zs_lay_nce_metrics = compute_zs_layout_infonce_loss(
                z_lay_proj, label_dist_v, args.zs_layout_infonce_temperature)

        # z_s pool InfoNCE — EXACT SAME mechanism as decoder hidden InfoNCE
        # head output: [B, 16, 32] L2-norm  (mirrors [B, 40000, 32] from decoder)
        # labels:      [B, 16]  dominant category broadcast to all 16 positions
        # loss:        compute_semantic_loss with same subsampling as decoder
        zs_pool_nce_loss    = torch.tensor(0., device=device)
        zs_pool_nce_metrics = {'zs_pool_infonce_loss': 0., 'zs_pool_num_cats': 0}
        if args.zs_pool_infonce_weight > 0:
            _pool_emb = raw_model.shape_model.last_zs_pool_proj
            if _pool_emb is None:
                _pool_emb = getattr(raw_model.shape_model,
                                    'last_z_layout_pool_proj', None)
            if _pool_emb is not None:
                # _pool_emb: [B, 16, 32] — same format as pg_features [B, 40000, 32]
                # Build labels: dominant category broadcast to all 16 positions
                _dom_cat = label_dist_v.float().argmax(dim=1)  # [B]
                _pool_labels = _dom_cat.unsqueeze(1).expand(
                    -1, _pool_emb.shape[1]).long()  # [B, 16]
                # Call EXACT same compute_semantic_loss as decoder InfoNCE
                # subsample/sampling_strategy args are identical
                zs_pool_nce_loss, _pool_metrics = compute_semantic_loss(
                    embeddings=_pool_emb,
                    segment_labels=_pool_labels,
                    instance_labels=None,
                    batch_size=B,
                    segment_weight=1.0,
                    instance_weight=0.0,
                    temperature=args.zs_pool_infonce_temperature,
                    subsample=_pool_emb.shape[1],   # 16 — no subsampling needed
                    sampling_strategy=args.sampling_strategy)
                zs_pool_nce_metrics = {
                    'zs_pool_infonce_loss': _pool_metrics.get('segment_loss', 0.),
                    'zs_pool_num_cats':     _pool_metrics.get('num_categories_in_batch', 0)}

        # Cross-reconstruction
        # With decoder_zs_cross_attn: we build z_cross = [z_s_B | z_g_A] reshaped to [B,512,32]
        # and call decode() — it will internally split into z_s (first 16 tokens) and z_g (last 496)
        # so the swapped z_s from scene B goes into the cross-attention conditioning
        # and z_g from scene A goes into the decoder sequence. Exactly what we want.
        cross_recon_loss = torch.tensor(0., device=device)
        if (args.latent_disentangle and args.cross_recon_weight > 0
                and _mu_s is not None and _mu_g is not None and B > 1):
            D_s = args.semantic_dims
            mu_s_shifted = torch.roll(_mu_s, shifts=1, dims=0)
            lv_s_shifted = torch.roll(log_var[:, :D_s], shifts=1, dims=0)
            z_s_swapped  = mu_s_shifted + torch.exp(0.5*lv_s_shifted) * torch.randn_like(mu_s_shifted)
            z_g_current  = _mu_g + torch.exp(0.5*log_var[:, D_s:]) * torch.randn_like(_mu_g)
            z_cross      = torch.cat([z_s_swapped, z_g_current], dim=-1)
            lat_cross    = z_cross.reshape(B, 512, 32)

            # Update layout pred for scene B before cross-recon decode
            if (raw_model.shape_model.scene_layout_module is not None and
                    args.semantic_token_heads):
                with torch.no_grad():
                    _ed = raw_model.shape_model.embed_dim
                    _sd = args.semantic_dims
                    if args.structured_layout_tokens:
                        # layout module expects tokens 9-15 only [B, 7*32=224]
                        _n_s   = raw_model.shape_model._n_sem_tokens  # 8
                        _start = _ed + _n_s * _ed   # 32 + 8*32 = 288
                        z_lay_B = z_s_swapped[:, _start:_sd]  # [B, 224]
                        raw_model.shape_model.last_scene_layout_pred =                             raw_model.shape_model.scene_layout_module(z_lay_B)
                    else:
                        # unstructured: layout module expects full tokens 1-15 [B, 480]
                        z_sem_B = z_s_swapped[:, _ed:_sd]     # [B, 480]
                        raw_model.shape_model.last_scene_layout_pred =                             raw_model.shape_model.scene_layout_module(z_sem_B)

            se_shifted = torch.roll(raw_model.shape_model._shape_embed_cache, shifts=1, dims=0)
            _mp = accelerator.mixed_precision
            _dtype = (torch.bfloat16 if _mp == 'bf16' else
                      torch.float16  if _mp == 'fp16' else torch.float32)
            # For Strategy B: shift z_layout as well so cross-recon uses shifted layout
            _z_layout_shifted = None
            _any_B_train = args.decoder_layout_cross_attn or args.decoder_layout_additive
            if _any_B_train and raw_model.shape_model.last_z_layout is not None:
                _z_layout_shifted = torch.roll(
                    raw_model.shape_model.last_z_layout, shifts=1, dims=0)
            with torch.autocast('cuda', dtype=_dtype, enabled=(_mp != 'no')):
                UV_cross, _ = raw_model.shape_model.decode(
                    lat_cross, volume_queries=None,
                    return_semantic_features=False, shape_embed=se_shifted,
                    scaffold_anchors=sa_gpu, scaffold_token_ids=sti_gpu,
                    z_layout=_z_layout_shifted)
            pred_cross_3d = UV_cross.reshape(B, -1, 14)

            if args.position_scaffold:
                cross_anch = raw_model.shape_model.last_predicted_anchors_from_tokens
                if cross_anch is not None and sti_gpu is not None:
                    idx_cr = sti_gpu.unsqueeze(-1).expand(-1,-1,3)
                    cross_dc = torch.gather(cross_anch, 1, idx_cr).detach()
                    pred_cross_for_loss = pred_cross_3d.clone()
                    pred_cross_for_loss[:,:,0:3] -= cross_dc
                else:
                    pred_cross_for_loss = pred_cross_3d
            else:
                pred_cross_for_loss = pred_cross_3d

            cross_recon_loss = compute_cross_recon_loss(pred_cross_for_loss, target, B)

            # Restore layout pred for scene A
            if (raw_model.shape_model.scene_layout_module is not None and
                    args.semantic_token_heads):
                raw_model.shape_model.last_scene_layout_pred = slp

        ortho_loss = torch.tensor(0., device=device)
        if (args.latent_disentangle and args.ortho_weight > 0
                and _mu_s is not None and _mu_g is not None and B > 1):
            ortho_loss = compute_orthogonality_loss(_mu_s, _mu_g)

        scale_pen = torch.tensor(0., device=device)
        if args.scale_penalty_weight > 0:
            scale_pen = compute_scale_penalty(UV_gs_recover.reshape(B,-1,14),
                                              threshold=args.scale_penalty_threshold)

        total_loss = (recon_loss
                      + args.kl_weight              * KL_loss
                      + args.mean_color_weight       * color_pred_loss
                      + args.scene_semantic_weight   * scene_sem_loss
                      + args.anchor_loss_weight      * anchor_loss
                      + args.layout_loss_weight      * layout_loss
                      + args.cross_recon_weight      * cross_recon_loss
                      + args.ortho_weight            * ortho_loss
                      + args.seg_pred_weight         * seg_pred_loss
                      + args.scale_penalty_weight    * scale_pen
                      + args.z_s_infonce_weight      * z_s_nce_loss
                      + args.zs_token_infonce_weight * zs_tok_nce_loss
                      + args.zs_layout_infonce_weight * zs_lay_nce_loss
                      + args.zs_pool_infonce_weight   * zs_pool_nce_loss
                      + semantic_loss)

        accelerator.backward(total_loss)
        optimizer.step()
        scheduler.step()

        ind = compute_individual_losses(pred_3d, target)
        e['loss']       += total_loss.item()
        e['recon']      += recon_loss.item()
        e['kl']         += KL_loss.item()
        e['sem']        += semantic_loss.item()
        e['color_pred'] += color_pred_loss.item()
        e['scene_sem']  += scene_sem_loss.item()
        e['anchor']     += anchor_loss.item()
        e['layout']     += layout_loss.item()
        e['cross_recon'] += cross_recon_loss.item()
        e['ortho']      += ortho_loss.item()
        e['seg_pred']   += seg_pred_loss.item()
        e['scale_pen']  += scale_pen.item()
        e['z_s_nce']    += z_s_nce_loss.item()
        e['z_s_npos']   += z_s_nce_metrics.get('z_s_num_positives', 0.)
        e['zs_tok_nce']   += zs_tok_nce_loss.item()
        e['zs_tok_ncats'] += zs_tok_nce_metrics.get('zs_tok_num_categories', 0)
        e['zs_lay_nce']    += zs_lay_nce_loss.item()
        e['zs_lay_ncats']  += zs_lay_nce_metrics.get('zs_layout_num_cats', 0)
        e['zs_pool_nce']   += zs_pool_nce_loss.item()
        e['zs_pool_ncats'] += zs_pool_nce_metrics.get('zs_pool_num_cats', 0)
        e['pos'] += ind['position']; e['col'] += ind['color']
        e['opa'] += ind['opacity'];  e['scl'] += ind['scale']
        e['rot'] += ind['rotation']

        if epoch == start_epoch and i_batch == 0 and accelerator.is_main_process:
            print(f"\nEPOCH {epoch} BATCH 0 DIAGNOSTIC:")
            print(f"  recon={recon_loss.item():.4f} | KL={KL_loss.item():.4f} | "
                  f"mu=[{mu.min().item():.3f},{mu.max().item():.3f}]")
            if args.decoder_zs_cross_attn:
                print(f"  [NEW DESIGN] z_g only in decoder sequence")
                print(f"  cross_recon={cross_recon_loss.item():.4f}  "
                      f"(gradient isolates z_g via architecture)")
            if args.z_s_infonce_weight > 0 and zsp is not None:
                print(f"  z_s_NCE={z_s_nce_loss.item():.4f}  "
                      f"n_pos={z_s_nce_metrics.get('z_s_num_positives',0):.1f}  "
                      f"frac_anch={z_s_nce_metrics.get('z_s_frac_anchors',0):.2f}")
            if args.zs_layout_infonce_weight > 0 and z_lay_proj is not None:
                print(f"  ZsLayNCE={zs_lay_nce_loss.item():.4f}  "
                      f"n_cats={zs_lay_nce_metrics.get('zs_layout_num_cats',0)}")
                print(f"  [Strategy B] z_layout from shape_embed → layout conditioning")
            if args.zs_token_infonce_weight > 0:
                print(f"  ZsTokNCE={zs_tok_nce_loss.item():.4f}  "
                      f"n_cats={zs_tok_nce_metrics.get('zs_tok_num_categories',0)}")
            if _mu_s is not None:
                print(f"  mu_s=[{_mu_s.min().item():.3f},{_mu_s.max().item():.3f}]  "
                      f"mu_g=[{_mu_g.min().item():.3f},{_mu_g.max().item():.3f}]")

        global_step += 1

    nb = len(trainDataLoader)
    lr_now = scheduler.get_last_lr()[0]
    if accelerator.is_main_process:
        print(f"\nEpoch {epoch:04d} | "
              f"Loss={e['loss']/nb:.4f} | "
              f"Recon={e['recon']/nb:.4f} | "
              f"KL={e['kl']/nb:.4f} | "
              f"ColorPred={e['color_pred']/nb:.6f} | "
              f"SceneSem={e['scene_sem']/nb:.4f} | "
              f"Layout={e['layout']/nb:.4f} | "
              f"CrossRecon={e['cross_recon']/nb:.4f} | "
              f"Ortho={e['ortho']/nb:.6f} | "
              f"Anchor={e['anchor']/nb:.4f} | "
              f"SegPred={e['seg_pred']/nb:.4f} | "
              f"ScalePen={e['scale_pen']/nb:.6f} | "
              f"Z_sNCE={e['z_s_nce']/nb:.4f} | "
              f"Z_sNPos={e['z_s_npos']/nb:.1f} | "
              f"ZsTokNCE={e['zs_tok_nce']/nb:.4f} | "
              f"ZsTokNCats={e['zs_tok_ncats']/nb:.1f} | "
              f"LayNCE={e['zs_lay_nce']/nb:.4f} | "
              f"LayNCats={e['zs_lay_ncats']/nb:.1f} | "
              f"PoolNCE={e['zs_pool_nce']/nb:.4f} | "
              f"PoolNCats={e['zs_pool_ncats']/nb:.1f} | "
              f"PgNCE={e['sem']/nb:.4f} | "
              f"LR={lr_now:.2e}")
        print(f"  Pos={e['pos']/nb:.3f} | Col={e['col']/nb:.3f} | "
              f"Opa={e['opa']/nb:.3f} | Scl={e['scl']/nb:.3f} | Rot={e['rot']/nb:.3f}")

    val_metrics = None
    if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:
        val_metrics = evaluate_model(gs_autoencoder, raw_model, valDataLoader,
                                     device, accelerator, epoch=epoch)
        if accelerator.is_main_process:
            print(f"\n--- Validation epoch {epoch} ---")
            print(f"  L2={val_metrics['avg_l2_error']:.4f}  "
                  f"Pos={val_metrics['position_loss']:.4f}  "
                  f"Col={val_metrics['color_loss']:.4f}  "
                  f"Opa={val_metrics['opacity_loss']:.4f}  "
                  f"Scl={val_metrics['scale_loss']:.4f}  "
                  f"Rot={val_metrics['rotation_loss']:.4f}")
            if args.color_residual:
                print(f"  ColorPredMSE={val_metrics['color_pred_loss']:.6f}")
            if args.scene_semantic_head:
                print(f"  SceneSemKL={val_metrics['scene_semantic_kl']:.4f}")
            if args.scene_layout_head:
                print(f"  LayoutMSE={val_metrics['layout_loss']:.4f}")
            if args.z_s_infonce_weight > 0:
                print(f"  Val Z_sNCE={val_metrics['z_s_infonce_loss']:.4f}")
            if args.zs_token_infonce_weight > 0:
                print(f"  Val ZsTokNCE={val_metrics['zs_tok_infonce_loss']:.4f}")
            if args.zs_layout_infonce_weight > 0:
                print(f"  Val LayNCE={val_metrics['zs_lay_infonce_loss']:.4f}")
            if args.zs_pool_infonce_weight > 0:
                print(f"  Val PoolNCE: see PoolNCE= in epoch log")

        if val_metrics['avg_l2_error'] < best_val_loss:
            best_val_loss = val_metrics['avg_l2_error']
            best_epoch    = epoch
            if accelerator.is_main_process:
                torch.save({
                    'epoch':                epoch,
                    'model_state_dict':     raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_l2_error':         val_metrics['avg_l2_error'],
                    **_ckpt_meta,
                }, os.path.join(save_path, "best_model.pth"))
                print(f"  [NEW BEST] L2={best_val_loss:.4f} saved")

    if epoch >= 10 and epoch % 500 == 0 and accelerator.is_main_process:
        torch.save({'epoch': epoch, 'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': e['loss']/nb, **_ckpt_meta},
                   os.path.join(save_path, f"epoch_{epoch}.pth"))

# ============================================================================
# FINAL SAVE
# ============================================================================
accelerator.wait_for_everyone()
final_metrics = evaluate_model(gs_autoencoder, raw_model, valDataLoader, device,
                               accelerator, epoch=args.num_epochs-1)
if accelerator.is_main_process:
    print(f"\nFinal L2: {final_metrics['avg_l2_error']:.4f}  "
          f"Best L2: {best_val_loss:.4f} (epoch {best_epoch})")
    torch.save({
        'epoch':            args.num_epochs - 1,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_l2':     final_metrics['avg_l2_error'],
        'best_val_l2':      best_val_loss,
        'best_epoch':       best_epoch,
        **_ckpt_meta,
        'individual_losses': {k: final_metrics[f'{k}_loss'] for k in PARAM_SLICES},
    }, os.path.join(save_path, "final.pth"))
    print(f"Saved: {save_path}final.pth")
if wandb_enabled and accelerator.is_main_process:
    wandb_run.summary.update({"final_val_l2": final_metrics['avg_l2_error'],
                               "best_val_l2": best_val_loss, "best_epoch": best_epoch})
    wandb_run.finish()
if accelerator.is_main_process: print("Done.")