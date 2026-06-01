"""
Can3Tok Training — MAIN NEW IDEA: decoder_zs_cross_attn
=========================================================
DATASET MODES (--train_data):
  "chunks"   — train_grid1.0cm_chunk8x8_stride6x6/ (default, 3888 chunks)
               Requires norm_factor.npy (run precompute_norm_from_chunks.py)
               Normalization: GLOBAL scene frame via norm_factor.npy
  "full"     — train/ (800 full scenes, per-scene normalization)
  "combined" — both sources concatenated (4688 total)
  Validation always uses val/ (100 held-out full scenes).

HELD-OUT CHUNK VALIDATION (when train_data="chunks" or "combined"):
  The first --train_scenes chunks (sorted) go to training.
  The remaining chunks (3888 - train_scenes) are held-out for chunk eval.
  gs_dataset skip_scenes parameter handles the split — no file changes needed.
  Normalization is IDENTICAL for train/val chunks: both use norm_factor.npy
  written by precompute_norm_from_chunks.py from the union of all chunks.

  This gives two evaluation streams every eval_every epochs:
    val_full  — 100 held-out full scenes  (primary metric, thesis target)
    val_chunk — held-out chunks            (in-distribution diagnostic)

  The gap  full_L2 / chunk_L2  quantifies the train→eval distribution shift.
  Best model checkpoint is saved on val_full (the thesis target).

BF16 MIXED PRECISION:
  torch.autocast wraps training forward, eval forward, cross-recon decode.
  Enable via mixed_precision: 'bf16' in accelerate_config.yaml.
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
    print("[WARNING] visualize_zs_tokens not found — z_s token PCA disabled.")
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
parser.add_argument('--kl_anneal_steps',      type=int,   default=0,
    help='Number of optimizer steps over which to ramp kl_weight from 0 to its '
         'target value (linear warm-up). 0 = no annealing (fixed kl_weight). '
         'Recommended: set to ~20× batches_per_epoch so the encoder builds a '
         'rough reconstruction prior before KL regularisation kicks in. '
         'Example: 4 GPUs, 3800 scenes, batch=90 → ~10 steps/epoch → '
         '2000 steps covers ~200 epochs of KL ramp.')
parser.add_argument('--weight_decay',         type=float, default=1e-2)
parser.add_argument('--warmup_steps',         type=int,   default=100)
parser.add_argument('--lr_min_ratio',         type=float, default=0.1)
parser.add_argument('--lr_restart_T0',        type=int,   default=0,
    help='Cosine warm restart period in EPOCHS. 0 = single cosine decay (old behaviour). '
         'When >0, LR decays from peak to floor over T0 epochs then rises back to peak '
         'and repeats. Proven in Run 3 (1500 chunks, T0=900) to achieve 4.5x better '
         'convergence than single cosine over same epoch budget. '
         'Recommended: ~800-1000 epochs for full-scene runs, ~500 for chunk runs.')
parser.add_argument('--eval_every',           type=int,   default=20)
parser.add_argument('--failure_threshold',    type=float, default=100.0)
parser.add_argument('--train_scenes',         type=int,   default=None)
parser.add_argument('--val_scenes',           type=int,   default=None)
parser.add_argument('--sampling_method',      type=str,   default='opacity',
                    choices=['random','opacity','hybrid'])
# ── Dataset source ────────────────────────────────────────────────────────────
parser.add_argument('--train_data',           type=str,   default='chunks',
                    choices=['chunks', 'full', 'combined'],
    help='"chunks" = train_grid/ (global norm via norm_factor.npy), '
         '"full" = train/ (per-scene norm), '
         '"combined" = both sources concatenated.')
# ── MAIN NEW IDEA ─────────────────────────────────────────────────────────────
parser.add_argument('--decoder_zs_cross_attn', action='store_true', default=False)
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
# ── Strategy B flags ──────────────────────────────────────────────────────────
parser.add_argument('--decoder_layout_cross_attn', action='store_true', default=False)
parser.add_argument('--decoder_layout_additive',   action='store_true', default=False)
parser.add_argument('--structured_layout_tokens',  action='store_true', default=False)
parser.add_argument('--zs_layout_infonce_weight',      type=float, default=0.0)
parser.add_argument('--zs_layout_infonce_temperature', type=float, default=0.07)
# ── z_s pool / token InfoNCE ──────────────────────────────────────────────────
parser.add_argument('--zs_pool_infonce_weight',      type=float, default=0.0)
parser.add_argument('--zs_pool_infonce_temperature', type=float, default=0.07)
parser.add_argument('--zs_token_infonce_weight',      type=float, default=0.0)
parser.add_argument('--zs_token_infonce_temperature', type=float, default=0.07)
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
# Legacy
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
chunk_norm_grp = parser.add_mutually_exclusive_group()
chunk_norm_grp.add_argument('--chunk_norm_factor', dest='chunk_norm_factor',
    action='store_true', default=True,
    help='[DEFAULT ON] Use norm_factor.npy global frame for grid chunks. '
         'All chunks of the same room share one norm_factor.npy computed from '
         'their union, so every chunk has the same coordinate system. This is '
         'critical for position loss convergence: without it each chunk is '
         'normalised into its own local sphere and gradient signals cancel. '
         'See normalize_with_norm_factor() in gs_dataset_scenesplat.py.')
chunk_norm_grp.add_argument('--no_chunk_norm_factor', dest='chunk_norm_factor',
    action='store_false',
    help='Disable norm_factor.npy for chunks: force per-scene normalisation '
         'even when norm_factor.npy is present. Ablation use only. '
         'Full scenes are unaffected regardless of this flag.')
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
# ── Multi-dataset support ─────────────────────────────────────────────────────
# Extra paths let you add ArkitScenes, ScanNet++, or any SceneSplat-format
# dataset on top of the main --train_data source. Semantics are disabled for
# extra paths because their label spaces differ from ScanNet72. Reconstruction
# and geometry losses are completely unaffected. All InfoNCE losses either
# filter zero-label scenes automatically (per-Gaussian, z_s scene) or are
# explicitly masked (pool NCE, token NCE — see _sem_valid in training loop).
parser.add_argument('--extra_train_paths',    type=str,   default='',
    help='Colon-separated list of extra scene root directories added on top of '
         '--train_data. Each must contain scene subdirs with the standard '
         'SceneSplat layout: coord.npy, color.npy, scale.npy, quat.npy, opacity.npy. '
         'Semantics disabled automatically (label_dist=zeros, segment=-1). '
         'Example: "/path/arkitscenes/train:/path/scannetpp/train"')
parser.add_argument('--extra_train_scenes',   type=str,   default='',
    help='Colon-separated max scenes per extra path (0 = all scenes). '
         'Must be empty or match the number of paths in --extra_train_paths. '
         'Example: "1290:906"')

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
        print("[INFO] zs_layout_infonce_weight > 0 with Strategy A: routing z_s as layout tokens")
    else:
        print("[WARNING] zs_layout_infonce_weight > 0 requires decoder_layout_* or latent_disentangle. Setting to 0.")
        args.zs_layout_infonce_weight = 0.0
if _any_B and args.latent_disentangle:
    print("[INFO] decoder_layout_cross/additive=True with latent_disentangle=True.")
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
# find_unused_parameters=FALSE is correct and required here.
#
# Root cause of the "marked ready twice" crash:
#   scene_layout_module is called TWICE inside a single gs_autoencoder() forward:
#     1. Encoder path:  scene_layout_module(z_s_tokens) → last_scene_layout_pred
#                       gradient flows: layout_loss → last_scene_layout_pred → slm
#     2. Decoder path:  decoder uses last_scene_layout_pred (or calls slm directly)
#                       for layout conditioning → UV_gs_recover → recon_loss → slm
#   Both paths contribute to total_loss, so backward() accumulates scene_layout_module
#   gradients from two sources. With find_unused_parameters=True, DDP registers an
#   AccumulateGrad hook per parameter that fires ONCE PER GRADIENT ACCUMULATION.
#   Two accumulations = two hook fires = "marked ready twice" crash.
#
# Why find_unused_parameters=False is SAFE:
#   All model parameters are used in every gs_autoencoder() forward — scene_semantic_head,
#   scene_layout_head, semantic_token_heads, fourier_pe, and the semantic projection head
#   all run unconditionally. The cross_recon block goes through raw_model (unwrapped),
#   which DDP cannot see, so it does not create conditional paths from DDP's perspective.
#   With find_unused_parameters=False, DDP does NOT register AccumulateGrad hooks.
#   Multiple gradient paths through the same parameter just accumulate normally. No crash.
#
# static_graph=False: kept False (True would require identical graph every iteration,
#   which is incompatible with the semantic sampling variability).
_ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=False)
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
        run_name += f"_{args.train_data}_inferencefixed"
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
tag += f"_{args.train_data}_inferencefixed"

save_path = f"/home/yli11/scratch-project/Hafeez_thesis/Can3Tok/checkpoints_stage1/{tag}/"
os.makedirs(save_path, exist_ok=True)

# ============================================================================
# STARTUP SUMMARY
# ============================================================================
if accelerator.is_main_process:
    print(f"\n{'='*70}")
    print(f"CAN3TOK — train_data='{args.train_data}'")
    print(f"  decoder_zs_cross_attn={args.decoder_zs_cross_attn}")
    print(f"  color_residual={args.color_residual}")
    print(f"  latent_disentangle={args.latent_disentangle} semantic_dims={args.semantic_dims}")
    print(f"  scene_layout_head={args.scene_layout_head}")
    print(f"  decoder_fourier_pe={args.decoder_fourier_pe}")
    print(f"  semantic_token_heads={args.semantic_token_heads}")
    print(f"  zs_pool_infonce_weight={args.zs_pool_infonce_weight}")
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
p.decoder_zs_cross_attn       = args.decoder_zs_cross_attn
p.decoder_layout_cross_attn   = args.decoder_layout_cross_attn
p.decoder_layout_additive     = args.decoder_layout_additive
p.structured_layout_tokens    = args.structured_layout_tokens
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
        print(f"  Architecture changed — loading strict=False")
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
    """Single cosine decay with linear warmup (original behaviour)."""
    cosine_steps = max(total_steps - warmup_steps, 1)
    def f(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(warmup_steps)
        t = step - warmup_steps
        return lr_min_ratio + (1-lr_min_ratio) * 0.5*(1 + math.cos(math.pi*t/cosine_steps))
    return f

def build_lr_lambda_restart(warmup_steps, restart_T0_steps, lr_min_ratio):
    """
    Cosine warm restarts with linear warmup.

    Proven superior in Run 3 (1500 chunks, T0≈900 epochs):
      - LR decays peak→floor over T0_steps after warmup, then RISES back to peak.
      - The re-ascent lets the optimizer escape whatever basin it settled in and
        find a lower minimum. Run 3 val L2=6.46 vs Run 1 (no restarts) val L2=29.04
        with LESS data and LOWER peak LR.
      - KL spikes during the rise (encoder re-explores latent space) then drops
        to a LOWER value than before — confirmed healthy re-exploration.

    Schedule:
      step < warmup_steps        : linear warmup  0 → 1.0
      step >= warmup_steps       : cosine cycles of length restart_T0_steps
                                   each cycle: lr_min_ratio → 1.0 (rising half)
                                               then 1.0 → lr_min_ratio (falling half)
    The cosine here uses a FULL cosine period (2π) per cycle so LR rises AND falls
    within one cycle — this matches CosineAnnealingWarmRestarts behaviour.
    """
    T = max(restart_T0_steps, 1)
    def f(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(warmup_steps)
        t = (step - warmup_steps) % T          # position within current cycle
        # Standard cosine: 1.0 at t=0, floor at t=T/2, 1.0 at t=T
        # This gives peak → floor → peak within one cycle.
        cosine_val = 0.5 * (1 + math.cos(math.pi * t / (T / 2)))
        return lr_min_ratio + (1 - lr_min_ratio) * cosine_val
    return f

# NOTE: _bpe, scheduler, and LR print are created AFTER datasets below
# so _bpe reflects the actual combined dataset size (main + extra_train_paths).

# ============================================================================
# DATASETS
# ============================================================================
# --train_data controls which scenes are used for TRAINING.
#
# NORMALIZATION (critical for correctness):
#   "chunks"   → norm_factor.npy present → GLOBAL scene frame
#                All 3888 chunks from the SAME scene share one norm_factor.npy
#                (written by precompute_norm_from_chunks.py from the union of
#                all chunk coords). Training chunks (first 3800 sorted) and
#                held-out val chunks (last 88 sorted) BOTH use the same
#                norm_factor.npy. The coordinate frame is fully consistent.
#
#   "full"     → norm_factor.npy absent → PER-SCENE fallback
#                Each full scene normalised independently. Correct for
#                train/ and val/ which contain complete room scans.
#
#   "combined" → chunks use global frame, full scenes use per-scene.
#
# VAL FULL SCENES (primary metric — thesis target):
#   Always val/ — 100 held-out full scenes, per-scene normalization.
#
# VAL HELD-OUT CHUNKS (in-distribution diagnostic):
#   When train_data="chunks" or "combined": the chunks sorted AFTER the
#   training portion (skip_scenes=train_scenes) are used as a second val set.
#   gs_dataset.skip_scenes handles the split — no file changes needed.
#   These chunks have norm_factor.npy → same global frame as training chunks.
# ============================================================================
from gs_dataset_scenesplat import gs_dataset

# Shared kwargs for all dataset instances
_ds_kwargs = dict(
    resol=200,
    sampling_method=args.sampling_method,
    normalize=args.use_canonical_norm,
    normalize_colors=args.normalize_colors,
    use_chunk_norm_factor=args.chunk_norm_factor,
    target_radius=10.0,
    scale_norm_mode=args.scale_norm_mode,
    label_input=args.label_input,
    color_residual=args.color_residual,
    position_scaffold=args.position_scaffold,
    scene_layout_head=args.scene_layout_head,
    jepa_idea1=args.jepa_idea1,
    position_layout_residual=args.position_layout_residual,
)

_chunk_root = os.path.join(data_path, "train_grid1.0cm_chunk8x8_stride6x6")
_full_root  = os.path.join(data_path, "train")

# ── Training dataset ──────────────────────────────────────────────────────────
if args.train_data == 'chunks':
    if accelerator.is_main_process:
        print(f"\n--- Training Dataset: CHUNKS ({_chunk_root}) ---")
    gs_dataset_train = gs_dataset(
        root=_chunk_root, random_permute=True, train=True,
        max_scenes=args.train_scenes, skip_scenes=None, **_ds_kwargs)
    # Record the actual number of chunks used for training (needed for skip_scenes below)
    _n_train_chunks = len(gs_dataset_train)

elif args.train_data == 'full':
    if accelerator.is_main_process:
        print(f"\n--- Training Dataset: FULL SCENES ({_full_root}) ---")
    gs_dataset_train = gs_dataset(
        root=_full_root, random_permute=True, train=True,
        max_scenes=args.train_scenes, skip_scenes=None, **_ds_kwargs)
    _n_train_chunks = 0   # no chunks used

else:  # combined
    _max_full  = max(1, args.train_scenes // 2) if args.train_scenes else None
    _max_chunk = (args.train_scenes - _max_full)  if args.train_scenes else None
    if accelerator.is_main_process:
        print(f"\n--- Training Dataset: COMBINED (full + chunks) ---")
    _ds_full  = gs_dataset(root=_full_root,  random_permute=True, train=True,
                           max_scenes=_max_full,  skip_scenes=None, **_ds_kwargs)
    _ds_chunk = gs_dataset(root=_chunk_root, random_permute=True, train=True,
                           max_scenes=_max_chunk, skip_scenes=None, **_ds_kwargs)
    gs_dataset_train = Data.ConcatDataset([_ds_full, _ds_chunk])
    _n_train_chunks  = len(_ds_chunk)

# ── Val full scenes (PRIMARY — thesis target) ─────────────────────────────────
if accelerator.is_main_process:
    print(f"\n--- Validation Dataset: val/ (held-out full scenes) ---")
gs_dataset_val = gs_dataset(
    root=os.path.join(data_path, "val"),
    random_permute=False, train=False,
    max_scenes=args.val_scenes, skip_scenes=None, **_ds_kwargs)

# ── Val held-out chunks (IN-DISTRIBUTION DIAGNOSTIC) ──────────────────────────
# These are the chunks sorted AFTER the training portion. They were never seen
# during training. Because all chunks share norm_factor.npy computed from the
# full scene union, held-out chunks use the SAME global coordinate frame as the
# training chunks → normalization is fully consistent, no file changes needed.
#
# Gap metric: full_L2 / chunk_L2
#   ≈ 1.0  → distribution shift is negligible
#   >> 1.0 → chunks are much easier; the model generalises poorly to full scenes
#             (which is the point — it quantifies what we knew qualitatively)
gs_dataset_val_chunk  = None
valChunkDataLoader    = None
_has_chunk_val        = False

if args.train_data in ('chunks', 'combined') and _n_train_chunks > 0:
    if accelerator.is_main_process:
        print(f"\n--- Validation Dataset: held-out chunks "
              f"(skip_scenes={_n_train_chunks}) ---")
    try:
        gs_dataset_val_chunk = gs_dataset(
            root=_chunk_root,
            random_permute=False, train=False,
            skip_scenes=_n_train_chunks,   # skip the first _n_train_chunks (training)
            max_scenes=None,               # all remaining chunks
            **_ds_kwargs)
        if len(gs_dataset_val_chunk) > 0:
            _has_chunk_val = True
        else:
            if accelerator.is_main_process:
                print(f"  [INFO] No held-out chunks available "
                      f"(train_scenes={_n_train_chunks} used all chunks). "
                      f"Chunk val disabled.")
            gs_dataset_val_chunk = None
    except Exception as e:
        if accelerator.is_main_process:
            print(f"  [WARNING] Could not create held-out chunk val dataset: {e}")
        gs_dataset_val_chunk = None

# ── Extra training datasets (multi-path support) ──────────────────────────────
# Scenes from extra paths use disable_semantics=True because ArkitScenes (137
# classes) and ScanNet++ (100 classes) use label spaces incompatible with the
# ScanNet72 model heads. Setting disable_semantics=True forces label_dist=zeros
# and segment=-1 for those scenes, which causes all InfoNCE losses to return 0
# for them automatically (see _sem_valid masking in the training loop for pool
# and token NCE which otherwise would misuse argmax(zeros)=0 as a category).
# All reconstruction and geometry losses are fully unaffected.
_extra_train_datasets = []
_extra_path_list      = []
_extra_n_scenes_map   = {}   # path → scene count (for summary print)

if args.extra_train_paths:
    _raw_paths  = [p.strip() for p in args.extra_train_paths.split(':') if p.strip()]
    _raw_scenes = ([s.strip() for s in args.extra_train_scenes.split(':') if s.strip()]
                   if args.extra_train_scenes else [])
    # Pad scene counts with '0' (= all scenes) when fewer entries than paths
    while len(_raw_scenes) < len(_raw_paths):
        _raw_scenes.append('0')
    _raw_scenes = _raw_scenes[:len(_raw_paths)]

    for _ep, _es_str in zip(_raw_paths, _raw_scenes):
        _max_s = (int(_es_str) if _es_str and _es_str != '0' else None)
        if accelerator.is_main_process:
            print(f"\n--- Extra Training Dataset: {os.path.basename(_ep)} ---")
            print(f"    Path       : {_ep}")
            print(f"    Max scenes : {'all' if _max_s is None else _max_s}")
            print(f"    Semantics  : disabled (label_dist=zeros, segment=-1)")
        try:
            _extra_ds = gs_dataset(
                root=_ep, random_permute=True, train=True,
                max_scenes=_max_s, skip_scenes=None,
                disable_semantics=True,
                **_ds_kwargs)
            _extra_train_datasets.append(_extra_ds)
            _extra_path_list.append(_ep)
            _extra_n_scenes_map[_ep] = len(_extra_ds)
        except Exception as _exc:
            if accelerator.is_main_process:
                print(f"  [WARNING] Could not load extra dataset at {_ep}: {_exc}  (skipping)")

# Combine main training dataset with all extra datasets
if _extra_train_datasets:
    _gs_dataset_train_combined = Data.ConcatDataset(
        [gs_dataset_train] + _extra_train_datasets)
else:
    _gs_dataset_train_combined = gs_dataset_train

# ── Scheduler (created here so _bpe uses the actual combined dataset size) ────
# When extra_train_paths are used, the combined dataset is larger than the main
# dataset alone. Creating the scheduler here gives cosine decay the correct
# total_steps so the LR floor is reached at epoch num_epochs, not earlier.
_bpe = max(1, math.ceil(
    len(_gs_dataset_train_combined) / (args.batch_size * accelerator.num_processes)))
_total_steps  = _bpe * args.num_epochs
_elapsed      = _bpe * start_epoch

if args.lr_restart_T0 > 0:
    # ── Cosine warm restarts ──────────────────────────────────────────────────
    # T0 is specified in EPOCHS; convert to optimizer steps.
    # _elapsed warmup adjustment: if we're resuming mid-run, the current cycle
    # position is (_elapsed % T0_steps). The lambda handles this via modulo
    # so we pass the CURRENT step offset, not total elapsed.
    _restart_T0_steps = args.lr_restart_T0 * _bpe
    # For warmup: only apply warmup on the very first cycle (step 0 to warmup_steps).
    # If resuming after the warmup is already done, warmup_steps_adjusted=0.
    _warmup_adj = max(0, args.warmup_steps - _elapsed)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda_restart(
            warmup_steps=_warmup_adj,
            restart_T0_steps=_restart_T0_steps,
            lr_min_ratio=args.lr_min_ratio))
    if accelerator.is_main_process:
        print(f"\n  LR: peak={args.lr:.2e} | floor={args.lr*args.lr_min_ratio:.2e}")
        print(f"  Scheduler: COSINE WARM RESTARTS  T0={args.lr_restart_T0} epochs "
              f"({_restart_T0_steps} steps)  _bpe={_bpe}")
        print(f"  Restart cycle: peak→floor→peak every {args.lr_restart_T0} epochs")
        print(f"  Expected restarts over {args.num_epochs} epochs: "
              f"{args.num_epochs // args.lr_restart_T0}")
else:
    # ── Original single cosine decay ─────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=build_lr_lambda(
        warmup_steps=max(0, args.warmup_steps - _elapsed),
        total_steps=max(_total_steps - _elapsed, 1),
        lr_min_ratio=args.lr_min_ratio))
    if accelerator.is_main_process:
        print(f"\n  LR: peak={args.lr:.2e} | floor={args.lr*args.lr_min_ratio:.2e}")
        print(f"  Scheduler: single cosine  _bpe={_bpe}  total_steps={_total_steps}  "
              f"combined_train_scenes={len(_gs_dataset_train_combined)}")

# ── DataLoaders ───────────────────────────────────────────────────────────────
trainDataLoader = Data.DataLoader(
    dataset=_gs_dataset_train_combined, batch_size=args.batch_size,
    shuffle=True, num_workers=9, pin_memory=True, persistent_workers=True)

valDataLoader = Data.DataLoader(
    dataset=gs_dataset_val, batch_size=args.batch_size,
    shuffle=False, num_workers=9, pin_memory=True, persistent_workers=True)

if _has_chunk_val:
    valChunkDataLoader = Data.DataLoader(
        dataset=gs_dataset_val_chunk, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

# ============================================================================
# NORMALIZATION VERIFICATION
# ============================================================================
# Confirms that norm_factor.npy is present and consistent across all splits.
# Run this check BEFORE accelerator.prepare to use raw dataset attributes.
# ============================================================================
if accelerator.is_main_process:
    print(f"\n{'='*70}")
    print(f"  NORMALIZATION VERIFICATION")
    print(f"{'='*70}")

    def _check_nf(label, dirs, expected_present):
        """Check norm_factor.npy presence and spot-check consistency."""
        sample = min(50, len(dirs))
        nf_ok  = sum(1 for d in dirs[:sample]
                     if os.path.exists(os.path.join(d, 'norm_factor.npy')))
        if expected_present:
            status = ('✓ ALL PRESENT — global frame' if nf_ok == sample
                      else f'✗ MISSING in {sample-nf_ok}/{sample} — position WILL NOT converge!')
        else:
            status = ('✓ ABSENT — per-scene fallback (correct for full scenes)'
                      if nf_ok == 0 else f'present in {nf_ok}/{sample} (unusual but OK)')
        print(f"  {label:<30s}: {nf_ok}/{sample}  {status}")
        # Spot-check: if chunks, verify two chunks from same scene share the same nf
        if expected_present and nf_ok >= 2:
            _ex = dirs[0]
            _nf = np.load(os.path.join(_ex, 'norm_factor.npy'))
            print(f"    Example {os.path.basename(_ex)}: "
                  f"center=({_nf[0]:.3f},{_nf[1]:.3f},{_nf[2]:.3f}) "
                  f"scale={_nf[3]:.4f}")
        return nf_ok == sample if expected_present else True

    _is_chunks_train = args.train_data in ('chunks', 'combined')

    # Training chunks
    if args.train_data == 'chunks':
        _ok_train = _check_nf("Training chunks", gs_dataset_train.scene_dirs, True)
    elif args.train_data == 'combined':
        _ok_train = _check_nf("Training chunks (combined)", _ds_chunk.scene_dirs, True)
        _check_nf("Training full scenes (combined)", _ds_full.scene_dirs, False)
    else:
        _check_nf("Training full scenes", gs_dataset_train.scene_dirs, False)

    # Val full scenes
    _check_nf("Val full scenes (primary)", gs_dataset_val.scene_dirs, False)

    # Val held-out chunks
    if _has_chunk_val:
        _ok_chunk_val = _check_nf("Val held-out chunks", gs_dataset_val_chunk.scene_dirs, True)
        # Cross-check: verify a held-out chunk and a training chunk from the same
        # parent scene share the same norm_factor (the whole point of the fix)
        _train_dirs = (gs_dataset_train.scene_dirs if args.train_data == 'chunks'
                       else _ds_chunk.scene_dirs)
        _train_bases = {os.path.basename(d).rsplit('_', 1)[0] for d in _train_dirs}
        _found_cross = False
        for val_dir in gs_dataset_val_chunk.scene_dirs[:20]:
            _scene_id = os.path.basename(val_dir).rsplit('_', 1)[0]
            if _scene_id in _train_bases:
                # Find a training chunk from the same scene
                _train_match = next(
                    (d for d in _train_dirs if os.path.basename(d).startswith(_scene_id)),
                    None)
                if _train_match:
                    nf_train = np.load(os.path.join(_train_match, 'norm_factor.npy'))
                    nf_val   = np.load(os.path.join(val_dir,      'norm_factor.npy'))
                    _match   = np.allclose(nf_train, nf_val, atol=1e-5)
                    print(f"  Cross-check (same scene, train vs val chunk):")
                    print(f"    Train: {os.path.basename(_train_match)} "
                          f"scale={nf_train[3]:.4f}")
                    print(f"    Val  : {os.path.basename(val_dir)} "
                          f"scale={nf_val[3]:.4f}")
                    print(f"    norm_factor match: {'✓ IDENTICAL' if _match else '✗ DIFFER — BUG!'}")
                    _found_cross = True
                    break
        if not _found_cross:
            print(f"  Cross-check: no overlapping scene found between train/val chunks "
                  f"(expected if train used all chunks of those scenes)")

    print(f"{'='*70}\n")

# ============================================================================
# DATASET SUMMARY
# ============================================================================
if accelerator.is_main_process:
    _n_train_main  = len(gs_dataset_train)
    _n_train_extra = sum(len(d) for d in _extra_train_datasets)
    _n_train_total = len(_gs_dataset_train_combined)
    n_val          = len(gs_dataset_val)
    print(f"{'='*70}")
    print(f"  DATASET SUMMARY  (train_data='{args.train_data}')")
    print(f"{'='*70}")
    if _extra_train_datasets:
        print(f"  Training scenes    : {_n_train_total}  "              f"({_n_train_main} main  +  {_n_train_extra} extra)")
        for _ep in _extra_path_list:
            print(f"    + {os.path.basename(_ep)}: {_extra_n_scenes_map[_ep]} scenes  "                  f"(semantics disabled)")
    else:
        print(f"  Training scenes    : {_n_train_main}")
    print(f"  Val full scenes    : {n_val}  (PRIMARY — thesis target)")
    if _has_chunk_val:
        print(f"  Val held-out chunks: {len(gs_dataset_val_chunk)}  "              f"(in-distribution diagnostic; skipped first {_n_train_chunks})")
    else:
        print(f"  Val held-out chunks: N/A  (train_data='{args.train_data}' "              f"or all chunks used for training)")
    if args.train_data in ('chunks', 'combined'):
        _cnf_str = "norm_factor.npy GLOBAL frame ✓" if args.chunk_norm_factor else "per-scene fallback (--no_chunk_norm_factor)"
        print(f"  Chunk norm mode    : {_cnf_str}")
    print(f"  Batches/epoch      : {_bpe}  "          f"(batch={args.batch_size} × {accelerator.num_processes} GPUs)")
    print(f"{'='*70}\n")

# ============================================================================
# ACCELERATE PREPARE
# ============================================================================
if _has_chunk_val:
    (gs_autoencoder, optimizer, trainDataLoader,
     valDataLoader, valChunkDataLoader, scheduler) = accelerator.prepare(
        gs_autoencoder, optimizer, trainDataLoader,
        valDataLoader, valChunkDataLoader, scheduler)
else:
    (gs_autoencoder, optimizer, trainDataLoader,
     valDataLoader, scheduler) = accelerator.prepare(
        gs_autoencoder, optimizer, trainDataLoader,
        valDataLoader, scheduler)

raw_model = accelerator.unwrap_model(gs_autoencoder)

# ============================================================================
# MIXED PRECISION SETUP
# ============================================================================
_mp             = accelerator.mixed_precision
_autocast_dtype = (torch.bfloat16 if _mp == 'bf16' else
                   torch.float16  if _mp == 'fp16' else torch.float32)
_use_autocast   = (_mp != 'no')
if accelerator.is_main_process:
    print(f"\n{'='*70}")
    print(f"  GPU / COMPUTE SETUP")
    print(f"{'='*70}")
    print(f"  Num GPUs (accelerator processes) : {accelerator.num_processes}")
    print(f"  Distributed type                 : {accelerator.distributed_type}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem_gb = props.total_memory / (1024 ** 3)
            print(f"  GPU {i}: {props.name}  |  {total_mem_gb:.1f} GB VRAM  "
                  f"|  SM {props.major}.{props.minor}  "
                  f"|  {props.multi_processor_count} SMs")
    else:
        print(f"  CUDA not available — running on CPU")
    print(f"  Mixed precision : {_mp}")
    print(f"  Autocast dtype  : {_autocast_dtype}")
    print(f"  Autocast enabled: {_use_autocast}")
    print(f"{'='*70}\n")

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
    'decoder_zs_cross_attn':      args.decoder_zs_cross_attn,
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
    'chunk_norm_factor':          args.chunk_norm_factor,
    'scale_norm_mode':            args.scale_norm_mode,
    'train_data':                 args.train_data,
    'n_train_chunks':             _n_train_chunks,
    'kl_anneal_steps':            args.kl_anneal_steps,
}

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def evaluate_model(model, raw_model, dataloader, device, accelerator,
                   epoch=None, do_vis=True):
    """
    Evaluate the autoencoder on a dataloader.

    Parameters
    ----------
    do_vis : bool
        Whether to save PLY / PCA visualisations. Pass False for the
        held-out chunk eval to avoid doubling visualisation overhead.
    """
    model.eval()
    _eval_dtype    = _autocast_dtype
    _eval_autocast = _use_autocast

    total_l2 = total_kl = total_color = total_scene_sem = 0.0
    total_anchor = total_layout = total_seg = total_z_s_nce = total_zs_tok_nce = total_zs_lay_nce = 0.0
    per_param    = {k: 0.0 for k in PARAM_SLICES}
    n_scenes     = 0

    recon_preds  = []; recon_means  = []
    pca_input    = []; pca_recon    = []
    pca_sem_feat = []
    z_s_proj_acc = []; label_dist_acc = []
    zs_tokens_acc = []; zs_layout_acc = []; zs_pool_acc = []

    # Visualisation only on full-scene val and only on the scheduled epochs
    _do_vis    = do_vis
    do_recon   = (_do_vis and epoch is not None and epoch % args.recon_ply_freq == 0)
    do_pca     = (_do_vis and epoch is not None and epoch % args.pca_vis_freq   == 0)
    do_sem_pca = (do_pca and enable_semantic)
    do_z_s_vis     = (do_pca and raw_model.shape_model.z_s_infonce_head is not None)
    do_zs_tok_vis  = (do_pca and args.zs_token_infonce_weight > 0 and args.latent_disentangle)
    _any_B_eval    = args.decoder_layout_cross_attn or args.decoder_layout_additive
    do_zs_lay_vis  = (do_pca and _any_B_eval)
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

            with torch.autocast('cuda', dtype=_eval_dtype, enabled=_eval_autocast):
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

            pred_abs   = UV_gs_recover.reshape(B,-1,14)
            recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight)
            kl_loss    = -0.5*torch.sum(1+log_var - mu.pow(2) - log_var.exp(), dim=1)

            if mcp is not None and args.color_residual:
                total_color += F.mse_loss(mcp.float(), mean_color_gt).item() * B
            if ssp is not None and args.scene_semantic_head:
                p_s = batch_data['label_dist'].float().to(device)
                total_scene_sem += scene_semantic_kl_loss(ssp.float(), p_s).item() * B
            if anch is not None and args.position_scaffold:
                total_anchor += F.mse_loss(anch.float(), sa_gpu).item() * B
            if slp is not None and args.scene_layout_head:
                gt_c = batch_data['category_centroids'].float().to(device)
                gt_v = batch_data['category_valid'].float().to(device)
                total_layout += compute_layout_loss(slp.float(), gt_c, gt_v).item() * B
            if args.predict_seg_labels and sgp is not None:
                total_seg += compute_seg_pred_loss(
                    sgp, batch_data['segment_labels'].long().to(device)).item() * B

            z_s_tokens_eval = None
            if args.latent_disentangle and args.semantic_dims > 0:
                _n_tok = args.semantic_dims // 32
                z_s_tokens_eval = z.reshape(B, -1, 32)[:, :_n_tok, :].detach()
            if args.zs_token_infonce_weight > 0 and z_s_tokens_eval is not None:
                zl_tok, _ = compute_zs_token_infonce_loss(
                    z_s_tokens_eval, label_dist_v, args.zs_token_infonce_temperature)
                total_zs_tok_nce += zl_tok.item() * B

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
                _pos_abs_min  = pred_abs[:,:,0:3].cpu().float().min().item()
                _pos_abs_max  = pred_abs[:,:,0:3].cpu().float().max().item()
                _pos_gt_range = (UV_gs_batch[:,:,4:7].cpu().max()-UV_gs_batch[:,:,4:7].cpu().min()).item()/2

            ind = compute_individual_losses(pred_3d, target)
            for k in per_param: per_param[k] += ind[k]

            if do_recon and len(recon_preds) < args.recon_ply_num_scenes:
                pnp = pred_abs.cpu().float().numpy(); mnp = mean_color_gt.cpu().numpy()
                for si in range(B):
                    if len(recon_preds) >= args.recon_ply_num_scenes: break
                    recon_preds.append(pnp[si]); recon_means.append(mnp[si])

            if do_pca and len(pca_input) < args.pca_num_scenes:
                for si in range(B):
                    if len(pca_input) >= args.pca_num_scenes: break
                    pca_input.append(UV_gs_batch.cpu().numpy()[si])
                    pca_recon.append(pred_abs.cpu().float().numpy()[si])
                    if do_sem_pca and pg_feats is not None:
                        pca_sem_feat.append(pg_feats.cpu().float().numpy()[si])

            if do_z_s_vis and zsp is not None:
                z_s_proj_acc.append(zsp.detach().cpu().float().numpy())
                label_dist_acc.append(label_dist_v.cpu().numpy())
            if do_zs_tok_vis and z_s_tokens_eval is not None:
                zs_tokens_acc.append(z_s_tokens_eval.cpu().float().numpy())
                if not do_z_s_vis:
                    label_dist_acc.append(label_dist_v.cpu().numpy())

            z_lay_raw_eval = raw_model.shape_model.last_z_layout
            if do_zs_lay_vis and z_lay_raw_eval is not None:
                zs_layout_acc.append(z_lay_raw_eval.detach().cpu().float().numpy())
                if not do_z_s_vis and not do_zs_tok_vis:
                    label_dist_acc.append(label_dist_v.cpu().numpy())

            if do_zs_pool_vis:
                _ph = getattr(raw_model.shape_model, 'last_zs_pool_hidden', None)
                if _ph is None:
                    _ph = getattr(raw_model.shape_model, 'last_z_layout_pool_hidden', None)
                if _ph is not None:
                    zs_pool_acc.append(_ph.detach().cpu().float().numpy())
                    if not label_dist_acc:
                        label_dist_acc.append(label_dist_v.cpu().numpy())

    # ── PLY / PCA saves (full-scene val only) ─────────────────────────────────
    if do_recon and recon_preds and accelerator.is_main_process:
        try:
            all_preds = np.stack(recon_preds, 0)
            if args.color_residual:
                for si in range(len(all_preds)):
                    all_preds[si,:,3:6] = np.clip(all_preds[si,:,3:6] + recon_means[si], 0, 1)
            recon_dir = Path(save_path)/"reconstructed_gaussians"/f"epoch_{epoch:03d}"
            save_reconstructed_gaussians(predictions=all_preds, output_dir=recon_dir,
                epoch=epoch, num_scenes=len(all_preds),
                max_sh_degree=args.recon_ply_max_sh, color_mode="1")
        except Exception as e: print(f"  PLY error: {e}")

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

    if do_z_s_vis and z_s_proj_acc and accelerator.is_main_process:
        try:
            all_z_s = np.concatenate(z_s_proj_acc, 0)
            all_ld  = np.concatenate(label_dist_acc, 0)
            vis_dir = Path(save_path)/"pca_visualisations"
            vis_dir.mkdir(parents=True, exist_ok=True)
            out = visualize_z_s_space(all_z_s, all_ld,
                str(vis_dir/f"z_s_space_epoch_{epoch:03d}.ply"), verbose=True)
            if out: print(f"  z_s space PLY: {out}")
        except Exception as e: print(f"  z_s vis error: {e}")

    if do_zs_tok_vis and zs_tokens_acc and accelerator.is_main_process:
        try:
            all_toks = np.concatenate(zs_tokens_acc, axis=0)
            all_ld   = np.concatenate(label_dist_acc, axis=0)
            vis_dir  = Path(save_path) / "pca_visualisations"
            vis_dir.mkdir(parents=True, exist_ok=True)
            out_tok = visualize_zs_tokens(zs_tokens=all_toks, label_dists=all_ld,
                output_path=str(vis_dir / f"zs_tokens_epoch_{epoch:03d}.ply"), verbose=True)
            if out_tok: print(f"  z_s token PLY: {out_tok}")
        except Exception as e: print(f"  z_s token vis error: {e}")

    if do_zs_lay_vis and zs_layout_acc and accelerator.is_main_process:
        try:
            all_lay = np.concatenate(zs_layout_acc, axis=0)
            all_ld  = np.concatenate(label_dist_acc, axis=0) if label_dist_acc else None
            if all_ld is not None:
                vis_dir = Path(save_path) / "pca_visualisations"
                vis_dir.mkdir(parents=True, exist_ok=True)
                out_lay = visualize_zs_tokens(zs_tokens=all_lay, label_dists=all_ld,
                    output_path=str(vis_dir / f"zs_layout_epoch_{epoch:03d}.ply"), verbose=True)
                if out_lay: print(f"  z_layout PLY: {out_lay}")
        except Exception as e: print(f"  z_layout vis error: {e}")

    if do_zs_pool_vis and zs_pool_acc and accelerator.is_main_process:
        try:
            all_pool = np.concatenate(zs_pool_acc, axis=0)
            all_ld   = np.concatenate(label_dist_acc, axis=0) if label_dist_acc else None
            if all_ld is not None:
                vis_dir = Path(save_path) / 'pca_visualisations'
                vis_dir.mkdir(parents=True, exist_ok=True)
                out_pool = visualize_z_s_space(z_s_proj=all_pool, label_dists=all_ld,
                    output_path=str(vis_dir / f'zs_pool_epoch_{epoch:03d}.ply'), verbose=True)
                if out_pool: print(f'  z_s pool PLY: {out_pool}')
        except Exception as e: print(f'  z_s pool vis error: {e}')

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
        'zs_pool_infonce_loss': 0.0,
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

# ── KL annealing ──────────────────────────────────────────────────────────────
# When kl_anneal_steps > 0: kl_weight ramps linearly from 0 → args.kl_weight
# over the first kl_anneal_steps optimizer steps, then holds at args.kl_weight.
#
# Why this prevents the epoch-50 KL explosion:
#   Without annealing the encoder has kl_anneal_steps=0 → full KL penalty from
#   step 0, but the gradient is so small (kl_weight=5e-5) that the encoder
#   ignores it for 40+ epochs. By epoch 46 the posterior is very non-Gaussian
#   (high mutual information) and the suddenly-relevant KL explodes to 75,502.
#
#   With annealing the encoder starts with KL_weight=0 → builds a reconstruction
#   prior with zero regularisation. As kl_weight ramps up, the encoder receives
#   an ever-growing gradient signal and adjusts gradually rather than all at once.
#   The KL rises and falls smoothly instead of spiking.
#
# Recommended value: kl_anneal_steps = 20 × batches_per_epoch
#   4 GPU, 3800 scenes, batch=90 → ~10 steps/epoch → 2000 steps = 200 epoch ramp
#   1 GPU, 3800 scenes, batch=90 → ~42 steps/epoch → 2000 steps ≈ 48 epoch ramp
#
# _kl_step_offset accounts for resumed training so the ramp is relative to
# the total steps taken across all runs, not just this run's steps.
_kl_anneal_active = (args.kl_anneal_steps > 0)
_kl_step_offset   = _bpe * start_epoch  # steps already taken before this run

if accelerator.is_main_process:
    print(f"  KL annealing : {'ENABLED' if _kl_anneal_active else 'DISABLED (fixed kl_weight)'}")
    if _kl_anneal_active:
        _ramp_epochs = args.kl_anneal_steps / max(_bpe, 1)
        print(f"  kl_anneal_steps={args.kl_anneal_steps}  "
              f"(≈ {_ramp_epochs:.0f} epochs at {_bpe} steps/epoch)")
        print(f"  kl_weight ramps: 0.0 → {args.kl_weight:.1e} over first {args.kl_anneal_steps} steps")
    else:
        print(f"  kl_weight fixed at {args.kl_weight:.1e} throughout")
    print()

global_step = _kl_step_offset  # continue counting from where we left off

# ── DDP FIX: scene_layout_module visibility hook ──────────────────────────────
#
# ROOT CAUSE (confirmed from model source code):
#
#   In Strategy A with token_cond=False, scene_layout_module (slm) is called
#   ONCE in forward() — in the structured_layout_tokens encoder branch:
#     self.last_scene_layout_pred = self.scene_layout_module(z_lay)
#
#   The output (last_scene_layout_pred / slp) is stored as a model attribute.
#   It is NOT returned in the 6-tuple (shape_embed, mu, log_var, z,
#   UV_gs_recover, per_gaussian_features).
#
#   DDP's prepare_for_backward() traverses only the 6 return tensors to find
#   which parameters were used. slp is NOT reachable from any of them.
#   → DDP marks slm as UNUSED and IMMEDIATELY PRE-FIRES "ready" for its params.
#
#   The training code then computes:
#     slp  = raw_model.shape_model.last_scene_layout_pred  (has grad_fn → slm)
#     layout_loss = compute_layout_loss(slp, gt_c, gt_v)
#   total_loss.backward() fires AccumulateGrad for slm → "ready" AGAIN.
#   → TWO "ready" signals → "marked ready twice" → crash.
#
# FIX: register a forward hook on raw_model.shape_model that adds
#   per_gaussian_features + slp.sum() * 0.0  (a zero-gradient graph path).
#
#   Why this works:
#   (a) "slp.sum() * 0.0" has grad_fn=MulBackward (graph path exists, value=0).
#   (b) DDP's prepare_for_backward traverses pf_modified and finds slm.
#       → DDP marks slm as USED → registers AccumulateGrad hook; no pre-fire.
#   (c) During backward, layout_loss and the zero-path BOTH trace through the
#       same slp tensor. PyTorch's autograd engine processes slp ONCE (summing
#       gradients from both consumers). AccumulateGrad for slm fires ONCE.
#       DDP hook fires ONCE. No crash.
#   (d) Gradient to slm = layout_loss gradient + 0 = layout_loss gradient.
#       Training is completely unaffected.

if raw_model.shape_model is not None:
    # ALL side-head modules store their outputs as model attributes (last_XXX) that
    # are NOT in the 6-tuple returned by forward(). DDP's prepare_for_backward()
    # traverses only the 6 return tensors, so it marks every side head as UNUSED
    # and pre-fires "ready" for its parameters. Then training losses (layout_loss,
    # semantic_loss, pool_nce_loss, etc.) trace backward through those same params,
    # firing "ready" a second time → "marked ready twice" crash.
    #
    # Fix: after forward() returns, add zero-gradient connections from every cached
    # side-head output to per_gaussian_features. "pred.sum() * 0.0" has value=0 but
    # a live grad_fn, so DDP's graph traversal reaches each side-head module and marks
    # it USED. During backward, all paths (loss path + zero path) share the SAME cached
    # tensor, so the autograd engine processes it once → AccumulateGrad fires ONCE.
    _SIDE_HEAD_ATTRS = [
        'last_mean_color_pred',     # mean_color_head
        'last_scene_semantic_pred', # scene_semantic_module
        'last_scene_layout_pred',   # scene_layout_module
        'last_z_s_infonce_proj',    # z_s_infonce_head
        'last_zs_pool_proj',        # zs_pool_proj_head (embeddings)
        'last_zs_pool_hidden',      # zs_pool_proj_head (hidden)
        'last_z_layout_proj',       # z_layout_infonce_head  (Strategy B)
        'last_z_layout_pool_proj',  # z_layout_pool_head     (Strategy B)
        'last_z_layout_pool_hidden',# z_layout_pool_head     (Strategy B)
        'last_seg_pred',            # seg_pred_head
    ]

    def _all_side_heads_ddp_visibility_hook(module, inp, output):
        # output = (shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features)
        pf = output[5]
        if pf is None:
            return output
        zero_sum = None
        for attr in _SIDE_HEAD_ATTRS:
            pred = getattr(module, attr, None)
            if pred is not None and isinstance(pred, torch.Tensor) and pred.requires_grad:
                term = pred.sum() * 0.0   # zero value, live grad_fn
                zero_sum = term if zero_sum is None else zero_sum + term
        if zero_sum is None:
            return output
        pf_modified = pf + zero_sum
        return (output[0], output[1], output[2], output[3], output[4], pf_modified)

    raw_model.shape_model.register_forward_hook(_all_side_heads_ddp_visibility_hook)
    if accelerator.is_main_process:
        print("  DDP visibility hook registered: ALL side-head outputs connected to "
              "model output graph via zero-gradient paths (fixes 'marked ready twice')")
# ─────────────────────────────────────────────────────────────────────────────

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

        # _sem_valid[b] = True when scene b has at least one known semantic category.
        # Scenes from extra_train_paths have label_dist=zeros (disable_semantics=True).
        # Pool NCE and token NCE must be restricted to _sem_valid scenes because
        # argmax(zeros)=0 assigns ALL no-semantic scenes to category 0, contaminating
        # that prototype and producing false InfoNCE signal. Per-Gaussian NCE filters
        # via segment >= 0, and z_s scene NCE is safe because F.normalize(zeros)=zeros
        # produces zero weights — both handle mixed batches without this mask.
        _sem_valid = label_dist_v.sum(dim=1) > 1e-6   # [B] bool

        seg_labels = inst_labels = None
        if need_segment_labels:
            seg_labels  = batch_data['segment_labels'].long().to(device)
            if enable_semantic:
                inst_labels = batch_data['instance_labels'].long().to(device)

        sa_gpu  = (batch_data['scaffold_anchors'].float().to(device) if need_scaffold_data else None)
        sti_gpu = (batch_data['scaffold_token_ids'].long().to(device) if args.position_scaffold else None)

        optimizer.zero_grad()

        # ── KL weight for this step (annealed or fixed) ───────────────────────
        # global_step counts total optimizer steps including any resumed steps.
        # _kl_current ramps from 0 → args.kl_weight over kl_anneal_steps steps,
        # then holds at args.kl_weight. When kl_anneal_steps=0 it is always
        # args.kl_weight (no annealing — backward compatible).
        if _kl_anneal_active and global_step < args.kl_anneal_steps:
            _kl_current = args.kl_weight * (global_step / args.kl_anneal_steps)
        else:
            _kl_current = args.kl_weight

        with torch.autocast('cuda', dtype=_autocast_dtype, enabled=_use_autocast):
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

        recon_loss  = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight)
        # Clamp log_var before KL computation. Without a KL penalty (early annealing)
        # the encoder can drive log_var to extreme values at high LR. Clamping to [-10, 10]
        # keeps exp(0.5*log_var) in [0.007, 148] — numerically safe while still expressive.
        # This has no effect once the KL penalty is large enough to self-regulate log_var.
        log_var_clamped = log_var.clamp(-10.0, 10.0)
        KL_loss     = -0.5*torch.sum(1+log_var_clamped-mu.pow(2)-log_var_clamped.exp(), dim=1).mean()

        color_pred_loss = torch.tensor(0., device=device)
        if mcp is not None and args.color_residual:
            color_pred_loss = F.mse_loss(mcp.float(), mean_color_gt)

        scene_sem_loss = torch.tensor(0., device=device)
        if ssp is not None and args.scene_semantic_head:
            p_s = batch_data['label_dist'].float().to(device)
            scene_sem_loss = scene_semantic_kl_loss(ssp.float(), p_s)

        anchor_loss = torch.tensor(0., device=device)
        if anch is not None and args.position_scaffold and sa_gpu is not None:
            anchor_loss = F.mse_loss(anch.float(), sa_gpu)

        layout_loss = torch.tensor(0., device=device)
        if slp is not None and args.scene_layout_head:
            gt_c = batch_data['category_centroids'].float().to(device)
            gt_v = batch_data['category_valid'].float().to(device)
            layout_loss = compute_layout_loss(slp.float(), gt_c, gt_v)

        seg_pred_loss = torch.tensor(0., device=device)
        if args.predict_seg_labels and sgp is not None and seg_labels is not None:
            seg_pred_loss = compute_seg_pred_loss(sgp, seg_labels)

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

        z_s_nce_loss    = torch.tensor(0., device=device)
        z_s_nce_metrics = {'z_s_infonce_loss': 0., 'z_s_num_positives': 0., 'z_s_frac_anchors': 0.}
        if args.z_s_infonce_weight > 0 and zsp is not None:
            z_s_nce_loss, z_s_nce_metrics = compute_scene_infonce_loss(
                zsp, label_dist_v, args.z_s_infonce_temperature, args.z_s_infonce_delta)

        zs_tok_nce_loss    = torch.tensor(0., device=device)
        zs_tok_nce_metrics = {'zs_tok_infonce_loss': 0., 'zs_tok_num_categories': 0}
        if args.zs_token_infonce_weight > 0 and args.latent_disentangle:
            _n_tok       = args.semantic_dims // 32
            _z_s_tok_all = z[:, :args.semantic_dims].reshape(B, _n_tok, 32)
            # Restrict to scenes with valid semantics — argmax(zeros)=0 hazard.
            _n_sem_tok   = int(_sem_valid.sum().item())
            if _n_sem_tok >= 2:
                zs_tok_nce_loss, zs_tok_nce_metrics = compute_zs_token_infonce_loss(
                    _z_s_tok_all[_sem_valid], label_dist_v[_sem_valid],
                    args.zs_token_infonce_temperature)

        zs_lay_nce_loss    = torch.tensor(0., device=device)
        zs_lay_nce_metrics = {'zs_layout_infonce_loss': 0., 'zs_layout_num_cats': 0}
        z_lay_proj = raw_model.shape_model.last_z_layout_proj
        if args.zs_layout_infonce_weight > 0 and z_lay_proj is not None:
            zs_lay_nce_loss, zs_lay_nce_metrics = compute_zs_layout_infonce_loss(
                z_lay_proj, label_dist_v, args.zs_layout_infonce_temperature)

        zs_pool_nce_loss    = torch.tensor(0., device=device)
        zs_pool_nce_metrics = {'zs_pool_infonce_loss': 0., 'zs_pool_num_cats': 0}
        if args.zs_pool_infonce_weight > 0:
            _pool_emb = raw_model.shape_model.last_zs_pool_proj
            if _pool_emb is None:
                _pool_emb = getattr(raw_model.shape_model, 'last_z_layout_pool_proj', None)
            # Restrict to scenes with valid semantics — argmax(zeros)=0 hazard:
            # if label_dist is all-zeros, argmax returns 0, causing all no-semantic
            # scenes to cluster under category 0 and contaminate its prototype.
            _n_sem_pool = int(_sem_valid.sum().item())
            if _pool_emb is not None and _n_sem_pool >= 2:
                _pe_v        = _pool_emb[_sem_valid]
                _ld_v        = label_dist_v[_sem_valid]
                _dom_cat     = _ld_v.float().argmax(dim=1)
                _pool_labels = _dom_cat.unsqueeze(1).expand(-1, _pe_v.shape[1]).long()
                zs_pool_nce_loss, _pool_metrics = compute_semantic_loss(
                    embeddings=_pe_v, segment_labels=_pool_labels,
                    instance_labels=None, batch_size=_n_sem_pool,
                    segment_weight=1.0, instance_weight=0.0,
                    temperature=args.zs_pool_infonce_temperature,
                    subsample=_pe_v.shape[1],
                    sampling_strategy=args.sampling_strategy)
                zs_pool_nce_metrics = {
                    'zs_pool_infonce_loss': _pool_metrics.get('segment_loss', 0.),
                    'zs_pool_num_cats':     _pool_metrics.get('num_categories_in_batch', 0)}

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

            if (raw_model.shape_model.scene_layout_module is not None and
                    args.semantic_token_heads):
                with torch.no_grad():
                    _ed = raw_model.shape_model.embed_dim
                    _sd = args.semantic_dims
                    if args.structured_layout_tokens:
                        _n_s   = raw_model.shape_model._n_sem_tokens
                        _start = _ed + _n_s * _ed
                        z_lay_B = z_s_swapped[:, _start:_sd]
                        raw_model.shape_model.last_scene_layout_pred = \
                            raw_model.shape_model.scene_layout_module(z_lay_B)
                    else:
                        z_sem_B = z_s_swapped[:, _ed:_sd]
                        raw_model.shape_model.last_scene_layout_pred = \
                            raw_model.shape_model.scene_layout_module(z_sem_B)

            se_shifted = torch.roll(raw_model.shape_model._shape_embed_cache, shifts=1, dims=0)
            _z_layout_shifted = None
            _any_B_train = args.decoder_layout_cross_attn or args.decoder_layout_additive
            if _any_B_train and raw_model.shape_model.last_z_layout is not None:
                _z_layout_shifted = torch.roll(
                    raw_model.shape_model.last_z_layout, shifts=1, dims=0)

            # ── DDP FIX: remove scene_layout_module during cross-recon decode ──────
            # ROOT CAUSE: scene_layout_module appears in TWO gradient paths in total_loss:
            #   Path 1 (main forward):  layout_loss → last_scene_layout_pred → slm
            #   Path 2 (cross_recon):   cross_recon_loss → UV_cross → decode() → slm
            # With find_unused_parameters=True, DDP registers an AccumulateGrad hook on
            # each leaf parameter. Leaf AccumulateGrad fires once per gradient accumulation.
            # Two paths → two firings for the same parameter → "marked ready twice" crash.
            #
            # Fix: temporarily set scene_layout_module to None so decode() cannot call it.
            # The model is built to handle None scene_layout_module (scene_layout_head is
            # an optional flag). Also detach last_scene_layout_pred to cut any indirect
            # gradient path decode() might take through the cached layout tensor.
            # After decode(), both are restored to their original state.
            _saved_slm = raw_model.shape_model.scene_layout_module
            _saved_slp = raw_model.shape_model.last_scene_layout_pred
            raw_model.shape_model.scene_layout_module = None
            if _saved_slp is not None:
                raw_model.shape_model.last_scene_layout_pred = _saved_slp.detach()

            with torch.autocast('cuda', dtype=_autocast_dtype, enabled=_use_autocast):
                UV_cross, _ = raw_model.shape_model.decode(
                    lat_cross, volume_queries=None,
                    return_semantic_features=False, shape_embed=se_shifted,
                    scaffold_anchors=sa_gpu, scaffold_token_ids=sti_gpu,
                    z_layout=_z_layout_shifted)

            # Restore both to original state before computing cross_recon_loss
            raw_model.shape_model.scene_layout_module = _saved_slm
            raw_model.shape_model.last_scene_layout_pred = _saved_slp
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
                      + _kl_current                * KL_loss
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
        # Gradient clipping at max_norm=10.
        # log_var clamping only protects the KL loss value. The reparameterisation
        # z = mu + exp(0.5*log_var)*eps runs INSIDE the model forward before any
        # clamping. Without clipping, lr=8e-4 can drive log_var to ~20 in one bad
        # step: exp(0.5×20)≈22026, z values reach ±66000, BF16 overflows (max 65504)
        # → NaN. This is exactly the epoch-30 NaN in doc 27.
        # max_norm=10: at natural ||g||=20 uses 50% of signal (not destructive);
        # at ||g||=200 (pathological batch) uses 5% (safely dampened).
        # max_norm=1 only used 2-10% → reconstruction plateau seen in doc 24.
        accelerator.clip_grad_norm_(gs_autoencoder.parameters(), max_norm=10.0)
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
                  f"kl_weight={_kl_current:.2e} | KL_contrib={_kl_current*KL_loss.item():.4f} | "
                  f"mu=[{mu.min().item():.3f},{mu.max().item():.3f}]")
            if _kl_anneal_active:
                _pct = min(100.0, 100.0 * global_step / args.kl_anneal_steps)
                print(f"  KL annealing: step {global_step}/{args.kl_anneal_steps} "
                      f"({_pct:.1f}% of ramp complete)")
            if _mu_s is not None:
                print(f"  mu_s=[{_mu_s.min().item():.3f},{_mu_s.max().item():.3f}]  "
                      f"mu_g=[{_mu_g.min().item():.3f},{_mu_g.max().item():.3f}]")

        global_step += 1

    nb = len(trainDataLoader)
    lr_now = scheduler.get_last_lr()[0]
    # kl_weight at the last step of this epoch (for logging)
    _kl_log = _kl_current
    if accelerator.is_main_process:
        print(f"\nEpoch {epoch:04d} | "
              f"Loss={e['loss']/nb:.4f} | "
              f"Recon={e['recon']/nb:.4f} | "
              f"KL={e['kl']/nb:.4f} | "
              f"KLw={_kl_log:.2e} | "
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

    # ── EVALUATION ────────────────────────────────────────────────────────────
    if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:

        # PRIMARY: full-scene val (thesis target, used for best model)
        val_metrics = evaluate_model(gs_autoencoder, raw_model, valDataLoader,
                                     device, accelerator, epoch=epoch, do_vis=True)

        if accelerator.is_main_process:
            print(f"\n--- Val FULL SCENES epoch {epoch} ---")
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
                print(f"  Z_sNCE={val_metrics['z_s_infonce_loss']:.4f}")
            if args.zs_token_infonce_weight > 0:
                print(f"  ZsTokNCE={val_metrics['zs_tok_infonce_loss']:.4f}")
            if args.zs_layout_infonce_weight > 0:
                print(f"  LayNCE={val_metrics['zs_lay_infonce_loss']:.4f}")

        # DIAGNOSTIC: held-out chunk val (in-distribution, no vis overhead)
        chunk_metrics = None
        if _has_chunk_val:
            chunk_metrics = evaluate_model(gs_autoencoder, raw_model, valChunkDataLoader,
                                           device, accelerator, epoch=epoch, do_vis=False)
            if accelerator.is_main_process:
                print(f"\n--- Val HELD-OUT CHUNKS epoch {epoch} "
                      f"(skip={_n_train_chunks}, n={len(gs_dataset_val_chunk)}) ---")
                print(f"  L2={chunk_metrics['avg_l2_error']:.4f}  "
                      f"Pos={chunk_metrics['position_loss']:.4f}  "
                      f"Col={chunk_metrics['color_loss']:.4f}  "
                      f"Opa={chunk_metrics['opacity_loss']:.4f}  "
                      f"Scl={chunk_metrics['scale_loss']:.4f}  "
                      f"Rot={chunk_metrics['rotation_loss']:.4f}")
                # Distribution gap metric: how much harder are full scenes vs chunks?
                # Values close to 1.0 mean the model generalises well.
                # Values >> 1.0 mean the training-eval distribution shift is large.
                if chunk_metrics['avg_l2_error'] > 1e-6:
                    _gap = val_metrics['avg_l2_error'] / chunk_metrics['avg_l2_error']
                    print(f"  DISTRIBUTION GAP  full_L2 / chunk_L2 = {_gap:.2f}×  "
                          f"({'negligible' if _gap < 1.3 else 'moderate' if _gap < 2.0 else 'large — chunks much easier'})")

        # Best model checkpoint on full-scene val (primary metric)
        if val_metrics['avg_l2_error'] < best_val_loss:
            best_val_loss = val_metrics['avg_l2_error']
            best_epoch    = epoch
            if accelerator.is_main_process:
                ckpt_dict = {
                    'epoch':                epoch,
                    'model_state_dict':     raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_l2_error':         val_metrics['avg_l2_error'],
                    **_ckpt_meta,
                }
                if chunk_metrics is not None:
                    ckpt_dict['chunk_val_l2_error'] = chunk_metrics['avg_l2_error']
                torch.save(ckpt_dict, os.path.join(save_path, "best_model.pth"))
                print(f"  [NEW BEST] full_L2={best_val_loss:.4f} saved")

        if accelerator.is_main_process and wandb_enabled:
            log_dict = {
                'epoch': epoch,
                'val_full_l2': val_metrics['avg_l2_error'],
                'val_full_pos': val_metrics['position_loss'],
                'val_full_col': val_metrics['color_loss'],
            }
            if chunk_metrics is not None:
                log_dict['val_chunk_l2']  = chunk_metrics['avg_l2_error']
                log_dict['val_chunk_pos'] = chunk_metrics['position_loss']
                if chunk_metrics['avg_l2_error'] > 1e-6:
                    log_dict['val_dist_gap'] = (val_metrics['avg_l2_error']
                                                / chunk_metrics['avg_l2_error'])
            wandb_run.log(log_dict)

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
                               accelerator, epoch=args.num_epochs-1, do_vis=True)
final_chunk_metrics = None
if _has_chunk_val:
    final_chunk_metrics = evaluate_model(gs_autoencoder, raw_model, valChunkDataLoader,
                                         device, accelerator, epoch=args.num_epochs-1,
                                         do_vis=False)

if accelerator.is_main_process:
    print(f"\nFinal full_L2 : {final_metrics['avg_l2_error']:.4f}")
    if final_chunk_metrics is not None:
        print(f"Final chunk_L2: {final_chunk_metrics['avg_l2_error']:.4f}")
        if final_chunk_metrics['avg_l2_error'] > 1e-6:
            print(f"Final gap     : {final_metrics['avg_l2_error']/final_chunk_metrics['avg_l2_error']:.2f}×")
    print(f"Best full_L2  : {best_val_loss:.4f}  (epoch {best_epoch})")

    final_dict = {
        'epoch':            args.num_epochs - 1,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_l2':     final_metrics['avg_l2_error'],
        'best_val_l2':      best_val_loss,
        'best_epoch':       best_epoch,
        **_ckpt_meta,
        'individual_losses': {k: final_metrics[f'{k}_loss'] for k in PARAM_SLICES},
    }
    if final_chunk_metrics is not None:
        final_dict['final_chunk_val_l2'] = final_chunk_metrics['avg_l2_error']
    torch.save(final_dict, os.path.join(save_path, "final.pth"))
    print(f"Saved: {save_path}final.pth")

if wandb_enabled and accelerator.is_main_process:
    summary = {"final_val_l2": final_metrics['avg_l2_error'],
               "best_val_l2": best_val_loss, "best_epoch": best_epoch}
    if final_chunk_metrics is not None:
        summary["final_chunk_val_l2"] = final_chunk_metrics['avg_l2_error']
    wandb_run.summary.update(summary)
    wandb_run.finish()

if accelerator.is_main_process: print("Done.")