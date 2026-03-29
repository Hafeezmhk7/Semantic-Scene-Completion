"""
Can3Tok Training
================
Step 1:    Color Residual             (--color_residual)
Move 1:    Scene Semantic Head        (--scene_semantic_head)
Scaffold:  Position Scaffold          (--position_scaffold)
Option 1:  Decoder Shape Prepend      (--decoder_shape_prepend)
Option 2:  Decoder Shape Cross-Attn   (--decoder_shape_cross_attn)
InfoNCE:   Per-Gaussian contrastive   (--semantic_mode hidden)

DISENTANGLEMENT SUITE:
  --latent_disentangle  --semantic_dims  --cross_recon_weight  --ortho_weight
  --scene_layout_head  --layout_loss_weight  --position_layout_residual

NEW: THREE SPATIAL INDUCTIVE BIAS IDEAS
  Idea 0: --decoder_pos_enc
    Learnable positional embeddings [512, width] before decoder transformer.

  Idea 1: --predict_seg_labels  --seg_pred_weight
    SegPredHead: Gaussian params [B,40000,14] -> seg logits [B,40000,72].
    CrossEntropy loss; backprop forces decoder positions to be categorically
    discriminative. At inference: soft centroid lookup from seg logits.

  Idea 2: --token_cond  --token_cond_approach (A / B / AB)
    A: Fourier(scaffold_anchors[B,512,3]) -> token spatial bias
    B: W[512,72] × pred_centroids[B,72,3] -> token spatial bias
    Both approaches inject spatial identity into decoder tokens before transformer.

  Idea 3: --query_decoder
    Replaces GS_decoder MLP with SpatialAwareDecoder.
    Per-Gaussian: gather(token_feat) + Fourier(voxel_anchor) -> MLP -> 14 params.

ABLATION TABLE:
  Run A:  color_residual only                               (done, L2=1.43)
  Run H:  color_residual + semantic + disentangle + layout  (done, L2=1.565)
  Run K:  Run H + position_layout_residual                  (done, L2~1.0-1.2)
  Run P:  Run K + decoder_pos_enc                           position PE ablation
  Run Q:  Run K + predict_seg_labels                        Idea 1 ablation
  Run R:  Run K + token_cond approach A                     Idea 2A ablation
  Run S:  Run K + token_cond approach B                     Idea 2B ablation
  Run T:  Run K + token_cond approach AB                    Idea 2AB ablation
  Run U:  Run K + query_decoder                             Idea 3 ablation
  Run V:  Run K + all three ideas                           full combination
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse
from pathlib import Path

from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
import torch.utils.data as Data

from semantic_losses import compute_semantic_loss
from distribution_loss import compute_distribution_loss
from pca_feature_visualization import visualize_comparison
from gs_ply_reconstructor import save_reconstructed_gaussians

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

# ============================================================================
# PARAMETER INDICES
# ============================================================================

PARAM_SLICES = {
    'position': slice(0, 3),
    'color':    slice(3, 6),
    'opacity':  slice(6, 7),
    'scale':    slice(7, 10),
    'rotation': slice(10, 14),
}

GEOMETRIC_INDICES = (
    list(range(4, 7)) + list(range(7, 10)) + [10]
    + list(range(11, 14)) + list(range(14, 18))
)

GEO_ONLY_SLICES = {
    'position': slice(0, 3),
    'opacity':  slice(6, 7),
    'scale':    slice(7, 10),
    'rotation': slice(10, 14),
}

# ============================================================================
# LOSS HELPERS
# ============================================================================

def compute_reconstruction_loss(prediction, target, batch_size, color_weight=1.0):
    if color_weight == 1.0:
        return torch.norm(prediction - target, p=2) / batch_size
    loss_pos   = torch.norm(prediction[:, :, 0:3] - target[:, :, 0:3], p=2)
    loss_color = torch.norm(prediction[:, :, 3:6] - target[:, :, 3:6], p=2) * color_weight
    loss_other = torch.norm(prediction[:, :, 6:]  - target[:, :, 6:],  p=2)
    return (loss_pos + loss_color + loss_other) / batch_size


def compute_individual_losses(prediction, target):
    return {
        name: torch.norm(prediction[:, :, sl] - target[:, :, sl], p=2).item()
        for name, sl in PARAM_SLICES.items()
    }


def scene_semantic_kl_loss(p_hat, p_s, eps=1e-8):
    p_hat_clamped = torch.clamp(p_hat, min=eps)
    return (p_s * (torch.log(p_s + eps) - torch.log(p_hat_clamped))).sum(dim=-1).mean()


def compute_cross_recon_loss(pred_cross_3d, target, batch_size):
    loss = torch.tensor(0.0, device=pred_cross_3d.device)
    for sl in GEO_ONLY_SLICES.values():
        loss = loss + torch.norm(
            pred_cross_3d[:, :, sl] - target[:, :, sl], p=2) / batch_size
    return loss


def compute_orthogonality_loss(mu_s, mu_g, proj_dim=64):
    B = mu_s.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=mu_s.device)
    with torch.no_grad():
        p_dim = min(proj_dim, B - 1, mu_s.shape[1], mu_g.shape[1])
        idx_s = torch.randperm(mu_s.shape[1], device=mu_s.device)[:p_dim]
        idx_g = torch.randperm(mu_g.shape[1], device=mu_g.device)[:p_dim]
    p_s = mu_s[:, idx_s]
    p_g = mu_g[:, idx_g]
    p_s = p_s - p_s.mean(dim=0, keepdim=True)
    p_g = p_g - p_g.mean(dim=0, keepdim=True)
    p_s = F.normalize(p_s, p=2, dim=0)
    p_g = F.normalize(p_g, p=2, dim=0)
    return ((p_s.T @ p_g) ** 2).mean()


def compute_layout_loss(pred_centroids, gt_centroids, gt_valid):
    diff    = (pred_centroids - gt_centroids) ** 2
    per_cat = diff.mean(dim=-1)
    masked  = per_cat * gt_valid
    return masked.sum() / (gt_valid.sum() + 1e-8)


def compute_spatial_semantic_loss(pred_voxel, gt_voxel, voxel_valid, eps=1e-8):
    p_hat        = torch.clamp(pred_voxel, min=eps)
    kl_per_voxel = (gt_voxel * (torch.log(gt_voxel + eps) - torch.log(p_hat))).sum(dim=-1)
    return (kl_per_voxel * voxel_valid).sum() / (voxel_valid.sum() + 1e-8)


def compute_seg_pred_loss(seg_logits, segment_labels):
    """
    Idea 1: CrossEntropy between predicted per-Gaussian segment logits
    and ground-truth segment labels.

    seg_logits:     [B, 40000, 72]  — raw logits
    segment_labels: [B, 40000]      — ground truth (>= 0 valid, -1 unlabelled)
    """
    B, N, C = seg_logits.shape
    flat_logits = seg_logits.reshape(B * N, C)
    flat_labels = segment_labels.reshape(B * N).long()
    valid       = flat_labels >= 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=seg_logits.device)
    return F.cross_entropy(flat_logits[valid], flat_labels[valid])


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Can3Tok Training')

# Core
parser.add_argument('--batch_size',           type=int,   default=64)
parser.add_argument('--num_epochs',           type=int,   default=1000)
parser.add_argument('--lr',                   type=float, default=1e-4)
parser.add_argument('--kl_weight',            type=float, default=1e-5)
parser.add_argument('--eval_every',           type=int,   default=20)
parser.add_argument('--failure_threshold',    type=float, default=100.0)

# Dataset
parser.add_argument('--train_scenes',         type=int,   default=None)
parser.add_argument('--val_scenes',           type=int,   default=None)
parser.add_argument('--sampling_method',      type=str,   default='opacity',
                    choices=['random', 'opacity', 'hybrid'])

# InfoNCE
parser.add_argument('--semantic_mode',        type=str,   default='none',
                    choices=['none', 'hidden', 'geometric', 'attention', 'dist'])
parser.add_argument('--segment_loss_weight',  type=float, default=0.0)
parser.add_argument('--instance_loss_weight', type=float, default=0.0)
parser.add_argument('--semantic_temperature', type=float, default=0.07)
parser.add_argument('--semantic_subsample',   type=int,   default=2000)
parser.add_argument('--sampling_strategy',    type=str,   default='balanced',
                    choices=['random', 'balanced'])

# Step 1
parser.add_argument('--color_residual',       action='store_true', default=False)
parser.add_argument('--mean_color_weight',    type=float, default=1.0)

# Move 1
parser.add_argument('--scene_semantic_head',   action='store_true', default=False)
parser.add_argument('--scene_semantic_weight', type=float, default=0.3)

# Scaffold
parser.add_argument('--position_scaffold',     action='store_true', default=False)
parser.add_argument('--anchor_loss_weight',    type=float, default=1.0)

# Decoder conditioning
parser.add_argument('--decoder_shape_prepend',     action='store_true', default=False)
parser.add_argument('--decoder_shape_cross_attn',  action='store_true', default=False)
parser.add_argument('--decoder_cross_attn_layers', type=int, default=4)

# Latent disentanglement
parser.add_argument('--latent_disentangle',   action='store_true', default=False)
parser.add_argument('--semantic_dims',        type=int, default=512)
parser.add_argument('--cross_recon_weight',   type=float, default=0.5)
parser.add_argument('--ortho_weight',         type=float, default=0.1)

# Scene layout head
parser.add_argument('--scene_layout_head',    action='store_true', default=False)
parser.add_argument('--layout_loss_weight',   type=float, default=0.3)

# Position layout residual
parser.add_argument('--position_layout_residual', action='store_true', default=False)

# JEPA Idea 1
parser.add_argument('--jepa_idea1',           action='store_true', default=False)
parser.add_argument('--jepa_idea1_weight',    type=float, default=1.0)

# ── NEW: THREE SPATIAL INDUCTIVE BIAS IDEAS ──────────────────────────────────

parser.add_argument('--decoder_pos_enc', action='store_true', default=False,
    help='[Idea 0] Add learnable positional embeddings PE[512,width] to decoder '
         'tokens before transformer. Gives tokens structural identity. '
         'Safe to combine with any other flag. '
         'Recommended as baseline companion for all Ideas 1-3.')

parser.add_argument('--predict_seg_labels', action='store_true', default=False,
    help='[Idea 1] SegPredHead: Gaussian params [B,40000,14] -> seg logits [B,40000,72]. '
         'CE loss vs gt segment labels. Backprop forces position to be categorically '
         'discriminative. Requires segment_labels (forces need_segment_labels=True). '
         'At inference: soft centroid lookup from seg_logits removes label dependency.')

parser.add_argument('--seg_pred_weight', type=float, default=0.3,
    help='Weight for Idea 1 segment prediction cross-entropy loss (default 0.3).')

parser.add_argument('--token_cond', action='store_true', default=False,
    help='[Idea 2] Inject spatial conditioning into decoder tokens before transformer. '
         'Use --token_cond_approach to choose approach.')

parser.add_argument('--token_cond_approach', type=str, default='A',
    choices=['A', 'B', 'AB'],
    help='[Idea 2] Spatial conditioning approach: '
         'A = scaffold anchor Fourier encoding (requires --position_scaffold), '
         'B = learned W[512,72] x pred_centroids (requires --scene_layout_head), '
         'AB = both applied additively.')

parser.add_argument('--query_decoder', action='store_true', default=False,
    help='[Idea 3] Replace flat GS_decoder MLP with SpatialAwareDecoder. '
         'Per-Gaussian: gather(token_feat) + Fourier(voxel_anchor) -> MLP -> 14. '
         'Requires --position_scaffold (needs scaffold_anchors + scaffold_token_ids). '
         'Falls back to GS_decoder if scaffold data not available. '
         'NOTE: may require --batch_size 32 due to higher memory usage.')

# Label input
parser.add_argument('--label_input',          action='store_true', default=False)
parser.add_argument('--no_label_input',       dest='label_input', action='store_false')

# Normalization
parser.add_argument('--scale_norm_mode',      type=str, default='linear',
                    choices=['log', 'linear'])
parser.add_argument('--color_loss_weight',    type=float, default=1.0)
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

# Visualization
parser.add_argument('--pca_vis_freq',         type=int,   default=50)
parser.add_argument('--pca_brightness',       type=float, default=1.25)
parser.add_argument('--pca_num_scenes',       type=int,   default=3)
parser.add_argument('--recon_ply_freq',       type=int,   default=50)
parser.add_argument('--recon_ply_num_scenes', type=int,   default=3)
parser.add_argument('--recon_ply_max_sh',     type=int,   default=3)

# W&B
parser.add_argument('--use_wandb',            action='store_true', default=False)
parser.add_argument('--wandb_project',        type=str, default='Can3Tok-SceenSplat-7K')
parser.add_argument('--wandb_entity',         type=str, default='3D-SSC')

# Resume
parser.add_argument('--resume_checkpoint',    type=str, default=None)
parser.add_argument('--resume_epoch',         type=int, default=None)

args = parser.parse_args()

# ── Flag validation ───────────────────────────────────────────────────────────
if args.cross_recon_weight > 0 and not args.latent_disentangle:
    print("[WARNING] --cross_recon_weight > 0 requires --latent_disentangle. Setting to 0.")
    args.cross_recon_weight = 0.0
if args.ortho_weight > 0 and not args.latent_disentangle:
    print("[WARNING] --ortho_weight > 0 requires --latent_disentangle. Setting to 0.")
    args.ortho_weight = 0.0
if args.jepa_idea1 and not args.position_scaffold:
    print("[INFO] --jepa_idea1 requires --position_scaffold. Enabling.")
    args.position_scaffold = True
if args.semantic_dims % 32 != 0:
    raise ValueError(f"--semantic_dims ({args.semantic_dims}) must be divisible by 32.")
if args.position_layout_residual and not args.scene_layout_head:
    print("[INFO] --position_layout_residual requires --scene_layout_head. Enabling.")
    args.scene_layout_head = True
if args.position_layout_residual and args.position_scaffold:
    raise ValueError("--position_layout_residual and --position_scaffold are mutually exclusive.")

# New idea validation
if args.token_cond and 'A' in args.token_cond_approach.upper() and not args.position_scaffold:
    print("[INFO] --token_cond approach A requires --position_scaffold. Enabling.")
    args.position_scaffold = True
if args.token_cond and 'B' in args.token_cond_approach.upper() and not args.scene_layout_head:
    print("[INFO] --token_cond approach B requires --scene_layout_head. Enabling.")
    args.scene_layout_head = True
if args.query_decoder and not args.position_scaffold:
    print("[INFO] --query_decoder requires --position_scaffold. Enabling.")
    args.position_scaffold = True

# need_scaffold_data: whether to load scaffold batch keys and pass to model
need_scaffold_data = (args.position_scaffold or
                      args.token_cond or
                      args.query_decoder)

semantic_requested      = (args.semantic_mode != 'none')
semantic_loss_enabled   = (args.segment_loss_weight > 0 or args.instance_loss_weight > 0)
enable_semantic         = semantic_requested and semantic_loss_enabled
effective_semantic_mode = args.semantic_mode if enable_semantic else 'none'
need_segment_labels     = (enable_semantic or args.scene_semantic_head or
                           args.jepa_idea1 or args.predict_seg_labels)

# ============================================================================
# W&B
# ============================================================================

wandb_enabled = False
if args.use_wandb:
    try:
        import wandb
        job_id   = os.environ.get('SLURM_JOB_ID', 'local')
        run_name = f"can3tok_job_{job_id}_{effective_semantic_mode}"
        if args.color_residual:               run_name += "_colorresidual"
        if args.scene_semantic_head:          run_name += "_scenesemantic"
        if args.position_scaffold:            run_name += "_scaffold"
        if args.decoder_shape_prepend:        run_name += "_shapeprepend"
        if args.decoder_shape_cross_attn:     run_name += "_shapecrossattn"
        if args.latent_disentangle:           run_name += f"_disentangle{args.semantic_dims}"
        if args.scene_layout_head:            run_name += "_layout"
        if args.position_layout_residual:     run_name += "_posresid"
        if args.jepa_idea1:                   run_name += "_jepa1"
        if args.decoder_pos_enc:              run_name += "_posenc"
        if args.predict_seg_labels:           run_name += "_segpred"
        if args.token_cond:                   run_name += f"_tokencond{args.token_cond_approach}"
        if args.query_decoder:                run_name += "_querydec"
        if enable_semantic:                   run_name += f"_beta{args.segment_loss_weight}"
        if args.resume_checkpoint:            run_name += "_resumed"
        wandb_run = wandb.init(
            entity=args.wandb_entity, project=args.wandb_project,
            name=run_name, config=vars(args))
        wandb_enabled = True
        print("W&B enabled")
    except Exception as e:
        print(f"W&B failed: {e}")

# ============================================================================
# DEVICE + PATHS
# ============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = "/home/yli7/scratch/datasets/gaussian_world/preprocessed/interior_gs"

job_id = os.environ.get('SLURM_JOB_ID', None)
tag    = (f"RGB_job_{job_id}_{effective_semantic_mode}" if job_id
          else f"RGB_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{effective_semantic_mode}")
if args.color_residual:               tag += "_colorresidual"
if args.scene_semantic_head:          tag += "_scenesemantic"
if args.position_scaffold:            tag += "_scaffold"
if args.decoder_shape_prepend:        tag += "_shapeprepend"
if args.decoder_shape_cross_attn:     tag += "_shapecrossattn"
if args.latent_disentangle:           tag += f"_disentangle{args.semantic_dims}"
if args.scene_layout_head:            tag += "_layout"
if args.position_layout_residual:     tag += "_posresid"
if args.jepa_idea1:                   tag += "_jepa1"
if args.decoder_pos_enc:              tag += "_posenc"
if args.predict_seg_labels:           tag += "_segpred"
if args.token_cond:                   tag += f"_tokencond{args.token_cond_approach}"
if args.query_decoder:                tag += "_querydec"
if enable_semantic:                   tag += f"_beta{args.segment_loss_weight}"
if not args.use_canonical_norm:       tag += "_raw"

save_path = f"/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/{tag}/"
os.makedirs(save_path, exist_ok=True)

# ============================================================================
# STARTUP SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print(f"CAN3TOK TRAINING")
print(f"{'='*70}")
print(f"  color_residual:            {args.color_residual}")
print(f"  scene_semantic_head:       {args.scene_semantic_head}")
print(f"  position_scaffold:         {args.position_scaffold}")
print(f"  decoder_shape_prepend:     {args.decoder_shape_prepend}")
print(f"  decoder_shape_cross_attn:  {args.decoder_shape_cross_attn}")
print(f"  latent_disentangle:        {args.latent_disentangle}  (semantic_dims={args.semantic_dims})")
print(f"  cross_recon_weight:        {args.cross_recon_weight}")
print(f"  ortho_weight:              {args.ortho_weight}")
print(f"  scene_layout_head:         {args.scene_layout_head}  (weight={args.layout_loss_weight})")
print(f"  position_layout_residual:  {args.position_layout_residual}")
print(f"  jepa_idea1:                {args.jepa_idea1}  (weight={args.jepa_idea1_weight})")
print(f"  semantic_mode:             {effective_semantic_mode}")
print(f"  ── NEW IDEAS ──────────────────────────────────────────────────────")
print(f"  decoder_pos_enc:           {args.decoder_pos_enc}  (PE before transformer)")
print(f"  predict_seg_labels:        {args.predict_seg_labels}  "
      f"(weight={args.seg_pred_weight})")
print(f"  token_cond:                {args.token_cond}  "
      f"approach={args.token_cond_approach}")
print(f"  query_decoder:             {args.query_decoder}  (per-Gaussian spatial MLP)")
print(f"  Save: {save_path}")
print(f"{'='*70}\n")

# ============================================================================
# MODEL
# ============================================================================

print("Loading model config...")
config_path  = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
model_config = get_config_from_file(config_path).model
p = model_config.params.shape_module_cfg.params
p.semantic_mode              = effective_semantic_mode
p.color_residual             = args.color_residual
p.scene_semantic_head        = args.scene_semantic_head
p.position_scaffold          = args.position_scaffold
p.decoder_shape_prepend      = args.decoder_shape_prepend
p.decoder_shape_cross_attn   = args.decoder_shape_cross_attn
p.decoder_cross_attn_layers  = args.decoder_cross_attn_layers
p.latent_disentangle         = args.latent_disentangle
p.semantic_dims              = args.semantic_dims
p.scene_layout_head          = args.scene_layout_head
p.jepa_idea1                 = args.jepa_idea1
# NEW
p.decoder_pos_enc            = args.decoder_pos_enc
p.predict_seg_labels         = args.predict_seg_labels
p.token_cond                 = args.token_cond
p.token_cond_approach        = args.token_cond_approach
p.query_decoder              = args.query_decoder

cfg_point_feats = p.point_feats
expected_feats  = 12 if args.label_input else 11
if cfg_point_feats != expected_feats:
    raise ValueError(f"point_feats mismatch: yaml={cfg_point_feats}, "
                     f"label_input={args.label_input} requires {expected_feats}.")
print(f"  point_feats={cfg_point_feats} OK")

gs_autoencoder = instantiate_from_config(model_config)
gs_autoencoder.to(device)
optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=args.lr, betas=[0.9, 0.999])

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

start_epoch   = 0
best_val_loss = float('inf')
best_epoch    = 0

if args.resume_checkpoint:
    print(f"\nResuming from: {args.resume_checkpoint}")
    ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)

    # Hard mismatches that break weight loading
    for flag_name, current_val, default_val in [
        ('color_residual',             args.color_residual,            False),
        ('label_input',                args.label_input,               False),
        ('position_scaffold',          args.position_scaffold,         False),
        ('decoder_shape_prepend',      args.decoder_shape_prepend,     False),
        ('decoder_shape_cross_attn',   args.decoder_shape_cross_attn,  False),
        ('decoder_cross_attn_layers',  args.decoder_cross_attn_layers, 4),
        ('latent_disentangle',         args.latent_disentangle,        False),
        ('semantic_dims',              args.semantic_dims,              512),
        ('position_layout_residual',   args.position_layout_residual,  False),
        # New ideas are soft — changing them only means strict=False
    ]:
        saved_val = ckpt.get(flag_name, default_val)
        if saved_val != current_val:
            raise ValueError(
                f"{flag_name} mismatch: checkpoint={saved_val}, current={current_val}. "
                f"Cannot resume across this architectural/target boundary.")

    # Soft flags: architecture may differ, use strict=False
    strict = all([
        ckpt.get('scene_semantic_head', False) == args.scene_semantic_head,
        ckpt.get('semantic_mode', 'none') == effective_semantic_mode,
        ckpt.get('scene_layout_head', False) == args.scene_layout_head,
        ckpt.get('jepa_idea1', False) == args.jepa_idea1,
        ckpt.get('decoder_pos_enc', False) == args.decoder_pos_enc,
        ckpt.get('predict_seg_labels', False) == args.predict_seg_labels,
        ckpt.get('token_cond', False) == args.token_cond,
        ckpt.get('token_cond_approach', 'A') == args.token_cond_approach,
        ckpt.get('query_decoder', False) == args.query_decoder,
    ])
    if not strict:
        print(f"  Architecture changed — loading strict=False")
    gs_autoencoder.load_state_dict(ckpt['model_state_dict'], strict=strict)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch   = ckpt.get('epoch', 0) + 1
    if args.resume_epoch is not None:
        start_epoch = args.resume_epoch
    best_val_loss = ckpt.get('val_l2_error', ckpt.get('best_val_l2', float('inf')))
    best_epoch    = ckpt.get('epoch', 0)
    print(f"  Resumed epoch {start_epoch} (saved val L2: {best_val_loss:.4f})")

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
    shuffle=True, num_workers=9, pin_memory=True, persistent_workers=True)

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
    shuffle=False, num_workers=9, pin_memory=True, persistent_workers=True)

print(f"\n{'='*70}")
print(f"  Train: {len(gs_dataset_train)} scenes, {len(trainDataLoader)} batches/epoch")
print(f"  Val:   {len(gs_dataset_val)} scenes,  {len(valDataLoader)} batches")
print(f"  Total steps: {len(trainDataLoader) * args.num_epochs:,}")
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
    'decoder_shape_prepend':      args.decoder_shape_prepend,
    'decoder_shape_cross_attn':   args.decoder_shape_cross_attn,
    'decoder_cross_attn_layers':  args.decoder_cross_attn_layers,
    'latent_disentangle':         args.latent_disentangle,
    'semantic_dims':              args.semantic_dims,
    'scene_layout_head':          args.scene_layout_head,
    'jepa_idea1':                 args.jepa_idea1,
    'position_layout_residual':   args.position_layout_residual,
    # NEW
    'decoder_pos_enc':            args.decoder_pos_enc,
    'predict_seg_labels':         args.predict_seg_labels,
    'token_cond':                 args.token_cond,
    'token_cond_approach':        args.token_cond_approach,
    'query_decoder':              args.query_decoder,
    # Weights
    'mean_color_weight':          args.mean_color_weight,
    'scene_semantic_weight':      args.scene_semantic_weight,
    'anchor_loss_weight':         args.anchor_loss_weight,
    'cross_recon_weight':         args.cross_recon_weight,
    'ortho_weight':               args.ortho_weight,
    'layout_loss_weight':         args.layout_loss_weight,
    'jepa_idea1_weight':          args.jepa_idea1_weight,
    'seg_pred_weight':            args.seg_pred_weight,
    'color_loss_weight':          args.color_loss_weight,
    'use_canonical_norm':         args.use_canonical_norm,
    'scale_norm_mode':            args.scale_norm_mode,
}

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, dataloader, device, epoch=None):
    model.eval()
    total_l2           = 0.0
    total_kl           = 0.0
    total_color_pred   = 0.0
    total_scene_sem_kl = 0.0
    total_anchor_loss  = 0.0
    total_layout_loss  = 0.0
    total_spatial_kl   = 0.0
    total_seg_pred     = 0.0
    per_param  = {k: 0.0 for k in PARAM_SLICES}
    n_scenes   = 0
    recon_preds_list = []
    recon_means_list = []
    recon_tids_list  = []
    recon_anch_list  = []
    recon_dc_list    = []
    do_recon = (epoch is not None and epoch % args.recon_ply_freq == 0)

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating", leave=False):
            UV_gs_batch   = batch_data['features'].float().to(device)
            mean_color_gt = batch_data['mean_color'].float().to(device)
            B = UV_gs_batch.shape[0]

            # Scaffold data for new ideas
            sa_gpu  = (batch_data['scaffold_anchors'].float().to(device)
                       if need_scaffold_data else None)
            sti_gpu = (batch_data['scaffold_token_ids'].long().to(device)
                       if args.query_decoder else None)

            (shape_embed, mu, log_var, z,
             UV_gs_recover, _) = model(
                UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3],
                scaffold_anchors=sa_gpu,
                scaffold_token_ids=sti_gpu)

            mean_color_pred     = model.shape_model.last_mean_color_pred
            scene_semantic_pred = model.shape_model.last_scene_semantic_pred
            anchor_pred         = model.shape_model.last_anchor_pred
            scene_layout_pred   = model.shape_model.last_scene_layout_pred
            seg_pred            = model.shape_model.last_seg_pred

            # Position target
            target_abs = UV_gs_batch[:, :, GEOMETRIC_INDICES]
            if args.position_scaffold:
                position_offsets = batch_data['position_offsets'].float().to(device)
                target = target_abs.clone()
                target[:, :, 0:3] = position_offsets
            elif args.position_layout_residual:
                pos_residuals = batch_data['position_residuals'].float().to(device)
                target = target_abs.clone()
                target[:, :, 0:3] = pos_residuals
            else:
                target = target_abs

            pred_3d    = UV_gs_recover.reshape(B, -1, 14)
            recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight)
            kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

            if mean_color_pred is not None and args.color_residual:
                total_color_pred += F.mse_loss(mean_color_pred, mean_color_gt).item() * B
            if scene_semantic_pred is not None and args.scene_semantic_head:
                p_s = batch_data['label_dist'].float().to(device)
                total_scene_sem_kl += scene_semantic_kl_loss(
                    scene_semantic_pred, p_s).item() * B
            if anchor_pred is not None and args.position_scaffold:
                scaffold_anchors_gt = batch_data['scaffold_anchors'].float().to(device)
                total_anchor_loss += F.mse_loss(anchor_pred, scaffold_anchors_gt).item() * B
            if scene_layout_pred is not None and args.scene_layout_head:
                gt_centroids = batch_data['category_centroids'].float().to(device)
                gt_valid     = batch_data['category_valid'].float().to(device)
                total_layout_loss += compute_layout_loss(
                    scene_layout_pred, gt_centroids, gt_valid).item() * B
            if args.predict_seg_labels and seg_pred is not None:
                seg_labels_gpu = batch_data['segment_labels'].long().to(device)
                total_seg_pred += compute_seg_pred_loss(seg_pred, seg_labels_gpu).item() * B

            total_l2 += recon_loss.item()
            total_kl += kl_loss.sum().item()
            n_scenes  += B
            ind = compute_individual_losses(pred_3d, target)
            for k in per_param:
                per_param[k] += ind[k]

            if do_recon and len(recon_preds_list) < args.recon_ply_num_scenes:
                preds_np = pred_3d.cpu().numpy()
                means_np = mean_color_gt.cpu().numpy()
                tids_np  = batch_data['scaffold_token_ids'].numpy()
                anch_np  = (anchor_pred.cpu().numpy() if anchor_pred is not None
                            else batch_data['scaffold_anchors'].numpy())
                dc_np    = (batch_data['dc_position'].numpy()
                            if args.position_layout_residual else None)
                for si in range(B):
                    if len(recon_preds_list) >= args.recon_ply_num_scenes:
                        break
                    recon_preds_list.append(preds_np[si])
                    recon_means_list.append(means_np[si])
                    recon_tids_list.append(tids_np[si])
                    recon_anch_list.append(anch_np[si])
                    if dc_np is not None:
                        recon_dc_list.append(dc_np[si])

    if do_recon and recon_preds_list and save_path:
        try:
            all_preds = np.stack(recon_preds_list, axis=0)
            if args.color_residual:
                for si in range(len(all_preds)):
                    all_preds[si, :, 3:6] += recon_means_list[si]
                    all_preds[si, :, 3:6]  = np.clip(all_preds[si, :, 3:6], 0, 1)
            if args.position_scaffold:
                for si in range(len(all_preds)):
                    all_preds[si, :, 0:3] += recon_anch_list[si][recon_tids_list[si]]
            elif args.position_layout_residual and recon_dc_list:
                for si in range(len(all_preds)):
                    all_preds[si, :, 0:3] += recon_dc_list[si]
            recon_dir = Path(save_path) / "reconstructed_gaussians" / f"epoch_{epoch:03d}"
            save_reconstructed_gaussians(
                predictions=all_preds, output_dir=recon_dir, epoch=epoch,
                num_scenes=len(all_preds), max_sh_degree=args.recon_ply_max_sh,
                color_mode="1", prefix="scene")
        except Exception as e:
            print(f"  PLY save error: {e}")

    model.train()
    n = max(n_scenes, 1)
    return {
        'avg_l2_error':      total_l2,
        'avg_kl':            total_kl / n,
        'color_pred_loss':   total_color_pred / n,
        'scene_semantic_kl': total_scene_sem_kl / n,
        'anchor_loss':       total_anchor_loss / n,
        'layout_loss':       total_layout_loss / n,
        'spatial_kl':        total_spatial_kl / n,
        'seg_pred_loss':     total_seg_pred / n,
        **{f'{k}_loss': v / n for k, v in per_param.items()},
    }


# ============================================================================
# TRAINING LOOP
# ============================================================================

print(f"{'='*70}")
print(f"STARTING TRAINING  (epoch {start_epoch} -> {args.num_epochs - 1})")
print(f"{'='*70}\n")

global_step = 0

for epoch in tqdm(range(start_epoch, args.num_epochs), desc="Training"):
    gs_autoencoder.train()

    epoch_loss = epoch_recon = epoch_kl = epoch_sem = 0.0
    epoch_color_pred = epoch_scene_semantic = epoch_anchor = 0.0
    epoch_layout = epoch_spatial = epoch_cross_recon = epoch_ortho = 0.0
    epoch_seg_pred = 0.0
    epoch_pos = epoch_col = epoch_opa = epoch_scl = epoch_rot = 0.0

    for i_batch, batch_data in enumerate(trainDataLoader):
        UV_gs_batch   = batch_data['features'].float().to(device)
        mean_color_gt = batch_data['mean_color'].float().to(device)
        B = UV_gs_batch.shape[0]

        # Segment labels (needed for various losses)
        segment_labels  = None
        instance_labels = None
        if need_segment_labels:
            segment_labels  = batch_data['segment_labels'].long().to(device)
            if enable_semantic:
                instance_labels = batch_data['instance_labels'].long().to(device)

        # Scaffold data
        scaffold_anchors    = None
        scaffold_token_ids  = None
        position_offsets    = None
        if need_scaffold_data:
            scaffold_anchors   = batch_data['scaffold_anchors'].float().to(device)
            scaffold_token_ids = batch_data['scaffold_token_ids'].long().to(device)
        if args.position_scaffold:
            position_offsets   = batch_data['position_offsets'].float().to(device)

        optimizer.zero_grad()

        # ── Forward ──────────────────────────────────────────────────────────
        # Pass scaffold data for Idea 2 (token conditioning) and Idea 3 (query decoder)
        sa_gpu  = scaffold_anchors   if need_scaffold_data else None
        sti_gpu = scaffold_token_ids if args.query_decoder  else None

        (shape_embed, mu, log_var, z,
         UV_gs_recover, per_gaussian_features) = gs_autoencoder(
            UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3],
            scaffold_anchors=sa_gpu,
            scaffold_token_ids=sti_gpu)

        mean_color_pred     = gs_autoencoder.shape_model.last_mean_color_pred
        scene_semantic_pred = gs_autoencoder.shape_model.last_scene_semantic_pred
        anchor_pred         = gs_autoencoder.shape_model.last_anchor_pred
        scene_layout_pred   = gs_autoencoder.shape_model.last_scene_layout_pred
        seg_pred_logits     = gs_autoencoder.shape_model.last_seg_pred
        _mu_s               = gs_autoencoder.shape_model._mu_s_cache
        _mu_g               = gs_autoencoder.shape_model._mu_g_cache

        # ── JEPA Idea 1 ───────────────────────────────────────────────────────
        spatial_semantic_pred = None
        if (args.jepa_idea1 and
                gs_autoencoder.shape_model.spatial_semantic_module is not None):
            scaffold_anchors_jepa = batch_data['scaffold_anchors'].float().to(device)
            spatial_semantic_pred = gs_autoencoder.shape_model.spatial_semantic_module(
                gs_autoencoder.shape_model._shape_embed_cache, scaffold_anchors_jepa)

        # ── Reconstruction target ─────────────────────────────────────────────
        target_abs = UV_gs_batch[:, :, GEOMETRIC_INDICES]
        if args.position_scaffold:
            target = target_abs.clone()
            target[:, :, 0:3] = position_offsets
        elif args.position_layout_residual:
            pos_residuals = batch_data['position_residuals'].float().to(device)
            target = target_abs.clone()
            target[:, :, 0:3] = pos_residuals
        else:
            target = target_abs

        pred_3d    = UV_gs_recover.reshape(B, -1, 14)
        recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight)

        # ── KL ────────────────────────────────────────────────────────────────
        KL_loss = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

        # ── Auxiliary losses ──────────────────────────────────────────────────
        color_pred_loss = torch.tensor(0.0, device=device)
        if mean_color_pred is not None and args.color_residual:
            color_pred_loss = F.mse_loss(mean_color_pred, mean_color_gt)

        scene_semantic_loss = torch.tensor(0.0, device=device)
        if scene_semantic_pred is not None and args.scene_semantic_head:
            p_s = batch_data['label_dist'].float().to(device)
            scene_semantic_loss = scene_semantic_kl_loss(scene_semantic_pred, p_s)

        anchor_loss = torch.tensor(0.0, device=device)
        if anchor_pred is not None and args.position_scaffold and scaffold_anchors is not None:
            anchor_loss = F.mse_loss(anchor_pred, scaffold_anchors)

        layout_loss = torch.tensor(0.0, device=device)
        if scene_layout_pred is not None and args.scene_layout_head:
            gt_centroids = batch_data['category_centroids'].float().to(device)
            gt_valid     = batch_data['category_valid'].float().to(device)
            layout_loss  = compute_layout_loss(scene_layout_pred, gt_centroids, gt_valid)

        spatial_loss = torch.tensor(0.0, device=device)
        if spatial_semantic_pred is not None and args.jepa_idea1:
            gt_voxel    = batch_data['voxel_label_dists'].float().to(device)
            voxel_valid = batch_data['voxel_valid'].float().to(device)
            spatial_loss = compute_spatial_semantic_loss(
                spatial_semantic_pred, gt_voxel, voxel_valid)

        # ── NEW: Idea 1 — Segment Prediction Loss ────────────────────────────
        # CE between predicted per-Gaussian segment logits and gt labels.
        # Gradient forces decoder positions to be categorically discriminative.
        seg_pred_loss = torch.tensor(0.0, device=device)
        if args.predict_seg_labels and seg_pred_logits is not None and segment_labels is not None:
            seg_pred_loss = compute_seg_pred_loss(seg_pred_logits, segment_labels)

        # ── InfoNCE ────────────────────────────────────────────────────────────
        semantic_loss    = torch.tensor(0.0, device=device)
        semantic_metrics = {}
        if enable_semantic and segment_labels is not None and per_gaussian_features is not None:
            if args.semantic_mode == 'dist':
                semantic_loss, semantic_metrics = compute_distribution_loss(
                    dist_logits=per_gaussian_features,
                    segment_labels=segment_labels,
                    weight=args.segment_loss_weight)
            else:
                semantic_loss, semantic_metrics = compute_semantic_loss(
                    embeddings=per_gaussian_features,
                    segment_labels=segment_labels,
                    instance_labels=instance_labels,
                    batch_size=B,
                    segment_weight=args.segment_loss_weight,
                    instance_weight=args.instance_loss_weight,
                    temperature=args.semantic_temperature,
                    subsample=args.semantic_subsample,
                    sampling_strategy=args.sampling_strategy)

        # ── Cross-reconstruction loss ─────────────────────────────────────────
        cross_recon_loss = torch.tensor(0.0, device=device)
        if (args.latent_disentangle and args.cross_recon_weight > 0
                and _mu_s is not None and _mu_g is not None and B > 1):
            D_s = args.semantic_dims
            mu_s_shifted = torch.roll(_mu_s, shifts=1, dims=0)
            lv_s_shifted = torch.roll(log_var[:, :D_s], shifts=1, dims=0)
            z_s_swapped  = (mu_s_shifted +
                            torch.exp(0.5 * lv_s_shifted) * torch.randn_like(mu_s_shifted))
            z_g_current  = (_mu_g +
                            torch.exp(0.5 * log_var[:, D_s:]) * torch.randn_like(_mu_g))
            z_cross   = torch.cat([z_s_swapped, z_g_current], dim=-1)
            lat_cross = z_cross.reshape(B, 512, 32)
            se_shifted = torch.roll(
                gs_autoencoder.shape_model._shape_embed_cache, shifts=1, dims=0)
            # Pass scaffold data for cross-recon decode too (Idea 2A and 3)
            UV_cross, _ = gs_autoencoder.shape_model.decode(
                lat_cross, volume_queries=None,
                return_semantic_features=False, shape_embed=se_shifted,
                scaffold_anchors=scaffold_anchors,
                scaffold_token_ids=scaffold_token_ids)
            pred_cross_3d    = UV_cross.reshape(B, -1, 14)
            cross_recon_loss = compute_cross_recon_loss(pred_cross_3d, target, B)

        # ── Orthogonality loss ────────────────────────────────────────────────
        ortho_loss = torch.tensor(0.0, device=device)
        if (args.latent_disentangle and args.ortho_weight > 0
                and _mu_s is not None and _mu_g is not None and B > 1):
            ortho_loss = compute_orthogonality_loss(_mu_s, _mu_g)

        # ── Total loss ────────────────────────────────────────────────────────
        loss = (recon_loss
                + args.kl_weight             * KL_loss
                + args.mean_color_weight     * color_pred_loss
                + args.scene_semantic_weight * scene_semantic_loss
                + args.anchor_loss_weight    * anchor_loss
                + args.layout_loss_weight    * layout_loss
                + args.jepa_idea1_weight     * spatial_loss
                + args.cross_recon_weight    * cross_recon_loss
                + args.ortho_weight          * ortho_loss
                + args.seg_pred_weight       * seg_pred_loss    # NEW
                + semantic_loss)
        loss.backward()
        optimizer.step()

        ind = compute_individual_losses(pred_3d, target)
        epoch_loss           += loss.item()
        epoch_recon          += recon_loss.item()
        epoch_kl             += KL_loss.item()
        epoch_sem            += semantic_loss.item()
        epoch_color_pred     += color_pred_loss.item()
        epoch_scene_semantic += scene_semantic_loss.item()
        epoch_anchor         += anchor_loss.item()
        epoch_layout         += layout_loss.item()
        epoch_spatial        += spatial_loss.item()
        epoch_cross_recon    += cross_recon_loss.item()
        epoch_ortho          += ortho_loss.item()
        epoch_seg_pred       += seg_pred_loss.item()
        epoch_pos += ind['position']
        epoch_col += ind['color']
        epoch_opa += ind['opacity']
        epoch_scl += ind['scale']
        epoch_rot += ind['rotation']

        # First batch diagnostic
        if epoch == start_epoch and i_batch == 0:
            print(f"\nEPOCH {epoch} DIAGNOSTIC (batch 0):")
            print(f"  mu range:    [{mu.min().item():.3f}, {mu.max().item():.3f}]")
            print(f"  recon_loss:  {recon_loss.item():.4f}")
            if args.position_layout_residual:
                pos_resid_np = batch_data['position_residuals'].numpy()
                pos_abs_np   = UV_gs_batch[:, :, 4:7].cpu().numpy()
                reduction    = pos_abs_np.std() / (pos_resid_np.std() + 1e-8)
                print(f"  [PLR] pos_residual range: [{pos_resid_np.min():.3f}, {pos_resid_np.max():.3f}]m")
                print(f"  [PLR] pos_absolute range: [{pos_abs_np.min():.3f}, {pos_abs_np.max():.3f}]m")
                print(f"  [PLR] dynamic range reduction: {reduction:.1f}x  (want >5x)")
            if args.latent_disentangle and _mu_s is not None:
                print(f"  mu_s range:  [{_mu_s.min().item():.3f}, {_mu_s.max().item():.3f}]")
                print(f"  mu_g range:  [{_mu_g.min().item():.3f}, {_mu_g.max().item():.3f}]")
                print(f"  cross_recon: {cross_recon_loss.item():.4f}")
                print(f"  ortho:       {ortho_loss.item():.6f}")
            if args.decoder_pos_enc:
                pe = gs_autoencoder.shape_model.decoder_pos_emb
                print(f"  [PE] decoder_pos_emb range: [{pe.min().item():.4f}, {pe.max().item():.4f}]")
            if args.predict_seg_labels and seg_pred_logits is not None:
                print(f"  [Idea1] seg_pred_loss: {seg_pred_loss.item():.4f}")
            if args.token_cond:
                print(f"  [Idea2] token_cond approach {args.token_cond_approach}")
                if scaffold_anchors is not None:
                    print(f"  [Idea2A] scaffold_anchors range: [{scaffold_anchors.min().item():.3f}, {scaffold_anchors.max().item():.3f}]")
            if args.query_decoder:
                print(f"  [Idea3] query_decoder active: per-Gaussian spatial MLP")
            if args.scene_layout_head and scene_layout_pred is not None:
                print(f"  layout_loss: {layout_loss.item():.4f}")
            if args.color_residual and mean_color_pred is not None:
                print(f"  color_pred:  {color_pred_loss.item():.6f}")

        if wandb_enabled:
            log = {
                "train/step_loss":           loss.item(),
                "train/step_recon":          recon_loss.item(),
                "train/step_kl":             KL_loss.item(),
                "train/step_color_pred":     color_pred_loss.item(),
                "train/step_scene_semantic": scene_semantic_loss.item(),
                "train/step_anchor":         anchor_loss.item(),
                "train/step_layout":         layout_loss.item(),
                "train/step_spatial_kl":     spatial_loss.item(),
                "train/step_cross_recon":    cross_recon_loss.item(),
                "train/step_ortho":          ortho_loss.item(),
                "train/step_seg_pred":       seg_pred_loss.item(),
                "train/step_position":       ind['position'],
                "train/step_color":          ind['color'],
                "train/step_opacity":        ind['opacity'],
                "train/step_scale":          ind['scale'],
                "train/step_rotation":       ind['rotation'],
            }
            if semantic_metrics:
                log.update({f"train/step_{k}": v for k, v in semantic_metrics.items()})
            wandb_run.log(log, step=global_step)

        global_step += 1

    # ── End-of-epoch logging ──────────────────────────────────────────────────
    nb = len(trainDataLoader)
    print(f"\nEpoch {epoch} | Loss={epoch_loss/nb:.4f} | Recon={epoch_recon/nb:.4f} | "
          f"KL={epoch_kl/nb:.4f} | InfoNCE={epoch_sem/nb:.4f} | "
          f"ColorPred={epoch_color_pred/nb:.6f} | SceneSem={epoch_scene_semantic/nb:.4f} | "
          f"Layout={epoch_layout/nb:.4f} | CrossRecon={epoch_cross_recon/nb:.4f} | "
          f"SegPred={epoch_seg_pred/nb:.4f} | Ortho={epoch_ortho/nb:.6f}")
    print(f"  Pos={epoch_pos/nb:.3f} | Col={epoch_col/nb:.3f} | "
          f"Opa={epoch_opa/nb:.3f} | Scl={epoch_scl/nb:.3f} | Rot={epoch_rot/nb:.3f}")

    # ── Validation ────────────────────────────────────────────────────────────
    val_metrics = None
    if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:
        print(f"\n--- Validation (epoch {epoch}) ---")
        val_metrics = evaluate_model(gs_autoencoder, valDataLoader, device, epoch=epoch)
        pos_label = ('(residuals)' if args.position_layout_residual else
                     '(offsets)'  if args.position_scaffold          else '(absolute)')
        print(f"  L2:              {val_metrics['avg_l2_error']:.4f}")
        print(f"  Position:        {val_metrics['position_loss']:.4f}  {pos_label}")
        print(f"  Color:           {val_metrics['color_loss']:.4f}"
              f"{'  (residuals)' if args.color_residual else '  (absolute)'}")
        print(f"  Opacity:         {val_metrics['opacity_loss']:.4f}")
        print(f"  Scale:           {val_metrics['scale_loss']:.4f}")
        print(f"  Rotation:        {val_metrics['rotation_loss']:.4f}")
        if args.color_residual:
            print(f"  ColorPredMSE:    {val_metrics['color_pred_loss']:.6f}")
        if args.scene_semantic_head:
            print(f"  SceneSemanticKL: {val_metrics['scene_semantic_kl']:.4f}")
        if args.position_scaffold:
            print(f"  AnchorMSE:       {val_metrics['anchor_loss']:.4f}")
        if args.scene_layout_head:
            print(f"  LayoutMSE:       {val_metrics['layout_loss']:.4f}")
        if args.predict_seg_labels:
            print(f"  SegPredCE:       {val_metrics['seg_pred_loss']:.4f}")

        if val_metrics['avg_l2_error'] < best_val_loss:
            best_val_loss = val_metrics['avg_l2_error']
            best_epoch    = epoch
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     gs_autoencoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_l2_error':         val_metrics['avg_l2_error'],
                **_ckpt_meta,
            }, os.path.join(save_path, "best_model.pth"))
            print(f"  [NEW BEST] L2={best_val_loss:.4f} saved")

    if wandb_enabled and val_metrics:
        wandb_run.log({
            "val/l2_error":          val_metrics['avg_l2_error'],
            "val/position_loss":     val_metrics['position_loss'],
            "val/color_loss":        val_metrics['color_loss'],
            "val/color_pred_mse":    val_metrics['color_pred_loss'],
            "val/scene_semantic_kl": val_metrics['scene_semantic_kl'],
            "val/anchor_mse":        val_metrics['anchor_loss'],
            "val/layout_mse":        val_metrics['layout_loss'],
            "val/seg_pred_ce":       val_metrics['seg_pred_loss'],
            "val/opacity_loss":      val_metrics['opacity_loss'],
            "val/scale_loss":        val_metrics['scale_loss'],
            "val/rotation_loss":     val_metrics['rotation_loss'],
            "best/val_l2":           best_val_loss,
            "best/epoch":            best_epoch,
            "train/epoch":           epoch,
        }, step=global_step)

    if epoch >= 10 and epoch % 500 == 0:
        torch.save({
            'epoch':      epoch,
            'model_state_dict':     gs_autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss / nb,
            **_ckpt_meta,
        }, os.path.join(save_path, f"epoch_{epoch}.pth"))
        print(f"  Checkpoint saved: epoch_{epoch}.pth")

# ============================================================================
# FINAL SAVE
# ============================================================================

print(f"\n{'='*70}\nTRAINING COMPLETE\n{'='*70}")
final_metrics = evaluate_model(gs_autoencoder, valDataLoader, device,
                               epoch=args.num_epochs - 1)
print(f"\nFinal L2:  {final_metrics['avg_l2_error']:.4f}")
print(f"Best L2:   {best_val_loss:.4f} (epoch {best_epoch})")

torch.save({
    'epoch':        args.num_epochs - 1,
    'model_state_dict':     gs_autoencoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_val_l2': final_metrics['avg_l2_error'],
    'best_val_l2':  best_val_loss,
    'best_epoch':   best_epoch,
    **_ckpt_meta,
    'individual_losses': {k: final_metrics[f'{k}_loss'] for k in PARAM_SLICES},
}, os.path.join(save_path, "final.pth"))

print(f"\nSaved: {save_path}final.pth")
if wandb_enabled:
    wandb_run.summary.update({
        "final_val_l2": final_metrics['avg_l2_error'],
        "best_val_l2":  best_val_loss, "best_epoch": best_epoch})
    wandb_run.finish()
print("Done.")