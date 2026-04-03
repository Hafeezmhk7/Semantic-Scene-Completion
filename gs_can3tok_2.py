"""
Can3Tok Training — INFERENCE-FIXED VERSION
==========================================
KEY CHANGES vs previous version:

1. AnchorPredFromTokens replaces AnchorPositionHead:
   - Previously: shape_embed -> AnchorPositionHead -> predicted_anchors (auxiliary only)
   - Now:        transformer_tokens -> AnchorPredFromTokens -> predicted_anchors (used in decode)
   - anchor_loss now supervises AnchorPredFromTokens via last_predicted_anchors_from_tokens

2. Training target is ABSOLUTE POSITIONS (not smooth offsets):
   - Previously: target[:,:,0:3] = smooth_anchor_offsets  (GT data leaked at inference)
   - Now:        target = target_abs  (absolute xyz, decoder outputs include DC)
   - DC (predicted anchor) is added to position output INSIDE decode()

3. scaffold_token_ids always passed to model when position_scaffold=True:
   - Enables accurate DC assignment at training time
   - At inference: pass scaffold_token_ids=None → fixed j→j*512//40000 used

4. PLY save simplified:
   - Previously: all_preds[:,:,0:3] += GT smooth_anchor  (GT leaked)
   - Now:        decoder output is already absolute (DC added inside decode())
   - Just save decoder output directly (after adding mean_color back for color)

ABLATION TABLE:
  Run A:  color_residual only                               (done, L2=1.43)
  Run H:  color_residual + semantic + disentangle + layout  (done, L2=1.565)
  Run K:  Run H + position_layout_residual                  (done, L2~1.0-1.2)
  Run P:  Run K + decoder_pos_enc                           (done, L2=1.38)
  Run Q:  Run K + predict_seg_labels                        (done, L2=1.54, no benefit)
  Run R:  Run K + token_cond approach A                     (done, L2=0.589)
  Run S:  Run K + token_cond approach B                     (done, unstable)
  Run T:  Run K + token_cond approach AB                    (done, best visual)
  Run T2: Run T + trilinear smoothing                       (done, L2=0.606)
  Run V:  Run T + AnchorPredFromTokens (INFERENCE FIX)      <- LAUNCH THIS
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
from pca_feature_visualization import visualize_semantic_features
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
    # Uses GEO_ONLY_SLICES: position, opacity, scale, rotation.
    # pred_cross_3d passed here must already have DC subtracted from position
    # (see cross-recon block in training loop) so comparison is offset vs offset.
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


def compute_scale_penalty(pred_3d, threshold=0.5):
    """
    Penalise Gaussians whose scale exceeds threshold (metres).
    pred_3d: [B, 40000, 14] — scale values at indices 7:10 are already post-exp
             (GS_decoder applies exp() internally so these are in metres).
    Only the excess above threshold is penalised, not the full scale value.
    This avoids penalising correctly-sized Gaussians.
    """
    scale_pred = pred_3d[:, :, 7:10]   # [B, 40000, 3], post-exp metres
    excess     = torch.clamp(scale_pred - threshold, min=0.0)
    return (excess ** 2).mean()


def compute_seg_pred_loss(seg_logits, segment_labels):
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

parser = argparse.ArgumentParser(description='Can3Tok Training (Inference-Fixed)')

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
                    choices=['random', 'opacity', 'hybrid'])
parser.add_argument('--semantic_mode',        type=str,   default='none',
                    choices=['none', 'hidden', 'geometric', 'attention', 'dist'])
parser.add_argument('--segment_loss_weight',  type=float, default=0.0)
parser.add_argument('--instance_loss_weight', type=float, default=0.0)
parser.add_argument('--semantic_temperature', type=float, default=0.07)
parser.add_argument('--semantic_subsample',   type=int,   default=2000)
parser.add_argument('--sampling_strategy',    type=str,   default='balanced',
                    choices=['random', 'balanced'])
parser.add_argument('--color_residual',       action='store_true', default=False)
parser.add_argument('--mean_color_weight',    type=float, default=1.0)
parser.add_argument('--scene_semantic_head',   action='store_true', default=False)
parser.add_argument('--scene_semantic_weight', type=float, default=0.3)
parser.add_argument('--position_scaffold',     action='store_true', default=False)
parser.add_argument('--anchor_loss_weight',    type=float, default=1.0)
parser.add_argument('--decoder_shape_prepend',     action='store_true', default=False)
parser.add_argument('--decoder_shape_cross_attn',  action='store_true', default=False)
parser.add_argument('--decoder_cross_attn_layers', type=int, default=4)
parser.add_argument('--latent_disentangle',   action='store_true', default=False)
parser.add_argument('--semantic_dims',        type=int, default=512)
parser.add_argument('--cross_recon_weight',   type=float, default=0.5)
parser.add_argument('--ortho_weight',         type=float, default=0.1)
parser.add_argument('--scene_layout_head',    action='store_true', default=False)
parser.add_argument('--layout_loss_weight',   type=float, default=0.3)
parser.add_argument('--position_layout_residual', action='store_true', default=False)
parser.add_argument('--jepa_idea1',           action='store_true', default=False)
parser.add_argument('--jepa_idea1_weight',    type=float, default=1.0)
parser.add_argument('--decoder_pos_enc', action='store_true', default=False)
parser.add_argument('--predict_seg_labels', action='store_true', default=False)
parser.add_argument('--seg_pred_weight', type=float, default=0.3)
parser.add_argument('--token_cond', action='store_true', default=False)
parser.add_argument('--token_cond_approach', type=str, default='A',
                    choices=['A', 'B', 'AB'])
parser.add_argument('--query_decoder', action='store_true', default=False)
parser.add_argument('--label_input',          action='store_true', default=False)
parser.add_argument('--no_label_input',       dest='label_input', action='store_false')
parser.add_argument('--scale_norm_mode',      type=str, default='linear',
                    choices=['log', 'linear'])
parser.add_argument('--color_loss_weight',    type=float, default=1.0)
parser.add_argument('--scale_penalty_weight',    type=float, default=0.0,
    help='Weight for scale penalty loss. 0 = disabled (default). '
         'Penalises Gaussians with scale > scale_penalty_threshold. '
         'Start with 0.1 and adjust. Safe to add to a trained checkpoint.')
parser.add_argument('--scale_penalty_threshold', type=float, default=0.5,
    help='Scale threshold in metres. Gaussians above this are penalised. '
         'Default 0.5m (50cm). For indoor SceneSplat scenes ~0.3-0.5m is reasonable.')
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
parser.add_argument('--wandb_project',        type=str, default='Can3Tok-SceenSplat-7K')
parser.add_argument('--wandb_entity',         type=str, default='3D-SSC')
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
if args.token_cond and 'A' in args.token_cond_approach.upper() and not args.position_scaffold:
    print("[INFO] --token_cond approach A requires --position_scaffold. Enabling.")
    args.position_scaffold = True
if args.token_cond and 'B' in args.token_cond_approach.upper() and not args.scene_layout_head:
    print("[INFO] --token_cond approach B requires --scene_layout_head. Enabling.")
    args.scene_layout_head = True
if args.query_decoder and not args.position_scaffold:
    print("[INFO] --query_decoder requires --position_scaffold. Enabling.")
    args.position_scaffold = True

need_scaffold_data = (args.position_scaffold or args.token_cond or args.query_decoder)

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
        if args.color_residual:           run_name += "_colorresidual"
        if args.scene_semantic_head:      run_name += "_scenesemantic"
        if args.position_scaffold:        run_name += "_scaffold"
        if args.latent_disentangle:       run_name += f"_disentangle{args.semantic_dims}"
        if args.scene_layout_head:        run_name += "_layout"
        if args.position_layout_residual: run_name += "_posresid"
        if args.decoder_pos_enc:          run_name += "_posenc"
        if args.predict_seg_labels:       run_name += "_segpred"
        if args.token_cond:               run_name += f"_tokencond{args.token_cond_approach}"
        if args.query_decoder:            run_name += "_querydec"
        if enable_semantic:               run_name += f"_beta{args.segment_loss_weight}"
        if args.resume_checkpoint:        run_name += "_resumed"
        run_name += "_inferencefixed"
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
if args.color_residual:           tag += "_colorresidual"
if args.scene_semantic_head:      tag += "_scenesemantic"
if args.position_scaffold:        tag += "_scaffold"
if args.decoder_shape_prepend:    tag += "_shapeprepend"
if args.decoder_shape_cross_attn: tag += "_shapecrossattn"
if args.latent_disentangle:       tag += f"_disentangle{args.semantic_dims}"
if args.scene_layout_head:        tag += "_layout"
if args.position_layout_residual: tag += "_posresid"
if args.jepa_idea1:               tag += "_jepa1"
if args.decoder_pos_enc:          tag += "_posenc"
if args.predict_seg_labels:       tag += "_segpred"
if args.token_cond:               tag += f"_tokencond{args.token_cond_approach}"
if args.query_decoder:            tag += "_querydec"
if enable_semantic:               tag += f"_beta{args.segment_loss_weight}"
if not args.use_canonical_norm:   tag += "_raw"
tag += "_inferencefixed"   # marks this as the inference-compatible version

save_path = f"/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/{tag}/"
os.makedirs(save_path, exist_ok=True)

# ============================================================================
# STARTUP SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print(f"CAN3TOK TRAINING — INFERENCE-FIXED")
print(f"{'='*70}")
print(f"  INFERENCE FIX:")
print(f"    AnchorPredFromTokens inside decode() predicts scaffold anchors from z")
print(f"    DC added to decoder positions → output is absolute positions")
print(f"    Training target positions: OFFSETS (coord - GT_hard_anchor, range ~±2m)")
print(f"    DC supervised separately via L_anchor (AnchorPredFromTokens vs GT anchors)")
print(f"    PLY save: decoder output used directly (absolute, DC already inside decode())")
print(f"    Second-stage inference: pass scaffold_token_ids=None → fixed assignment")
print(f"  color_residual:            {args.color_residual}")
print(f"  scene_semantic_head:       {args.scene_semantic_head}")
print(f"  position_scaffold:         {args.position_scaffold}")
print(f"  latent_disentangle:        {args.latent_disentangle}  (semantic_dims={args.semantic_dims})")
print(f"  scene_layout_head:         {args.scene_layout_head}  (weight={args.layout_loss_weight})")
print(f"  position_layout_residual:  {args.position_layout_residual}")
print(f"  decoder_pos_enc:           {args.decoder_pos_enc}")
print(f"  predict_seg_labels:        {args.predict_seg_labels}")
print(f"  token_cond:                {args.token_cond}  approach={args.token_cond_approach}")
print(f"  query_decoder:             {args.query_decoder}")
print(f"  scale_penalty_weight:      {args.scale_penalty_weight}  "
      f"threshold={args.scale_penalty_threshold}m  (0=disabled)")
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
optimizer = torch.optim.AdamW(
    gs_autoencoder.parameters(),
    lr=args.lr, betas=[0.9, 0.999], weight_decay=args.weight_decay)

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
        ('color_residual',             args.color_residual,            False),
        ('label_input',                args.label_input,               False),
        ('position_scaffold',          args.position_scaffold,         False),
        ('decoder_shape_prepend',      args.decoder_shape_prepend,     False),
        ('decoder_shape_cross_attn',   args.decoder_shape_cross_attn,  False),
        ('decoder_cross_attn_layers',  args.decoder_cross_attn_layers, 4),
        ('latent_disentangle',         args.latent_disentangle,        False),
        ('semantic_dims',              args.semantic_dims,              512),
        ('position_layout_residual',   args.position_layout_residual,  False),
    ]:
        saved_val = ckpt.get(flag_name, default_val)
        if saved_val != current_val:
            raise ValueError(
                f"{flag_name} mismatch: checkpoint={saved_val}, current={current_val}.")

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
# LR SCHEDULER
# ============================================================================

import math

def build_lr_lambda(warmup_steps, total_steps, lr_min_ratio):
    cosine_steps = max(total_steps - warmup_steps, 1)
    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        t = current_step - warmup_steps
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * t / cosine_steps))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine_factor
    return lr_lambda

_approx_batches_per_epoch = max(1, (args.train_scenes or 300) // args.batch_size)
_total_steps_full         = _approx_batches_per_epoch * args.num_epochs
_elapsed_steps            = _approx_batches_per_epoch * start_epoch

_lr_lambda = build_lr_lambda(
    warmup_steps  = max(0, args.warmup_steps - _elapsed_steps),
    total_steps   = _total_steps_full - _elapsed_steps,
    lr_min_ratio  = args.lr_min_ratio)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

print(f"\n  LR SCHEDULER: linear warmup + cosine decay")
print(f"    peak LR: {args.lr:.2e}  |  floor LR: {args.lr * args.lr_min_ratio:.2e}")
print(f"    warmup steps: {args.warmup_steps}")

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
    'decoder_pos_enc':            args.decoder_pos_enc,
    'predict_seg_labels':         args.predict_seg_labels,
    'token_cond':                 args.token_cond,
    'token_cond_approach':        args.token_cond_approach,
    'query_decoder':              args.query_decoder,
    'inference_fixed':            True,   # marks this checkpoint as inference-compatible
    'mean_color_weight':          args.mean_color_weight,
    'scene_semantic_weight':      args.scene_semantic_weight,
    'anchor_loss_weight':         args.anchor_loss_weight,
    'cross_recon_weight':         args.cross_recon_weight,
    'ortho_weight':               args.ortho_weight,
    'layout_loss_weight':         args.layout_loss_weight,
    'jepa_idea1_weight':          args.jepa_idea1_weight,
    'seg_pred_weight':            args.seg_pred_weight,
    'color_loss_weight':          args.color_loss_weight,
    'scale_penalty_weight':       args.scale_penalty_weight,
    'scale_penalty_threshold':    args.scale_penalty_threshold,
    'use_canonical_norm':         args.use_canonical_norm,
    'scale_norm_mode':            args.scale_norm_mode,
    'weight_decay':               args.weight_decay,
    'warmup_steps':               args.warmup_steps,
    'lr_min_ratio':               args.lr_min_ratio,
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
    total_seg_pred     = 0.0
    per_param  = {k: 0.0 for k in PARAM_SLICES}
    n_scenes   = 0
    recon_preds_list   = []
    recon_means_list   = []
    do_recon = (epoch is not None and epoch % args.recon_ply_freq == 0)

    pca_input_list = []
    pca_recon_list = []
    pca_seg_list   = []
    do_pca = (epoch is not None and epoch % args.pca_vis_freq == 0)

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating", leave=False):
            UV_gs_batch   = batch_data['features'].float().to(device)
            mean_color_gt = batch_data['mean_color'].float().to(device)
            B = UV_gs_batch.shape[0]

            sa_gpu  = (batch_data['scaffold_anchors'].float().to(device)
                       if need_scaffold_data else None)
            # Always pass scaffold_token_ids when position_scaffold for accurate DC
            sti_gpu = (batch_data['scaffold_token_ids'].long().to(device)
                       if args.position_scaffold else None)

            (shape_embed, mu, log_var, z,
             UV_gs_recover, _) = model(
                UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3],
                scaffold_anchors=sa_gpu,
                scaffold_token_ids=sti_gpu)

            mean_color_pred     = model.shape_model.last_mean_color_pred
            scene_semantic_pred = model.shape_model.last_scene_semantic_pred
            # CHANGED: use AnchorPredFromTokens prediction
            anchor_pred         = model.shape_model.last_predicted_anchors_from_tokens
            scene_layout_pred   = model.shape_model.last_scene_layout_pred
            seg_pred            = model.shape_model.last_seg_pred

            # TARGET — matches training: offset supervision for scaffold path
            target_abs = UV_gs_batch[:, :, GEOMETRIC_INDICES]

            if args.position_scaffold:
                # Hard offsets as target; subtract predicted DC from pred to compare offset vs offset
                pos_offset_gt = batch_data['position_offsets'].float().to(device)
                target = target_abs.clone()
                target[:, :, 0:3] = pos_offset_gt

                pred_3d = UV_gs_recover.reshape(B, -1, 14).clone()
                if anchor_pred is not None and sti_gpu is not None:
                    idx_3d  = sti_gpu.unsqueeze(-1).expand(-1, -1, 3)
                    pred_dc = torch.gather(anchor_pred, 1, idx_3d)
                    pred_3d[:, :, 0:3] = pred_3d[:, :, 0:3] - pred_dc
            elif args.position_layout_residual:
                pos_residuals = batch_data['position_residuals'].float().to(device)
                target = target_abs.clone()
                target[:, :, 0:3] = pos_residuals
                pred_3d = UV_gs_recover.reshape(B, -1, 14)
            else:
                target  = target_abs
                pred_3d = UV_gs_recover.reshape(B, -1, 14)

            recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight)
            kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

            if mean_color_pred is not None and args.color_residual:
                total_color_pred += F.mse_loss(mean_color_pred, mean_color_gt).item() * B
            if scene_semantic_pred is not None and args.scene_semantic_head:
                p_s = batch_data['label_dist'].float().to(device)
                total_scene_sem_kl += scene_semantic_kl_loss(
                    scene_semantic_pred, p_s).item() * B
            # CHANGED: anchor_loss now supervises AnchorPredFromTokens
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

            # PLY save: decoder output is already absolute (DC added inside decode())
            # Just add mean_color back if color_residual, then save directly.
            if do_recon and len(recon_preds_list) < args.recon_ply_num_scenes:
                preds_np = pred_3d.cpu().numpy()
                means_np = mean_color_gt.cpu().numpy()
                for si in range(B):
                    if len(recon_preds_list) >= args.recon_ply_num_scenes:
                        break
                    recon_preds_list.append(preds_np[si])
                    recon_means_list.append(means_np[si])

            if do_pca and len(pca_input_list) < args.pca_num_scenes:
                seg_np = batch_data['segment_labels'].numpy()
                inp_np = UV_gs_batch.cpu().numpy()
                rec_np = pred_3d.cpu().numpy()
                for si in range(B):
                    if len(pca_input_list) >= args.pca_num_scenes:
                        break
                    pca_input_list.append(inp_np[si])
                    pca_recon_list.append(rec_np[si])
                    pca_seg_list.append(seg_np[si])

    # PLY reconstruction: decoder output positions are already absolute
    if do_recon and recon_preds_list and save_path:
        try:
            all_preds = np.stack(recon_preds_list, axis=0)

            # Add mean color back (color residual path)
            if args.color_residual:
                for si in range(len(all_preds)):
                    all_preds[si, :, 3:6] += recon_means_list[si]
                    all_preds[si, :, 3:6]  = np.clip(all_preds[si, :, 3:6], 0, 1)

            # CHANGED: NO smooth_anchor addition here.
            # Positions are already absolute — AnchorPredFromTokens DC was added inside decode().
            # This is exactly what second-stage inference will do: use decoder output directly.

            recon_dir = Path(save_path) / "reconstructed_gaussians" / f"epoch_{epoch:03d}"
            save_reconstructed_gaussians(
                predictions=all_preds, output_dir=recon_dir, epoch=epoch,
                num_scenes=len(all_preds), max_sh_degree=args.recon_ply_max_sh,
                color_mode="1", prefix="scene")
        except Exception as e:
            print(f"  PLY save error: {e}")

    # PCA visualisation
    if do_pca and pca_input_list and save_path:
        try:
            pca_dir = Path(save_path) / "pca_visualisations" / f"epoch_{epoch:03d}"
            all_inputs = np.stack(pca_input_list, axis=0)
            all_recons = np.stack(pca_recon_list, axis=0)
            all_segs   = np.stack(pca_seg_list,   axis=0)
            pca_dir.mkdir(parents=True, exist_ok=True)

            for si in range(len(pca_input_list)):
                coords_in  = all_inputs[si, :, 4:7]
                feats_in   = all_inputs[si]
                feats_rec  = all_recons[si]

                out_input = str(pca_dir / f"scene{si:02d}_input.ply")
                visualize_semantic_features(coords=coords_in, features=feats_in,
                                            output_path=out_input, brightness=args.pca_brightness)

                # CHANGED: decoder output positions are already absolute, no smooth_anchor needed
                coords_rec = feats_rec[:, 0:3].copy()   # already absolute
                out_recon = str(pca_dir / f"scene{si:02d}_recon.ply")
                visualize_semantic_features(coords=coords_rec, features=feats_rec,
                                            output_path=out_recon, brightness=args.pca_brightness)

            print(f"  PCA PLY saved → {pca_dir}  ({len(pca_input_list)} scenes)")
        except Exception as e:
            import traceback
            print(f"  PCA error: {e}")
            traceback.print_exc()

    model.train()
    n = max(n_scenes, 1)
    return {
        'avg_l2_error':      total_l2,
        'avg_kl':            total_kl / n,
        'color_pred_loss':   total_color_pred / n,
        'scene_semantic_kl': total_scene_sem_kl / n,
        'anchor_loss':       total_anchor_loss / n,
        'layout_loss':       total_layout_loss / n,
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
    epoch_seg_pred = epoch_scale_penalty = 0.0
    epoch_pos = epoch_col = epoch_opa = epoch_scl = epoch_rot = 0.0

    for i_batch, batch_data in enumerate(trainDataLoader):
        UV_gs_batch   = batch_data['features'].float().to(device)
        mean_color_gt = batch_data['mean_color'].float().to(device)
        B = UV_gs_batch.shape[0]

        segment_labels  = None
        instance_labels = None
        if need_segment_labels:
            segment_labels  = batch_data['segment_labels'].long().to(device)
            if enable_semantic:
                instance_labels = batch_data['instance_labels'].long().to(device)

        scaffold_anchors   = None
        scaffold_token_ids = None
        if need_scaffold_data:
            scaffold_anchors   = batch_data['scaffold_anchors'].float().to(device)
            scaffold_token_ids = batch_data['scaffold_token_ids'].long().to(device)

        optimizer.zero_grad()

        sa_gpu  = scaffold_anchors   if need_scaffold_data      else None
        # CHANGED: always pass scaffold_token_ids when position_scaffold
        # This enables accurate DC assignment inside decode()
        sti_gpu = scaffold_token_ids if args.position_scaffold   else None

        (shape_embed, mu, log_var, z,
         UV_gs_recover, per_gaussian_features) = gs_autoencoder(
            UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3],
            scaffold_anchors=sa_gpu,
            scaffold_token_ids=sti_gpu)

        mean_color_pred     = gs_autoencoder.shape_model.last_mean_color_pred
        scene_semantic_pred = gs_autoencoder.shape_model.last_scene_semantic_pred
        # CHANGED: AnchorPredFromTokens replaces AnchorPositionHead
        anchor_pred         = gs_autoencoder.shape_model.last_predicted_anchors_from_tokens
        scene_layout_pred   = gs_autoencoder.shape_model.last_scene_layout_pred
        seg_pred_logits     = gs_autoencoder.shape_model.last_seg_pred
        _mu_s               = gs_autoencoder.shape_model._mu_s_cache
        _mu_g               = gs_autoencoder.shape_model._mu_g_cache

        spatial_semantic_pred = None
        if (args.jepa_idea1 and
                gs_autoencoder.shape_model.spatial_semantic_module is not None):
            scaffold_anchors_jepa = batch_data['scaffold_anchors'].float().to(device)
            spatial_semantic_pred = gs_autoencoder.shape_model.spatial_semantic_module(
                gs_autoencoder.shape_model._shape_embed_cache, scaffold_anchors_jepa)

        # RECONSTRUCTION TARGET
        # OFFSET SUPERVISION (position_scaffold path):
        #   L_recon supervises raw position offsets (coord - GT_hard_anchor, range ~±2m).
        #   L_anchor separately supervises the DC term (AnchorPredFromTokens vs GT anchors).
        #   This keeps the two losses orthogonal and reduces position dynamic range 5×.
        #   pred_3d positions = raw_offset + predicted_DC (added inside decode()).
        #   We subtract predicted_DC before computing L_recon so we compare offset vs offset.
        #   PLY save is unaffected — decoder output is still absolute (DC stays in output).
        target_abs = UV_gs_batch[:, :, GEOMETRIC_INDICES]

        if args.position_scaffold:
            # Build offset target: coord - GT_scaffold_anchors[scaffold_token_ids]
            # This is the hard offset already in the batch as 'position_offsets'.
            pos_offset_gt = batch_data['position_offsets'].float().to(device)  # [B, 40000, 3]
            target = target_abs.clone()
            target[:, :, 0:3] = pos_offset_gt   # range ~±2m vs ±10m absolute

            # Subtract predicted DC from pred_3d positions so loss is offset vs offset.
            # anchor_pred is None only if position_scaffold=False, which cannot happen here.
            pred_3d = UV_gs_recover.reshape(B, -1, 14).clone()
            if anchor_pred is not None:
                # Gather predicted anchor for each Gaussian using GT spatial assignment
                idx_3d  = scaffold_token_ids.long().unsqueeze(-1).expand(-1, -1, 3)
                pred_dc = torch.gather(anchor_pred, 1, idx_3d)   # [B, 40000, 3]
                pred_3d[:, :, 0:3] = pred_3d[:, :, 0:3] - pred_dc  # back to raw offset
        elif args.position_layout_residual:
            pos_residuals = batch_data['position_residuals'].float().to(device)
            target = target_abs.clone()
            target[:, :, 0:3] = pos_residuals
            pred_3d = UV_gs_recover.reshape(B, -1, 14)
        else:
            target  = target_abs
            pred_3d = UV_gs_recover.reshape(B, -1, 14)

        recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight)

        KL_loss = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

        color_pred_loss = torch.tensor(0.0, device=device)
        if mean_color_pred is not None and args.color_residual:
            color_pred_loss = F.mse_loss(mean_color_pred, mean_color_gt)

        scene_semantic_loss = torch.tensor(0.0, device=device)
        if scene_semantic_pred is not None and args.scene_semantic_head:
            p_s = batch_data['label_dist'].float().to(device)
            scene_semantic_loss = scene_semantic_kl_loss(scene_semantic_pred, p_s)

        # CHANGED: anchor_loss now supervises AnchorPredFromTokens (inside decoder)
        # vs GT scaffold_anchors from dataset.
        # Gradient: L_anchor → AnchorPredFromTokens → transformer tokens → post_kl → z
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

        seg_pred_loss = torch.tensor(0.0, device=device)
        if args.predict_seg_labels and seg_pred_logits is not None and segment_labels is not None:
            seg_pred_loss = compute_seg_pred_loss(seg_pred_logits, segment_labels)

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
            UV_cross, _ = gs_autoencoder.shape_model.decode(
                lat_cross, volume_queries=None,
                return_semantic_features=False, shape_embed=se_shifted,
                scaffold_anchors=scaffold_anchors,
                scaffold_token_ids=scaffold_token_ids)
            pred_cross_3d = UV_cross.reshape(B, -1, 14)

            # CROSS-RECON POSITION FIX — offset space comparison.
            # pred_cross_3d positions = raw_offset + cross_DC, where cross_DC comes
            # from AnchorPredFromTokens run on the mixed latent z_cross=[mu_s_B,mu_g_A].
            # That DC is spatially incoherent (neither scene A nor B).
            # Solution: subtract cross_DC (DETACHED) from pred positions to recover
            # raw offsets, then compare against the offset target.
            # Detaching cross_DC ensures gradients from cross-recon do NOT flow through
            # AnchorPredFromTokens — preventing the anchor-collapse-to-zero that caused
            # all positions to compress into a blob in SuperSplat.
            # This gives position cross-recon in offset space, consistent with
            # offset supervision, without contaminating the DC gradient path.
            if args.position_scaffold:
                cross_anchors = gs_autoencoder.shape_model.last_predicted_anchors_from_tokens
                if cross_anchors is not None and scaffold_token_ids is not None:
                    idx_3d_cr = scaffold_token_ids.long().unsqueeze(-1).expand(-1, -1, 3)
                    cross_dc  = torch.gather(cross_anchors, 1, idx_3d_cr).detach()
                    pred_cross_for_loss = pred_cross_3d.clone()
                    pred_cross_for_loss[:, :, 0:3] = pred_cross_3d[:, :, 0:3] - cross_dc
                else:
                    pred_cross_for_loss = pred_cross_3d
            else:
                pred_cross_for_loss = pred_cross_3d

            cross_recon_loss = compute_cross_recon_loss(pred_cross_for_loss, target, B)

        ortho_loss = torch.tensor(0.0, device=device)
        if (args.latent_disentangle and args.ortho_weight > 0
                and _mu_s is not None and _mu_g is not None and B > 1):
            ortho_loss = compute_orthogonality_loss(_mu_s, _mu_g)

        # Scale penalty — penalises Gaussians with scale > threshold (metres).
        # pred_3d[:,:,7:10] are post-exp scale values. DC subtraction only affected
        # position (0:3) so scale values here are the true decoder outputs.
        # Safe to enable on a pre-trained checkpoint — acts as fine-tuning signal.
        scale_penalty_loss = torch.tensor(0.0, device=device)
        if args.scale_penalty_weight > 0:
            # Use original UV_gs_recover (not pred_3d with DC subtracted) for scale.
            # Both are identical for scale — DC subtraction only touches position — but
            # using the raw output makes the intent clearer.
            raw_pred_for_scale = UV_gs_recover.reshape(B, -1, 14)
            scale_penalty_loss = compute_scale_penalty(
                raw_pred_for_scale, threshold=args.scale_penalty_threshold)

        loss = (recon_loss
                + args.kl_weight             * KL_loss
                + args.mean_color_weight     * color_pred_loss
                + args.scene_semantic_weight * scene_semantic_loss
                + args.anchor_loss_weight    * anchor_loss
                + args.layout_loss_weight    * layout_loss
                + args.jepa_idea1_weight     * spatial_loss
                + args.cross_recon_weight    * cross_recon_loss
                + args.ortho_weight          * ortho_loss
                + args.seg_pred_weight       * seg_pred_loss
                + args.scale_penalty_weight  * scale_penalty_loss
                + semantic_loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

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
        epoch_scale_penalty  += scale_penalty_loss.item()
        epoch_pos += ind['position']
        epoch_col += ind['color']
        epoch_opa += ind['opacity']
        epoch_scl += ind['scale']
        epoch_rot += ind['rotation']

        if epoch == start_epoch and i_batch == 0:
            print(f"\nEPOCH {epoch} DIAGNOSTIC (batch 0):")
            print(f"  mu range:        [{mu.min().item():.3f}, {mu.max().item():.3f}]")
            print(f"  recon_loss:      {recon_loss.item():.4f}  (vs position OFFSETS ±~2m)")
            if args.position_scaffold:
                pos_abs_np    = UV_gs_batch[:, :, 4:7].cpu().numpy()
                pos_offset_np = batch_data['position_offsets'].numpy()
                pred_pos_off  = pred_3d[:, :, 0:3].detach().cpu().numpy()
                print(f"  [OFFSET SUPERVISION] GT abs pos range:   [{pos_abs_np.min():.3f}, {pos_abs_np.max():.3f}]m")
                print(f"  [OFFSET SUPERVISION] GT offset range:    [{pos_offset_np.min():.3f}, {pos_offset_np.max():.3f}]m  (~5x smaller)")
                print(f"  [OFFSET SUPERVISION] Pred offset range:  [{pred_pos_off.min():.3f}, {pred_pos_off.max():.3f}]m")
                if anchor_pred is not None:
                    anch_np = anchor_pred.detach().cpu().numpy()
                    print(f"  [AnchorPredFromTokens] range: [{anch_np.min():.3f}, {anch_np.max():.3f}]m")
                    print(f"  anchor_loss: {anchor_loss.item():.4f}  (DC supervised separately)")
            if args.latent_disentangle and _mu_s is not None:
                print(f"  mu_s range:  [{_mu_s.min().item():.3f}, {_mu_s.max().item():.3f}]")
                print(f"  cross_recon: {cross_recon_loss.item():.4f}")
            if args.scale_penalty_weight > 0:
                scale_np = raw_pred_for_scale[:, :, 7:10].detach().cpu().numpy()
                print(f"  [SCALE PENALTY] mean scale: {scale_np.mean():.4f}m  "
                      f"max: {scale_np.max():.4f}m  threshold: {args.scale_penalty_threshold}m")
                print(f"  [SCALE PENALTY] frac above threshold: "
                      f"{(scale_np > args.scale_penalty_threshold).mean()*100:.1f}%")
                print(f"  scale_penalty_loss: {scale_penalty_loss.item():.6f}")

        if wandb_enabled:
            log = {
                "train/step_loss":           loss.item(),
                "train/step_recon":          recon_loss.item(),
                "train/step_kl":             KL_loss.item(),
                "train/step_color_pred":     color_pred_loss.item(),
                "train/step_scene_semantic": scene_semantic_loss.item(),
                "train/step_anchor":         anchor_loss.item(),
                "train/step_layout":         layout_loss.item(),
                "train/step_cross_recon":    cross_recon_loss.item(),
                "train/step_ortho":          ortho_loss.item(),
                "train/step_seg_pred":       seg_pred_loss.item(),
                "train/step_scale_penalty":  scale_penalty_loss.item(),
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

    nb = len(trainDataLoader)
    current_lr = scheduler.get_last_lr()[0]
    print(f"\nEpoch {epoch} | Loss={epoch_loss/nb:.4f} | Recon={epoch_recon/nb:.4f} | "
          f"KL={epoch_kl/nb:.4f} | Anchor={epoch_anchor/nb:.4f} | "
          f"ColorPred={epoch_color_pred/nb:.6f} | SceneSem={epoch_scene_semantic/nb:.4f} | "
          f"Layout={epoch_layout/nb:.4f} | CrossRecon={epoch_cross_recon/nb:.4f} | "
          f"SegPred={epoch_seg_pred/nb:.4f} | ScalePenalty={epoch_scale_penalty/nb:.6f} | "
          f"Ortho={epoch_ortho/nb:.6f} | LR={current_lr:.2e}")
    print(f"  Pos={epoch_pos/nb:.3f} | Col={epoch_col/nb:.3f} | "
          f"Opa={epoch_opa/nb:.3f} | Scl={epoch_scl/nb:.3f} | Rot={epoch_rot/nb:.3f}")

    val_metrics = None
    if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:
        print(f"\n--- Validation (epoch {epoch}) ---")
        val_metrics = evaluate_model(gs_autoencoder, valDataLoader, device, epoch=epoch)
        pos_label = ('(offsets ±~2m, DC separate)' if args.position_scaffold else
                     '(residuals)' if args.position_layout_residual else '(absolute)')
        print(f"  L2:              {val_metrics['avg_l2_error']:.4f}")
        print(f"  Position:        {val_metrics['position_loss']:.4f}  {pos_label}")
        print(f"  Color:           {val_metrics['color_loss']:.4f}")
        print(f"  Opacity:         {val_metrics['opacity_loss']:.4f}")
        print(f"  Scale:           {val_metrics['scale_loss']:.4f}")
        print(f"  Rotation:        {val_metrics['rotation_loss']:.4f}")
        if args.color_residual:
            print(f"  ColorPredMSE:    {val_metrics['color_pred_loss']:.6f}")
        if args.scene_semantic_head:
            print(f"  SceneSemanticKL: {val_metrics['scene_semantic_kl']:.4f}")
        if args.position_scaffold:
            print(f"  AnchorMSE:       {val_metrics['anchor_loss']:.4f}  (AnchorPredFromTokens)")
        if args.scene_layout_head:
            print(f"  LayoutMSE:       {val_metrics['layout_loss']:.4f}")
        if args.predict_seg_labels:
            print(f"  SegPredCE:       {val_metrics['seg_pred_loss']:.4f}")
        if args.scale_penalty_weight > 0:
            print(f"  ScalePenalty:    {val_metrics['scale_loss']:.4f}  "
                  f"(Scl loss shown; threshold={args.scale_penalty_threshold}m)")

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
            "train/lr":              current_lr,
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
print(f"\nINFERENCE NOTE:")
print(f"  At second-stage diffusion inference, call decode() with scaffold_token_ids=None.")
print(f"  AnchorPredFromTokens uses fixed assignment j→j*512//40000 for DC.")
print(f"  Decoder output positions are absolute (raw_offset + predicted_DC).")
print(f"  Add mean_color back if color_residual=True, then save PLY directly.")
print(f"  L_recon was trained on offsets; L_anchor trained DC separately — both are baked into weights.")
if wandb_enabled:
    wandb_run.summary.update({
        "final_val_l2": final_metrics['avg_l2_error'],
        "best_val_l2":  best_val_loss, "best_epoch": best_epoch})
    wandb_run.finish()
print("Done.")