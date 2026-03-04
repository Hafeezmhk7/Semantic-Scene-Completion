"""
Can3Tok Training — Step 1: Color Residual + Move 1: Scene Semantic Head
========================================================================

WHAT THIS FILE IMPLEMENTS
--------------------------
This is the complete training script for Can3Tok VAE with:
  - Step 1 Color Residual (--color_residual)
  - Move 1 Scene Semantic Head (--scene_semantic_head) ← NEW

MOVE 1 — SCENE SEMANTIC HEAD
------------------------------
Adds SceneSemanticHead to shape_embed: a second explicit gradient path
into the global scene token alongside MeanColorHead.

  MODEL (sal_perceiver_II_initialization.py):
    SceneSemanticHead: shape_embed [B,384] -> MLP(128->128) -> [B,72] softmax
    Stored as self.last_scene_semantic_pred (instance attribute).

  TRAINING LOOP (this file):
    p_s   = compute_label_distribution(segment_labels)   # [B, 72] gt dist
    p_hat = gs_autoencoder.shape_model.last_scene_semantic_pred  # [B, 72]
    scene_semantic_loss = scene_semantic_kl_loss(p_hat, p_s)
    total_loss += scene_semantic_weight * scene_semantic_loss

  Gradient path: KL_loss -> SceneSemanticHead -> shape_embed -> encoder token 0
  Independent of MeanColorHead and InfoNCE paths.

CHECKPOINT METADATA
-------------------
  scene_semantic_head: bool — stored in every checkpoint so probe/resume
  scripts can auto-detect and build the correct model architecture.

ABLATION TABLE (runs to compare):
  Run A: color_residual only              (done: job 20207686, L2=1.43)
  Run B: color_residual + InfoNCE hidden  (done: job 20231337, L2=1.99)
  Run C: color_residual + scene_semantic  (NEW: isolates Move 1)
  Run D: color_residual + scene_semantic + InfoNCE hidden (NEW: full model)
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
# PARAMETER INDICES  (identical for label_input=True/False)
# ============================================================================

PARAM_SLICES = {
    'position': slice(0, 3),
    'color':    slice(3, 6),
    'opacity':  slice(6, 7),
    'scale':    slice(7, 10),
    'rotation': slice(10, 14),
}

GEOMETRIC_INDICES = (
    list(range(4, 7))   # xyz
  + list(range(7, 10))  # rgb / residuals
  + [10]                # opacity
  + list(range(11, 14)) # scale
  + list(range(14, 18)) # quaternion
)  # 14 values, cols 4-17

# ============================================================================
# MOVE 1 HELPERS
# ============================================================================

# def compute_label_distribution(
#     segment_labels: torch.Tensor,
#     n_labels: int = 72,
# ) -> torch.Tensor:
#     """
#     Compute per-scene ScanNet72 label distribution from raw segment labels.

#     segment_labels: [B, N]  long tensor, values 0-71 (valid) or <0 (missing)
#     returns:        [B, 72] float tensor, rows sum to 1.0
#                     All-zero row if scene has no valid labels.

#     Uses vectorised torch ops — no Python loop over batch items.
#     This is the ground truth that SceneSemanticHead is trained to predict.
#     Same computation as in probe_label_distribution.py LabelDistributionDataset.
#     """
#     B, N = segment_labels.shape
#     device = segment_labels.device

#     p_s = torch.zeros(B, n_labels, dtype=torch.float32, device=device)

#     # One-hot encode valid labels, ignore negative (missing) labels
#     valid_mask = segment_labels >= 0                             # [B, N]
#     safe_labels = segment_labels.clone()
#     safe_labels[~valid_mask] = 0                                 # clamp for one_hot, masked out next

#     one_hot = F.one_hot(safe_labels, num_classes=n_labels).float()  # [B, N, 72]
#     one_hot = one_hot * valid_mask.unsqueeze(-1).float()             # zero out invalid

#     counts     = one_hot.sum(dim=1)                              # [B, 72]
#     n_valid    = valid_mask.float().sum(dim=1, keepdim=True)     # [B, 1]
#     n_valid    = n_valid.clamp(min=1.0)                          # avoid div by zero

#     p_s = counts / n_valid                                       # [B, 72]  sums to 1

#     return p_s


def scene_semantic_kl_loss(
    p_hat: torch.Tensor,
    p_s:   torch.Tensor,
    eps:   float = 1e-8,
) -> torch.Tensor:
    """
    KL divergence loss: D_KL(p_s || p_hat) for scene-level semantic distribution.

    p_hat: [B, 72]  model predictions (softmax, sums to 1)
    p_s:   [B, 72]  ground truth distributions (sums to 1)
    returns: scalar mean KL divergence across batch

    D_KL(p_s || p_hat) = Σ p_s * log(p_s / p_hat)

    Scenes where p_s is all-zero (no valid labels) contribute zero loss
    because p_s * log(...) = 0. No masking needed.
    """
    p_hat_clamped = torch.clamp(p_hat, min=eps)
    kl_per_scene  = (p_s * (torch.log(p_s + eps) - torch.log(p_hat_clamped))).sum(dim=-1)
    return kl_per_scene.mean()

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

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Can3Tok Training')

# Core training
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

# Semantic (per-Gaussian InfoNCE)
parser.add_argument('--semantic_mode',        type=str,   default='none',
                    choices=['none', 'hidden', 'geometric', 'attention', 'dist'])
parser.add_argument('--segment_loss_weight',  type=float, default=0.0)
parser.add_argument('--instance_loss_weight', type=float, default=0.0)
parser.add_argument('--semantic_temperature', type=float, default=0.07)
parser.add_argument('--semantic_subsample',   type=int,   default=2000)
parser.add_argument('--sampling_strategy',    type=str,   default='balanced',
                    choices=['random', 'balanced'])

# Step 1: Color residual
parser.add_argument('--color_residual',       action='store_true', default=False,
    help='Enable Step 1 color residual. Dataset stores color-mean_color; '
         'model predicts mean_color from shape_embed via MeanColorHead.')
parser.add_argument('--mean_color_weight',    type=float, default=1.0,
    help='Weight on MSE(mean_color_pred, mean_color_gt).')

# Move 1: Scene semantic head  ← NEW
parser.add_argument('--scene_semantic_head',   action='store_true', default=False,
    help='Move 1: Add SceneSemanticHead to shape_embed. Predicts per-scene '
         'ScanNet72 label distribution [B,72] from shape_embed. '
         'Loss: KL(p_s || p_hat) weighted by --scene_semantic_weight. '
         'Gives shape_embed a 2nd gradient path alongside MeanColorHead.')
parser.add_argument('--scene_semantic_weight', type=float, default=0.3,
    help='Weight for SceneSemanticHead KL loss (default 0.3). '
         'total_loss += scene_semantic_weight * KL(p_s || p_hat).')

# Option 1: label as encoder input
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

# Resuming
parser.add_argument('--resume_checkpoint',    type=str, default=None)
parser.add_argument('--resume_epoch',         type=int, default=None)

args = parser.parse_args()

# ── Semantic activation ───────────────────────────────────────────────────────
semantic_requested    = (args.semantic_mode != 'none')
semantic_loss_enabled = (args.segment_loss_weight > 0 or args.instance_loss_weight > 0)
enable_semantic       = semantic_requested and semantic_loss_enabled
effective_semantic_mode = args.semantic_mode if enable_semantic else 'none'

# Need segment_labels if InfoNCE OR SceneSemanticHead is active
need_segment_labels = enable_semantic or args.scene_semantic_head

# ============================================================================
# W&B
# ============================================================================

wandb_enabled = False
if args.use_wandb:
    try:
        import wandb
        job_id   = os.environ.get('SLURM_JOB_ID', 'local')
        run_name = f"can3tok_job_{job_id}_{effective_semantic_mode}"
        if args.color_residual:       run_name += "_colorresidual"
        if args.scene_semantic_head:  run_name += "_scenesemantic"
        if args.label_input:          run_name += "_labelinput"
        if enable_semantic:           run_name += f"_beta{args.segment_loss_weight}"
        if args.resume_checkpoint:    run_name += "_resumed"
        wandb_run = wandb.init(
            entity=args.wandb_entity, project=args.wandb_project, name=run_name,
            config=vars(args)
        )
        wandb_enabled = True
        print("W&B enabled")
    except Exception as e:
        print(f"W&B failed: {e}")

# ============================================================================
# DEVICE
# ============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================================
# DATA PATHS
# ============================================================================

data_path = "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs"

# ============================================================================
# CHECKPOINT FOLDER
# ============================================================================

job_id = os.environ.get('SLURM_JOB_ID', None)
tag = f"RGB_job_{job_id}_{effective_semantic_mode}" if job_id else \
      f"RGB_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{effective_semantic_mode}"

if args.color_residual:       tag += "_colorresidual"
if args.scene_semantic_head:  tag += "_scenesemantic"
if args.label_input:          tag += "_labelinput"
if enable_semantic:           tag += f"_beta{args.segment_loss_weight}"
if not args.use_canonical_norm: tag += "_raw"

save_path = f"/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/{tag}/"
os.makedirs(save_path, exist_ok=True)

# ============================================================================
# STARTUP SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print(f"CAN3TOK TRAINING")
print(f"{'='*70}")
print(f"  Semantic mode:         {effective_semantic_mode}")
print(f"  Enable semantic:       {enable_semantic}")
print(f"  Color residual:        {args.color_residual}  (Step 1 — MeanColorHead)")
print(f"  Mean color weight:     {args.mean_color_weight}")
print(f"  Scene semantic head:   {args.scene_semantic_head}  (Move 1 — SceneSemanticHead)")
print(f"  Scene semantic weight: {args.scene_semantic_weight}")
print(f"  Label input:           {args.label_input}")
print(f"  Scale norm mode:       {args.scale_norm_mode}")
print(f"  Canonical norm:        {args.use_canonical_norm}")
print(f"  Device:                {device}")
print(f"  Save path:             {save_path}")
if args.resume_checkpoint:
    print(f"  Resume from:           {args.resume_checkpoint}")
if args.scene_semantic_head:
    print(f"\n  Move 1 ACTIVE: shape_embed -> SceneSemanticHead -> [B,72] softmax")
    print(f"    Loss: KL(p_s || p_hat), weight={args.scene_semantic_weight}")
    print(f"    Supervision: ScanNet72 label dist from segment.npy")
    print(f"    Gradient: independent of MeanColorHead and InfoNCE paths")
print(f"{'='*70}\n")

# ============================================================================
# MODEL
# ============================================================================

print("Loading model config...")
config_path = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
model_config_perceiver = get_config_from_file(config_path)
model_config = model_config_perceiver.model
model_config.params.shape_module_cfg.params.semantic_mode     = effective_semantic_mode
model_config.params.shape_module_cfg.params.color_residual    = args.color_residual
model_config.params.shape_module_cfg.params.scene_semantic_head = args.scene_semantic_head  # Move 1

cfg_point_feats = model_config.params.shape_module_cfg.params.point_feats
expected_feats  = 12 if args.label_input else 11
if cfg_point_feats != expected_feats:
    raise ValueError(
        f"point_feats mismatch: yaml={cfg_point_feats}, "
        f"label_input={args.label_input} requires {expected_feats}. "
        f"Update shapevae-256.yaml."
    )
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
    if not os.path.exists(args.resume_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume_checkpoint}")
    ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)

    # Verify color_residual compatibility
    saved_cr = ckpt.get('color_residual', False)
    if saved_cr != args.color_residual:
        raise ValueError(
            f"color_residual mismatch: checkpoint={saved_cr}, current={args.color_residual}. "
            f"MeanColorHead exists only when color_residual=True — incompatible."
        )
    # Verify label_input
    saved_li = ckpt.get('label_input', False)
    if saved_li != args.label_input:
        raise ValueError(
            f"label_input mismatch: checkpoint={saved_li}, current={args.label_input}. "
            f"input_proj dimension differs — cannot resume across this boundary."
        )
    # Verify scene_semantic_head
    saved_ssh = ckpt.get('scene_semantic_head', False)
    if saved_ssh != args.scene_semantic_head:
        print(f"  scene_semantic_head changed: {saved_ssh} -> {args.scene_semantic_head} "
              f"(strict=False)")
        gs_autoencoder.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        saved_mode = ckpt.get('semantic_mode', 'none')
        if saved_mode != effective_semantic_mode:
            print(f"  Semantic mode changed: {saved_mode} -> {effective_semantic_mode} "
                  f"(strict=False)")
            gs_autoencoder.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            gs_autoencoder.load_state_dict(ckpt['model_state_dict'])

    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    saved_epoch = ckpt.get('epoch', 0)
    start_epoch = args.resume_epoch if args.resume_epoch is not None else saved_epoch + 1
    best_val_loss = ckpt.get('val_l2_error', ckpt.get('best_val_l2', float('inf')))
    best_epoch    = saved_epoch
    print(f"  Resumed at epoch {start_epoch} (saved val L2: {best_val_loss:.4f})")

# ============================================================================
# DATASETS
# ============================================================================

from gs_dataset_scenesplat import gs_dataset

print(f"\n--- Training Dataset ---")
gs_dataset_train = gs_dataset(
    root=os.path.join(data_path, "train_grid1.0cm_chunk8x8_stride6x6"),
    resol=200, random_permute=True, train=True,
    sampling_method=args.sampling_method,
    max_scenes=args.train_scenes,
    normalize=args.use_canonical_norm,
    normalize_colors=args.normalize_colors,
    target_radius=10.0,
    scale_norm_mode=args.scale_norm_mode,
    label_input=args.label_input,
    color_residual=args.color_residual,
)
trainDataLoader = Data.DataLoader(
    dataset=gs_dataset_train, batch_size=args.batch_size,
    shuffle=True, num_workers=9, pin_memory=True, persistent_workers=True,
)

print(f"\n--- Validation Dataset ---")
gs_dataset_val = gs_dataset(
    root=os.path.join(data_path, "train_grid1.0cm_chunk8x8_stride6x6"),
    resol=200, random_permute=False, train=True,
    sampling_method=args.sampling_method,
    max_scenes=args.val_scenes,
    normalize=args.use_canonical_norm,
    normalize_colors=args.normalize_colors,
    target_radius=10.0,
    scale_norm_mode=args.scale_norm_mode,
    label_input=args.label_input,
    color_residual=args.color_residual,
)
valDataLoader = Data.DataLoader(
    dataset=gs_dataset_val, batch_size=args.batch_size,
    shuffle=False, num_workers=9, pin_memory=True, persistent_workers=True,
)

print(f"\n{'='*70}")
print(f"  Train: {len(gs_dataset_train)} scenes, {len(trainDataLoader)} batches/epoch")
print(f"  Val:   {len(gs_dataset_val)} scenes,  {len(valDataLoader)} batches")
print(f"  Total steps: {len(trainDataLoader) * args.num_epochs:,}")
print(f"{'='*70}\n")

# ============================================================================
# CHECKPOINT METADATA  (written into every saved .pth)
# ============================================================================

_ckpt_meta = {
    'semantic_mode':        effective_semantic_mode,
    'enable_semantic':      enable_semantic,
    'label_input':          args.label_input,
    'color_residual':       args.color_residual,
    'scene_semantic_head':  args.scene_semantic_head,   # Move 1
    'mean_color_weight':    args.mean_color_weight,
    'scene_semantic_weight': args.scene_semantic_weight,
    'color_loss_weight':    args.color_loss_weight,
    'use_canonical_norm':   args.use_canonical_norm,
    'scale_norm_mode':      args.scale_norm_mode,
}

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, dataloader, device, epoch=None):
    model.eval()
    total_l2 = total_kl = 0.0
    total_color_pred_loss    = 0.0
    total_scene_semantic_kl  = 0.0
    per_param = {k: 0.0 for k in PARAM_SLICES}
    n_scenes  = 0

    scene_data_list  = []
    recon_preds_list = []
    recon_means_list = []

    do_pca   = (epoch is not None and epoch % args.pca_vis_freq  == 0 and enable_semantic)
    do_recon = (epoch is not None and epoch % args.recon_ply_freq == 0)

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating", leave=False):
            UV_gs_batch    = batch_data['features'].float().to(device)
            mean_color_gt  = batch_data['mean_color'].float().to(device)
            B = UV_gs_batch.shape[0]

            # # Load segment labels if needed for Move 1 eval metric
            # segment_labels_cpu = batch_data.get('segment_labels', None)
            # segment_labels_gpu = None
            # if need_segment_labels and segment_labels_cpu is not None:
            #     segment_labels_gpu = segment_labels_cpu.long().to(device)
            segment_labels_cpu = batch_data.get('segment_labels', None)
            need_sem = (do_pca and len(scene_data_list) < args.pca_num_scenes
                        and segment_labels_cpu is not None)
            was_train = model.training
            if need_sem:
                model.train()

            (shape_embed, mu, log_var, z,
             UV_gs_recover, per_gaussian_features) = model(
                UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3]
            )
            mean_color_pred       = model.shape_model.last_mean_color_pred
            scene_semantic_pred   = model.shape_model.last_scene_semantic_pred

            if need_sem and not was_train:
                model.eval()

            target     = UV_gs_batch[:, :, GEOMETRIC_INDICES]
            pred_3d    = UV_gs_recover.reshape(B, -1, 14)
            recon_loss = compute_reconstruction_loss(
                pred_3d, target, B, args.color_loss_weight)

            kl_loss = -0.5 * torch.sum(
                1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

            # Step 1: color prediction error
            if mean_color_pred is not None and args.color_residual:
                cp_loss = F.mse_loss(mean_color_pred, mean_color_gt).item()
                total_color_pred_loss += cp_loss * B

            # Move 1: scene semantic KL error
            if scene_semantic_pred is not None and args.scene_semantic_head:
                p_s   = batch_data['label_dist'].float().to(device)       # ← precomputed
                ss_kl = scene_semantic_kl_loss(scene_semantic_pred, p_s).item()
                total_scene_semantic_kl += ss_kl * B

            total_l2 += recon_loss.item()
            total_kl += kl_loss.sum().item()
            n_scenes  += B

            ind = compute_individual_losses(pred_3d, target)
            for k in per_param:
                per_param[k] += ind[k]

            # Collect for PCA
            if need_sem and per_gaussian_features is not None:
                pos = UV_gs_batch[:, :, 4:7].cpu().numpy()
                col = UV_gs_batch[:, :, 7:10].cpu().numpy()
                sem = per_gaussian_features.cpu().numpy()
                for si in range(B):
                    if len(scene_data_list) >= args.pca_num_scenes:
                        break
                    scene_data_list.append({
                        'semantic_features': sem[si],
                        'positions': pos[si], 'colors': col[si],
                        'coords': pos[si], 'scene_id': len(scene_data_list),
                    })

            # Collect for PLY reconstruction
            if do_recon and len(recon_preds_list) < args.recon_ply_num_scenes:
                preds_np = pred_3d.cpu().numpy()
                means_np = mean_color_gt.cpu().numpy()
                for si in range(B):
                    if len(recon_preds_list) >= args.recon_ply_num_scenes:
                        break
                    recon_preds_list.append(preds_np[si])
                    recon_means_list.append(means_np[si])

    # PCA visualizations
    if do_pca and scene_data_list and save_path:
        vis_dir = Path(save_path) / "pca_visualizations" / f"epoch_{epoch:03d}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        for si, sd in enumerate(scene_data_list):
            try:
                visualize_comparison(
                    coords=sd['coords'], semantic_features=sd['semantic_features'],
                    positions=sd['positions'], colors=sd['colors'],
                    output_dir=vis_dir, scene_name=f"scene_{si:03d}",
                    brightness=args.pca_brightness,
                )
            except Exception as e:
                print(f"  PCA error scene {si}: {e}")

    # PLY reconstruction — add mean_color back to residuals before saving
    if do_recon and recon_preds_list and save_path:
        try:
            all_preds = np.stack(recon_preds_list, axis=0)
            if args.color_residual:
                for si in range(len(all_preds)):
                    all_preds[si, :, 3:6] += recon_means_list[si]
                    all_preds[si, :, 3:6]  = np.clip(all_preds[si, :, 3:6], 0, 1)
            recon_dir = Path(save_path) / "reconstructed_gaussians" / f"epoch_{epoch:03d}"
            save_reconstructed_gaussians(
                predictions=all_preds, output_dir=recon_dir, epoch=epoch,
                num_scenes=len(all_preds), max_sh_degree=args.recon_ply_max_sh,
                color_mode="1", prefix="scene",
            )
        except Exception as e:
            print(f"  PLY save error: {e}")

    model.train()
    return {
        'avg_l2_error':           total_l2,
        'avg_kl':                 total_kl / max(n_scenes, 1),
        'color_pred_loss':        total_color_pred_loss / max(n_scenes, 1),
        'scene_semantic_kl':      total_scene_semantic_kl / max(n_scenes, 1),
        **{f'{k}_loss': v / max(n_scenes, 1) for k, v in per_param.items()},
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
    epoch_color_pred = epoch_scene_semantic = 0.0
    epoch_pos = epoch_col = epoch_opa = epoch_scl = epoch_rot = 0.0

    for i_batch, batch_data in enumerate(trainDataLoader):
        UV_gs_batch   = batch_data['features'].float().to(device)
        mean_color_gt = batch_data['mean_color'].float().to(device)
        B = UV_gs_batch.shape[0]

        # Load segment labels when needed for InfoNCE or SceneSemanticHead
        segment_labels  = None
        instance_labels = None
        if need_segment_labels:
            segment_labels  = batch_data['segment_labels'].long().to(device)
            if enable_semantic:
                instance_labels = batch_data['instance_labels'].long().to(device)

        optimizer.zero_grad()

        # ── Forward ──────────────────────────────────────────────────────────
        (shape_embed, mu, log_var, z,
         UV_gs_recover, per_gaussian_features) = gs_autoencoder(
            UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3]
        )
        mean_color_pred     = gs_autoencoder.shape_model.last_mean_color_pred
        scene_semantic_pred = gs_autoencoder.shape_model.last_scene_semantic_pred

        # ── Reconstruction loss ───────────────────────────────────────────────
        target   = UV_gs_batch[:, :, GEOMETRIC_INDICES]
        pred_3d  = UV_gs_recover.reshape(B, -1, 14)
        recon_loss = compute_reconstruction_loss(
            pred_3d, target, B, args.color_loss_weight)

        # ── KL loss ───────────────────────────────────────────────────────────
        KL_loss = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

        # ── Step 1: mean color prediction loss ───────────────────────────────
        # Gradient: MSE -> MeanColorHead -> shape_embed -> encoder token 0
        color_pred_loss = torch.tensor(0.0, device=device)
        if mean_color_pred is not None and args.color_residual:
            color_pred_loss = F.mse_loss(mean_color_pred, mean_color_gt)

        # ── Move 1: scene semantic distribution loss ──────────────────────────
        # Gradient: KL -> SceneSemanticHead -> shape_embed -> encoder token 0
        # Independent of MeanColorHead (different output head, different loss).
        # Independent of InfoNCE (different network path entirely).
        scene_semantic_loss = torch.tensor(0.0, device=device)
        if scene_semantic_pred is not None and args.scene_semantic_head:
            p_s = batch_data['label_dist'].float().to(device)   # [B, 72] — precomputed
            scene_semantic_loss = scene_semantic_kl_loss(scene_semantic_pred, p_s)

        # ── Semantic loss (per-Gaussian InfoNCE) ─────────────────────────────
        semantic_loss    = torch.tensor(0.0, device=device)
        semantic_metrics = {}
        if enable_semantic and segment_labels is not None and per_gaussian_features is not None:
            if args.semantic_mode == 'dist':
                semantic_loss, semantic_metrics = compute_distribution_loss(
                    dist_logits=per_gaussian_features,
                    segment_labels=segment_labels,
                    weight=args.segment_loss_weight,
                )
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
                    sampling_strategy=args.sampling_strategy,
                )

        # ── Total loss ────────────────────────────────────────────────────────
        loss = (recon_loss
                + args.kl_weight            * KL_loss
                + args.mean_color_weight    * color_pred_loss
                + args.scene_semantic_weight * scene_semantic_loss
                + semantic_loss)
        loss.backward()
        optimizer.step()

        ind = compute_individual_losses(pred_3d, target)
        epoch_loss            += loss.item()
        epoch_recon           += recon_loss.item()
        epoch_kl              += KL_loss.item()
        epoch_sem             += semantic_loss.item()
        epoch_color_pred      += color_pred_loss.item()
        epoch_scene_semantic  += scene_semantic_loss.item()
        epoch_pos += ind['position']
        epoch_col += ind['color']
        epoch_opa += ind['opacity']
        epoch_scl += ind['scale']
        epoch_rot += ind['rotation']

        # Epoch 0 first-batch diagnostic
        if epoch == start_epoch and i_batch == 0:
            print(f"\nEPOCH {epoch} DIAGNOSTIC (batch 0):")
            print(f"  mu range:              [{mu.min().item():.3f}, {mu.max().item():.3f}]")
            print(f"  recon_loss:            {recon_loss.item():.4f}")
            print(f"  KL_loss:               {KL_loss.item():.4f}")
            if args.color_residual and mean_color_pred is not None:
                print(f"  color_pred_loss:       {color_pred_loss.item():.6f}")
                mc_pred = mean_color_pred[0].detach().cpu().numpy()
                mc_gt   = mean_color_gt[0].cpu().numpy()
                print(f"  mean_color gt:         {mc_gt.round(3)}")
                print(f"  mean_color pred:       {mc_pred.round(3)}")
                color_col = UV_gs_batch[:, :, 7:10].detach().cpu().numpy()
                print(f"  color col range:       [{color_col.min():.3f}, {color_col.max():.3f}]"
                      f"  (residuals — should be negative)")
            if args.scene_semantic_head and scene_semantic_pred is not None:
                print(f"  scene_semantic_loss:   {scene_semantic_loss.item():.4f}")
                pred_top = scene_semantic_pred[0].detach().cpu()
                top3_idx = pred_top.topk(3).indices.numpy()
                top3_val = pred_top.topk(3).values.numpy()
                print(f"  scene_semantic top3:   labels={top3_idx}, probs={top3_val.round(3)}")
            if semantic_loss.item() > 0:
                print(f"  semantic_loss (InfoNCE):{semantic_loss.item():.4f}")

        if wandb_enabled:
            log = {
                "train/step_loss":            loss.item(),
                "train/step_recon":           recon_loss.item(),
                "train/step_kl":              KL_loss.item(),
                "train/step_color_pred":      color_pred_loss.item(),
                "train/step_scene_semantic":  scene_semantic_loss.item(),
                "train/step_position":        ind['position'],
                "train/step_color":           ind['color'],
                "train/step_opacity":         ind['opacity'],
                "train/step_scale":           ind['scale'],
                "train/step_rotation":        ind['rotation'],
            }
            if semantic_metrics:
                log.update({f"train/step_{k}": v for k, v in semantic_metrics.items()})
            wandb_run.log(log, step=global_step)

        global_step += 1

    # ── End-of-epoch logging ──────────────────────────────────────────────────
    nb = len(trainDataLoader)
    print(f"\nEpoch {epoch} | Loss={epoch_loss/nb:.4f} | "
          f"Recon={epoch_recon/nb:.4f} | KL={epoch_kl/nb:.4f} | "
          f"InfoNCE={epoch_sem/nb:.4f} | ColorPred={epoch_color_pred/nb:.6f} | "
          f"SceneSem={epoch_scene_semantic/nb:.4f}")
    print(f"  Pos={epoch_pos/nb:.3f} | Col={epoch_col/nb:.3f} | "
          f"Opa={epoch_opa/nb:.3f} | Scl={epoch_scl/nb:.3f} | "
          f"Rot={epoch_rot/nb:.3f}")
    print(f"  mu range: [{mu.min().item():.3f}, {mu.max().item():.3f}]")

    # ── Validation ────────────────────────────────────────────────────────────
    val_metrics = None
    if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:
        print(f"\n--- Validation (epoch {epoch}) ---")
        val_metrics = evaluate_model(gs_autoencoder, valDataLoader, device, epoch=epoch)
        print(f"  L2:                {val_metrics['avg_l2_error']:.4f}")
        print(f"  Position:          {val_metrics['position_loss']:.4f}")
        print(f"  Color:             {val_metrics['color_loss']:.4f}"
              f"  {'(residuals)' if args.color_residual else '(absolute)'}")
        print(f"  Opacity:           {val_metrics['opacity_loss']:.4f}")
        print(f"  Scale:             {val_metrics['scale_loss']:.4f}")
        print(f"  Rotation:          {val_metrics['rotation_loss']:.4f}")
        if args.color_residual:
            print(f"  ColorPredMSE:      {val_metrics['color_pred_loss']:.6f}"
                  f"  (MeanColorHead: shape_embed -> mean RGB)")
        if args.scene_semantic_head:
            print(f"  SceneSemanticKL:   {val_metrics['scene_semantic_kl']:.4f}"
                  f"  (SceneSemanticHead: shape_embed -> label dist)")

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
            "val/l2_error":             val_metrics['avg_l2_error'],
            "val/position_loss":        val_metrics['position_loss'],
            "val/color_loss":           val_metrics['color_loss'],
            "val/color_pred_mse":       val_metrics['color_pred_loss'],
            "val/scene_semantic_kl":    val_metrics['scene_semantic_kl'],
            "val/opacity_loss":         val_metrics['opacity_loss'],
            "val/scale_loss":           val_metrics['scale_loss'],
            "val/rotation_loss":        val_metrics['rotation_loss'],
            "best/val_l2":              best_val_loss,
            "best/epoch":               best_epoch,
            "train/epoch":              epoch,
        }, step=global_step)

    # ── Periodic checkpoint ───────────────────────────────────────────────────
    if epoch >= 10 and epoch % 50 == 0:
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     gs_autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss':           epoch_loss / nb,
            **_ckpt_meta,
        }, os.path.join(save_path, f"epoch_{epoch}.pth"))
        print(f"  Checkpoint saved: epoch_{epoch}.pth")

# ============================================================================
# FINAL SAVE
# ============================================================================

print(f"\n{'='*70}\nTRAINING COMPLETE\n{'='*70}")

final_metrics = evaluate_model(gs_autoencoder, valDataLoader, device,
                               epoch=args.num_epochs - 1)

print(f"\nFinal Results:")
print(f"  Final L2:   {final_metrics['avg_l2_error']:.4f}")
print(f"  Best L2:    {best_val_loss:.4f} (epoch {best_epoch})")
print(f"  Position:   {final_metrics['position_loss']:.4f}")
print(f"  Color:      {final_metrics['color_loss']:.4f}"
      f"  {'(residuals)' if args.color_residual else '(absolute)'}")
if args.color_residual:
    print(f"  ColorPredMSE:    {final_metrics['color_pred_loss']:.6f}")
if args.scene_semantic_head:
    print(f"  SceneSemanticKL: {final_metrics['scene_semantic_kl']:.4f}")

torch.save({
    'epoch':                args.num_epochs - 1,
    'model_state_dict':     gs_autoencoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_val_l2':         final_metrics['avg_l2_error'],
    'best_val_l2':          best_val_loss,
    'best_epoch':           best_epoch,
    **_ckpt_meta,
    'individual_losses': {k: final_metrics[f'{k}_loss'] for k in PARAM_SLICES},
}, os.path.join(save_path, "final.pth"))

print(f"\nSaved: {save_path}final.pth")

if wandb_enabled:
    wandb_run.summary.update({
        "final_val_l2":   final_metrics['avg_l2_error'],
        "best_val_l2":    best_val_loss,
        "best_epoch":     best_epoch,
    })
    wandb_run.finish()

print("Done.")