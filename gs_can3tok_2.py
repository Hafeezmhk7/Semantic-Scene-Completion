"""
Can3Tok Training — Step 1: Color Residual + Hidden Semantic Baseline
======================================================================

WHAT THIS FILE IMPLEMENTS
--------------------------
This is the complete training script for the 2000-scene hidden semantic
baseline with Step 1 color residual encoding.

Step 1 adds three coordinated changes vs the plain hidden baseline:

  DATASET (gs_dataset_scenesplat.py, color_residual=True):
    After top-40k sampling:
      mean_color  = color.mean(axis=0)    # [3] scene DC term
      color       = color - mean_color    # [N,3] AC residuals ~[-0.3, +0.3]
    mean_color returned in batch dict.

  MODEL (sal_perceiver.py, color_residual=True):
    MeanColorHead: shape_embed [B,384] -> Linear(384,64) -> ReLU
                                       -> Linear(64,3) -> Sigmoid -> [B,3]
    Stored as self.last_mean_color_pred (instance attribute) — NOT a 7th return
    value. Keeps the 6-value return signature that asl_pl_module.py expects.
    Gradient path: MSE_color -> MeanColorHead -> shape_embed -> encoder token 0.

  TRAINING LOOP (this file):
    mean_color_gt   = batch['mean_color'].to(device)                   # [B,3]
    mean_color_pred = gs_autoencoder.shape_model.last_mean_color_pred  # [B,3]
    color_pred_loss = F.mse_loss(mean_color_pred, mean_color_gt)
    total_loss += mean_color_weight * color_pred_loss

    At PLY save time: add mean_color back to residuals before saving.
      absolute_color = predicted_residual + mean_color_gt

SEMANTIC MODE
-------------
  semantic_mode='hidden'  → InfoNCE on GS_decoder hidden state [B,1024]
  segment_weight=0.3      → beta, must be >0 to activate
  Both conditions required (gs_can3tok_2.py checks: semantic_mode != 'none'
  AND segment_loss_weight > 0).

  shape_embed -> MeanColorHead path and GS_decoder hidden -> InfoNCE path
  are completely independent. No interference.

MODEL FORWARD RETURN VALUES  (6 values, unchanged)
---------------------------------------------------
  shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features
  mean_color_pred accessed via: gs_autoencoder.shape_model.last_mean_color_pred

CHECKPOINT
----------
  Stores color_residual=True/False so probe and resume scripts can verify
  compatibility automatically.
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
# Feature tensor cols 4-17 → 14-dim reconstruction target:
#   [0:3] xyz  [3:6] rgb-or-residual  [6] opacity  [7:10] scale  [10:14] quat

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

# Semantic
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
    help='Weight on MSE(mean_color_pred, mean_color_gt). '
         'total_loss += mean_color_weight * color_pred_loss')

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

# ============================================================================
# W&B
# ============================================================================

wandb_enabled = False
if args.use_wandb:
    try:
        import wandb
        job_id   = os.environ.get('SLURM_JOB_ID', 'local')
        run_name = f"can3tok_job_{job_id}_{effective_semantic_mode}"
        if args.color_residual:   run_name += "_colorresidual"
        if args.label_input:      run_name += "_labelinput"
        if enable_semantic:       run_name += f"_beta{args.segment_loss_weight}"
        if args.resume_checkpoint: run_name += "_resumed"
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

if args.color_residual:   tag += "_colorresidual"
if args.label_input:      tag += "_labelinput"
if enable_semantic:       tag += f"_beta{args.segment_loss_weight}"
if not args.use_canonical_norm: tag += "_raw"

save_path = f"/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/{tag}/"
os.makedirs(save_path, exist_ok=True)

# ============================================================================
# STARTUP SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print(f"CAN3TOK TRAINING")
print(f"{'='*70}")
print(f"  Semantic mode:    {effective_semantic_mode}")
print(f"  Enable semantic:  {enable_semantic}")
print(f"  Color residual:   {args.color_residual}  "
      f"(Step 1 — shape_embed -> MeanColorHead)")
print(f"  Mean color weight:{args.mean_color_weight}  (weight on color pred loss)")
print(f"  Label input:      {args.label_input}")
print(f"  Scale norm mode:  {args.scale_norm_mode}")
print(f"  Canonical norm:   {args.use_canonical_norm}")
print(f"  Device:           {device}")
print(f"  Save path:        {save_path}")
if args.resume_checkpoint:
    print(f"  Resume from:      {args.resume_checkpoint}")
print(f"{'='*70}\n")

# ============================================================================
# MODEL
# ============================================================================

print("Loading model config...")
config_path = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
model_config_perceiver = get_config_from_file(config_path)
model_config = model_config_perceiver.model
model_config.params.shape_module_cfg.params.semantic_mode = effective_semantic_mode
model_config.params.shape_module_cfg.params.color_residual = args.color_residual  # Step 1

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

    # Verify color_residual compatibility — changes input to decoder color head
    saved_cr = ckpt.get('color_residual', False)
    if saved_cr != args.color_residual:
        raise ValueError(
            f"color_residual mismatch: checkpoint={saved_cr}, current={args.color_residual}. "
            f"These are incompatible — the MeanColorHead exists only when color_residual=True."
        )
    # Verify label_input — changes input_proj dimension
    saved_li = ckpt.get('label_input', False)
    if saved_li != args.label_input:
        raise ValueError(
            f"label_input mismatch: checkpoint={saved_li}, current={args.label_input}. "
            f"input_proj dimension differs — cannot resume across this boundary."
        )

    saved_mode = ckpt.get('semantic_mode', 'none')
    if saved_mode != effective_semantic_mode:
        print(f"  Semantic mode changed: {saved_mode} -> {effective_semantic_mode} (strict=False)")
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
    'semantic_mode':      effective_semantic_mode,
    'enable_semantic':    enable_semantic,
    'label_input':        args.label_input,
    'color_residual':     args.color_residual,
    'mean_color_weight':  args.mean_color_weight,
    'color_loss_weight':  args.color_loss_weight,
    'use_canonical_norm': args.use_canonical_norm,
    'scale_norm_mode':    args.scale_norm_mode,
}

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, dataloader, device, epoch=None):
    model.eval()
    total_l2 = total_kl = 0.0
    total_color_pred_loss = 0.0
    per_param = {k: 0.0 for k in PARAM_SLICES}
    n_scenes  = 0

    scene_data_list  = []
    recon_preds_list = []
    recon_means_list = []   # mean_color for each collected scene (Step 1)

    do_pca   = (epoch is not None and epoch % args.pca_vis_freq  == 0 and enable_semantic)
    do_recon = (epoch is not None and epoch % args.recon_ply_freq == 0)

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating", leave=False):
            UV_gs_batch    = batch_data['features'].float().to(device)
            mean_color_gt  = batch_data['mean_color'].float().to(device)  # [B,3]
            segment_labels = batch_data.get('segment_labels', None)
            B = UV_gs_batch.shape[0]

            # Need model.train() to get per_gaussian_features for PCA collection
            need_sem = (do_pca and len(scene_data_list) < args.pca_num_scenes
                        and segment_labels is not None)
            was_train = model.training
            if need_sem:
                model.train()

            (shape_embed, mu, log_var, z,
             UV_gs_recover, per_gaussian_features) = model(
                UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3]
            )
            mean_color_pred = model.shape_model.last_mean_color_pred

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

    # PLY reconstruction
    # Step 1: add mean_color back to residuals before saving
    if do_recon and recon_preds_list and save_path:
        try:
            all_preds = np.stack(recon_preds_list, axis=0)   # [S, N, 14]
            if args.color_residual:
                for si in range(len(all_preds)):
                    all_preds[si, :, 3:6] += recon_means_list[si]   # add DC term
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
        'avg_l2_error':        total_l2,
        'avg_kl':              total_kl / max(n_scenes, 1),
        'color_pred_loss':     total_color_pred_loss / max(n_scenes, 1),
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

    epoch_loss = epoch_recon = epoch_kl = epoch_sem = epoch_color_pred = 0.0
    epoch_pos = epoch_col = epoch_opa = epoch_scl = epoch_rot = 0.0

    for i_batch, batch_data in enumerate(trainDataLoader):
        UV_gs_batch   = batch_data['features'].float().to(device)
        mean_color_gt = batch_data['mean_color'].float().to(device)   # [B,3]
        B = UV_gs_batch.shape[0]

        segment_labels = instance_labels = None
        if enable_semantic:
            segment_labels  = batch_data['segment_labels'].long().to(device)
            instance_labels = batch_data['instance_labels'].long().to(device)

        optimizer.zero_grad()

        # ── Forward ──────────────────────────────────────────────────────────
        # asl_pl_module.py wraps shape_model and expects exactly 6 return values.
        # mean_color_pred is stored as shape_model.last_mean_color_pred to avoid
        # touching asl_pl_module.py.
        (shape_embed, mu, log_var, z,
         UV_gs_recover, per_gaussian_features) = gs_autoencoder(
            UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3]
        )
        mean_color_pred = gs_autoencoder.shape_model.last_mean_color_pred

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
        # Completely independent from reconstruction and InfoNCE paths.
        color_pred_loss = torch.tensor(0.0, device=device)
        if mean_color_pred is not None and args.color_residual:
            color_pred_loss = F.mse_loss(mean_color_pred, mean_color_gt)

        # ── Semantic loss ─────────────────────────────────────────────────────
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
                + args.kl_weight    * KL_loss
                + semantic_loss
                + args.mean_color_weight * color_pred_loss)
        loss.backward()
        optimizer.step()

        ind = compute_individual_losses(pred_3d, target)
        epoch_loss       += loss.item()
        epoch_recon      += recon_loss.item()
        epoch_kl         += KL_loss.item()
        epoch_sem        += semantic_loss.item()
        epoch_color_pred += color_pred_loss.item()
        epoch_pos += ind['position']
        epoch_col += ind['color']
        epoch_opa += ind['opacity']
        epoch_scl += ind['scale']
        epoch_rot += ind['rotation']

        # Epoch 0 first-batch diagnostic
        if epoch == start_epoch and i_batch == 0:
            print(f"\nEPOCH {epoch} DIAGNOSTIC (batch 0):")
            print(f"  mu range:         [{mu.min().item():.3f}, {mu.max().item():.3f}]")
            print(f"  recon_loss:       {recon_loss.item():.4f}")
            print(f"  KL_loss:          {KL_loss.item():.4f}")
            if args.color_residual and mean_color_pred is not None:
                print(f"  color_pred_loss:  {color_pred_loss.item():.6f}"
                      f"  (MSE of mean color prediction)")
                mc_pred = mean_color_pred[0].detach().cpu().numpy()
                mc_gt   = mean_color_gt[0].cpu().numpy()
                print(f"  mean_color gt:    {mc_gt.round(3)}")
                print(f"  mean_color pred:  {mc_pred.round(3)}")
                color_col = UV_gs_batch[:, :, 7:10].detach().cpu().numpy()
                print(f"  color col range:  [{color_col.min():.3f}, {color_col.max():.3f}]"
                      f"  (should be negative — these are residuals)")
            else:
                color_col = UV_gs_batch[:, :, 7:10].detach().cpu().numpy()
                print(f"  color col range:  [{color_col.min():.3f}, {color_col.max():.3f}]"
                      f"  (absolute [0,1])")
            if semantic_loss.item() > 0:
                print(f"  semantic_loss:    {semantic_loss.item():.4f}")

        if wandb_enabled:
            log = {
                "train/step_loss":       loss.item(),
                "train/step_recon":      recon_loss.item(),
                "train/step_kl":         KL_loss.item(),
                "train/step_color_pred": color_pred_loss.item(),
                "train/step_position":   ind['position'],
                "train/step_color":      ind['color'],
                "train/step_opacity":    ind['opacity'],
                "train/step_scale":      ind['scale'],
                "train/step_rotation":   ind['rotation'],
            }
            if semantic_metrics:
                log.update({f"train/step_{k}": v for k, v in semantic_metrics.items()})
            wandb_run.log(log, step=global_step)

        global_step += 1

    # ── End-of-epoch logging (EVERY epoch) ───────────────────────────────────
    nb = len(trainDataLoader)
    print(f"\nEpoch {epoch} | Loss={epoch_loss/nb:.4f} | "
          f"Recon={epoch_recon/nb:.4f} | KL={epoch_kl/nb:.4f} | "
          f"Sem={epoch_sem/nb:.4f} | ColorPred={epoch_color_pred/nb:.6f}")
    print(f"  Pos={epoch_pos/nb:.3f} | Col={epoch_col/nb:.3f} | "
          f"Opa={epoch_opa/nb:.3f} | Scl={epoch_scl/nb:.3f} | "
          f"Rot={epoch_rot/nb:.3f}")
    print(f"  mu range: [{mu.min().item():.3f}, {mu.max().item():.3f}]")

    # ── Validation ────────────────────────────────────────────────────────────
    val_metrics = None
    if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:
        print(f"\n--- Validation (epoch {epoch}) ---")
        val_metrics = evaluate_model(gs_autoencoder, valDataLoader, device, epoch=epoch)
        print(f"  L2:           {val_metrics['avg_l2_error']:.4f}")
        print(f"  Position:     {val_metrics['position_loss']:.4f}")
        print(f"  Color:        {val_metrics['color_loss']:.4f}"
              f"  {'(residuals)' if args.color_residual else '(absolute)'}")
        print(f"  Opacity:      {val_metrics['opacity_loss']:.4f}")
        print(f"  Scale:        {val_metrics['scale_loss']:.4f}")
        print(f"  Rotation:     {val_metrics['rotation_loss']:.4f}")
        if args.color_residual:
            print(f"  ColorPredMSE: {val_metrics['color_pred_loss']:.6f}"
                  f"  (shape_embed -> mean RGB prediction)")

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
            "val/l2_error":       val_metrics['avg_l2_error'],
            "val/position_loss":  val_metrics['position_loss'],
            "val/color_loss":     val_metrics['color_loss'],
            "val/color_pred_mse": val_metrics['color_pred_loss'],
            "val/opacity_loss":   val_metrics['opacity_loss'],
            "val/scale_loss":     val_metrics['scale_loss'],
            "val/rotation_loss":  val_metrics['rotation_loss'],
            "best/val_l2":        best_val_loss,
            "best/epoch":         best_epoch,
            "train/epoch":        epoch,
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

final_metrics = evaluate_model(gs_autoencoder, valDataLoader, device, epoch=args.num_epochs - 1)

print(f"\nFinal Results:")
print(f"  Final L2:  {final_metrics['avg_l2_error']:.4f}")
print(f"  Best L2:   {best_val_loss:.4f} (epoch {best_epoch})")
print(f"  Position:  {final_metrics['position_loss']:.4f}")
print(f"  Color:     {final_metrics['color_loss']:.4f}"
      f"  {'(residuals)' if args.color_residual else '(absolute)'}")
if args.color_residual:
    print(f"  ColorPredMSE: {final_metrics['color_pred_loss']:.6f}")

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
        "final_val_l2": final_metrics['avg_l2_error'],
        "best_val_l2":  best_val_loss,
        "best_epoch":   best_epoch,
    })
    wandb_run.finish()

print("Done.")