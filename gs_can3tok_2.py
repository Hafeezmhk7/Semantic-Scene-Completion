"""
Can3Tok Training - SIMPLE COLOR WEIGHTING
==========================================
RGB COLOR + INDIVIDUAL PARAMETER TRACKING + PCA VISUALIZATION

SIMPLIFIED:
- Standard L2 loss (no complex balanced loss)
- Optional color weighting (multiply color loss by weight)
- Dataset normalization controlled by flags

CHECKPOINT RESUMING:
- Use --resume_checkpoint /path/to/checkpoint.pth to resume
- Use --resume_epoch N to override the epoch counter (optional)
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime
import argparse
from pathlib import Path

# Michelangelo imports (VAE model)
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file

# Data loading
import torch.utils.data as Data

# Semantic loss functions
from semantic_losses import compute_semantic_loss

# PCA Feature Visualization
from pca_feature_visualization import visualize_comparison

# 3DGS PLY Reconstruction
from gs_ply_reconstructor import save_reconstructed_gaussians

import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group

# Unbuffered output
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

# ============================================================================
# PARAMETER INDICES FOR 14-DIM GAUSSIANS
# ============================================================================
"""
Input format [B, N, features]:
  Columns 0-3:   voxel_centers + uniq_idx
  Columns 4-7:   xyz positions  
  Columns 7-10:  rgb colors
  Columns 10:    opacity
  Columns 11-14: scale (sx, sy, sz)
  Columns 14-18: rotation quaternion (qw, qx, qy, qz)

Target format [B, N, 14]:
  [0:3]   xyz positions
  [3:6]   rgb colors
  [6]     opacity
  [7:10]  scale
  [10:14] rotation quaternion
"""

# Define parameter slices
PARAM_SLICES = {
    'position': slice(0, 3),      # xyz
    'color': slice(3, 6),         # rgb
    'opacity': slice(6, 7),       # alpha
    'scale': slice(7, 10),        # sx, sy, sz
    'rotation': slice(10, 14),    # quat
}

# Indices from input to extract for target
GEOMETRIC_INDICES = list(range(4, 7)) + list(range(7, 10)) + [10] + list(range(11, 14)) + list(range(14, 18))

print(f"PARAMETER CONFIGURATION:")
print(f"   Total params: 14")
print(f"   Position (xyz): indices {PARAM_SLICES['position']}")
print(f"   Color (rgb):    indices {PARAM_SLICES['color']}")
print(f"   Opacity (a):    indices {PARAM_SLICES['opacity']}")
print(f"   Scale (s):      indices {PARAM_SLICES['scale']}")
print(f"   Rotation (q):   indices {PARAM_SLICES['rotation']}")
print()

# ============================================================================
# INDIVIDUAL LOSS COMPUTATION (FOR TRACKING ONLY)
# ============================================================================

def compute_individual_losses(prediction, target):
    losses = {}
    pos_diff      = prediction[:, :, PARAM_SLICES['position']] - target[:, :, PARAM_SLICES['position']]
    losses['position'] = torch.norm(pos_diff, p=2).item()
    color_diff    = prediction[:, :, PARAM_SLICES['color']] - target[:, :, PARAM_SLICES['color']]
    losses['color'] = torch.norm(color_diff, p=2).item()
    opacity_diff  = prediction[:, :, PARAM_SLICES['opacity']] - target[:, :, PARAM_SLICES['opacity']]
    losses['opacity'] = torch.norm(opacity_diff, p=2).item()
    scale_diff    = prediction[:, :, PARAM_SLICES['scale']] - target[:, :, PARAM_SLICES['scale']]
    losses['scale'] = torch.norm(scale_diff, p=2).item()
    rotation_diff = prediction[:, :, PARAM_SLICES['rotation']] - target[:, :, PARAM_SLICES['rotation']]
    losses['rotation'] = torch.norm(rotation_diff, p=2).item()
    return losses


# ============================================================================
# SIMPLE RECONSTRUCTION LOSS WITH COLOR WEIGHTING
# ============================================================================

def compute_reconstruction_loss(prediction, target, batch_size, color_weight=1.0):
    if color_weight == 1.0:
        diff = prediction - target
        loss = torch.norm(diff, p=2)
        return loss / batch_size
    else:
        diff_pos   = prediction[:, :, 0:3] - target[:, :, 0:3]
        diff_color = prediction[:, :, 3:6] - target[:, :, 3:6]
        diff_other = prediction[:, :, 6:]  - target[:, :, 6:]
        loss_pos   = torch.norm(diff_pos, p=2)
        loss_color = torch.norm(diff_color, p=2) * color_weight
        loss_other = torch.norm(diff_other, p=2)
        total_loss = loss_pos + loss_color + loss_other
        return total_loss / batch_size


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Can3Tok Training - Simple Color Weighting')
parser.add_argument('--use_wandb', action='store_true', default=False)
parser.add_argument('--wandb_project', type=str, default='Can3Tok-SceenSplat-7K')
parser.add_argument('--wandb_entity', type=str, default='3D-SSC')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--kl_weight', type=float, default=1e-5)
parser.add_argument('--eval_every', type=int, default=10)
parser.add_argument('--failure_threshold', type=float, default=8000.0)
parser.add_argument('--train_scenes', type=int, default=None)
parser.add_argument('--val_scenes', type=int, default=None)
parser.add_argument('--sampling_method', type=str, default='opacity', choices=['random', 'opacity', 'hybrid'])
parser.add_argument('--segment_loss_weight', type=float, default=0.0)
parser.add_argument('--instance_loss_weight', type=float, default=0.0)
parser.add_argument('--semantic_temperature', type=float, default=0.07)
parser.add_argument('--semantic_subsample', type=int, default=2000)
parser.add_argument('--semantic_mode', type=str, default='none',
                    choices=['none', 'hidden', 'geometric', 'attention'])
parser.add_argument('--sampling_strategy', type=str, default='balanced',
                    choices=['random', 'balanced'])
parser.add_argument('--pca_vis_freq', type=int, default=10)
parser.add_argument('--pca_brightness', type=float, default=1.25)
parser.add_argument('--pca_num_scenes', type=int, default=10)
parser.add_argument('--recon_ply_freq', type=int, default=10)
parser.add_argument('--recon_ply_num_scenes', type=int, default=5)
parser.add_argument('--recon_ply_max_sh', type=int, default=3)
parser.add_argument('--color_loss_weight', type=float, default=1.0)
parser.add_argument('--scale_norm_mode', type=str, default='linear', choices=['log', 'linear'])

norm_group = parser.add_mutually_exclusive_group()
norm_group.add_argument('--use_canonical_norm', dest='use_canonical_norm', action='store_true', default=True)
norm_group.add_argument('--no_canonical_norm', dest='use_canonical_norm', action='store_false')

color_norm_group = parser.add_mutually_exclusive_group()
color_norm_group.add_argument('--normalize_colors', dest='normalize_colors', action='store_true', default=True)
color_norm_group.add_argument('--no_normalize_colors', dest='normalize_colors', action='store_false')

# ============================================================================
# CHECKPOINT RESUMING ARGUMENTS
# ============================================================================
# --resume_checkpoint  Path to .pth file. Loads weights + optimizer state.
#                      Training resumes at saved_epoch + 1 automatically.
#
# --resume_epoch       Optional integer to manually override epoch counter.
#                      Useful for fine-tuning experiments where you want
#                      to reset the epoch count to 0.
# ============================================================================
parser.add_argument('--resume_checkpoint', type=str, default=None,
                    help='Path to checkpoint .pth to resume from. '
                         'Example: --resume_checkpoint /path/to/epoch_950.pth')
parser.add_argument('--resume_epoch', type=int, default=None,
                    help='Override start epoch (default: saved_epoch + 1). '
                         'Use --resume_epoch 0 to reset counter for fine-tuning.')

args = parser.parse_args()

# Semantic detection
semantic_requested = (args.semantic_mode != 'none')
semantic_loss_enabled = (args.segment_loss_weight > 0 or args.instance_loss_weight > 0)
enable_semantic = semantic_requested and semantic_loss_enabled

if not semantic_loss_enabled:
    effective_semantic_mode = 'none'
    enable_semantic = False
else:
    effective_semantic_mode = args.semantic_mode

# ============================================================================
# WEIGHTS & BIASES SETUP
# ============================================================================

wandb_enabled = False
if args.use_wandb:
    try:
        import wandb
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        run_name = f"can3tok_RGB_job_{job_id}_{effective_semantic_mode}"
        if enable_semantic:
            run_name += f"_beta{args.segment_loss_weight}"
        if args.color_loss_weight > 1.0:
            run_name += f"_color{args.color_loss_weight}"
        if not args.use_canonical_norm:
            run_name += "_raw"
        if args.resume_checkpoint is not None:
            run_name += "_resumed"
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                "learning_rate": args.lr,
                "architecture": "Can3Tok-RGB-Simple",
                "dataset": "SceneSplat-7K",
                "batch_size": args.batch_size,
                "epochs": args.num_epochs,
                "kl_weight": args.kl_weight,
                "semantic_mode": effective_semantic_mode,
                "enable_semantic": enable_semantic,
                "num_params": 14,
                "color_loss_weight": args.color_loss_weight,
                "use_canonical_norm": args.use_canonical_norm,
                "resumed_from": args.resume_checkpoint,
            },
            tags=["rgb-color", "simple-loss", "color-weighting"],
        )
        print("W&B enabled")
        wandb_enabled = True
    except Exception as e:
        print(f"W&B failed: {e}")
        wandb_enabled = False
else:
    print("W&B disabled")

# ============================================================================
# GPU SETUP
# ============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================================
# CONFIGURATION
# ============================================================================

loss_usage = "L2"
random_permute = 0
random_rotation = 0

resol = 200
data_path = "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs"

num_epochs = args.num_epochs
bch_size = args.batch_size
kl_weight = args.kl_weight
eval_every = args.eval_every
failure_threshold = args.failure_threshold
train_scenes = args.train_scenes
val_scenes = args.val_scenes
sampling_method = args.sampling_method

# ============================================================================
# CHECKPOINT FOLDER
# ============================================================================

job_id = os.environ.get('SLURM_JOB_ID', None)
if job_id:
    checkpoint_folder = f"RGB_job_{job_id}_{effective_semantic_mode}"
    if enable_semantic:
        checkpoint_folder += f"_beta{args.segment_loss_weight}"
    if args.color_loss_weight > 1.0:
        checkpoint_folder += f"_color{args.color_loss_weight}"
    if not args.use_canonical_norm:
        checkpoint_folder += "_raw"
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_folder = f"RGB_local_{timestamp}_{effective_semantic_mode}"
    if enable_semantic:
        checkpoint_folder += f"_beta{args.segment_loss_weight}"
    if args.color_loss_weight > 1.0:
        checkpoint_folder += f"_color{args.color_loss_weight}"
    if not args.use_canonical_norm:
        checkpoint_folder += "_raw"

save_path = f"/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/{checkpoint_folder}/"
os.makedirs(save_path, exist_ok=True)

print(f"\n{'='*70}")
print(f"CAN3TOK TRAINING - SIMPLE COLOR WEIGHTING")
print(f"{'='*70}")
print(f"  Job ID:            {job_id or 'local'}")
print(f"  Semantic Mode:     {effective_semantic_mode}")
print(f"  Color Weight:      {args.color_loss_weight}x")
print(f"  Scale Norm Mode:   {args.scale_norm_mode}")
print(f"  Canonical Norm:    {'ENABLED' if args.use_canonical_norm else 'DISABLED'}")
print(f"  Color Norm:        {'[0,1]' if args.normalize_colors else '[0,255]'}")
print(f"  Device:            {device}")
print(f"  Save path:         {save_path}")
if args.resume_checkpoint:
    print(f"  Resume from:       {args.resume_checkpoint}")
print(f"{'='*70}\n")

# ============================================================================
# MODEL SETUP
# ============================================================================

print("Loading model configuration...")
config_path_perceiver = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
model_config_perceiver = get_config_from_file(config_path_perceiver)
model_config = model_config_perceiver.model
model_config.params.shape_module_cfg.params.semantic_mode = effective_semantic_mode

print(f"\n{'='*70}")
print("INSTANTIATING MODEL")
print(f"{'='*70}")
perceiver_encoder_decoder = instantiate_from_config(model_config)
print(f"Model instantiated successfully")
print(f"{'='*70}\n")

gs_autoencoder = perceiver_encoder_decoder
gs_autoencoder.to(device)

optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=args.lr, betas=[0.9, 0.999])

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================
#
# CONTROL GUIDE:
#
#   Fresh training (no flag needed):
#     python gs_can3tok_2.py [other args]
#     → Starts epoch 0 with random weights
#
#   Resume normally (most common case):
#     python gs_can3tok_2.py --resume_checkpoint /path/to/epoch_950.pth [other args]
#     → Loads weights + optimizer, starts at epoch 951
#     → best_val_loss is restored so good checkpoints are not overwritten
#
#   Resume with epoch reset (fine-tuning / ablation):
#     python gs_can3tok_2.py --resume_checkpoint /path/to/best_model.pth --resume_epoch 0 [other args]
#     → Loads weights + optimizer, but epoch counter starts at 0
#     → Useful when you want to run a second experiment starting from a pretrained base
#
# ============================================================================

start_epoch = 0

if args.resume_checkpoint is not None:
    print(f"\n{'='*70}")
    print(f"RESUMING FROM CHECKPOINT")
    print(f"{'='*70}")

    if not os.path.exists(args.resume_checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.resume_checkpoint}\n"
            f"Please check the path and try again."
        )

    print(f"  Loading: {args.resume_checkpoint}")
    checkpoint = torch.load(args.resume_checkpoint, map_location=device)

    # Load model weights
    gs_autoencoder.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Model weights loaded")

    # Load optimizer state (preserves momentum buffers, LR, etc.)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"  Optimizer state loaded")

    # Determine start epoch
    saved_epoch = checkpoint.get('epoch', 0)
    if args.resume_epoch is not None:
        start_epoch = args.resume_epoch
        print(f"  Epoch counter: overridden to {start_epoch} (--resume_epoch flag)")
    else:
        start_epoch = saved_epoch + 1
        print(f"  Epoch counter: saved={saved_epoch}, resuming at {start_epoch}")

    # Restore best_val_loss so we don't overwrite a better checkpoint
    best_val_loss = checkpoint.get('val_l2_error',
                    checkpoint.get('best_val_l2', float('inf')))
    best_epoch = saved_epoch

    print(f"")
    print(f"  Checkpoint info:")
    print(f"    Saved at epoch:  {saved_epoch}")
    print(f"    Val L2 at save:  {best_val_loss:.4f}")
    print(f"    Semantic mode:   {checkpoint.get('semantic_mode', 'unknown')}")
    print(f"    Color weight:    {checkpoint.get('color_loss_weight', 'unknown')}")
    print(f"    Scale norm mode: {checkpoint.get('scale_norm_mode', 'unknown')}")
    print(f"    Canonical norm:  {checkpoint.get('use_canonical_norm', 'unknown')}")
    print(f"")
    print(f"  Will train from epoch {start_epoch} to {num_epochs - 1}")
    print(f"{'='*70}\n")

else:
    print(f"Fresh training — starting from epoch 0 with random weights\n")
    best_val_loss = float('inf')
    best_epoch = 0

# ============================================================================
# DATASET LOADING
# ============================================================================

print("Loading datasets...")
from gs_dataset_scenesplat import gs_dataset

print("\n" + "="*70)
print("TRAINING DATASET")
print("="*70)
gs_dataset_train = gs_dataset(
    root=os.path.join(data_path, "train_grid1.0cm_chunk8x8_stride6x6"),
    resol=resol,
    random_permute=True,
    train=True,
    sampling_method=sampling_method,
    max_scenes=train_scenes,
    normalize=args.use_canonical_norm,
    normalize_colors=args.normalize_colors,
    target_radius=10.0,
    scale_norm_mode=args.scale_norm_mode,
)

trainDataLoader = Data.DataLoader(
    dataset=gs_dataset_train,
    batch_size=bch_size,
    shuffle=True,
    num_workers=9,
    pin_memory=True,
    persistent_workers=True
)

print("="*70)
print("VALIDATION DATASET")
print("="*70)

gs_dataset_val = gs_dataset(
    root=os.path.join(data_path, "train_grid1.0cm_chunk8x8_stride6x6"),
    resol=resol,
    random_permute=False,
    train=True,
    sampling_method=sampling_method,
    max_scenes=val_scenes,
    normalize=args.use_canonical_norm,
    normalize_colors=args.normalize_colors,
    target_radius=10.0,
    scale_norm_mode=args.scale_norm_mode,
)

valDataLoader = Data.DataLoader(
    dataset=gs_dataset_val,
    batch_size=bch_size,
    shuffle=False,
    num_workers=9,
    pin_memory=True,
    persistent_workers=True
)

print("="*70)
print("DATASET SUMMARY")
print("="*70)
print(f"Training: {len(gs_dataset_train)} scenes, {len(trainDataLoader)} batches")
print(f"Validation: {len(gs_dataset_val)} scenes, {len(valDataLoader)} batches")
print(f"Sampling: {sampling_method}")
print("="*70)
print()

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, dataloader, device, failure_threshold,
                   save_path=None, epoch=None, enable_pca_vis=True, num_vis_scenes=10):
    model.eval()

    total_l2_error = 0.0
    total_kl = 0.0
    per_scene_l2_errors = []
    num_failures = 0
    num_scenes = 0

    total_position_loss = 0.0
    total_color_loss = 0.0
    total_opacity_loss = 0.0
    total_scale_loss = 0.0
    total_rotation_loss = 0.0

    scene_data_list = []
    scenes_collected = 0
    max_scenes_for_visualization = num_vis_scenes

    recon_preds_list = []
    recon_scenes_collected = 0
    max_scenes_for_recon = args.recon_ply_num_scenes

    with torch.no_grad():
        for i_batch, batch_data in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            if isinstance(batch_data, dict):
                UV_gs_batch = batch_data['features'].type(torch.float32).to(device)
                segment_labels = batch_data.get('segment_labels', None)
            else:
                UV_gs_batch = batch_data[0].type(torch.float32).to(device)
                segment_labels = None

            batch_size = UV_gs_batch.shape[0]

            if i_batch == 0:
                voxel_centers = UV_gs_batch[:, :, 0:3].detach().cpu().numpy()
                gauss_positions = UV_gs_batch[:, :, 4:7].detach().cpu().numpy()
                print(f"\nCOORDINATE SYSTEM CHECK:")
                print(f"  Voxel centers range:      [{voxel_centers.min():.2f}, {voxel_centers.max():.2f}]")
                print(f"  Gaussian positions range: [{gauss_positions.min():.2f}, {gauss_positions.max():.2f}]")

            need_semantic = (enable_pca_vis and
                             scenes_collected < max_scenes_for_visualization and
                             segment_labels is not None and
                             epoch is not None and
                             epoch % args.pca_vis_freq == 0)

            was_training = model.training
            if need_semantic:
                model.train()

            shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features = model(
                UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3]
            )

            if need_semantic and not was_training:
                model.eval()

            target = UV_gs_batch[:, :, GEOMETRIC_INDICES]
            UV_gs_recover_reshaped = UV_gs_recover.reshape(UV_gs_batch.shape[0], -1, 14)

            batch_l2_error = compute_reconstruction_loss(
                UV_gs_recover_reshaped, target,
                batch_size=batch_size, color_weight=args.color_loss_weight
            )
            individual_losses = compute_individual_losses(UV_gs_recover_reshaped, target)

            per_scene_norms = torch.norm(UV_gs_recover_reshaped - target, p=2, dim=(1, 2))
            per_scene_errors_scaled = per_scene_norms / np.sqrt(batch_size)

            kl_loss = -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=1)

            total_l2_error += batch_l2_error.item()
            total_kl += kl_loss.sum().item()
            per_scene_l2_errors.extend(per_scene_errors_scaled.cpu().numpy().tolist())
            num_failures += (per_scene_errors_scaled.cpu().numpy() > failure_threshold).sum()
            num_scenes += batch_size

            total_position_loss += individual_losses['position']
            total_color_loss    += individual_losses['color']
            total_opacity_loss  += individual_losses['opacity']
            total_scale_loss    += individual_losses['scale']
            total_rotation_loss += individual_losses['rotation']

            if (enable_pca_vis and
                scenes_collected < max_scenes_for_visualization and
                per_gaussian_features is not None and
                segment_labels is not None and
                epoch is not None and
                epoch % args.pca_vis_freq == 0):
                try:
                    pos = UV_gs_batch[:, :, 4:7].cpu().numpy()
                    col = UV_gs_batch[:, :, 7:10].cpu().numpy()
                    sem_feat = per_gaussian_features.cpu().numpy()
                    for scene_idx in range(batch_size):
                        if scenes_collected >= max_scenes_for_visualization:
                            break
                        scene_data_list.append({
                            'semantic_features': sem_feat[scene_idx],
                            'positions': pos[scene_idx],
                            'colors': col[scene_idx],
                            'coords': pos[scene_idx],
                            'scene_id': scenes_collected
                        })
                        scenes_collected += 1
                except Exception as e:
                    print(f"Could not extract features for PCA: {e}")

            if (recon_scenes_collected < max_scenes_for_recon and
                epoch is not None and
                epoch % args.recon_ply_freq == 0):
                try:
                    pred_np = UV_gs_recover_reshaped.cpu().numpy()
                    for scene_idx in range(batch_size):
                        if recon_scenes_collected >= max_scenes_for_recon:
                            break
                        recon_preds_list.append(pred_np[scene_idx])
                        recon_scenes_collected += 1
                except Exception as e:
                    print(f"Could not collect predictions for PLY: {e}")

    avg_l2_error = total_l2_error
    avg_kl = total_kl / num_scenes
    failure_rate = (num_failures / num_scenes) * 100.0
    per_scene_l2_errors = np.array(per_scene_l2_errors)

    # PCA Visualizations
    pca_paths = {}
    if (enable_pca_vis and len(scene_data_list) > 0 and
        save_path is not None and epoch is not None and
        epoch % args.pca_vis_freq == 0):

        vis_dir = Path(save_path) / "pca_visualizations" / f"epoch_{epoch:03d}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        for scene_idx, scene_data in enumerate(scene_data_list):
            try:
                scene_pca_paths = visualize_comparison(
                    coords=scene_data['coords'],
                    semantic_features=scene_data['semantic_features'],
                    positions=scene_data['positions'],
                    colors=scene_data['colors'],
                    output_dir=vis_dir,
                    scene_name=f"scene_{scene_idx:03d}",
                    brightness=args.pca_brightness
                )
                pca_paths[f'scene_{scene_idx:03d}'] = scene_pca_paths
            except Exception as e:
                print(f"PCA error scene {scene_idx}: {e}")

    # 3DGS PLY Reconstruction
    recon_paths = {}
    if (len(recon_preds_list) > 0 and save_path is not None and
        epoch is not None and epoch % args.recon_ply_freq == 0):
        try:
            all_preds = np.stack(recon_preds_list, axis=0)
            recon_dir = Path(save_path) / "reconstructed_gaussians" / f"epoch_{epoch:03d}"
            recon_paths = save_reconstructed_gaussians(
                predictions=all_preds, output_dir=recon_dir, epoch=epoch,
                num_scenes=len(recon_preds_list), max_sh_degree=args.recon_ply_max_sh,
                color_mode="1", prefix="scene"
            )
        except Exception as e:
            print(f"PLY save error: {e}")

    model.train()

    return {
        'avg_l2_error': avg_l2_error,
        'l2_std': per_scene_l2_errors.std(),
        'failure_rate': failure_rate,
        'avg_kl': avg_kl,
        'per_scene_errors': per_scene_l2_errors,
        'position_loss': total_position_loss / num_scenes,
        'color_loss':    total_color_loss    / num_scenes,
        'opacity_loss':  total_opacity_loss  / num_scenes,
        'scale_loss':    total_scale_loss    / num_scenes,
        'rotation_loss': total_rotation_loss / num_scenes,
        'pca_paths': pca_paths,
        'recon_paths': recon_paths,
    }

# ============================================================================
# TRAINING SETUP
# ============================================================================

volume_dims = 40
resolution = 16.0 / volume_dims
origin_offset = torch.tensor(
    np.array([
        (volume_dims - 1) / 2,
        (volume_dims - 1) / 2,
        (volume_dims - 1) / 2
    ]) * resolution,
    dtype=torch.float32
).to(device)

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("="*70)
print("STARTING TRAINING")
if start_epoch > 0:
    print(f"RESUMING FROM EPOCH {start_epoch} --> TARGET EPOCH {num_epochs - 1}")
else:
    print("FRESH START FROM EPOCH 0")
print("="*70)
print()

global_step = 0
# Note: best_val_loss and best_epoch were already set in the checkpoint
# loading section above (either restored from checkpoint or initialized to
# float('inf') and 0 for fresh training)

# KEY CHANGE: range(start_epoch, num_epochs) so resumed runs don't repeat epochs
for epoch in tqdm(range(start_epoch, num_epochs), desc="Training"):
    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0
    epoch_semantic_loss = 0.0
    epoch_position_loss = 0.0
    epoch_color_loss = 0.0
    epoch_opacity_loss = 0.0
    epoch_scale_loss = 0.0
    epoch_rotation_loss = 0.0

    gs_autoencoder.train()

    for i_batch, batch_data in enumerate(trainDataLoader):
        if isinstance(batch_data, dict):
            UV_gs_batch = batch_data['features'].type(torch.float32).to(device)
            segment_labels  = batch_data['segment_labels'].type(torch.int64).to(device)  if enable_semantic else None
            instance_labels = batch_data['instance_labels'].type(torch.int64).to(device) if enable_semantic else None
        else:
            UV_gs_batch = batch_data[0].type(torch.float32).to(device)
            segment_labels  = None
            instance_labels = None

        if epoch % 10 == 0 and i_batch == 0 and random_permute == 1:
            perm_indices = torch.randperm(UV_gs_batch.size()[1])
            UV_gs_batch = UV_gs_batch[:, perm_indices]
            if segment_labels  is not None: segment_labels  = segment_labels[:, perm_indices]
            if instance_labels is not None: instance_labels = instance_labels[:, perm_indices]

        if epoch % 5 == 0 and epoch > 1 and random_rotation == 1:
            rand_rot_comp = special_ortho_group.rvs(3)
            rand_rot = torch.tensor(np.dot(rand_rot_comp, rand_rot_comp.T),
                                    dtype=torch.float32).to(UV_gs_batch.device)
            UV_gs_batch[:, :, 4:7] = UV_gs_batch[:, :, 4:7] @ rand_rot
            for bcbc in range(UV_gs_batch.shape[0]):
                shifted_points   = UV_gs_batch[bcbc, :, 4:7] + origin_offset
                voxel_indices    = torch.floor(shifted_points / resolution)
                voxel_indices    = torch.clip(voxel_indices, 0, volume_dims - 1)
                voxel_centers_b  = (voxel_indices - (volume_dims - 1) / 2) * resolution
                UV_gs_batch[bcbc, :, :3] = voxel_centers_b

        optimizer.zero_grad()

        shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features = gs_autoencoder(
            UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3]
        )

        KL_loss = -0.5 * torch.sum(
            1.0 + log_var - mu.pow(2) - log_var.exp(), dim=1
        ).mean()

        target = UV_gs_batch[:, :, GEOMETRIC_INDICES]
        UV_gs_recover_reshaped = UV_gs_recover.reshape(UV_gs_batch.shape[0], -1, 14)

        recon_loss = compute_reconstruction_loss(
            UV_gs_recover_reshaped, target,
            batch_size=UV_gs_batch.shape[0],
            color_weight=args.color_loss_weight
        )

        if i_batch == 0:
            coord_pred   = UV_gs_recover_reshaped[:, :, 0:3].detach().cpu().numpy()
            coord_target = target[:, :, 0:3].detach().cpu().numpy()
            print(f"\nEPOCH {epoch} - DIAGNOSTIC:")
            print(f"  Target coord range: [{coord_target.min():.2f}, {coord_target.max():.2f}]")
            print(f"  Output coord range: [{coord_pred.min():.2f}, {coord_pred.max():.2f}]")

        individual_losses = compute_individual_losses(UV_gs_recover_reshaped, target)

        semantic_loss    = torch.tensor(0.0, device=device)
        semantic_metrics = {}
        if enable_semantic and segment_labels is not None and per_gaussian_features is not None:
            semantic_loss, semantic_metrics = compute_semantic_loss(
                embeddings=per_gaussian_features,
                segment_labels=segment_labels,
                instance_labels=instance_labels,
                batch_size=UV_gs_batch.shape[0],
                segment_weight=args.segment_loss_weight,
                instance_weight=args.instance_loss_weight,
                temperature=args.semantic_temperature,
                subsample=args.semantic_subsample,
                sampling_strategy=args.sampling_strategy
            )

        loss = recon_loss + kl_weight * KL_loss + semantic_loss

        loss_value          = loss.item()
        recon_loss_value    = recon_loss.item()
        kl_loss_value       = KL_loss.item()
        semantic_loss_value = semantic_loss.item()

        loss.backward()
        optimizer.step()

        epoch_loss          += loss_value
        epoch_recon_loss    += recon_loss_value
        epoch_kl_loss       += kl_loss_value
        epoch_semantic_loss += semantic_loss_value
        epoch_position_loss += individual_losses['position']
        epoch_color_loss    += individual_losses['color']
        epoch_opacity_loss  += individual_losses['opacity']
        epoch_scale_loss    += individual_losses['scale']
        epoch_rotation_loss += individual_losses['rotation']

        if wandb_enabled:
            log_dict = {
                "train/step_loss":          loss_value,
                "train/step_recon_loss":    recon_loss_value,
                "train/step_kl_loss":       kl_loss_value,
                "train/step_position_loss": individual_losses['position'],
                "train/step_color_loss":    individual_losses['color'],
                "train/step_opacity_loss":  individual_losses['opacity'],
                "train/step_scale_loss":    individual_losses['scale'],
                "train/step_rotation_loss": individual_losses['rotation'],
            }
            if semantic_metrics:
                for key, value in semantic_metrics.items():
                    log_dict[f"train/step_{key}"] = value
            wandb_run.log(log_dict, step=global_step)

        global_step += 1

        if i_batch == 0:
            msg = (f"Epoch {epoch}/{num_epochs} | Loss: {loss_value:.2f} | "
                   f"Recon: {recon_loss_value:.2f} | KL: {kl_loss_value:.2f}")
            if semantic_loss_value > 0:
                msg += f" | Semantic: {semantic_loss_value:.4f}"
            msg += (f"\n  Pos: {individual_losses['position']:.2f} | "
                    f"Color: {individual_losses['color']:.2f} | "
                    f"Opacity: {individual_losses['opacity']:.2f} | "
                    f"Scale: {individual_losses['scale']:.2f} | "
                    f"Rot: {individual_losses['rotation']:.2f}")
            if args.color_loss_weight > 1.0:
                msg += f" [COLOR_WEIGHT={args.color_loss_weight}]"
            print(msg)

    avg_train_loss    = epoch_loss          / len(trainDataLoader)
    avg_train_recon   = epoch_recon_loss    / len(trainDataLoader)
    avg_train_kl      = epoch_kl_loss       / len(trainDataLoader)
    avg_train_semantic= epoch_semantic_loss / len(trainDataLoader)
    avg_position      = epoch_position_loss / len(trainDataLoader)
    avg_color         = epoch_color_loss    / len(trainDataLoader)
    avg_opacity       = epoch_opacity_loss  / len(trainDataLoader)
    avg_scale         = epoch_scale_loss    / len(trainDataLoader)
    avg_rotation      = epoch_rotation_loss / len(trainDataLoader)

    val_metrics = None
    if epoch % eval_every == 0 or epoch == num_epochs - 1:
        print(f"\n{'='*70}")
        print(f"VALIDATION (Epoch {epoch})")
        print(f"{'='*70}")

        val_metrics = evaluate_model(
            gs_autoencoder, valDataLoader, device, failure_threshold,
            save_path=save_path, epoch=epoch,
            enable_pca_vis=enable_semantic,
            num_vis_scenes=args.pca_num_scenes
        )

        print(f"  L2 Error:  {val_metrics['avg_l2_error']:.2f}")
        print(f"  Position:  {val_metrics['position_loss']:.2f}")
        print(f"  Color:     {val_metrics['color_loss']:.2f}")
        print(f"  Opacity:   {val_metrics['opacity_loss']:.2f}")
        print(f"  Scale:     {val_metrics['scale_loss']:.2f}")
        print(f"  Rotation:  {val_metrics['rotation_loss']:.2f}")

        if val_metrics['avg_l2_error'] < best_val_loss:
            best_val_loss = val_metrics['avg_l2_error']
            best_epoch    = epoch
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    gs_autoencoder.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'val_l2_error':        val_metrics['avg_l2_error'],
                'semantic_mode':       effective_semantic_mode,
                'enable_semantic':     enable_semantic,
                'color_loss_weight':   args.color_loss_weight,
                'use_canonical_norm':  args.use_canonical_norm,
                'scale_norm_mode':     args.scale_norm_mode,
            }, best_model_path)
            print(f"  New best model! (L2: {best_val_loss:.2f})")
        print(f"{'='*70}\n")

    if wandb_enabled:
        log_dict = {
            "train/epoch_loss":     avg_train_loss,
            "train/epoch_recon":    avg_train_recon,
            "train/epoch_kl":       avg_train_kl,
            "train/epoch_semantic": avg_train_semantic,
            "train/epoch":          epoch,
            "train/epoch_position": avg_position,
            "train/epoch_color":    avg_color,
            "train/epoch_opacity":  avg_opacity,
            "train/epoch_scale":    avg_scale,
            "train/epoch_rotation": avg_rotation,
            "best/val_l2_error":    best_val_loss,
            "best/epoch":           best_epoch,
        }
        if val_metrics is not None:
            log_dict.update({
                "val/l2_error":       val_metrics['avg_l2_error'],
                "val/failure_rate":   val_metrics['failure_rate'],
                "val/position_loss":  val_metrics['position_loss'],
                "val/color_loss":     val_metrics['color_loss'],
                "val/opacity_loss":   val_metrics['opacity_loss'],
                "val/scale_loss":     val_metrics['scale_loss'],
                "val/rotation_loss":  val_metrics['rotation_loss'],
            })
        wandb_run.log(log_dict, step=global_step)

    if epoch >= 10 and epoch % 50 == 0:
        checkpoint_path = os.path.join(save_path, f"epoch_{epoch}.pth")
        torch.save({
            'epoch':               epoch,
            'model_state_dict':    gs_autoencoder.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'train_loss':          avg_train_loss,
            'semantic_mode':       effective_semantic_mode,
            'enable_semantic':     enable_semantic,
            'color_loss_weight':   args.color_loss_weight,
            'use_canonical_norm':  args.use_canonical_norm,
            'scale_norm_mode':     args.scale_norm_mode,
        }, checkpoint_path)
        print(f"Checkpoint saved: epoch_{epoch}.pth")

# ============================================================================
# FINAL SAVE
# ============================================================================

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

final_val_metrics = evaluate_model(
    gs_autoencoder, valDataLoader, device, failure_threshold,
    save_path=save_path, epoch=num_epochs-1,
    enable_pca_vis=enable_semantic, num_vis_scenes=args.pca_num_scenes
)

print(f"\nFinal Results:")
print(f"  Final L2:  {final_val_metrics['avg_l2_error']:.2f}")
print(f"  Best L2:   {best_val_loss:.2f} (epoch {best_epoch})")
print(f"\nIndividual Losses:")
print(f"  Position:  {final_val_metrics['position_loss']:.2f}")
print(f"  Color:     {final_val_metrics['color_loss']:.2f}")
print(f"  Opacity:   {final_val_metrics['opacity_loss']:.2f}")
print(f"  Scale:     {final_val_metrics['scale_loss']:.2f}")
print(f"  Rotation:  {final_val_metrics['rotation_loss']:.2f}")

final_path = os.path.join(save_path, "final.pth")
torch.save({
    'epoch':               num_epochs - 1,
    'model_state_dict':    gs_autoencoder.state_dict(),
    'optimizer_state_dict':optimizer.state_dict(),
    'final_val_l2':        final_val_metrics['avg_l2_error'],
    'best_val_l2':         best_val_loss,
    'best_epoch':          best_epoch,
    'semantic_mode':       effective_semantic_mode,
    'enable_semantic':     enable_semantic,
    'color_loss_weight':   args.color_loss_weight,
    'use_canonical_norm':  args.use_canonical_norm,
    'scale_norm_mode':     args.scale_norm_mode,
    'individual_losses': {
        'position': final_val_metrics['position_loss'],
        'color':    final_val_metrics['color_loss'],
        'opacity':  final_val_metrics['opacity_loss'],
        'scale':    final_val_metrics['scale_loss'],
        'rotation': final_val_metrics['rotation_loss'],
    },
}, final_path)

print(f"\nSaved: {final_path}")

if wandb_enabled:
    wandb_run.summary.update({
        "final_val_l2_error":   final_val_metrics['avg_l2_error'],
        "best_val_l2_error":    best_val_loss,
        "best_epoch":           best_epoch,
        "semantic_mode":        effective_semantic_mode,
        "enable_semantic":      enable_semantic,
        "color_loss_weight":    args.color_loss_weight,
        "use_canonical_norm":   args.use_canonical_norm,
        "final_position_loss":  final_val_metrics['position_loss'],
        "final_color_loss":     final_val_metrics['color_loss'],
        "final_opacity_loss":   final_val_metrics['opacity_loss'],
        "final_scale_loss":     final_val_metrics['scale_loss'],
        "final_rotation_loss":  final_val_metrics['rotation_loss'],
    })
    wandb_run.finish()

print("\nTraining complete!")
print(f"  Canonical norm:  {args.use_canonical_norm}")
print(f"  Color weight:    {args.color_loss_weight}")
print(f"  Scale norm mode: {args.scale_norm_mode}")
if args.resume_checkpoint:
    print(f"  Resumed from:    {args.resume_checkpoint}")