"""
Can3Tok Training - WITH RGB COLOR + INDIVIDUAL PARAMETER TRACKING + PCA VISUALIZATION
=====================================================================================
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

# PCA Feature Visualization (NEW!)
from pca_feature_visualization import visualize_comparison

# 3DGS PLY Reconstruction (NEW! View on supersplat.at)
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
    'opacity': slice(6, 7),       # α
    'scale': slice(7, 10),        # sx, sy, sz
    'rotation': slice(10, 14),    # quat
}

# Indices from input to extract for target
GEOMETRIC_INDICES = list(range(4, 7)) + list(range(7, 10)) + [10] + list(range(11, 14)) + list(range(14, 18))

print(f"📐 PARAMETER CONFIGURATION:")
print(f"   Total params: 14")
print(f"   Position (xyz): indices {PARAM_SLICES['position']}")
print(f"   Color (rgb):    indices {PARAM_SLICES['color']}")
print(f"   Opacity (α):    indices {PARAM_SLICES['opacity']}")
print(f"   Scale (s):      indices {PARAM_SLICES['scale']}")
print(f"   Rotation (q):   indices {PARAM_SLICES['rotation']}")
print()

# ============================================================================
# INDIVIDUAL LOSS COMPUTATION FUNCTION
# ============================================================================

def compute_individual_losses(prediction, target, batch_size):
    """
    Compute L2 loss for each parameter type separately.
    
    Args:
        prediction: [B, 40000, 14] reconstructed Gaussians
        target: [B, 40000, 14] ground truth Gaussians
        batch_size: B
    
    Returns:
        dict with individual losses
    """
    losses = {}
    
    # Position loss (xyz)
    pos_pred = prediction[:, :, PARAM_SLICES['position']]
    pos_target = target[:, :, PARAM_SLICES['position']]
    losses['position'] = torch.norm(pos_pred - pos_target, p=2) / batch_size
    
    # Color loss (rgb)
    color_pred = prediction[:, :, PARAM_SLICES['color']]
    color_target = target[:, :, PARAM_SLICES['color']]
    losses['color'] = torch.norm(color_pred - color_target, p=2) / batch_size
    
    # Opacity loss (α)
    opacity_pred = prediction[:, :, PARAM_SLICES['opacity']]
    opacity_target = target[:, :, PARAM_SLICES['opacity']]
    losses['opacity'] = torch.norm(opacity_pred - opacity_target, p=2) / batch_size
    
    # Scale loss (sx, sy, sz)
    scale_pred = prediction[:, :, PARAM_SLICES['scale']]
    scale_target = target[:, :, PARAM_SLICES['scale']]
    losses['scale'] = torch.norm(scale_pred - scale_target, p=2) / batch_size
    
    # Rotation loss (quaternion)
    rotation_pred = prediction[:, :, PARAM_SLICES['rotation']]
    rotation_target = target[:, :, PARAM_SLICES['rotation']]
    losses['rotation'] = torch.norm(rotation_pred - rotation_target, p=2) / batch_size
    # # With this:
    # dot = torch.sum(rotation_pred * rotation_target, dim=-1).clamp(-1, 1)
    # losses['rotation'] = (1.0 - dot.abs()).mean()  # geodesic, handles q/-q symmetry
    
    # Total (for verification)
    losses['total_individual'] = sum(losses.values())
    
    return losses


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Can3Tok Training - RGB + Individual Tracking + PCA Visualization')
parser.add_argument('--use_wandb', action='store_true', default=False)
parser.add_argument('--wandb_project', type=str, default='Can3Tok-RGB-PCA')
parser.add_argument('--wandb_entity', type=str, default='3D-SSC')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--kl_weight', type=float, default=1e-5)
parser.add_argument('--eval_every', type=int, default=10)
parser.add_argument('--failure_threshold', type=float, default=8000.0)
parser.add_argument('--train_scenes', type=int, default=None)
parser.add_argument('--val_scenes', type=int, default=None)
parser.add_argument('--sampling_method', type=str, default='opacity', choices=['random', 'opacity'])
parser.add_argument('--segment_loss_weight', type=float, default=0.0)
parser.add_argument('--instance_loss_weight', type=float, default=0.0)
parser.add_argument('--semantic_temperature', type=float, default=0.07)
parser.add_argument('--semantic_subsample', type=int, default=2000)
parser.add_argument('--recon_scale', type=float, default=1000.0)
parser.add_argument('--semantic_mode', type=str, default='none', 
                    choices=['none', 'hidden', 'geometric', 'attention'])
parser.add_argument('--sampling_strategy', type=str, default='balanced',
                    choices=['random', 'balanced'])
parser.add_argument('--pca_vis_freq', type=int, default=10,
                    help='Generate PCA visualizations every N epochs')
parser.add_argument('--pca_brightness', type=float, default=1.25,
                    help='Brightness multiplier for PCA colors')
parser.add_argument('--pca_num_scenes', type=int, default=10,
                    help='Number of scenes to visualize with PCA (default: 10)')
parser.add_argument('--recon_ply_freq', type=int, default=10,
                    help='Save reconstructed 3DGS PLY files every N epochs')
parser.add_argument('--recon_ply_num_scenes', type=int, default=5,
                    help='Number of scenes to save as 3DGS PLY (default: 5)')
parser.add_argument('--recon_ply_max_sh', type=int, default=3,
                    help='Max SH degree for reconstructed PLY (default: 3)')
                    
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
        
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                "learning_rate": args.lr,
                "architecture": "Can3Tok-RGB-Individual-Tracking-PCA",
                "dataset": "SceneSplat-7K",
                "batch_size": args.batch_size,
                "epochs": args.num_epochs,
                "kl_weight": args.kl_weight,
                "semantic_mode": effective_semantic_mode,
                "enable_semantic": enable_semantic,
                "num_params": 14,  # With RGB!
                "individual_tracking": True,
                "pca_visualization": True,
                "pca_vis_freq": args.pca_vis_freq,
            },
            tags=["rgb-color", "individual-tracking", "pca-visualization"],
        )
        print("✓ Weights & Biases enabled")
        wandb_enabled = True
    except Exception as e:
        print(f"✗ Weights & Biases failed: {e}")
        wandb_enabled = False
else:
    print("✗ Weights & Biases disabled")

# ============================================================================
# GPU SETUP
# ============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================================
# CONFIGURATION
# ============================================================================

loss_usage = "L1"
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
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_folder = f"RGB_local_{timestamp}_{effective_semantic_mode}"
    if enable_semantic:
        checkpoint_folder += f"_beta{args.segment_loss_weight}"

save_path = f"/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/{checkpoint_folder}/"
os.makedirs(save_path, exist_ok=True)

print(f"\n{'='*70}")
print(f"🚀 CAN3TOK TRAINING - RGB + INDIVIDUAL TRACKING + PCA VISUALIZATION")
print(f"{'='*70}")
print(f"Configuration:")
print(f"  Job ID: {job_id or 'local'}")
print(f"  Semantic Mode: {effective_semantic_mode}")
print(f"  Semantic Enabled: {enable_semantic}")
print(f"  PCA Visualization: ENABLED (every {args.pca_vis_freq} epochs)")
print(f"  3DGS Reconstruction: ENABLED (every {args.recon_ply_freq} epochs, {args.recon_ply_num_scenes} scenes)")
# print(f"  View reconstructions at: https://supersplat.at")
print(f"  Parameters: 14 (WITH RGB COLOR!)")
print(f"  Individual tracking: ENABLED")
print(f"  Device: {device}")
print(f"  Save path: {save_path}")
print(f"{'='*70}\n")

# ============================================================================
# MODEL SETUP
# ============================================================================

print("Loading model configuration...")
config_path_perceiver = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
model_config_perceiver = get_config_from_file(config_path_perceiver)

print(f"✓ Loaded YAML config")
print(f"✓ Overriding with command-line arguments...")

model_config = model_config_perceiver.model
model_config.params.shape_module_cfg.params.semantic_mode = effective_semantic_mode

print(f"   semantic_mode: {effective_semantic_mode}")

print(f"\n{'='*70}")
print("INSTANTIATING MODEL")
print(f"{'='*70}")
perceiver_encoder_decoder = instantiate_from_config(model_config)
print(f"✓ Model instantiated successfully")
print(f"{'='*70}\n")

gs_autoencoder = perceiver_encoder_decoder
gs_autoencoder.to(device)

optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=args.lr, betas=[0.9, 0.999])

print("✓ Model loaded successfully\n")

# ============================================================================
# DATASET LOADING
# ============================================================================

print("Loading datasets...")
from gs_dataset_scenesplat import gs_dataset

# Training dataset
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
    normalize=True,
    target_radius=10.0,
)

trainDataLoader = Data.DataLoader(
    dataset=gs_dataset_train, 
    batch_size=bch_size,
    shuffle=True, 
    num_workers=9,
    pin_memory=True,
    persistent_workers=True
)

# Validation dataset
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
    normalize=True,
    target_radius=10.0,
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
print(f"✓ Training: {len(gs_dataset_train)} scenes, {len(trainDataLoader)} batches")
print(f"✓ Validation: {len(gs_dataset_val)} scenes, {len(valDataLoader)} batches")
print(f"✓ Sampling method: {sampling_method}")
print("="*70)
print()

# ============================================================================
# EVALUATION FUNCTION WITH PCA VISUALIZATION
# ============================================================================

def evaluate_model(model, dataloader, device, failure_threshold, 
                   save_path=None, epoch=None, enable_pca_vis=True, num_vis_scenes=10):
    """
    Evaluate model with individual parameter tracking + PCA visualization.
    
    Args:
        model: The model to evaluate
        dataloader: Validation dataloader
        device: Device to use
        failure_threshold: Threshold for failure detection
        save_path: Directory to save visualizations
        epoch: Current epoch number
        enable_pca_vis: Whether to generate PCA visualizations
        num_vis_scenes: Number of scenes to visualize (default: 10)
    """
    model.eval()
    
    total_l2_error = 0.0
    total_kl = 0.0
    per_scene_l2_errors = []
    num_failures = 0
    num_scenes = 0
    
    # Individual parameter tracking
    total_position_loss = 0.0
    total_color_loss = 0.0
    total_opacity_loss = 0.0
    total_scale_loss = 0.0
    total_rotation_loss = 0.0
    
    # PCA visualization: collect first N scenes separately
    scene_data_list = []  # List of dicts, one per scene
    scenes_collected = 0
    max_scenes_for_visualization = num_vis_scenes  # First 10 scenes

    # 3DGS reconstruction: collect first M scenes separately
    recon_preds_list = []   # List of (N, 14) arrays, one per scene
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
            
            # ================================================================
            # Temporarily enable training mode for semantic features
            # ================================================================
            need_semantic = (enable_pca_vis and 
                           scenes_collected < max_scenes_for_visualization and
                           segment_labels is not None and
                           epoch is not None and
                           epoch % args.pca_vis_freq == 0)
            
            was_training = model.training
            if need_semantic:
                model.train()
            
            shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features = model(
                UV_gs_batch,
                UV_gs_batch,
                UV_gs_batch,
                UV_gs_batch[:, :, :3]
            )
            
            # Restore eval mode
            if need_semantic and not was_training:
                model.eval()
            # ================================================================
            
            target = UV_gs_batch[:, :, GEOMETRIC_INDICES]
            UV_gs_recover_reshaped = UV_gs_recover.reshape(UV_gs_batch.shape[0], -1, 14)

            # # Normalize to [-1, 1] for loss computation
            # target_scaled = target.clone()
            # target_scaled[:, :, 0:3] /= 10.0   # positions now in [-0.9, +0.9]

            # pred_scaled = UV_gs_recover_reshaped.clone()
            # pred_scaled[:, :, 0:3] /= 10.0     # predictions now in comparable range

            # # recon_loss_raw = torch.norm(pred_scaled - target_scaled, p=2) / batch_size

            # # Total L2 error
            # batch_l2_error = torch.norm(
            #     pred_scaled - target_scaled,
            #     p=2
            # ) / batch_size


            
            # Total L2 error
            batch_l2_error = torch.norm(
                UV_gs_recover_reshaped - target,
                p=2
            ) / batch_size
            
            # Individual losses
            individual_losses = compute_individual_losses(
                UV_gs_recover_reshaped,
                target,
                batch_size
            )
            
            per_scene_norms = torch.norm(
                UV_gs_recover_reshaped - target,
                p=2,
                dim=(1, 2)
            )
            per_scene_errors_scaled = per_scene_norms / np.sqrt(batch_size)
            
            kl_loss = -0.5 * torch.sum(
                1.0 + log_var - mu.pow(2) - log_var.exp(),
                dim=1
            )
            
            total_l2_error += batch_l2_error.item() * batch_size
            total_kl += kl_loss.sum().item()
            per_scene_l2_errors.extend(per_scene_errors_scaled.cpu().numpy().tolist())
            num_failures += (per_scene_errors_scaled.cpu().numpy() > failure_threshold).sum()
            num_scenes += batch_size
            
            # Accumulate individual losses
            total_position_loss += individual_losses['position'].item() * batch_size
            total_color_loss += individual_losses['color'].item() * batch_size
            total_opacity_loss += individual_losses['opacity'].item() * batch_size
            total_scale_loss += individual_losses['scale'].item() * batch_size
            total_rotation_loss += individual_losses['rotation'].item() * batch_size
            
            # ============================================================
            # Collect features for PCA visualization (first N scenes)
            # ============================================================
            if (enable_pca_vis and 
                scenes_collected < max_scenes_for_visualization and
                per_gaussian_features is not None and
                segment_labels is not None and
                epoch is not None and
                epoch % args.pca_vis_freq == 0):
                
                try:
                    # Extract position and color from input
                    # Format: [voxel(3), idx(1), xyz(3), rgb(3), opacity(1), scale(3), quat(4)]
                    pos = UV_gs_batch[:, :, 4:7].cpu().numpy()  # [B, 40000, 3]
                    col = UV_gs_batch[:, :, 7:10].cpu().numpy()  # [B, 40000, 3]
                    
                    # Extract semantic features [B, 40000, 32]
                    sem_feat = per_gaussian_features.cpu().numpy()
                    
                    # Process each scene in the batch individually
                    for scene_idx in range(batch_size):
                        if scenes_collected >= max_scenes_for_visualization:
                            break
                        
                        # Extract data for this scene
                        scene_dict = {
                            'semantic_features': sem_feat[scene_idx],      # [40000, 32]
                            'positions': pos[scene_idx],                   # [40000, 3]
                            'colors': col[scene_idx],                      # [40000, 3]
                            'coords': pos[scene_idx],                      # [40000, 3] (same as positions)
                            'scene_id': scenes_collected
                        }
                        
                        scene_data_list.append(scene_dict)
                        scenes_collected += 1
                    
                except Exception as e:
                    print(f"⚠️  Could not extract features for PCA visualization: {e}")

            # ============================================================
            # Collect predictions for 3DGS PLY reconstruction
            # ============================================================
            if (recon_scenes_collected < max_scenes_for_recon and
                epoch is not None and
                epoch % args.recon_ply_freq == 0):

                try:
                    # UV_gs_recover_reshaped is [B, N, 14]
                    pred_np = UV_gs_recover_reshaped.cpu().numpy()  # [B, N, 14]

                    for scene_idx in range(batch_size):
                        if recon_scenes_collected >= max_scenes_for_recon:
                            break
                        recon_preds_list.append(pred_np[scene_idx])  # (N, 14)
                        recon_scenes_collected += 1

                except Exception as e:
                    print(f"⚠️  Could not collect predictions for 3DGS reconstruction: {e}")
    
    avg_l2_error = total_l2_error / num_scenes
    avg_kl = total_kl / num_scenes
    failure_rate = (num_failures / num_scenes) * 100.0
    
    per_scene_l2_errors = np.array(per_scene_l2_errors)
    
    # ============================================================
    # Generate PCA Visualizations (Per-Scene)
    # ============================================================
    pca_paths = {}
    if (enable_pca_vis and 
        len(scene_data_list) > 0 and 
        save_path is not None and 
        epoch is not None and 
        epoch % args.pca_vis_freq == 0):
        
        print("\n" + "="*70)
        print(f"PCA FEATURE VISUALIZATION (Epoch {epoch})")
        print("="*70)
        print(f"Generating PCA visualizations for {len(scene_data_list)} scenes...")
        
        # Create epoch-specific directory
        vis_dir = Path(save_path) / "pca_visualizations" / f"epoch_{epoch:03d}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualization for each scene
        for scene_idx, scene_data in enumerate(scene_data_list):
            try:
                print(f"\nProcessing scene {scene_idx}/{len(scene_data_list)-1}...")
                
                # Generate comparison visualizations for this scene
                scene_pca_paths = visualize_comparison(
                    coords=scene_data['coords'],
                    semantic_features=scene_data['semantic_features'],
                    positions=scene_data['positions'],
                    colors=scene_data['colors'],
                    output_dir=vis_dir,
                    scene_name=f"scene_{scene_idx:03d}",
                    brightness=args.pca_brightness
                )
                
                # Store paths
                pca_paths[f'scene_{scene_idx:03d}'] = scene_pca_paths
                
            except Exception as e:
                print(f"⚠️  Error creating PCA visualization for scene {scene_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*70)
        print(f"✓ Saved PCA visualizations for {len(pca_paths)} scenes")
        print(f"  Location: {vis_dir}")
        print("="*70)

    # ============================================================
    # Save Reconstructed 3DGS PLY Files (supersplat.at viewable)
    # ============================================================
    recon_paths = {}
    if (len(recon_preds_list) > 0 and
        save_path is not None and
        epoch is not None and
        epoch % args.recon_ply_freq == 0):

        try:
            # Stack into [M, N, 14]
            all_preds = np.stack(recon_preds_list, axis=0)

            # Save into epoch-specific subdirectory
            recon_dir = Path(save_path) / "reconstructed_gaussians" / f"epoch_{epoch:03d}"

            recon_paths = save_reconstructed_gaussians(
                predictions=all_preds,
                output_dir=recon_dir,
                epoch=epoch,
                num_scenes=len(recon_preds_list),
                max_sh_degree=args.recon_ply_max_sh,
                color_mode="1",
                # max_scale_act=0.5,  # clamp scale_act before log → max 0.5m splat
                prefix="scene"
            )

        except Exception as e:
            print(f"⚠️  Error saving reconstructed Gaussian PLY files: {e}")
            import traceback
            traceback.print_exc()

    model.train()
    
    return {
        'avg_l2_error': avg_l2_error,
        'l2_std': per_scene_l2_errors.std(),
        'failure_rate': failure_rate,
        'avg_kl': avg_kl,
        'per_scene_errors': per_scene_l2_errors,
        # Individual losses
        'position_loss': total_position_loss / num_scenes,
        'color_loss': total_color_loss / num_scenes,
        'opacity_loss': total_opacity_loss / num_scenes,
        'scale_loss': total_scale_loss / num_scenes,
        'rotation_loss': total_rotation_loss / num_scenes,
        # PCA visualization paths
        'pca_paths': pca_paths,
        # 3DGS reconstruction paths (viewable on supersplat.at)
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
print("STARTING TRAINING WITH INDIVIDUAL PARAMETER TRACKING + PCA VISUALIZATION")
print("="*70)
print()

global_step = 0
best_val_loss = float('inf')
best_epoch = 0

for epoch in tqdm(range(num_epochs), desc="Training"):
    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0
    epoch_semantic_loss = 0.0
    
    # Individual parameter losses
    epoch_position_loss = 0.0
    epoch_color_loss = 0.0
    epoch_opacity_loss = 0.0
    epoch_scale_loss = 0.0
    epoch_rotation_loss = 0.0
    
    gs_autoencoder.train()
    
    for i_batch, batch_data in enumerate(trainDataLoader):
        # Extract data
        if isinstance(batch_data, dict):
            UV_gs_batch = batch_data['features'].type(torch.float32).to(device)
            segment_labels = batch_data['segment_labels'].type(torch.int64).to(device) if enable_semantic else None
            instance_labels = batch_data['instance_labels'].type(torch.int64).to(device) if enable_semantic else None
        else:
            UV_gs_batch = batch_data[0].type(torch.float32).to(device)
            segment_labels = None
            instance_labels = None
        
        # Random permutation (reduced frequency)
        if epoch % 10 == 0 and i_batch == 0 and random_permute == 1:
            perm_indices = torch.randperm(UV_gs_batch.size()[1])
            UV_gs_batch = UV_gs_batch[:, perm_indices]
            
            if segment_labels is not None:
                segment_labels = segment_labels[:, perm_indices]
            if instance_labels is not None:
                instance_labels = instance_labels[:, perm_indices]
        
        if epoch % 5 == 0 and epoch > 1 and random_rotation == 1:
            rand_rot_comp = special_ortho_group.rvs(3)
            rand_rot = torch.tensor(np.dot(rand_rot_comp, rand_rot_comp.T), 
                                dtype=torch.float32).to(UV_gs_batch.device)
            UV_gs_batch[:,:,4:7] = UV_gs_batch[:,:,4:7] @ rand_rot
            
            for bcbc in range(UV_gs_batch.shape[0]):
                shifted_points = UV_gs_batch[bcbc, :, 4:7] + origin_offset
                voxel_indices = torch.floor(shifted_points / resolution)
                voxel_indices = torch.clip(voxel_indices, 0, volume_dims - 1)
                voxel_centers_batch = (voxel_indices - (volume_dims - 1) / 2) * resolution
                UV_gs_batch[bcbc, :, :3] = voxel_centers_batch
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features = gs_autoencoder(
            UV_gs_batch, 
            UV_gs_batch, 
            UV_gs_batch, 
            UV_gs_batch[:, :, :3]
        )

        
  

        
        # Compute losses
        KL_loss = -0.5 * torch.sum(
            1.0 + log_var - mu.pow(2) - log_var.exp(), 
            dim=1
        ).mean()
        
        target = UV_gs_batch[:, :, GEOMETRIC_INDICES]
        UV_gs_recover_reshaped = UV_gs_recover.reshape(UV_gs_batch.shape[0], -1, 14)


        # Around line 856, after:
        # shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features = gs_autoencoder(...)

        

        
        # Normalize to [-1, 1] for loss computation
        # target_scaled = target.clone()
        # target_scaled[:, :, 0:3] /= 10.0   # positions now in [-0.9, +0.9]

        # pred_scaled = UV_gs_recover_reshaped.clone()
        # pred_scaled[:, :, 0:3] /= 10.0     # predictions now in comparable range

            # recon_loss_raw = torch.norm(pred_scaled - target_scaled, p=2) / batch_size

        # recon_loss_raw = torch.norm(
        #     pred_scaled - target_scaled,
        #     p=2
        # ) / UV_gs_batch.shape[0]

        recon_loss_raw = torch.norm(
            UV_gs_recover_reshaped - target,
            p=2
        ) / UV_gs_batch.shape[0]


        # 🔍 ADD THIS DEBUG BLOCK:
        if i_batch == 0:
            UV_gs_recover_reshaped = UV_gs_recover.reshape(UV_gs_batch.shape[0], -1, 14)
            coord_pred = UV_gs_recover_reshaped[:, :, 0:3].detach().cpu().numpy()
            coord_target = target[:, :, 0:3].detach().cpu().numpy()

            # # UV_gs_recover_reshaped = UV_gs_recover.reshape(UV_gs_batch.shape[0], -1, 14)
            # coord_pred = pred_scaled[:, :, 0:3].detach().cpu().numpy()
            # coord_target = target_scaled[:, :, 0:3].detach().cpu().numpy()
            
            print(f"\n🔍 EPOCH {epoch} - MODEL OUTPUT DIAGNOSTIC:")
            print(f"  Input (target) coord range:  [{coord_target.min():.2f}, {coord_target.max():.2f}]")
            print(f"  Model output coord range:    [{coord_pred.min():.2f}, {coord_pred.max():.2f}]")
            
            # if coord_pred.min() > -1.0 and coord_pred.max() < 2.0:
            #     # print(f"  ⚠️  MODEL OUTPUTS [0, 1] RANGE!")
            #     # print(f"  ⚠️  This model was trained with OLD normalization!")
            #     # print(f"  ⚠️  Need to RETRAIN from scratch with canonical sphere dataset!")
            # elif coord_pred.min() < -5.0 and coord_pred.max() > 5.0:
            #     print(f"  ✓ Model outputs canonical sphere range!")
            # else:
            #     print(f"  ⚠️  Unexpected output range - might be early in training")
        
        # Compute individual losses
        individual_losses = compute_individual_losses(
            UV_gs_recover_reshaped,
            target,
            UV_gs_batch.shape[0]
        )
        
        recon_loss = recon_loss_raw / args.recon_scale
        
        # Semantic contrastive loss
        semantic_loss = torch.tensor(0.0, device=device)
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
        
        # Combined loss
        loss = recon_loss + kl_weight * KL_loss + semantic_loss
        
        # Save values for logging
        loss_value = loss.item()
        recon_loss_raw_value = recon_loss_raw.item()
        kl_loss_value = KL_loss.item()
        semantic_loss_value = semantic_loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        epoch_loss += loss_value
        epoch_recon_loss += recon_loss_raw_value
        epoch_kl_loss += kl_loss_value
        epoch_semantic_loss += semantic_loss_value
        
        # Accumulate individual losses
        epoch_position_loss += individual_losses['position'].item()
        epoch_color_loss += individual_losses['color'].item()
        epoch_opacity_loss += individual_losses['opacity'].item()
        epoch_scale_loss += individual_losses['scale'].item()
        epoch_rotation_loss += individual_losses['rotation'].item()
        
        # Log to wandb
        if wandb_enabled:
            log_dict = {
                "train/step_loss": loss_value,
                "train/step_recon_loss": recon_loss_raw_value,
                "train/step_kl_loss": kl_loss_value,
                # Individual parameter losses
                "train/step_position_loss": individual_losses['position'].item(),
                "train/step_color_loss": individual_losses['color'].item(),
                "train/step_opacity_loss": individual_losses['opacity'].item(),
                "train/step_scale_loss": individual_losses['scale'].item(),
                "train/step_rotation_loss": individual_losses['rotation'].item(),
            }
            
            if semantic_metrics:
                for key, value in semantic_metrics.items():
                    log_dict[f"train/step_{key}"] = value
            
            wandb_run.log(log_dict, step=global_step)
        
        global_step += 1
        
        # Print first batch
        if i_batch == 0:
            print_msg = (f"Epoch {epoch}/{num_epochs} | "
                        f"Loss: {loss_value:.2f} | "
                        f"Recon: {recon_loss_raw_value:.2f} | "
                        f"KL: {kl_loss_value:.2f}")
            
            if semantic_loss_value > 0:
                print_msg += f" | Semantic: {semantic_loss_value:.4f}"
            
            # Print individual losses
            print_msg += f"\n  └─ Pos: {individual_losses['position'].item():.2f} | "
            print_msg += f"Color: {individual_losses['color'].item():.2f} | "
            print_msg += f"Opacity: {individual_losses['opacity'].item():.2f} | "
            print_msg += f"Scale: {individual_losses['scale'].item():.2f} | "
            print_msg += f"Rot: {individual_losses['rotation'].item():.2f}"
            
            print(print_msg)
    
    # Epoch summary
    avg_train_loss = epoch_loss / len(trainDataLoader)
    avg_train_recon = epoch_recon_loss / len(trainDataLoader)
    avg_train_kl = epoch_kl_loss / len(trainDataLoader)
    avg_train_semantic = epoch_semantic_loss / len(trainDataLoader)
    
    # Average individual losses
    avg_position = epoch_position_loss / len(trainDataLoader)
    avg_color = epoch_color_loss / len(trainDataLoader)
    avg_opacity = epoch_opacity_loss / len(trainDataLoader)
    avg_scale = epoch_scale_loss / len(trainDataLoader)
    avg_rotation = epoch_rotation_loss / len(trainDataLoader)
    
    # Validation
    val_metrics = None
    if epoch % eval_every == 0 or epoch == num_epochs - 1:
        print(f"\n{'='*70}")
        print(f"VALIDATION (Epoch {epoch})")
        print(f"{'='*70}")
        
        val_metrics = evaluate_model(
            gs_autoencoder, 
            valDataLoader, 
            device, 
            failure_threshold,
            save_path=save_path,
            epoch=epoch,
            enable_pca_vis=enable_semantic,  # Only if semantic mode enabled
            num_vis_scenes=args.pca_num_scenes
        )
        
        print(f"  L2 Error: {val_metrics['avg_l2_error']:.2f}")
        print(f"  Individual losses:")
        print(f"    Position: {val_metrics['position_loss']:.2f}")
        print(f"    Color:    {val_metrics['color_loss']:.2f}")
        print(f"    Opacity:  {val_metrics['opacity_loss']:.2f}")
        print(f"    Scale:    {val_metrics['scale_loss']:.2f}")
        print(f"    Rotation: {val_metrics['rotation_loss']:.2f}")
        
        # Track best model
        if val_metrics['avg_l2_error'] < best_val_loss:
            best_val_loss = val_metrics['avg_l2_error']
            best_epoch = epoch
            
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': gs_autoencoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_l2_error': val_metrics['avg_l2_error'],
                'semantic_mode': effective_semantic_mode,
                'enable_semantic': enable_semantic,
            }, best_model_path)
            print(f"  ✓ New best model! (L2: {best_val_loss:.2f})")
        
        print(f"{'='*70}\n")
    
    # Log epoch metrics
    if wandb_enabled:
        log_dict = {
            "train/epoch_loss": avg_train_loss,
            "train/epoch_recon": avg_train_recon,
            "train/epoch_kl": avg_train_kl,
            "train/epoch_semantic": avg_train_semantic,
            "train/epoch": epoch,
            # Individual epoch losses
            "train/epoch_position": avg_position,
            "train/epoch_color": avg_color,
            "train/epoch_opacity": avg_opacity,
            "train/epoch_scale": avg_scale,
            "train/epoch_rotation": avg_rotation,
            "best/val_l2_error": best_val_loss,
            "best/epoch": best_epoch,
        }
        
        if val_metrics is not None:
            log_dict.update({
                "val/l2_error": val_metrics['avg_l2_error'],
                "val/failure_rate": val_metrics['failure_rate'],
                "val/position_loss": val_metrics['position_loss'],
                "val/color_loss": val_metrics['color_loss'],
                "val/opacity_loss": val_metrics['opacity_loss'],
                "val/scale_loss": val_metrics['scale_loss'],
                "val/rotation_loss": val_metrics['rotation_loss'],
            })
        
        wandb_run.log(log_dict, step=global_step)
    
    # Save checkpoints
    if epoch >= 10 and epoch % 10 == 0:
        checkpoint_path = os.path.join(save_path, f"epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': gs_autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'semantic_mode': effective_semantic_mode,
            'enable_semantic': enable_semantic,
        }, checkpoint_path)
        print(f"✓ Checkpoint saved: epoch_{epoch}.pth")

# ============================================================================
# FINAL SAVE
# ============================================================================

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

final_val_metrics = evaluate_model(
    gs_autoencoder, 
    valDataLoader, 
    device, 
    failure_threshold,
    save_path=save_path,
    epoch=num_epochs-1,
    enable_pca_vis=enable_semantic,
    num_vis_scenes=args.pca_num_scenes
)

print(f"\nFinal Results:")
print(f"  Final L2: {final_val_metrics['avg_l2_error']:.2f}")
print(f"  Best L2: {best_val_loss:.2f} (epoch {best_epoch})")
print(f"\nIndividual Parameter Losses:")
print(f"  Position: {final_val_metrics['position_loss']:.2f}")
print(f"  Color:    {final_val_metrics['color_loss']:.2f}")
print(f"  Opacity:  {final_val_metrics['opacity_loss']:.2f}")
print(f"  Scale:    {final_val_metrics['scale_loss']:.2f}")
print(f"  Rotation: {final_val_metrics['rotation_loss']:.2f}")

final_path = os.path.join(save_path, "final.pth")
torch.save({
    'epoch': num_epochs - 1,
    'model_state_dict': gs_autoencoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_val_l2': final_val_metrics['avg_l2_error'],
    'best_val_l2': best_val_loss,
    'best_epoch': best_epoch,
    'semantic_mode': effective_semantic_mode,
    'enable_semantic': enable_semantic,
    'individual_losses': {
        'position': final_val_metrics['position_loss'],
        'color': final_val_metrics['color_loss'],
        'opacity': final_val_metrics['opacity_loss'],
        'scale': final_val_metrics['scale_loss'],
        'rotation': final_val_metrics['rotation_loss'],
    },
}, final_path)

print(f"\n✓ Saved: {final_path}")
print("="*70)

if wandb_enabled:
    wandb_run.summary.update({
        "final_val_l2_error": final_val_metrics['avg_l2_error'],
        "best_val_l2_error": best_val_loss,
        "best_epoch": best_epoch,
        "semantic_mode": effective_semantic_mode,
        "enable_semantic": enable_semantic,
        "final_position_loss": final_val_metrics['position_loss'],
        "final_color_loss": final_val_metrics['color_loss'],
        "final_opacity_loss": final_val_metrics['opacity_loss'],
        "final_scale_loss": final_val_metrics['scale_loss'],
        "final_rotation_loss": final_val_metrics['rotation_loss'],
    })
    
    wandb_run.finish()

print("\n🎉 Training complete with PCA feature visualization!")
print(f"Check {save_path}/pca_visualizations/ for semantic feature visualizations!")