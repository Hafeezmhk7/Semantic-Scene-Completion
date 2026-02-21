"""
Can3Tok Training with SceneSplat-7K Dataset + Validation
Version: 3.0 - FIXED VALIDATION BUG
Critical Fix: Validation now uses same L2 calculation as training
Features:
  - FIXED: Validation metric now matches training (was 10x too large)
  - Per-scene statistics for analysis
  - Sanity check mode (--use_train_as_val)
  - Configurable sampling method
  - Job-specific checkpoint folders
  - Weights & Biases integration
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

# Michelangelo imports (VAE model)
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file

# Data loading
import torch.utils.data as Data
from scipy.stats import special_ortho_group

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Can3Tok Training with FIXED Validation')
parser.add_argument('--use_wandb', action='store_true', default=False,
                    help='Enable Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='Can3Tok-SceneSplat',
                    help='W&B project name')
parser.add_argument('--wandb_entity', type=str, default='3D-SSC',
                    help='W&B entity/team name')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=1000,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--kl_weight', type=float, default=1e-5,
                    help='KL divergence loss weight')
parser.add_argument('--eval_every', type=int, default=10,
                    help='Evaluate on validation set every N epochs')
parser.add_argument('--failure_threshold', type=float, default=8000.0,
                    help='L2 error threshold for failure rate (adjusted for fixed metric)')

# Scene count parameters
parser.add_argument('--train_scenes', type=int, default=None,
                    help='Number of training scenes to use (None = all)')
parser.add_argument('--val_scenes', type=int, default=None,
                    help='Number of validation scenes to use (None = all)')

# Sampling method
parser.add_argument('--sampling_method', type=str, default='opacity',
                    choices=['random', 'opacity'],
                    help='Sampling method: random or opacity (default: opacity)')

# Sanity check mode
parser.add_argument('--use_train_as_val', action='store_true', default=False,
                    help='SANITY CHECK: Use training data as validation (test overfitting)')

args = parser.parse_args()

# ============================================================================
# WEIGHTS & BIASES SETUP
# ============================================================================

wandb_enabled = False
if args.use_wandb:
    try:
        import wandb
        
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        
        # Add sanity check info to run name
        run_name = f"can3tok_job_{job_id}_FIXED"
        if args.use_train_as_val:
            run_name += "_SANITYCHECK"
        
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                "learning_rate": args.lr,
                "architecture": "Can3Tok-Perceiver-VAE",
                "dataset": "SceneSplat-7K",
                "batch_size": args.batch_size,
                "epochs": args.num_epochs,
                "kl_weight": args.kl_weight,
                "model_config": "shapevae-256",
                "num_gaussians": 40000,
                "sampling_method": args.sampling_method,
                "eval_every": args.eval_every,
                "failure_threshold": args.failure_threshold,
                "train_scenes": args.train_scenes,
                "val_scenes": args.val_scenes,
                "use_train_as_val": args.use_train_as_val,
                "validation_bug_fixed": True,  # NEW FLAG
            },
            tags=["scenesplat", "vae", "gaussian-splatting", "validation", "FIXED"] + 
                 (["sanity-check"] if args.use_train_as_val else []),
        )
        print("✓ Weights & Biases enabled")
        wandb_enabled = True
    except Exception as e:
        print(f"✗ Weights & Biases failed to initialize: {e}")
        print("  Continuing training without W&B...")
        wandb_enabled = False
else:
    print("✗ Weights & Biases disabled")

# ============================================================================
# GPU SETUP
# ============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================================
# CONFIGURATION
# ============================================================================

loss_usage = "L1"
random_permute = 0
random_rotation = 1

resol = 200
data_path = "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs"

num_epochs = args.num_epochs
bch_size = args.batch_size
kl_weight = args.kl_weight
eval_every = args.eval_every
failure_threshold = args.failure_threshold
train_scenes = args.train_scenes
val_scenes = args.val_scenes
use_train_as_val = args.use_train_as_val
sampling_method = args.sampling_method

# ============================================================================
# JOB-SPECIFIC CHECKPOINT FOLDER
# ============================================================================

job_id = os.environ.get('SLURM_JOB_ID', None)
if job_id:
    checkpoint_folder = f"job_{job_id}_FIXED"
    if use_train_as_val:
        checkpoint_folder += "_sanitycheck"
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_folder = f"local_{timestamp}_FIXED"
    if use_train_as_val:
        checkpoint_folder += "_sanitycheck"

save_path = f"/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/{checkpoint_folder}/"
os.makedirs(save_path, exist_ok=True)

# Save run configuration
config_file = os.path.join(save_path, "config.txt")
with open(config_file, 'w') as f:
    f.write(f"Can3Tok Training Configuration (VALIDATION BUG FIXED)\n")
    f.write(f"=" * 70 + "\n")
    f.write(f"Job ID: {job_id or 'local'}\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"MODE: {'SANITY CHECK (train=val)' if use_train_as_val else 'NORMAL'}\n")
    f.write(f"VALIDATION BUG: FIXED\n")
    f.write(f"Device: {device}\n")
    f.write(f"Batch size: {bch_size}\n")
    f.write(f"Epochs: {num_epochs}\n")
    f.write(f"Learning rate: {args.lr}\n")
    f.write(f"KL weight: {kl_weight}\n")
    f.write(f"Eval every: {eval_every} epochs\n")
    f.write(f"Failure threshold: {failure_threshold}\n")
    f.write(f"Train scenes limit: {train_scenes or 'All'}\n")
    f.write(f"Val scenes limit: {val_scenes or 'All'}\n")
    f.write(f"Sampling method: {sampling_method}\n")
    f.write(f"Use train as val: {use_train_as_val}\n")
    f.write(f"Dataset: {data_path}\n")
    f.write(f"Checkpoint folder: {save_path}\n")
    f.write(f"Weights & Biases: {'Enabled' if wandb_enabled else 'Disabled'}\n")
    f.write(f"=" * 70 + "\n")

print(f"Configuration:")
print(f"  Job ID: {job_id or 'local'}")
print(f"  VERSION: FIXED (validation bug corrected)")
if use_train_as_val:
    print(f"  MODE: 🔍 SANITY CHECK (train=val)")
    print(f"  ⚠️  Using TRAINING data for validation")
    print(f"  ⚠️  This tests if model can overfit")
else:
    print(f"  MODE: NORMAL TRAINING")
print(f"  Device: {device}")
print(f"  Batch size: {bch_size}")
print(f"  Epochs: {num_epochs}")
print(f"  Learning rate: {args.lr}")
print(f"  KL weight: {kl_weight}")
print(f"  Sampling method: {sampling_method}")
print(f"  Eval every: {eval_every} epochs")
print(f"  Failure threshold: {failure_threshold}")
print(f"  Train scenes: {train_scenes or 'All'}")
print(f"  Val scenes: {val_scenes or 'All'}")
print(f"  Save path: {save_path}")
print(f"  W&B enabled: {wandb_enabled}")
print()

# ============================================================================
# MODEL SETUP
# ============================================================================

print("Loading model...")
config_path_perceiver = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
model_config_perceiver = get_config_from_file(config_path_perceiver)

if hasattr(model_config_perceiver, "model"):
    model_config_perceiver = model_config_perceiver.model

perceiver_encoder_decoder = instantiate_from_config(model_config_perceiver)

# Multi-GPU setup
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    gs_autoencoder = nn.DataParallel(perceiver_encoder_decoder)
else:
    print("Using single GPU")
    gs_autoencoder = perceiver_encoder_decoder

gs_autoencoder.to(device)

# Optimizer
optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=args.lr, betas=[0.9, 0.999])

print("✓ Model loaded successfully")
print()

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
    root=os.path.join(data_path, "train"),
    resol=resol,
    random_permute=True,
    train=True,
    sampling_method=sampling_method,
    max_scenes=train_scenes
)

trainDataLoader = Data.DataLoader(
    dataset=gs_dataset_train, 
    batch_size=bch_size,
    shuffle=True, 
    num_workers=12,
    pin_memory=True
)

# Validation dataset - CONDITIONALLY USE TRAIN DATA
print("="*70)
if use_train_as_val:
    print("VALIDATION DATASET (USING TRAIN DATA - SANITY CHECK)")
    print("="*70)
    print("⚠️  WARNING: Using TRAINING data as validation!")
    print("⚠️  This is a SANITY CHECK to test overfitting")
    print("="*70)
    
    # Use same dataset as training
    gs_dataset_val = gs_dataset(
        root=os.path.join(data_path, "train"),  # ← SAME as train!
        resol=resol,
        random_permute=False,  # No augmentation for validation
        train=False,
        sampling_method=sampling_method,
        max_scenes=val_scenes if val_scenes else train_scenes  # Match train size
    )
else:
    print("VALIDATION DATASET (SEPARATE DATA)")
    print("="*70)
    
    # Use separate val dataset (normal mode)
    gs_dataset_val = gs_dataset(
        root=os.path.join(data_path, "val"),  # ← DIFFERENT from train
        resol=resol,
        random_permute=False,
        train=False,
        sampling_method=sampling_method,
        max_scenes=val_scenes
    )

valDataLoader = Data.DataLoader(
    dataset=gs_dataset_val,
    batch_size=bch_size,
    shuffle=False,
    num_workers=12,
    pin_memory=True
)

print("="*70)
print("DATASET SUMMARY")
print("="*70)
print(f"✓ Training: {len(gs_dataset_train)} scenes, {len(trainDataLoader)} batches")
print(f"✓ Validation: {len(gs_dataset_val)} scenes, {len(valDataLoader)} batches")
print(f"✓ Sampling method: {sampling_method}")
if use_train_as_val:
    print(f"⚠️  MODE: SANITY CHECK (train=val)")
    print(f"⚠️  Expected: Val loss should match train loss (overfitting)")
    print(f"⚠️  If val loss stays high → BUG in model or data loading!")
print("="*70)
print()

# Log dataset info to wandb
if wandb_enabled:
    wandb_run.config.update({
        "num_train_scenes": len(gs_dataset_train),
        "num_val_scenes": len(gs_dataset_val),
        "batches_per_epoch": len(trainDataLoader),
        "sanity_check_mode": use_train_as_val,
        "sampling_method": sampling_method,
    })

# ============================================================================
# EVALUATION FUNCTION (FIXED!)
# ============================================================================

def evaluate_model(model, dataloader, device, failure_threshold):
    """
    Evaluate model on validation set.
    
    FIXED: Now uses the SAME L2 calculation as training!
    
    Returns:
        dict with metrics: avg_l2_error, failure_rate, avg_kl, per_scene_errors
    """
    model.eval()
    
    total_l2_error = 0.0
    total_kl = 0.0
    per_scene_l2_errors = []
    num_failures = 0
    num_scenes = 0
    
    with torch.no_grad():
        for i_batch, UV_gs_batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            UV_gs_batch = UV_gs_batch[0].type(torch.float32).to(device)
            batch_size = UV_gs_batch.shape[0]
            
            # Forward pass
            shape_embed, mu, log_var, z, UV_gs_recover = model(
                UV_gs_batch,
                UV_gs_batch,
                UV_gs_batch,
                UV_gs_batch[:, :, :3]
            )
            
            # Prepare target
            target = UV_gs_batch[:, :, 4:18]  # [batch, 40000, 14]
            UV_gs_recover_reshaped = UV_gs_recover.reshape(batch_size, -1, 14)
            
            # ============================================================
            # FIX: Use the SAME calculation as training loss!
            # ============================================================
            # Training uses: torch.norm(..., p=2) / batch_size
            # We do the same here to get comparable metrics
            
            batch_l2_error = torch.norm(
                UV_gs_recover_reshaped - target,
                p=2  # Euclidean norm over ALL elements
            ) / batch_size
            
            # For per-scene analysis, compute individual scene errors
            # (This is just for statistics, not for the main metric)
            per_scene_norms = torch.norm(
                UV_gs_recover_reshaped - target,
                p=2,
                dim=(1, 2)  # Per-scene norm
            )
            
            # Scale per-scene norms to match the batch-averaged metric
            # per_scene_norms are sqrt(batch_size) times larger than batch_l2_error
            # So we divide by sqrt(batch_size) to make them comparable
            per_scene_errors_scaled = per_scene_norms / np.sqrt(batch_size)
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(
                1.0 + log_var - mu.pow(2) - log_var.exp(),
                dim=1
            )
            
            # Accumulate metrics
            total_l2_error += batch_l2_error.item() * batch_size
            total_kl += kl_loss.sum().item()
            
            # Track per-scene errors (scaled to match training metric)
            per_scene_l2_errors.extend(per_scene_errors_scaled.cpu().numpy().tolist())
            
            # Count failures using the scaled per-scene errors
            num_failures += (per_scene_errors_scaled.cpu().numpy() > failure_threshold).sum()
            num_scenes += batch_size
    
    # Compute average metrics
    avg_l2_error = total_l2_error / num_scenes
    avg_kl = total_kl / num_scenes
    failure_rate = (num_failures / num_scenes) * 100.0  # Percentage
    
    # Statistics on per-scene errors
    per_scene_l2_errors = np.array(per_scene_l2_errors)
    l2_std = per_scene_l2_errors.std()
    l2_min = per_scene_l2_errors.min()
    l2_max = per_scene_l2_errors.max()
    l2_median = np.median(per_scene_l2_errors)
    
    model.train()
    
    return {
        'avg_l2_error': avg_l2_error,
        'l2_std': l2_std,
        'l2_min': l2_min,
        'l2_max': l2_max,
        'l2_median': l2_median,
        'failure_rate': failure_rate,
        'num_failures': num_failures,
        'num_scenes': num_scenes,
        'avg_kl': avg_kl,
        'per_scene_errors': per_scene_l2_errors,
    }

# ============================================================================
# TRAINING SETUP
# ============================================================================

# Voxelization parameters for rotation augmentation
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
if use_train_as_val:
    print("STARTING TRAINING (SANITY CHECK MODE - VALIDATION BUG FIXED)")
    print("⚠️  Validation uses TRAIN data - testing overfitting")
else:
    print("STARTING TRAINING (NORMAL MODE - VALIDATION BUG FIXED)")
print("="*70)
print()

global_step = 0
best_val_loss = float('inf')
best_epoch = 0

for epoch in tqdm(range(num_epochs), desc="Training"):
    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0
    
    # ====================
    # TRAINING
    # ====================
    gs_autoencoder.train()
    
    for i_batch, UV_gs_batch in enumerate(trainDataLoader):
        UV_gs_batch = UV_gs_batch[0].type(torch.float32).to(device)
        
        # Random permutation augmentation
        if epoch % 1 == 0 and random_permute == 1:
            UV_gs_batch = UV_gs_batch[:, torch.randperm(UV_gs_batch.size()[1])]
        
        # Random rotation augmentation (every 5 epochs)
        if epoch % 5 == 0 and epoch > 1 and random_rotation == 1:
            rand_rot_comp = special_ortho_group.rvs(3)
            rand_rot = torch.tensor(
                np.dot(rand_rot_comp, rand_rot_comp.T), 
                dtype=torch.float32
            ).to(UV_gs_batch.device)
            
            UV_gs_batch[:, :, 4:7] = UV_gs_batch[:, :, 4:7] @ rand_rot
            
            for bcbc in range(UV_gs_batch.shape[0]):
                shifted_points = UV_gs_batch[bcbc, :, 4:7] + origin_offset
                voxel_indices = torch.floor(shifted_points / resolution)
                voxel_indices = torch.clip(voxel_indices, 0, volume_dims - 1)
                voxel_centers_batch = (voxel_indices - (volume_dims - 1) / 2) * resolution
                UV_gs_batch[bcbc, :, :3] = voxel_centers_batch
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        shape_embed, mu, log_var, z, UV_gs_recover = gs_autoencoder(
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
        
        recon_loss = torch.norm(
            UV_gs_recover.reshape(UV_gs_batch.shape[0], -1, 14) - UV_gs_batch[:, :, 4:], 
            p=2
        ) / UV_gs_batch.shape[0]
        
        loss = recon_loss + kl_weight * KL_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        epoch_loss += loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += KL_loss.item()
        
        # Log to wandb (per step)
        if wandb_enabled:
            wandb_run.log({
                "train/step_loss": loss.item(),
                "train/step_recon_loss": recon_loss.item(),
                "train/step_kl_loss": KL_loss.item(),
            }, step=global_step)
        
        global_step += 1
        
        # Print first batch of each epoch
        if i_batch == 0:
            print(f"Epoch {epoch}/{num_epochs} | "
                  f"Batch {i_batch}/{len(trainDataLoader)} | "
                  f"Loss: {loss.item():.2f} | "
                  f"Recon: {recon_loss.item():.2f} | "
                  f"KL: {KL_loss.item():.2f}")
    
    # ====================
    # EPOCH SUMMARY
    # ====================
    avg_train_loss = epoch_loss / len(trainDataLoader)
    avg_train_recon = epoch_recon_loss / len(trainDataLoader)
    avg_train_kl = epoch_kl_loss / len(trainDataLoader)
    
    # ====================
    # VALIDATION
    # ====================
    val_metrics = None
    if epoch % eval_every == 0 or epoch == num_epochs - 1:
        print(f"\n{'='*70}")
        if use_train_as_val:
            print(f"RUNNING VALIDATION (Epoch {epoch}) - SANITY CHECK (FIXED METRIC)")
            print(f"⚠️  Using TRAIN data - should match train loss!")
        else:
            print(f"RUNNING VALIDATION (Epoch {epoch}) - FIXED METRIC")
        print(f"{'='*70}")
        
        val_metrics = evaluate_model(gs_autoencoder, valDataLoader, device, failure_threshold)
        
        print(f"Validation Results:")
        print(f"  L2 Error: {val_metrics['avg_l2_error']:.2f} ± {val_metrics['l2_std']:.2f}")
        print(f"  L2 Range: [{val_metrics['l2_min']:.2f}, {val_metrics['l2_max']:.2f}]")
        print(f"  L2 Median: {val_metrics['l2_median']:.2f}")
        print(f"  Failure Rate: {val_metrics['failure_rate']:.2f}% ({val_metrics['num_failures']}/{val_metrics['num_scenes']} scenes)")
        print(f"  Avg KL: {val_metrics['avg_kl']:.2f}")
        
        # Compare train vs val
        train_val_gap = abs(val_metrics['avg_l2_error'] - avg_train_recon)
        gap_percent = (train_val_gap / avg_train_recon) * 100 if avg_train_recon > 0 else 0
        
        print(f"\n  📊 TRAIN vs VAL COMPARISON:")
        print(f"  Train Recon: {avg_train_recon:.2f}")
        print(f"  Val L2:      {val_metrics['avg_l2_error']:.2f}")
        print(f"  Gap:         {train_val_gap:.2f} ({gap_percent:.1f}%)")
        
        if use_train_as_val:
            print(f"\n  🔍 SANITY CHECK ANALYSIS:")
            if gap_percent < 10:
                print(f"  ✅ PASS: Gap <10% - Model CAN overfit! Code is working!")
            elif gap_percent < 50:
                print(f"  ⚠️  WARNING: Gap {gap_percent:.1f}% - Expected <10% for sanity check")
            else:
                print(f"  ❌ FAIL: Gap {gap_percent:.1f}% - Possible BUG in model or data!")
        else:
            if gap_percent < 50:
                print(f"  ✅ GOOD: Val is similar to train ({gap_percent:.1f}%)")
            elif gap_percent < 100:
                print(f"  ⚠️  OK: Val is somewhat higher ({gap_percent:.1f}%)")
            else:
                print(f"  ⚠️  WARNING: Large gap ({gap_percent:.1f}%) - val set may be much harder")
        
        # Track best model
        if val_metrics['avg_l2_error'] < best_val_loss:
            best_val_loss = val_metrics['avg_l2_error']
            best_epoch = epoch
            
            # Save best model
            if torch.cuda.device_count() > 1:
                model_state = gs_autoencoder.module.state_dict()
            else:
                model_state = gs_autoencoder.state_dict()
            
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_l2_error': val_metrics['avg_l2_error'],
                'val_failure_rate': val_metrics['failure_rate'],
                'train_loss': avg_train_loss,
                'train_recon': avg_train_recon,
                'validation_bug_fixed': True,
            }, best_model_path)
            print(f"  ✓ New best model saved! (L2: {best_val_loss:.2f})")
        
        print(f"{'='*70}\n")
    
    # Log epoch metrics to wandb
    if wandb_enabled:
        log_dict = {
            "train/epoch_loss": avg_train_loss,
            "train/epoch_recon": avg_train_recon,
            "train/epoch_kl": avg_train_kl,
            "train/epoch": epoch,
            "best/val_l2_error": best_val_loss,
            "best/epoch": best_epoch,
        }
        
        if val_metrics is not None:
            log_dict.update({
                "val/l2_error": val_metrics['avg_l2_error'],
                "val/l2_std": val_metrics['l2_std'],
                "val/l2_min": val_metrics['l2_min'],
                "val/l2_max": val_metrics['l2_max'],
                "val/l2_median": val_metrics['l2_median'],
                "val/failure_rate": val_metrics['failure_rate'],
                "val/num_failures": val_metrics['num_failures'],
                "val/kl": val_metrics['avg_kl'],
            })
            
            # Add train-val gap metrics
            train_val_gap = abs(val_metrics['avg_l2_error'] - avg_train_recon)
            log_dict["metrics/train_val_gap"] = train_val_gap
            log_dict["metrics/gap_percent"] = (train_val_gap / avg_train_recon) * 100 if avg_train_recon > 0 else 0
        
        wandb_run.log(log_dict, step=global_step)
    
    # Print training summary every 20 epochs
    if epoch % 20 == 0:
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*70}")
        print(f"Avg Train Loss: {avg_train_loss:.2f}")
        print(f"Avg Train Recon: {avg_train_recon:.2f}")
        print(f"Avg Train KL: {avg_train_kl:.2f}")
        print(f"Best Val L2 Error: {best_val_loss:.2f} (epoch {best_epoch})")
        print(f"{'='*70}\n")
    
    # Save checkpoints every 10 epochs
    if epoch >= 10 and epoch % 10 == 0:
        gs_autoencoder.eval()
        
        # Save embeddings
        torch.save(mu, f"{save_path}gs_mu_{epoch}.pt")
        torch.save(log_var, f"{save_path}gs_var_{epoch}.pt")
        torch.save(z, f"{save_path}gs_emb_{epoch}.pt")
        
        # Save model checkpoint
        if torch.cuda.device_count() > 1:
            model_state = gs_autoencoder.module.state_dict()
        else:
            model_state = gs_autoencoder.state_dict()
        
        checkpoint_path = os.path.join(save_path, f"{int(epoch)}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'train_recon': avg_train_recon,
            'val_l2_error': val_metrics['avg_l2_error'] if val_metrics else None,
            'best_val_l2': best_val_loss,
            'validation_bug_fixed': True,
        }, checkpoint_path)
        
        print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        gs_autoencoder.train()

# ============================================================================
# FINAL EVALUATION & SAVE
# ============================================================================

print("\n" + "="*70)
if use_train_as_val:
    print("TRAINING COMPLETE - FINAL SANITY CHECK")
else:
    print("TRAINING COMPLETE - FINAL EVALUATION")
print("="*70)

# Final validation
final_val_metrics = evaluate_model(gs_autoencoder, valDataLoader, device, failure_threshold)

print(f"\nFinal Validation Results:")
print(f"  L2 Error: {final_val_metrics['avg_l2_error']:.2f} ± {final_val_metrics['l2_std']:.2f}")
print(f"  Failure Rate: {final_val_metrics['failure_rate']:.2f}%")
print(f"  Best Val L2: {best_val_loss:.2f} (epoch {best_epoch})")

# Final comparison
final_train_recon = epoch_recon_loss / len(trainDataLoader)
final_gap = abs(final_val_metrics['avg_l2_error'] - final_train_recon)
final_gap_percent = (final_gap / final_train_recon) * 100 if final_train_recon > 0 else 0

print(f"\n📊 FINAL TRAIN vs VAL:")
print(f"  Train Recon: {final_train_recon:.2f}")
print(f"  Val L2:      {final_val_metrics['avg_l2_error']:.2f}")
print(f"  Gap:         {final_gap:.2f} ({final_gap_percent:.1f}%)")

if use_train_as_val:
    print(f"\n🔍 FINAL SANITY CHECK:")
    if final_gap_percent < 10:
        print(f"  ✅ PASS: Model CAN overfit - training pipeline works!")
    else:
        print(f"  ❌ FAIL: Model can't overfit properly - check for bugs!")
print()

# Save final model
torch.save(mu, f"{save_path}gs_mu_final.pt")
torch.save(log_var, f"{save_path}gs_var_final.pt")
torch.save(z, f"{save_path}gs_emb_final.pt")

if torch.cuda.device_count() > 1:
    model_state = gs_autoencoder.module.state_dict()
else:
    model_state = gs_autoencoder.state_dict()

final_path = os.path.join(save_path, "final.pth")
torch.save({
    'epoch': num_epochs - 1,
    'model_state_dict': model_state,
    'optimizer_state_dict': optimizer.state_dict(),
    'final_val_l2': final_val_metrics['avg_l2_error'],
    'final_train_recon': final_train_recon,
    'final_failure_rate': final_val_metrics['failure_rate'],
    'final_gap_percent': final_gap_percent,
    'best_val_l2': best_val_loss,
    'best_epoch': best_epoch,
    'sanity_check_mode': use_train_as_val,
    'validation_bug_fixed': True,
    'sampling_method': sampling_method,
}, final_path)

print(f"✓ Final checkpoint saved: {final_path}")
print(f"✓ Best model saved: {os.path.join(save_path, 'best_model.pth')}")
print(f"✓ All training artifacts saved to: {save_path}")
print("="*70)

# Finish wandb run
if wandb_enabled:
    # Log final summary
    wandb_run.summary.update({
        "final_val_l2_error": final_val_metrics['avg_l2_error'],
        "final_train_recon": final_train_recon,
        "final_gap_percent": final_gap_percent,
        "final_failure_rate": final_val_metrics['failure_rate'],
        "best_val_l2_error": best_val_loss,
        "best_epoch": best_epoch,
        "sanity_check_mode": use_train_as_val,
        "validation_bug_fixed": True,
        "sampling_method": sampling_method,
    })
    
    wandb_run.finish()
    print("✓ Weights & Biases run finished")