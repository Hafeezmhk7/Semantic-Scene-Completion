"""
can3tok_test_eval.py
====================
Evaluate one or more trained Can3Tok checkpoints on UNSEEN test scenes.

Metrics reported are IDENTICAL to evaluate_model() in gs_can3tok_2.py:
  L2          — total reconstruction L2 norm / n_scenes
  Position    — L2 on xyz [0:3]
  Color       — L2 on rgb / residuals [3:6]
  Opacity     — L2 on opacity [6:7]
  Scale       — L2 on scale [7:10]
  Rotation    — L2 on quaternion [10:14]
  ColorPredMSE — MSE(MeanColorHead output, gt mean_color)   [only if color_residual=True]
  SceneSemanticKL — KL(p_s || p_hat) from SceneSemanticHead [only if scene_semantic_head=True]

All flags (color_residual, scene_semantic_head, semantic_mode, scale_norm_mode,
use_canonical_norm) are read directly from the checkpoint — no need to specify them.

USAGE
-----
  # Evaluate one model on 20 test scenes:
  python can3tok_test_eval.py \\
    --checkpoint checkpoints/RGB_job_20207686_none_colorresidual/best_model.pth \\
    --n_scenes 20

  # Evaluate multiple models side-by-side (produces comparison table):
  python can3tok_test_eval.py \\
    --checkpoint checkpoints/run_a/best_model.pth \\
                 checkpoints/run_b/best_model.pth \\
                 checkpoints/run_c/best_model.pth \\
    --checkpoint_names "Run A" "Run B" "Run C" \\
    --n_scenes 30

  # Save PLY reconstructions for visual inspection:
  python can3tok_test_eval.py \\
    --checkpoint checkpoints/run_c/best_model.pth \\
    --n_scenes 20 --save_ply --ply_n_scenes 5

  # Use a different dataset root or split:
  python can3tok_test_eval.py \\
    --checkpoint checkpoints/run_a/best_model.pth \\
    --dataset_root /path/to/interior_gs \\
    --split test \\
    --n_scenes 20
"""

import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
from gs_dataset_scenesplat import gs_dataset
import torch.utils.data as Data

# ============================================================================
# CONSTANTS  — must match gs_can3tok_2.py exactly
# ============================================================================

PARAM_SLICES = {
    'position': slice(0, 3),
    'color':    slice(3, 6),
    'opacity':  slice(6, 7),
    'scale':    slice(7, 10),
    'rotation': slice(10, 14),
}

# Input feature columns used as reconstruction target
# Columns 4-17 of the [40000, 18] feature tensor:
#   4:7  → xyz (actual position)
#   7:10 → rgb or color residuals
#   10   → opacity
#   11:14→ scale
#   14:18→ quaternion
GEOMETRIC_INDICES = (
    list(range(4, 7))
  + list(range(7, 10))
  + [10]
  + list(range(11, 14))
  + list(range(14, 18))
)  # 14 values total

# ============================================================================
# LOSS HELPERS  — copied verbatim from gs_can3tok_2.py
# ============================================================================

def compute_reconstruction_loss(prediction, target, batch_size, color_weight=1.0):
    """L2 norm / batch_size — identical to training loop."""
    if color_weight == 1.0:
        return torch.norm(prediction - target, p=2) / batch_size
    loss_pos   = torch.norm(prediction[:, :, 0:3] - target[:, :, 0:3], p=2)
    loss_color = torch.norm(prediction[:, :, 3:6] - target[:, :, 3:6], p=2) * color_weight
    loss_other = torch.norm(prediction[:, :, 6:]  - target[:, :, 6:],  p=2)
    return (loss_pos + loss_color + loss_other) / batch_size


def compute_individual_losses(prediction, target):
    """Per-parameter L2 norms — identical to training loop."""
    return {
        name: torch.norm(prediction[:, :, sl] - target[:, :, sl], p=2).item()
        for name, sl in PARAM_SLICES.items()
    }


def scene_semantic_kl_loss(p_hat, p_s, eps=1e-8):
    """KL divergence — identical to training loop."""
    p_hat_clamped = torch.clamp(p_hat, min=eps)
    kl_per_scene  = (p_s * (torch.log(p_s + eps) - torch.log(p_hat_clamped))).sum(dim=-1)
    return kl_per_scene.mean()


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(ckpt_path: str, device: torch.device):
    """
    Load checkpoint and reconstruct model with EXACTLY the architecture
    that was saved — reads all flags from the checkpoint dict.
    """
    print(f"  Loading: {Path(ckpt_path).parent.name} / {Path(ckpt_path).name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    flags = {
        'semantic_mode':       ckpt.get('semantic_mode',       'none'),
        'color_residual':      ckpt.get('color_residual',      False),
        'scene_semantic_head': ckpt.get('scene_semantic_head', False),
        'label_input':         ckpt.get('label_input',         False),
        'scale_norm_mode':     ckpt.get('scale_norm_mode',     'linear'),
        'use_canonical_norm':  ckpt.get('use_canonical_norm',  True),
        'color_loss_weight':   ckpt.get('color_loss_weight',   1.0),
    }

    config_path  = './model/configs/aligned_shape_latents/shapevae-256.yaml'
    model_config = get_config_from_file(config_path).model
    p = model_config.params.shape_module_cfg.params
    p.semantic_mode       = flags['semantic_mode']
    p.color_residual      = flags['color_residual']
    p.scene_semantic_head = flags['scene_semantic_head']

    model = instantiate_from_config(model_config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    epoch  = ckpt.get('epoch', '?')
    val_l2 = ckpt.get('val_l2_error', ckpt.get('best_val_l2', ckpt.get('final_val_l2', '?')))
    print(f"    epoch={epoch}  train_val_L2={val_l2}")
    print(f"    semantic_mode={flags['semantic_mode']}  "
          f"color_residual={flags['color_residual']}  "
          f"scene_semantic_head={flags['scene_semantic_head']}")
    return model, flags, ckpt


# ============================================================================
# DATASET LOADING
# ============================================================================

def build_test_dataloader(dataset_root: str, split: str, flags: dict,
                          n_scenes: int, batch_size: int):
    """
    Build a DataLoader for the test split using exactly the same
    normalization / color_residual / label_input settings as training.
    """
    split_dir = os.path.join(dataset_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Test split directory not found: {split_dir}\n"
            f"Available splits: {os.listdir(dataset_root)}"
        )

    ds = gs_dataset(
        root             = split_dir,
        resol            = 200,
        random_permute   = False,      # deterministic order
        train            = False,
        sampling_method  = 'opacity',  # deterministic top-40k
        max_scenes       = n_scenes,
        normalize        = flags['use_canonical_norm'],
        normalize_colors = True,
        target_radius    = 10.0,
        scale_norm_mode  = flags['scale_norm_mode'],
        label_input      = flags['label_input'],
        color_residual   = flags['color_residual'],
    )

    print(f"  Test dataset: {len(ds)} scenes loaded from {split_dir}")

    loader = Data.DataLoader(
        dataset    = ds,
        batch_size = batch_size,
        shuffle    = False,
        num_workers= 4,
        pin_memory = True,
    )
    return loader, len(ds)


# ============================================================================
# EVALUATION  — mirrors evaluate_model() from gs_can3tok_2.py exactly
# ============================================================================

@torch.no_grad()
def evaluate(model, dataloader, flags, device,
             save_ply=False, ply_dir=None, ply_n_scenes=5):
    """
    Evaluate model on all batches in dataloader.
    Returns dict of metrics matching evaluate_model() output.
    """
    model.eval()

    total_l2              = 0.0
    total_kl              = 0.0
    total_color_pred_mse  = 0.0
    total_scene_sem_kl    = 0.0
    per_param             = {k: 0.0 for k in PARAM_SLICES}
    n_scenes              = 0

    # For PLY saving
    recon_preds_list = []
    recon_means_list = []
    scene_ids_list   = []

    color_weight = flags.get('color_loss_weight', 1.0)

    for batch_data in tqdm(dataloader, desc="  Evaluating", leave=False):

        UV_gs_batch   = batch_data['features'].float().to(device)
        mean_color_gt = batch_data['mean_color'].float().to(device)
        B = UV_gs_batch.shape[0]

        # Forward — same call signature as training loop
        (shape_embed, mu, log_var, z,
         UV_gs_recover, per_gaussian_features) = model(
            UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3]
        )

        mean_color_pred     = model.shape_model.last_mean_color_pred
        scene_semantic_pred = model.shape_model.last_scene_semantic_pred

        # Reconstruction target — GEOMETRIC_INDICES (14 values, cols 4-17)
        target  = UV_gs_batch[:, :, GEOMETRIC_INDICES]
        pred_3d = UV_gs_recover.reshape(B, -1, 14)

        # L2 reconstruction loss
        recon_loss = compute_reconstruction_loss(pred_3d, target, B, color_weight)
        total_l2  += recon_loss.item()

        # KL loss (per scene, sum over latent dims)
        kl_per_scene = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp(), dim=1
        )
        total_kl += kl_per_scene.sum().item()

        # Per-parameter losses
        ind = compute_individual_losses(pred_3d, target)
        for k in per_param:
            per_param[k] += ind[k]

        # ColorPredMSE — Step 1 metric
        if mean_color_pred is not None and flags['color_residual']:
            cp_mse = F.mse_loss(mean_color_pred, mean_color_gt).item()
            total_color_pred_mse += cp_mse * B

        # SceneSemanticKL — Move 1 metric
        if scene_semantic_pred is not None and flags['scene_semantic_head']:
            p_s   = batch_data['label_dist'].float().to(device)
            ss_kl = scene_semantic_kl_loss(scene_semantic_pred, p_s).item()
            total_scene_sem_kl += ss_kl * B

        n_scenes += B

        # Collect for PLY output
        if save_ply and len(recon_preds_list) < ply_n_scenes:
            preds_np = pred_3d.cpu().numpy()
            means_np = mean_color_gt.cpu().numpy()
            sids     = batch_data.get('scene_id', [f"scene_{n_scenes-B+i}" for i in range(B)])
            for si in range(B):
                if len(recon_preds_list) >= ply_n_scenes:
                    break
                recon_preds_list.append(preds_np[si])
                recon_means_list.append(means_np[si])
                scene_ids_list.append(sids[si] if isinstance(sids[si], str)
                                      else f"scene_{len(recon_preds_list):03d}")

    # Save PLY reconstructions
    if save_ply and recon_preds_list and ply_dir is not None:
        try:
            from gs_ply_reconstructor import save_reconstructed_gaussians
            all_preds = np.stack(recon_preds_list, axis=0)
            if flags['color_residual']:
                # Add mean color back before saving
                for si in range(len(all_preds)):
                    all_preds[si, :, 3:6] += recon_means_list[si]
                    all_preds[si, :, 3:6]  = np.clip(all_preds[si, :, 3:6], 0, 1)
            Path(ply_dir).mkdir(parents=True, exist_ok=True)
            save_reconstructed_gaussians(
                predictions   = all_preds,
                output_dir    = ply_dir,
                epoch         = 0,
                num_scenes    = len(all_preds),
                max_sh_degree = 3,
                color_mode    = "1",
                prefix        = "test_scene",
            )
            print(f"  PLY saved to: {ply_dir}")
        except Exception as e:
            print(f"  [WARN] PLY save failed: {e}")

    ns = max(n_scenes, 1)
    return {
        'n_scenes':           n_scenes,
        'avg_l2':             total_l2,
        'avg_kl':             total_kl / ns,
        'color_pred_mse':     total_color_pred_mse / ns,
        'scene_semantic_kl':  total_scene_sem_kl / ns,
        **{f'{k}_loss': v / ns for k, v in per_param.items()},
    }


# ============================================================================
# PRETTY PRINTING
# ============================================================================

def print_metrics(name: str, metrics: dict, flags: dict, train_val_l2: float = None):
    """Print one model's metrics in the same format as the training log."""
    print(f"\n  {'─'*60}")
    print(f"  {name}")
    print(f"  {'─'*60}")
    if train_val_l2 is not None:
        print(f"  Train val L2 (seen):       {train_val_l2:.4f}")
    print(f"  Test L2 (unseen, N={metrics['n_scenes']:3d}): {metrics['avg_l2']:.4f}"
          + (f"  [Δ = {metrics['avg_l2'] - train_val_l2:+.4f}]"
             if train_val_l2 is not None else ""))
    print(f"  Position:                  {metrics['position_loss']:.4f}")
    print(f"  Color:                     {metrics['color_loss']:.4f}"
          + ("  (residuals)" if flags['color_residual'] else "  (absolute)"))
    print(f"  Opacity:                   {metrics['opacity_loss']:.4f}")
    print(f"  Scale:                     {metrics['scale_loss']:.4f}")
    print(f"  Rotation:                  {metrics['rotation_loss']:.4f}")
    if flags['color_residual']:
        print(f"  ColorPredMSE:              {metrics['color_pred_mse']:.6f}"
              f"  (MeanColorHead → mean RGB)")
    if flags['scene_semantic_head']:
        print(f"  SceneSemanticKL:           {metrics['scene_semantic_kl']:.4f}"
              f"  (SceneSemanticHead → label dist)")


def print_comparison_table(results: list):
    """
    Print side-by-side comparison table for multiple models.
    results: list of (name, metrics, flags, train_val_l2)
    """
    metrics_keys = [
        ('avg_l2',            'L2 (test)'),
        ('position_loss',     'Position'),
        ('color_loss',        'Color'),
        ('opacity_loss',      'Opacity'),
        ('scale_loss',        'Scale'),
        ('rotation_loss',     'Rotation'),
        ('color_pred_mse',    'ColorPredMSE'),
        ('scene_semantic_kl', 'SceneSemanticKL'),
    ]

    # Column widths
    name_w = max(len(r[0]) for r in results) + 2
    val_w  = 12

    header = f"  {'Metric':<22}" + "".join(f"{r[0]:>{name_w}}" for r in results)
    print(f"\n  {'='*len(header)}")
    print(f"  COMPARISON TABLE — Test Set Generalization")
    print(f"  {'='*len(header)}")
    print(header)
    print(f"  {'─'*len(header)}")

    # Train val L2 row
    row = f"  {'Train val L2 (seen)':<22}"
    for _, _, _, tvl2 in results:
        s = f"{tvl2:.4f}" if tvl2 is not None else "N/A"
        row += f"{s:>{name_w}}"
    print(row)
    print(f"  {'─'*len(header)}")

    for key, label in metrics_keys:
        # Skip rows where no model has this metric
        has_values = any(r[1].get(key, 0) > 0 for r in results)
        if not has_values:
            continue
        row = f"  {label:<22}"
        for _, metrics, _, _ in results:
            val = metrics.get(key, 0.0)
            row += f"{val:>{name_w}.4f}"
        print(row)

    print(f"  {'─'*len(header)}")

    # Generalization gap row (test L2 - train val L2)
    row = f"  {'Gen. gap (Δ L2)':<22}"
    for _, metrics, _, tvl2 in results:
        if tvl2 is not None:
            gap = metrics['avg_l2'] - tvl2
            s   = f"{gap:+.4f}"
        else:
            s = "N/A"
        row += f"{s:>{name_w}}"
    print(row)
    print(f"  {'='*len(header)}\n")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description='Can3Tok — test set generalization evaluation'
    )
    ap.add_argument('--checkpoint', nargs='+', required=True,
                    help='Path(s) to checkpoint .pth file(s). '
                         'Multiple paths → comparison table.')
    ap.add_argument('--checkpoint_names', nargs='+', default=None,
                    help='Display names for each checkpoint (optional). '
                         'Defaults to parent directory name.')
    ap.add_argument('--dataset_root', type=str,
                    default='/home/yli11/scratch/datasets/gaussian_world/'
                            'preprocessed/interior_gs',
                    help='Root dataset directory containing split subdirectories.')
    ap.add_argument('--split', type=str, default='test',
                    help='Test split subdirectory name (default: test).')
    ap.add_argument('--n_scenes', type=int, default=20,
                    help='Number of test scenes to evaluate (default: 20).')
    ap.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for evaluation (default: 16).')
    ap.add_argument('--save_ply', action='store_true', default=True,
                    help='Save PLY reconstructions for visual inspection.')
    ap.add_argument('--ply_n_scenes', type=int, default=5,
                    help='Number of scenes to save as PLY (default: 5).')
    ap.add_argument('--output_dir', type=str, default='test_eval_results',
                    help='Directory for outputs (metrics JSON, PLYs).')
    return ap.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args   = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve display names
    names = args.checkpoint_names
    if names is None:
        names = [Path(c).parent.name for c in args.checkpoint]
    if len(names) < len(args.checkpoint):
        names += [Path(c).parent.name for c in args.checkpoint[len(names):]]

    print(f"\n{'='*65}")
    print(f"Can3Tok — Test Set Generalization Evaluation")
    print(f"{'='*65}")
    print(f"  Device:       {device}")
    print(f"  Dataset root: {args.dataset_root}")
    print(f"  Split:        {args.split}")
    print(f"  N scenes:     {args.n_scenes}")
    print(f"  Models:       {len(args.checkpoint)}")
    print(f"{'='*65}")

    all_results = []

    for ckpt_path, name in zip(args.checkpoint, names):

        print(f"\n{'─'*65}")
        print(f"  Model: {name}")
        print(f"{'─'*65}")

        # Load model
        model, flags, ckpt = load_model(ckpt_path, device)
        train_val_l2 = ckpt.get('val_l2_error',
                       ckpt.get('best_val_l2',
                       ckpt.get('final_val_l2', None)))

        # Build test dataloader with same flags as training
        try:
            loader, n_loaded = build_test_dataloader(
                args.dataset_root, args.split, flags,
                args.n_scenes, args.batch_size,
            )
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            continue

        # PLY output directory per model
        ply_dir = None
        if args.save_ply:
            safe_name = name.replace(' ', '_').replace('\n', '_').replace('/', '_')
            ply_dir = os.path.join(args.output_dir, 'ply', safe_name)

        # Evaluate
        metrics = evaluate(
            model, loader, flags, device,
            save_ply    = args.save_ply,
            ply_dir     = ply_dir,
            ply_n_scenes= args.ply_n_scenes,
        )

        print_metrics(name, metrics, flags, train_val_l2)
        all_results.append((name, metrics, flags, train_val_l2))

        # Clean up GPU memory between models
        del model
        torch.cuda.empty_cache()

    # Comparison table (only when 2+ models)
    if len(all_results) >= 2:
        print_comparison_table(all_results)

    # Save metrics as JSON
    json_path = os.path.join(args.output_dir, 'test_metrics.json')
    json_data = {}
    for name, metrics, flags, tvl2 in all_results:
        json_data[name] = {
            'train_val_l2':  tvl2,
            'generalization_gap': (metrics['avg_l2'] - tvl2
                                   if tvl2 is not None else None),
            **{k: float(v) for k, v in metrics.items()},
        }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Metrics saved: {json_path}")

    print(f"\nDone.")
    print(f"  scp user@snellius:$(pwd)/{args.output_dir}/test_metrics.json .")


if __name__ == '__main__':
    main()