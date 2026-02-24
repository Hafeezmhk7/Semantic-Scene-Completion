"""
Input Scene PCA Visualizer
==========================
Standalone script to generate PCA feature visualizations for the first N
scenes directly from the SceneSplat dataset — no model forward pass needed.

Creates 3 PLY files per scene:
  1. scene_XXX_position_pca.ply     — positions PCA-colored (spatial structure)
  2. scene_XXX_color_original.ply   — original RGB colors
  3. scene_XXX_opacity_pca.ply      — opacity values as grayscale

Usage:
    python visualize_input_pca.py --num_scenes 5 --output_dir ./input_pca_vis
    python visualize_input_pca.py --num_scenes 5 --output_dir ./input_pca_vis --scale_norm_mode log
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import sys
import os

# ── Add project root to path so we can import project modules ──────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pca_feature_visualization import (
    get_pca_color_torch,
    write_ply_with_colors,
    visualize_semantic_features
)

# ── Argument parsing ────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="PCA visualization of input scenes")
parser.add_argument("--num_scenes", type=int, default=5,
                    help="Number of scenes to visualize (default: 5)")
parser.add_argument("--output_dir", type=str, default="./input_pca_vis",
                    help="Output directory for PLY files")
parser.add_argument("--data_path", type=str,
                    default="/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs",
                    help="Root path to the dataset")
parser.add_argument("--sampling_method", type=str, default="opacity",
                    choices=["opacity", "random", "hybrid"],
                    help="Gaussian sampling method (default: opacity)")
parser.add_argument("--scale_norm_mode", type=str, default="linear",
                    choices=["log", "linear"],
                    help="Scale normalization mode (default: linear)")
parser.add_argument("--use_canonical_norm", action="store_true", default=True,
                    help="Use canonical sphere normalization (default: True)")
parser.add_argument("--brightness", type=float, default=1.25,
                    help="PCA color brightness multiplier (default: 1.25)")
parser.add_argument("--device", type=str, default="cpu",
                    help="Torch device for PCA computation (default: cpu)")
args = parser.parse_args()

# ── Parameter layout in the feature tensor ──────────────────────────────────
# Input format [B, N, features]:
#   0-3:   voxel_centers + uniq_idx
#   4-6:   xyz positions
#   7-9:   rgb colors
#   10:    opacity
#   11-13: scale (sx, sy, sz)
#   14-17: rotation quaternion

FEAT_POSITION = slice(4, 7)
FEAT_COLOR    = slice(7, 10)
FEAT_OPACITY  = 10
FEAT_SCALE    = slice(11, 14)
FEAT_ROTATION = slice(14, 18)

# ── Load dataset ─────────────────────────────────────────────────────────────

print("=" * 70)
print("INPUT SCENE PCA VISUALIZER")
print("=" * 70)
print(f"  Scenes:        {args.num_scenes}")
print(f"  Output dir:    {args.output_dir}")
print(f"  Scale mode:    {args.scale_norm_mode}")
print(f"  Canonical norm:{args.use_canonical_norm}")
print(f"  Brightness:    {args.brightness}")
print("=" * 70)
print()

print("Loading dataset...")
from gs_dataset_scenesplat import gs_dataset
import torch.utils.data as Data

dataset = gs_dataset(
    root=os.path.join(args.data_path, "train_grid1.0cm_chunk8x8_stride6x6"),
    resol=200,
    random_permute=False,
    train=True,
    sampling_method=args.sampling_method,
    max_scenes=args.num_scenes,
    normalize=args.use_canonical_norm,
    normalize_colors=True,
    target_radius=10.0,
    scale_norm_mode=args.scale_norm_mode,
)

loader = Data.DataLoader(
    dataset=dataset,
    batch_size=1,        # one scene at a time for clean indexing
    shuffle=False,
    num_workers=0,
)

print(f"✓ Dataset loaded: {len(dataset)} scenes")
print()

# ── Output directory ─────────────────────────────────────────────────────────

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Output directory: {output_dir.resolve()}")
print()

# ── Main loop ────────────────────────────────────────────────────────────────

for scene_idx, batch_data in enumerate(loader):
    if scene_idx >= args.num_scenes:
        break

    print(f"{'='*70}")
    print(f"SCENE {scene_idx:03d}")
    print(f"{'='*70}")

    # Extract the raw feature tensor [1, N, F]
    if isinstance(batch_data, dict):
        feat = batch_data['features'][0]   # [N, F]
    else:
        feat = batch_data[0][0]            # [N, F]

    feat_np = feat.numpy()                 # [N, F]
    N = feat_np.shape[0]
    print(f"  Gaussians: {N}")

    # ── 1. Positions ─────────────────────────────────────────────────────────
    positions = feat_np[:, FEAT_POSITION]   # [N, 3]
    print(f"  Position range: [{positions.min():.2f}, {positions.max():.2f}]")

    pos_tensor = torch.from_numpy(positions).float().to(args.device)
    with torch.no_grad():
        pos_colors = get_pca_color_torch(pos_tensor, brightness=args.brightness)
    pos_colors_np = pos_colors.cpu().numpy()

    pos_path = output_dir / f"scene_{scene_idx:03d}_position_pca.ply"
    write_ply_with_colors(positions, pos_colors_np, str(pos_path))
    print(f"  ✓ Position PCA saved: {pos_path.name}")

    # ── 2. Original RGB colors ───────────────────────────────────────────────
    colors = feat_np[:, FEAT_COLOR]         # [N, 3], already in [0,1]
    colors = np.clip(colors, 0.0, 1.0)
    print(f"  Color range: [{colors.min():.3f}, {colors.max():.3f}]  "
          f"std={colors.std():.3f}")

    color_path = output_dir / f"scene_{scene_idx:03d}_original_colors.ply"
    write_ply_with_colors(positions, colors, str(color_path))
    print(f"  ✓ Original colors saved: {color_path.name}")

    # ── 3. Scale PCA ─────────────────────────────────────────────────────────
    scales = feat_np[:, FEAT_SCALE]         # [N, 3]
    print(f"  Scale range:  [{scales.min():.4f}, {scales.max():.4f}]  "
          f"mean={scales.mean():.4f}")

    scale_tensor = torch.from_numpy(scales).float().to(args.device)
    with torch.no_grad():
        scale_colors = get_pca_color_torch(scale_tensor, brightness=args.brightness)
    scale_colors_np = scale_colors.cpu().numpy()

    scale_path = output_dir / f"scene_{scene_idx:03d}_scale_pca.ply"
    write_ply_with_colors(positions, scale_colors_np, str(scale_path))
    print(f"  ✓ Scale PCA saved: {scale_path.name}")

    # ── 4. Opacity as grayscale ───────────────────────────────────────────────
    opacity = feat_np[:, FEAT_OPACITY]      # [N]
    # Apply sigmoid to get actual opacity values
    opacity_sigmoid = 1.0 / (1.0 + np.exp(-opacity))
    opacity_gray = np.stack([opacity_sigmoid] * 3, axis=1)  # [N, 3] grayscale
    print(f"  Opacity range: [{opacity_sigmoid.min():.3f}, {opacity_sigmoid.max():.3f}]  "
          f"mean={opacity_sigmoid.mean():.3f}")

    opacity_path = output_dir / f"scene_{scene_idx:03d}_opacity_gray.ply"
    write_ply_with_colors(positions, opacity_gray, str(opacity_path))
    print(f"  ✓ Opacity grayscale saved: {opacity_path.name}")

    # ── 5. Combined feature PCA (all 14 Gaussian params) ────────────────────
    # Extract the 14-dim target vector: pos + color + opacity + scale + rot
    GEOMETRIC_INDICES = list(range(4, 7)) + list(range(7, 10)) + [10] + \
                        list(range(11, 14)) + list(range(14, 18))
    all_params = feat_np[:, GEOMETRIC_INDICES]  # [N, 14]

    all_tensor = torch.from_numpy(all_params).float().to(args.device)
    with torch.no_grad():
        all_colors = get_pca_color_torch(all_tensor, brightness=args.brightness)
    all_colors_np = all_colors.cpu().numpy()

    all_path = output_dir / f"scene_{scene_idx:03d}_all_params_pca.ply"
    write_ply_with_colors(positions, all_colors_np, str(all_path))
    print(f"  ✓ All-params PCA saved: {all_path.name}")

    print()

# ── Summary ───────────────────────────────────────────────────────────────────

print("=" * 70)
print("DONE")
print("=" * 70)
print(f"  Saved {args.num_scenes} scenes × 5 PLY files each")
print(f"  Output: {output_dir.resolve()}")
print()
print("Files per scene:")
print("  scene_XXX_position_pca.ply    — positions PCA-colored")
print("  scene_XXX_original_colors.ply — original RGB colors")
print("  scene_XXX_scale_pca.ply       — scale distribution PCA-colored")
print("  scene_XXX_opacity_gray.ply    — opacity as grayscale")
print("  scene_XXX_all_params_pca.ply  — all 14 Gaussian params PCA-colored")
print()
print("Load all PLY files into CloudCompare to compare input vs reconstruction.")
print("=" * 70)