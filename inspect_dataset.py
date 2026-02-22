"""
RAW Dataset Inspector - NO NORMALIZATION
=========================================

Shows EXACTLY what's stored in the .npy files on disk.
NO canonical sphere normalization, NO /255, NO sampling.
Just pure raw values as saved by the dataset creator.
"""

import numpy as np
from pathlib import Path


def inspect_raw_scene(scene_dir):
    """Load and display raw .npy files without ANY normalization."""
    
    print(f"\n{'='*70}")
    print(f"RAW SCENE: {scene_dir.name}")
    print(f"{'='*70}")
    
    # ========================================================================
    # LOAD RAW FILES (NO PROCESSING)
    # ========================================================================
    
    coord   = np.load(scene_dir / "coord.npy")
    color   = np.load(scene_dir / "color.npy")
    opacity = np.load(scene_dir / "opacity.npy")
    scale   = np.load(scene_dir / "scale.npy")
    quat    = np.load(scene_dir / "quat.npy")
    
    # ========================================================================
    # DISPLAY RAW VALUES
    # ========================================================================
    
    print(f"\n📍 POSITION (coord.npy) - RAW")
    print(f"  Shape:      {coord.shape}")
    print(f"  dtype:      {coord.dtype}")
    print(f"  Min:        {coord.min():.6f}")
    print(f"  Max:        {coord.max():.6f}")
    print(f"  Mean:       {coord.mean():.6f}")
    print(f"  Std:        {coord.std():.6f}")
    print(f"  First 3 Gaussians (xyz):")
    for i in range(min(3, len(coord))):
        print(f"    [{i}]: [{coord[i,0]:8.4f}, {coord[i,1]:8.4f}, {coord[i,2]:8.4f}]")
    
    print(f"\n🎨 COLOR (color.npy) - RAW")
    print(f"  Shape:      {color.shape}")
    print(f"  dtype:      {color.dtype}")
    print(f"  Min:        {color.min():.6f}")
    print(f"  Max:        {color.max():.6f}")
    print(f"  Mean:       {color.mean():.6f}")
    print(f"  Std:        {color.std():.6f}")
    print(f"  First 3 Gaussians (RGB):")
    for i in range(min(3, len(color))):
        print(f"    [{i}]: [{color[i,0]:7.2f}, {color[i,1]:7.2f}, {color[i,2]:7.2f}]")
    
    # Check if it's 0-255 or 0-1
    if color.max() > 2.0:
        print(f"  ⚠️  Colors are in [0, 255] range!")
        print(f"  ✓  Dataset WILL normalize with /255")
    else:
        print(f"  ⚠️  Colors already in [0, 1] range!")
        print(f"  ✓  Dataset will NOT normalize (already normalized)")
    
    print(f"\n👁️  OPACITY (opacity.npy) - RAW")
    print(f"  Shape:      {opacity.shape}")
    print(f"  dtype:      {opacity.dtype}")
    print(f"  Min:        {opacity.min():.6f}")
    print(f"  Max:        {opacity.max():.6f}")
    print(f"  Mean:       {opacity.mean():.6f}")
    print(f"  Std:        {opacity.std():.6f}")
    print(f"  First 10 values:")
    print(f"    {opacity[:10]}")
    
    # Check if post-sigmoid
    if opacity.min() >= 0 and opacity.max() <= 1.0:
        print(f"  ✓  Already in [0, 1] - POST-SIGMOID format")
    else:
        print(f"  ⚠️  Not in [0, 1] - pre-sigmoid format?")
    
    print(f"\n📏 SCALE (scale.npy) - RAW")
    print(f"  Shape:      {scale.shape}")
    print(f"  dtype:      {scale.dtype}")
    print(f"  Min:        {scale.min():.6f}")
    print(f"  Max:        {scale.max():.6f}")
    print(f"  Mean:       {scale.mean():.6f}")
    print(f"  Std:        {scale.std():.6f}")
    print(f"  First 3 Gaussians (sx, sy, sz):")
    for i in range(min(3, len(scale))):
        print(f"    [{i}]: [{scale[i,0]:.6f}, {scale[i,1]:.6f}, {scale[i,2]:.6f}]")
    
    # Check if all positive (post-exp)
    if scale.min() >= 0:
        print(f"  ✓  All positive - POST-EXP format (meters)")
    else:
        print(f"  ⚠️  Has negative values - pre-exp format (log-space)")
    
    print(f"\n🔄 QUATERNION (quat.npy) - RAW")
    print(f"  Shape:      {quat.shape}")
    print(f"  dtype:      {quat.dtype}")
    print(f"  Min:        {quat.min():.6f}")
    print(f"  Max:        {quat.max():.6f}")
    print(f"  Mean:       {quat.mean():.6f}")
    print(f"  First 3 quaternions (qw, qx, qy, qz):")
    for i in range(min(3, len(quat))):
        norm = np.linalg.norm(quat[i])
        print(f"    [{i}]: [{quat[i,0]:7.4f}, {quat[i,1]:7.4f}, {quat[i,2]:7.4f}, {quat[i,3]:7.4f}]  norm={norm:.6f}")
    
    # Check normalization
    quat_norms = np.linalg.norm(quat, axis=1)
    print(f"  Norms: min={quat_norms.min():.6f}, max={quat_norms.max():.6f}, mean={quat_norms.mean():.6f}")
    
    if quat_norms.min() > 0.99 and quat_norms.max() < 1.01:
        print(f"  ✓  Normalized (||q|| ≈ 1.0)")
    else:
        print(f"  ⚠️  Not normalized")
    
    return {
        'coord': coord,
        'color': color,
        'opacity': opacity,
        'scale': scale,
        'quat': quat,
    }


def compare_raw_vs_normalized(scene_dir):
    """Show side-by-side comparison of raw vs normalized values."""
    
    print(f"\n{'='*70}")
    print(f"COMPARISON: RAW vs DATASET NORMALIZATION")
    print(f"{'='*70}")
    
    # Load raw
    coord_raw   = np.load(scene_dir / "coord.npy")
    color_raw   = np.load(scene_dir / "color.npy")
    scale_raw   = np.load(scene_dir / "scale.npy")
    
    # Apply dataset normalization
    # 1. Canonical sphere for position & scale
    center = coord_raw.mean(axis=0)
    coord_centered = coord_raw - center
    distances = np.linalg.norm(coord_centered, axis=1)
    max_dist = distances.max()
    if max_dist < 1e-6:
        max_dist = 1.0
    scale_factor = 10.0 / (max_dist * 1.1)
    
    coord_norm = coord_centered * scale_factor
    scale_norm = scale_raw * scale_factor
    
    # 2. Color normalization
    color_norm = color_raw / 255.0
    
    # Print comparison
    print(f"\n📍 POSITION:")
    print(f"  RAW:        [{coord_raw.min():8.3f}, {coord_raw.max():8.3f}]m  mean={coord_raw.mean():7.3f}")
    print(f"  NORMALIZED: [{coord_norm.min():8.3f}, {coord_norm.max():8.3f}]m  mean={coord_norm.mean():7.3f}")
    print(f"  Scale factor: {scale_factor:.6f}")
    
    print(f"\n🎨 COLOR:")
    print(f"  RAW:        [{color_raw.min():8.3f}, {color_raw.max():8.3f}]  mean={color_raw.mean():7.3f}")
    print(f"  NORMALIZED: [{color_norm.min():8.3f}, {color_norm.max():8.3f}]  mean={color_norm.mean():7.3f}")
    if color_raw.max() > 2.0:
        print(f"  ✓  Divided by 255 (was integer RGB)")
    else:
        print(f"  ✓  No change (already in [0, 1])")
    
    print(f"\n📏 SCALE:")
    print(f"  RAW:        [{scale_raw.min():8.6f}, {scale_raw.max():8.6f}]m  mean={scale_raw.mean():8.6f}")
    print(f"  NORMALIZED: [{scale_norm.min():8.6f}, {scale_norm.max():8.6f}]m  mean={scale_norm.mean():8.6f}")
    print(f"  Scale factor: {scale_factor:.6f} (same as position)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect RAW .npy files (no normalization)')
    parser.add_argument('--dataset-dir', type=Path,
                       default=Path("/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs/train_grid1.0cm_chunk8x8_stride6x6"),
                       help='Dataset directory')
    parser.add_argument('--num-scenes', type=int, default=3,
                       help='Number of scenes to inspect')
    parser.add_argument('--compare', action='store_true',
                       help='Show raw vs normalized comparison')
    
    args = parser.parse_args()
    
    if not args.dataset_dir.exists():
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        return
    
    # Find scenes
    scene_dirs = sorted([
        d for d in args.dataset_dir.iterdir()
        if d.is_dir() and (d / "coord.npy").exists()
    ])
    
    if not scene_dirs:
        print(f"❌ No scenes found in {args.dataset_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"RAW DATASET INSPECTION - NO NORMALIZATION")
    print(f"{'='*70}")
    print(f"  Path: {args.dataset_dir}")
    print(f"  Found: {len(scene_dirs)} scenes")
    print(f"  Inspecting: {min(args.num_scenes, len(scene_dirs))} scenes")
    print(f"{'='*70}")
    
    # Inspect each scene
    for i, scene_dir in enumerate(scene_dirs[:args.num_scenes]):
        inspect_raw_scene(scene_dir)
        
        if i == 0 and args.compare:
            compare_raw_vs_normalized(scene_dir)
    
    # ========================================================================
    # AGGREGATE STATISTICS ACROSS ALL SCENES
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(f"AGGREGATE STATISTICS (RAW VALUES ONLY)")
    print(f"{'='*70}")
    
    all_coords = []
    all_colors = []
    all_opacities = []
    all_scales = []
    
    print(f"\nLoading {min(args.num_scenes, len(scene_dirs))} scenes...")
    for scene_dir in scene_dirs[:args.num_scenes]:
        all_coords.append(np.load(scene_dir / "coord.npy"))
        all_colors.append(np.load(scene_dir / "color.npy"))
        all_opacities.append(np.load(scene_dir / "opacity.npy"))
        all_scales.append(np.load(scene_dir / "scale.npy"))
    
    all_coords = np.concatenate(all_coords, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    all_opacities = np.concatenate(all_opacities, axis=0)
    all_scales = np.concatenate(all_scales, axis=0)
    
    print(f"\n📊 POSITION (raw .npy values):")
    print(f"  Total Gaussians: {len(all_coords):,}")
    print(f"  Range: [{all_coords.min():.3f}, {all_coords.max():.3f}]m")
    print(f"  Mean:  {all_coords.mean():.3f}m")
    print(f"  Std:   {all_coords.std():.3f}m")
    
    print(f"\n🎨 COLOR (raw .npy values):")
    print(f"  Range: [{all_colors.min():.3f}, {all_colors.max():.3f}]")
    print(f"  Mean:  {all_colors.mean():.3f}")
    print(f"  Std:   {all_colors.std():.3f}")
    if all_colors.max() > 2.0:
        print(f"  ✓  Stored as [0, 255] (integer RGB)")
        print(f"  ✓  Dataset will apply /255 normalization")
    else:
        print(f"  ✓  Stored as [0, 1] (already normalized)")
        print(f"  ✓  Dataset will NOT apply /255")
    
    print(f"\n👁️  OPACITY (raw .npy values):")
    print(f"  Range: [{all_opacities.min():.6f}, {all_opacities.max():.6f}]")
    print(f"  Mean:  {all_opacities.mean():.6f}")
    print(f"  Std:   {all_opacities.std():.6f}")
    if all_opacities.max() <= 1.0:
        print(f"  ✓  Stored in POST-SIGMOID format [0, 1]")
        print(f"  ✓  Dataset will NOT apply sigmoid")
    else:
        print(f"  ⚠️  Stored in raw format (> 1.0)")
    
    print(f"\n📏 SCALE (raw .npy values):")
    print(f"  Range: [{all_scales.min():.6f}, {all_scales.max():.6f}]m")
    print(f"  Mean:  {all_scales.mean():.6f}m = {all_scales.mean()*100:.2f}cm")
    print(f"  Std:   {all_scales.std():.6f}m")
    if all_scales.min() >= 0:
        print(f"  ✓  Stored in POST-EXP format (meters, positive)")
        print(f"  ✓  Dataset will NOT apply exp")
    else:
        print(f"  ⚠️  Stored in log-space (has negative values)")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY - WHAT'S STORED IN .NPY FILES")
    print(f"{'='*70}")
    print(f"\nThe .npy files contain:")
    print(f"  Position:   Raw meters (scene-specific scale)")
    if all_colors.max() > 2.0:
        print(f"  Color:      [0, 255] integer RGB")
    else:
        print(f"  Color:      [0, 1] normalized RGB")
    print(f"  Opacity:    [0, 1] post-sigmoid")
    print(f"  Scale:      Positive meters (post-exp)")
    print(f"  Quaternion: Normalized (||q|| = 1)")
    
    print(f"\nDataset normalization applies:")
    print(f"  Position:   ✓ Canonical sphere (center + scale to 10m)")
    print(f"  Scale:      ✓ Proportional scaling (same factor as position)")
    if all_colors.max() > 2.0:
        print(f"  Color:      ✓ Divide by 255 → [0, 1]")
    else:
        print(f"  Color:      ✗ No change (already [0, 1])")
    print(f"  Opacity:    ✗ No change (already post-sigmoid)")
    print(f"  Quaternion: ✗ No change (already normalized)")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()