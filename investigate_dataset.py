"""
Dataset Scene Investigation Script — OPACITY-SELECTED TOP-40K
==============================================================
Analyzes ONLY the Gaussians that the model actually sees during training.
Uses the EXACT same sampling logic as gs_dataset_scenesplat.py.

This is critical — analyzing all Gaussians gives misleading statistics
because training only sees the top-40k by opacity, which are biased
toward large, opaque structural elements (walls, floors, ceilings).

Usage:
    python investigate_dataset.py \\
        --data_path /path/to/scenes \\
        --num_scenes 500 \\
        --scale_norm_mode linear
"""

import numpy as np
import os
import argparse
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Sampling — EXACTLY matches gs_dataset_scenesplat.py
# ─────────────────────────────────────────────────────────────────────────────

TARGET_POINTS = 40_000

def sample_top40k_by_opacity(coord, color, scale, quat, opacity,
                              target=TARGET_POINTS):
    """
    Select top `target` Gaussians by opacity.
    EXACTLY matches gs_dataset_scenesplat.py __getitem__ Step 5.
    """
    N = len(coord)
    importance = opacity  # opacity sampling

    sorted_indices = np.argsort(importance)  # ascending

    if N >= target:
        selected = sorted_indices[-target:]  # top target
    else:
        n_extra  = target - N
        extra    = np.full(n_extra, sorted_indices[-1], dtype=np.int64)
        selected = np.concatenate([sorted_indices, extra])

    return (
        coord  [selected],
        color  [selected],
        scale  [selected],
        quat   [selected],
        opacity[selected],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Normalization — EXACTLY matches gs_dataset_scenesplat.py
# ─────────────────────────────────────────────────────────────────────────────

def normalize_to_canonical_sphere(coord, scale, target_radius=10.0,
                                   scale_norm_mode='linear'):
    """
    Matches gs_dataset_scenesplat.normalize_to_canonical_sphere exactly.
    """
    center        = coord.mean(axis=0)
    coord_centered = coord - center
    distances     = np.linalg.norm(coord_centered, axis=1)
    max_dist      = distances.max()
    if max_dist < 1e-6:
        max_dist = 1.0
    scale_factor  = target_radius / (max_dist * 1.1)
    coord_norm    = coord_centered * scale_factor

    if scale_norm_mode == 'log':
        scale_norm = np.log(scale + 1e-7) + np.log(scale_factor)
    else:
        scale_norm = scale * scale_factor

    return coord_norm, scale_norm, scale_factor


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_scenes(data_path, num_scenes=500, scale_norm_mode='linear',
                   target_radius=10.0):

    scene_dirs = sorted([
        os.path.join(data_path, d)
        for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])[:num_scenes]

    print(f"Analyzing {len(scene_dirs)} scenes")
    print(f"Sampling:  top-40k by opacity  (MATCHES training pipeline)")
    print(f"Scale mode: {scale_norm_mode}")
    print(f"Target radius: {target_radius}m\n")

    # ── Accumulators ─────────────────────────────────────────────────────────
    all_scales_raw  = []
    all_scales_norm = []
    all_colors      = []
    all_opacities   = []
    all_pos_spread  = []

    scene_stats     = []
    skipped         = 0

    for scene_dir in tqdm(scene_dirs, desc="Loading scenes"):
        try:
            coord   = np.load(os.path.join(scene_dir, 'coord.npy'))
            color   = np.load(os.path.join(scene_dir, 'color.npy'))
            scale   = np.load(os.path.join(scene_dir, 'scale.npy'))
            quat    = np.load(os.path.join(scene_dir, 'quat.npy'))
            opacity = np.load(os.path.join(scene_dir, 'opacity.npy'))
        except FileNotFoundError:
            skipped += 1
            continue

        # ── Sanity check — skip corrupted scenes ─────────────────────────────
        if (np.any(~np.isfinite(coord))  or
            np.any(~np.isfinite(scale))  or
            np.any(~np.isfinite(opacity)) or
            scale.max() > 100.0):          # extreme outlier guard
            skipped += 1
            continue

        # ── STEP 1: Sample top-40k by opacity (matches training) ─────────────
        coord, color, scale, quat, opacity = sample_top40k_by_opacity(
            coord, color, scale, quat, opacity
        )

        # ── STEP 2: Normalize (matches training) ─────────────────────────────
        coord_norm, scale_norm, scale_factor = normalize_to_canonical_sphere(
            coord, scale,
            target_radius=target_radius,
            scale_norm_mode=scale_norm_mode,
        )

        # ── STEP 3: Color normalization ───────────────────────────────────────
        color_norm = color / 255.0

        # ── Per-Gaussian scale magnitudes ─────────────────────────────────────
        scale_mag_raw  = np.linalg.norm(scale,      axis=1)
        scale_mag_norm = np.linalg.norm(scale_norm, axis=1)

        # For log mode, exp back to get metres
        if scale_norm_mode == 'log':
            scale_mag_metres = np.exp(scale_mag_norm)
        else:
            scale_mag_metres = scale_mag_norm

        # ── Per-scene stats ───────────────────────────────────────────────────
        scene_stats.append({
            'name':             os.path.basename(scene_dir),
            'scale_factor':     scale_factor,
            'n_total':          len(coord),

            # Raw scale (metres, before normalization)
            'scale_raw_mean':   scale_mag_raw.mean(),
            'scale_raw_median': np.median(scale_mag_raw),
            'scale_raw_max':    scale_mag_raw.max(),
            'scale_raw_p75':    np.percentile(scale_mag_raw, 75),
            'scale_raw_p90':    np.percentile(scale_mag_raw, 90),
            'scale_raw_p95':    np.percentile(scale_mag_raw, 95),

            # Normalized scale
            'scale_norm_mean':   scale_mag_metres.mean(),
            'scale_norm_median': np.median(scale_mag_metres),
            'scale_norm_max':    scale_mag_metres.max(),
            'scale_norm_p75':    np.percentile(scale_mag_metres, 75),
            'scale_norm_p90':    np.percentile(scale_mag_metres, 90),
            'scale_norm_p95':    np.percentile(scale_mag_metres, 95),

            # Size buckets (raw metres)
            'frac_below_5cm':   (scale_mag_raw < 0.05).mean(),
            'frac_5_10cm':      ((scale_mag_raw >= 0.05) & (scale_mag_raw < 0.10)).mean(),
            'frac_10_30cm':     ((scale_mag_raw >= 0.10) & (scale_mag_raw < 0.30)).mean(),
            'frac_30_100cm':    ((scale_mag_raw >= 0.30) & (scale_mag_raw < 1.00)).mean(),
            'frac_above_100cm': (scale_mag_raw >= 1.00).mean(),

            # Opacity of selected Gaussians
            'opacity_mean':    opacity.mean(),
            'opacity_median':  np.median(opacity),
            'opacity_min':     opacity.min(),

            # Color
            'color_mean':      color_norm.mean(),
            'color_std':       color_norm.std(),

            # Position spread
            'pos_spread_norm': np.linalg.norm(coord_norm, axis=1).max() * 2,
        })

        # ── Accumulate (subsample to keep memory reasonable) ──────────────────
        idx = np.random.choice(len(coord), min(1000, len(coord)), replace=False)
        all_scales_raw .append(scale_mag_raw [idx])
        all_scales_norm.append(scale_mag_metres[idx])
        all_colors     .append(color_norm     [idx])
        all_opacities  .append(opacity        [idx])
        all_pos_spread .append(
            np.linalg.norm(coord_norm, axis=1).max() * 2
        )

    if skipped > 0:
        print(f"\n⚠️  Skipped {skipped} corrupted/missing scenes")

    all_scales_raw  = np.concatenate(all_scales_raw)
    all_scales_norm = np.concatenate(all_scales_norm)
    all_colors      = np.concatenate(all_colors)
    all_opacities   = np.concatenate(all_opacities)

    # ═════════════════════════════════════════════════════════════════════════
    # REPORT
    # ═════════════════════════════════════════════════════════════════════════

    sep = "=" * 70

    print(f"\n{sep}")
    print("SCALE ANALYSIS — RAW (metres, AFTER opacity-based sampling)")
    print(sep)
    print(f"  Mean:    {all_scales_raw.mean():.4f}m")
    print(f"  Median:  {np.median(all_scales_raw):.4f}m")
    print(f"  Std:     {all_scales_raw.std():.4f}m")
    print(f"  Min:     {all_scales_raw.min():.6f}m")
    print(f"  Max:     {all_scales_raw.max():.4f}m")
    print(f"  P25:     {np.percentile(all_scales_raw, 25):.4f}m")
    print(f"  P50:     {np.percentile(all_scales_raw, 50):.4f}m")
    print(f"  P75:     {np.percentile(all_scales_raw, 75):.4f}m")
    print(f"  P90:     {np.percentile(all_scales_raw, 90):.4f}m")
    print(f"  P95:     {np.percentile(all_scales_raw, 95):.4f}m")
    print(f"  P99:     {np.percentile(all_scales_raw, 99):.4f}m")
    print()
    print("  Size distribution (raw metres, training-selected Gaussians):")
    print(f"    < 5cm:       {(all_scales_raw <  0.05).mean()*100:.1f}%")
    print(f"    5 – 10cm:    {((all_scales_raw >= 0.05) & (all_scales_raw < 0.10)).mean()*100:.1f}%")
    print(f"    10 – 30cm:   {((all_scales_raw >= 0.10) & (all_scales_raw < 0.30)).mean()*100:.1f}%")
    print(f"    30 – 100cm:  {((all_scales_raw >= 0.30) & (all_scales_raw < 1.00)).mean()*100:.1f}%")
    print(f"    > 100cm:     {(all_scales_raw >= 1.00).mean()*100:.1f}%")

    print(f"\n{sep}")
    print(f"SCALE ANALYSIS — NORMALIZED ({scale_norm_mode} mode, training targets)")
    print(sep)
    print(f"  Mean:    {all_scales_norm.mean():.4f}m")
    print(f"  Median:  {np.median(all_scales_norm):.4f}m")
    print(f"  Std:     {all_scales_norm.std():.4f}m")
    print(f"  Min:     {all_scales_norm.min():.6f}m")
    print(f"  Max:     {all_scales_norm.max():.4f}m")
    print(f"  P75:     {np.percentile(all_scales_norm, 75):.4f}m")
    print(f"  P90:     {np.percentile(all_scales_norm, 90):.4f}m")
    print(f"  P95:     {np.percentile(all_scales_norm, 95):.4f}m")
    print(f"  P99:     {np.percentile(all_scales_norm, 99):.4f}m")
    print()
    print("  Size distribution (normalized, what model learns to predict):")
    print(f"    < 5cm:       {(all_scales_norm <  0.05).mean()*100:.1f}%")
    print(f"    5 – 10cm:    {((all_scales_norm >= 0.05) & (all_scales_norm < 0.10)).mean()*100:.1f}%")
    print(f"    10 – 30cm:   {((all_scales_norm >= 0.10) & (all_scales_norm < 0.30)).mean()*100:.1f}%")
    print(f"    30 – 100cm:  {((all_scales_norm >= 0.30) & (all_scales_norm < 1.00)).mean()*100:.1f}%")
    print(f"    > 100cm:     {(all_scales_norm >= 1.00).mean()*100:.1f}%")

    print(f"\n{sep}")
    print("OPACITY ANALYSIS — Selected Gaussians (training-selected only)")
    print(sep)
    print(f"  Mean:    {all_opacities.mean():.4f}")
    print(f"  Median:  {np.median(all_opacities):.4f}")
    print(f"  Std:     {all_opacities.std():.4f}")
    print(f"  Min:     {all_opacities.min():.4f}")
    print(f"  Max:     {all_opacities.max():.4f}")
    print(f"  < 0.5:   {(all_opacities < 0.5).mean()*100:.1f}%")
    print(f"  > 0.9:   {(all_opacities > 0.9).mean()*100:.1f}%")
    print()
    print("  Note: These are the MINIMUM opacity values in the top-40k.")
    print("  A high minimum means all selected Gaussians are opaque.")
    mins = [s['opacity_min'] for s in scene_stats]
    print(f"  Per-scene min opacity — mean: {np.mean(mins):.4f}  "
          f"std: {np.std(mins):.4f}  min: {np.min(mins):.4f}")

    print(f"\n{sep}")
    print("COLOR ANALYSIS — [0,1] normalized, training-selected Gaussians")
    print(sep)
    print(f"  Mean (all channels):  {all_colors.mean():.4f}")
    print(f"  Std  (all channels):  {all_colors.std():.4f}")
    print(f"  Mean R:  {all_colors[:,0].mean():.4f}   Std R: {all_colors[:,0].std():.4f}")
    print(f"  Mean G:  {all_colors[:,1].mean():.4f}   Std G: {all_colors[:,1].std():.4f}")
    print(f"  Mean B:  {all_colors[:,2].mean():.4f}   Std B: {all_colors[:,2].std():.4f}")
    print()
    print("  Color distribution:")
    print(f"    [0.0-0.2]:  {(all_colors < 0.2).mean()*100:.1f}%  (dark)")
    print(f"    [0.2-0.4]:  {((all_colors >= 0.2) & (all_colors < 0.4)).mean()*100:.1f}%")
    print(f"    [0.4-0.6]:  {((all_colors >= 0.4) & (all_colors < 0.6)).mean()*100:.1f}%  (mid-gray)")
    print(f"    [0.6-0.8]:  {((all_colors >= 0.6) & (all_colors < 0.8)).mean()*100:.1f}%")
    print(f"    [0.8-1.0]:  {(all_colors >= 0.8).mean()*100:.1f}%  (bright)")

    print(f"\n{sep}")
    print("SCALE FACTOR DISTRIBUTION (per-scene normalization strength)")
    print(sep)
    factors = [s['scale_factor'] for s in scene_stats]
    print(f"  Mean:  {np.mean(factors):.4f}")
    print(f"  Std:   {np.std(factors):.4f}")
    print(f"  Min:   {np.min(factors):.4f}")
    print(f"  Max:   {np.max(factors):.4f}")
    print()
    print("  This is how much each scene's scales are multiplied by.")
    print("  High variation here means same physical scale → different targets")
    print("  across scenes, making learning harder.")

    print(f"\n{sep}")
    print("PER-SCENE SUMMARY (across all valid scenes)")
    print(sep)
    raw_medians  = [s['scale_raw_median']  for s in scene_stats]
    norm_medians = [s['scale_norm_median'] for s in scene_stats]
    raw_p90s     = [s['scale_raw_p90']     for s in scene_stats]
    norm_p90s    = [s['scale_norm_p90']    for s in scene_stats]

    print(f"  Raw median per scene  — mean: {np.mean(raw_medians):.4f}m  "
          f"std: {np.std(raw_medians):.4f}m")
    print(f"  Norm median per scene — mean: {np.mean(norm_medians):.4f}m  "
          f"std: {np.std(norm_medians):.4f}m")
    print(f"  Raw P90 per scene     — mean: {np.mean(raw_p90s):.4f}m  "
          f"std: {np.std(raw_p90s):.4f}m")
    print(f"  Norm P90 per scene    — mean: {np.mean(norm_p90s):.4f}m  "
          f"std: {np.std(norm_p90s):.4f}m")

    print(f"\n{sep}")
    print("WHAT THIS MEANS FOR TRAINING DECISIONS")
    print(sep)

    raw_median   = np.median(all_scales_raw)
    norm_median  = np.median(all_scales_norm)
    norm_p75     = np.percentile(all_scales_norm, 75)
    norm_p90     = np.percentile(all_scales_norm, 90)
    color_mean   = all_colors.mean()
    color_std    = all_colors.std()
    factor_std   = np.std(factors)

    print(f"  Model currently outputs ~1.0m mean scale")
    print(f"  Ground truth raw median (training Gaussians): {raw_median:.4f}m")
    print(f"  → Model is {1.0/raw_median:.1f}× too large")
    print()
    print(f"  Normalized scale targets the model must learn:")
    print(f"    Median: {norm_median:.4f}m")
    print(f"    P75:    {norm_p75:.4f}m  ← reasonable reg threshold")
    print(f"    P90:    {norm_p90:.4f}m  ← aggressive reg threshold")
    print()
    print(f"  Scale factor std = {factor_std:.4f}")
    if factor_std > 0.3:
        print(f"  → HIGH variation. Same physical scale maps to very different")
        print(f"    training targets across scenes. Consider global normalization.")
    else:
        print(f"  → LOW variation. Per-scene normalization is consistent enough.")
    print()
    print(f"  Color ground truth std: {color_std:.4f}")
    print(f"  Model color collapse std: ~0.02-0.05")
    print(f"  → Anti-collapse variance target: ~{color_std**2 * 0.3:.4f} "
          f"(30% of ground truth variance)")

    return scene_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze dataset statistics using SAME sampling as training"
    )
    parser.add_argument('--data_path', type=str,
                        default="/home/yli11/scratch/datasets/gaussian_world/"
                                "preprocessed/interior_gs/"
                                "train_grid1.0cm_chunk8x8_stride6x6")
    parser.add_argument('--num_scenes', type=int, default=500)
    parser.add_argument('--scale_norm_mode', type=str, default='linear',
                        choices=['log', 'linear'])
    parser.add_argument('--target_radius', type=float, default=10.0)
    args = parser.parse_args()

    analyze_scenes(
        data_path=args.data_path,
        num_scenes=args.num_scenes,
        scale_norm_mode=args.scale_norm_mode,
        target_radius=args.target_radius,
    )