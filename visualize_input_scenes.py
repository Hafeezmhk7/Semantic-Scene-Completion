"""
Visualize Input 3DGS Scenes → PLY
===================================
Converts raw .npy scene files to PLY using EXACTLY the same pipeline
as training so the input and reconstruction are directly comparable.

CRITICAL MATCHES WITH gs_dataset_scenesplat.py:
  1. Sampling:      top-40k by opacity (argsort ascending, take last 40k)
  2. Normalization: same canonical sphere function
  3. Scale mode:    same scale_norm_mode argument
  4. Color:         /255 normalization

WHY THIS MATTERS:
  If the input visualization uses different Gaussians than training,
  you are comparing apples to oranges in SuperSplat. The reconstruction
  is trained on specific 40k Gaussians — the input must show those same
  Gaussians at the same scale for a fair comparison.

USAGE:
    python visualize_input_scenes.py \\
        --dataset-dir /path/to/val \\
        --output-dir  ./input_scenes_ply \\
        --num-scenes  5 \\
        --scale-norm-mode linear \\
        --target-radius 10.0
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

# ── Constants ─────────────────────────────────────────────────────────────────

C0            = 0.28209479177387814
EPS           = 1e-7
TARGET_POINTS = 40_000


# ─────────────────────────────────────────────────────────────────────────────
# Sampling — EXACTLY matches gs_dataset_scenesplat.py Step 5
# ─────────────────────────────────────────────────────────────────────────────

def sample_top40k_by_opacity(coord, color, scale, quat, opacity,
                              target=TARGET_POINTS):
    """
    Select top `target` Gaussians by opacity.

    This is a line-for-line match of __getitem__ Step 5 in
    gs_dataset_scenesplat.py. Do NOT change the logic here without
    also changing the dataset file.

    Ascending argsort → take last `target` = highest opacity Gaussians.
    If scene has fewer than `target` Gaussians, pad by repeating the
    highest-opacity Gaussian (same as dataset).
    """
    N          = len(coord)
    importance = opacity                        # opacity sampling

    sorted_indices = np.argsort(importance)     # ascending

    if N >= target:
        selected = sorted_indices[-target:]     # top target by opacity
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

    scale_norm_mode='linear': scale_norm = scale * factor
    scale_norm_mode='log':    scale_norm = log(scale) + log(factor)

    Must use the same mode as your training run.
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
# PLY helpers
# ─────────────────────────────────────────────────────────────────────────────

def logit(p, eps=EPS):
    """Inverse sigmoid → raw PLY opacity field."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float32)


def color_to_f_dc(color_arr):
    """
    Convert color.npy (uint8 [0,255]) → f_dc SH coefficients.

    color.npy stores values as f_pc * 255 where f_pc = f_dc * C0 + 0.5
    Inversion: f_dc = (color/255 - 0.5) / C0

    This matches gs_dataset_scenesplat.py which does color/255 for training.
    """
    f_pc = np.clip(color_arr.astype(np.float32) / 255.0, 0.0, 1.0)
    return ((f_pc - 0.5) / C0).astype(np.float32)


def scale_to_raw(scale_norm, scale_norm_mode='linear'):
    """
    Convert normalized scale back to raw PLY log-scale format.

    PLY stores: log(scale_metres)
    
    For linear mode: scale_norm is in metres → log(scale_norm)
    For log mode:    scale_norm is already log(metres) + log(factor)
                     → already in log space, use directly
    """
    if scale_norm_mode == 'linear':
        # scale_norm is metres, PLY wants log(metres)
        raw = np.log(np.maximum(scale_norm, EPS))
    else:
        # scale_norm is already log(metres * factor)
        # PLY wants log(metres), same thing since factor absorbed in norm
        raw = scale_norm.copy()
    return raw.astype(np.float32)


def make_vertex_struct(coord, f_dc, raw_opacity, raw_scales, quat,
                        max_sh_degree=3):
    """Build PLY vertex structured array."""
    N          = coord.shape[0]
    num_f_dc   = f_dc.shape[1]
    num_scale  = raw_scales.shape[1]
    num_f_rest = 3 * ((max_sh_degree + 1) ** 2 - 1)

    dtype_list = (
        [("x", "f4"), ("y", "f4"), ("z", "f4")]
      + [("nx","f4"), ("ny","f4"), ("nz","f4")]
      + [(f"f_dc_{i}",   "f4") for i in range(num_f_dc)]
      + [(f"f_rest_{i}", "f4") for i in range(num_f_rest)]
      + [("opacity", "f4")]
      + [(f"scale_{i}", "f4") for i in range(num_scale)]
      + [(f"rot_{i}",   "f4") for i in range(4)]
    )

    vert = np.empty(N, dtype=dtype_list)

    vert["x"], vert["y"], vert["z"] = (
        coord[:, 0], coord[:, 1], coord[:, 2]
    )
    vert["nx"] = vert["ny"] = vert["nz"] = 0.0  # no normals stored

    for i in range(num_f_dc):
        vert[f"f_dc_{i}"] = f_dc[:, i]

    for i in range(num_f_rest):
        vert[f"f_rest_{i}"] = 0.0

    vert["opacity"] = raw_opacity.reshape(N)

    for i in range(num_scale):
        vert[f"scale_{i}"] = raw_scales[:, i]

    # Normalize quaternion
    qnorm = np.linalg.norm(quat, axis=1, keepdims=True)
    qnorm = np.where(qnorm > 0, qnorm, 1.0)
    quat  = (quat / qnorm).astype(np.float32)
    for i in range(4):
        vert[f"rot_{i}"] = quat[:, i]

    return vert


# ─────────────────────────────────────────────────────────────────────────────
# Main scene conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_scene(scene_dir, output_path, target_radius=10.0,
                  scale_norm_mode='linear', max_sh_degree=3, verbose=True):
    """
    Load scene → apply EXACT training pipeline → write PLY.

    Pipeline order matches gs_dataset_scenesplat.py __getitem__:
      Step 1: Load raw npy files
      Step 2: Canonical sphere normalization (positions + scales)
      Step 3: Color /255 normalization
      Step 4: Load semantics (skipped here)
      Step 5: Deterministic top-40k by opacity sampling
      Step 6: Voxelisation (skipped, not needed for PLY)
    """
    try:
        # ── Step 1: Load ──────────────────────────────────────────────────────
        coord   = np.load(scene_dir / 'coord.npy')
        color   = np.load(scene_dir / 'color.npy')   # uint8 [0,255]
        scale   = np.load(scene_dir / 'scale.npy')   # metres
        quat    = np.load(scene_dir / 'quat.npy')
        opacity = np.load(scene_dir / 'opacity.npy')

        N_raw = len(coord)
        if verbose:
            print(f"  Scene:       {scene_dir.name}")
            print(f"  Total N:     {N_raw:,} Gaussians (before sampling)")

        # ── Step 2: Canonical sphere normalization ────────────────────────────
        # MUST match training: normalize BEFORE sampling
        coord_norm, scale_norm, scale_factor = normalize_to_canonical_sphere(
            coord, scale,
            target_radius=target_radius,
            scale_norm_mode=scale_norm_mode,
        )

        if verbose:
            max_dist = np.linalg.norm(coord_norm, axis=1).max()
            print(f"  Scale factor:{scale_factor:.4f}")
            print(f"  Coord range: [{coord_norm.min():.2f}, {coord_norm.max():.2f}]m")
            if scale_norm_mode == 'linear':
                print(f"  Scale range: [{scale_norm.min():.4f}, "
                      f"{scale_norm.max():.4f}]m  (linear)")
            else:
                print(f"  Scale range (log): [{scale_norm.min():.4f}, "
                      f"{scale_norm.max():.4f}]")
                print(f"  Scale range (exp): [{np.exp(scale_norm.min()):.4f}, "
                      f"{np.exp(scale_norm.max()):.4f}]m")

        # ── Step 3: Color normalization (for diagnostics only) ────────────────
        # We keep original color_arr for f_dc conversion (needs uint8 [0,255])

        # ── Step 5: Sample top-40k by opacity ─────────────────────────────────
        # MUST match training: sampling happens AFTER normalization
        coord_norm, color, scale_norm, quat, opacity = sample_top40k_by_opacity(
            coord_norm, color, scale_norm, quat, opacity,
            target=TARGET_POINTS,
        )

        if verbose:
            print(f"  Sampled N:   {len(coord_norm):,} Gaussians "
                  f"(top-{TARGET_POINTS} by opacity)")
            print(f"  Opacity min: {opacity.min():.4f}  "
                  f"(lowest opacity in selected set)")
            color_01 = color / 255.0
            print(f"  Color mean:  {color_01.mean():.4f}  "
                  f"std: {color_01.std():.4f}")

        # ── Convert to PLY format ─────────────────────────────────────────────
        f_dc        = color_to_f_dc(color)
        raw_opacity = logit(opacity)
        raw_scales  = scale_to_raw(scale_norm, scale_norm_mode)

        vertex = make_vertex_struct(
            coord       = coord_norm.astype(np.float32),
            f_dc        = f_dc,
            raw_opacity = raw_opacity,
            raw_scales  = raw_scales,
            quat        = quat.astype(np.float32),
            max_sh_degree = max_sh_degree,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        PlyData(
            [PlyElement.describe(vertex, "vertex")],
            text=False
        ).write(str(output_path))

        file_mb = output_path.stat().st_size / 1e6
        if verbose:
            print(f"  ✓ Saved: {output_path.name}  ({file_mb:.1f} MB)\n")

        return True

    except Exception as e:
        print(f"  ⚠  Error processing {scene_dir.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Convert dataset scenes → PLY using EXACT training pipeline"
    )
    ap.add_argument("--dataset-dir",      type=Path, required=True)
    ap.add_argument("--output-dir",       type=Path, default=Path("./input_scenes_ply"))
    ap.add_argument("--num-scenes",       type=int,  default=5)
    ap.add_argument("--target-radius",    type=float, default=10.0)
    ap.add_argument("--scale-norm-mode",  type=str,  default="linear",
                    choices=["linear", "log"],
                    help="Must match SCALE_NORM_MODE in your training job script")
    ap.add_argument("--max-sh-degree",    type=int,  default=3)
    args = ap.parse_args()

    if not args.dataset_dir.exists():
        print(f"[error] Not found: {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    scene_dirs = sorted([
        d for d in args.dataset_dir.iterdir()
        if d.is_dir() and (d / "coord.npy").exists()
    ])

    if not scene_dirs:
        print(f"[error] No scene folders in {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    n = min(args.num_scenes, len(scene_dirs))

    print(f"\n{'='*70}")
    print(f"INPUT 3DGS → PLY  (EXACT training pipeline)")
    print(f"{'='*70}")
    print(f"  Dataset:         {args.dataset_dir}")
    print(f"  Output:          {args.output_dir}")
    print(f"  Scenes:          {n} of {len(scene_dirs)} available")
    print(f"  Target radius:   {args.target_radius}m")
    print(f"  Scale norm mode: {args.scale_norm_mode}  "
          f"← must match SCALE_NORM_MODE in job script")
    print(f"  Sampling:        top-{TARGET_POINTS} by opacity")
    print(f"")
    print(f"  Pipeline order (matches gs_dataset_scenesplat.py):")
    print(f"    1. Load npy files")
    print(f"    2. Canonical sphere normalization")
    print(f"    3. Sample top-40k by opacity")
    print(f"    4. Write PLY")
    print(f"{'='*70}\n")

    saved = 0
    for i, scene_dir in enumerate(scene_dirs[:n]):
        out_name = f"input_scene_{i:03d}_{scene_dir.name}.ply"
        out_path = args.output_dir / out_name
        print(f"[{i+1}/{n}] {scene_dir.name}")
        if convert_scene(
            scene_dir, out_path,
            target_radius    = args.target_radius,
            scale_norm_mode  = args.scale_norm_mode,
            max_sh_degree    = args.max_sh_degree,
        ):
            saved += 1

    print(f"{'='*70}")
    print(f"✓ Converted {saved}/{n} scenes  →  {args.output_dir}")
    print()
    print(f"Comparing in SuperSplat:")
    print(f"  1. Go to https://supersplat.at")
    print(f"  2. Load: input_scene_000_<name>.ply     ← input (same 40k Gaussians)")
    print(f"  3. Load: scene_000_epoch_XXX.ply        ← reconstruction")
    print(f"  4. Both use same sampling + normalization → fair comparison")
    print(f"  5. Toggle visibility to compare side by side")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()