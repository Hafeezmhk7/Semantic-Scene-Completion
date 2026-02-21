"""
Visualize Input 3DGS Scenes → PLY (with Canonical Sphere Normalization)
=========================================================================

Converts raw .npy scene files to PLY using:
  1. Same sampling as training (deterministic top-40k by opacity)
  2. CANONICAL SPHERE NORMALIZATION (Can3Tok's method)
  3. Same as training pipeline for direct comparison

🎯 KEY: Applies SAME normalization as training so input and reconstruction
        are at the SAME SCALE (both in ~10m sphere).

WHY CANONICAL SPHERE NORMALIZATION:
  - Training normalizes to 10m sphere
  - Reconstructions are in 10m sphere  
  - Input MUST also be in 10m sphere for comparison!
  - NO denormalization needed (output is already at good scale)

USAGE:
    python visualize_input_scenes.py \\
        --dataset-dir /path/to/val \\
        --output-dir  ./input_scenes_ply \\
        --num-scenes  5 \\
        --target-radius 10.0

    Then in SuperSplat:
      1. Load input_scene_000.ply     ← Input (in 10m sphere)
      2. Load scene_000_epoch_090.ply ← Reconstruction (in 10m sphere)
      3. Both at same scale → easy comparison!
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

# ── constants ─────────────────────────────────────────────────────────────────

C0  = 0.28209479177387814
EPS = 1e-7
TARGET_POINTS = 40000
DEFAULT_TARGET_RADIUS = 10.0  # 🎯 Canonical sphere radius


# ─────────────────────────────────────────────────────────────────────────────
# 🎯 Canonical Sphere Normalization (Can3Tok's Method)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_to_canonical_sphere(coord, scale, target_radius=10.0):
    """
    Normalize scene to canonical sphere (Can3Tok ICCV 2025).
    
    MUST match training pipeline exactly so input and reconstruction
    are at the same scale!
    
    Args:
        coord: [N, 3] positions
        scale: [N, 3] scales
        target_radius: Fixed radius (default 10.0m)
    
    Returns:
        coord_norm: Centered and scaled to fit in sphere of radius target_radius
        scale_norm: Scaled proportionally
    """
    # Center at origin
    center = coord.mean(axis=0)
    coord_centered = coord - center
    
    # Find max distance
    distances = np.linalg.norm(coord_centered, axis=1)
    max_dist = distances.max()
    
    if max_dist < 1e-6:
        max_dist = 1.0
    
    # Scale to target radius
    scale_factor = target_radius / (max_dist * 1.1)
    
    coord_norm = coord_centered * scale_factor
    scale_norm = scale * scale_factor
    
    return coord_norm, scale_norm


# ── PLY conversion functions ──────────────────────────────────────────────────

def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Inverse sigmoid → PLY opacity field."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float32)


def decode_f_dc_from_color(color_arr: np.ndarray, mode: str = "255") -> np.ndarray:
    """
    Convert color.npy → f_dc (SH DC coefficients).
    
    color.npy stores f_pc * 255, where f_pc = f_dc * C0 + 0.5.
    Inversion: f_dc = (f_pc - 0.5) / C0
    """
    color = color_arr.astype(np.float32, copy=False)

    if mode == "auto":
        scale = 255.0 if (color_arr.dtype == np.uint8 or float(np.nanmax(color)) > 1.5) else 1.0
    elif mode == "255":
        scale = 255.0
    elif mode == "1":
        scale = 1.0
    else:
        raise ValueError(f"Unknown color mode: {mode!r}")

    f_pc = np.clip(color / scale, 0.0, 1.0)
    return ((f_pc - 0.5) / C0).astype(np.float32)


# ── sampling (matches training) ───────────────────────────────────────────────

def sample_like_training(
    coord:   np.ndarray,
    color:   np.ndarray,
    opacity: np.ndarray,
    scale:   np.ndarray,
    quat:    np.ndarray,
    normals: np.ndarray,
    target:  int = TARGET_POINTS,
) -> tuple:
    """
    Sample top `target` Gaussians by opacity (deterministic).
    Matches training pipeline's deterministic sampling.
    """
    N = len(coord)
    importance = opacity.ravel()

    if N > target * 2:
        # Take top 2× by opacity, then best target from those
        top_indices = np.argsort(importance)[-(target * 2):]
        sub_importance = importance[top_indices]
        best = np.argsort(sub_importance)[-target:]
        selected = top_indices[best]
    elif N > target:
        selected = np.argsort(importance)[-target:]
    else:
        # Pad
        extra    = np.random.choice(N, target - N, replace=True)
        selected = np.concatenate([np.arange(N), extra])

    arrays = [coord, color, scale, quat, opacity.reshape(-1, 1)]
    results = [a[selected] for a in arrays]
    
    if normals is not None:
        results.append(normals[selected])
    else:
        results.append(np.zeros((target, 3), dtype=np.float32))

    return tuple(results)


# ── PLY vertex struct ─────────────────────────────────────────────────────────

def make_vertex_struct(
    coord:       np.ndarray,
    normals:     np.ndarray,
    f_dc:        np.ndarray,
    raw_opacity: np.ndarray,
    raw_scales:  np.ndarray,
    quat:        np.ndarray,
    max_sh_degree: int = 3,
) -> np.ndarray:
    """Build PLY vertex structure."""
    N          = coord.shape[0]
    num_f_dc   = f_dc.shape[1]
    num_scale  = raw_scales.shape[1]
    num_f_rest = 3 * ((max_sh_degree + 1) ** 2 - 1)

    dtype_list = (
        [("x","f4"), ("y","f4"), ("z","f4")]
      + [("nx","f4"), ("ny","f4"), ("nz","f4")]
      + [(f"f_dc_{i}",   "f4") for i in range(num_f_dc)]
      + [(f"f_rest_{i}", "f4") for i in range(num_f_rest)]
      + [("opacity", "f4")]
      + [(f"scale_{i}", "f4") for i in range(num_scale)]
      + [(f"rot_{i}",   "f4") for i in range(4)]
    )

    vert = np.empty(N, dtype=dtype_list)
    vert["x"],  vert["y"],  vert["z"]  = coord[:,0],   coord[:,1],   coord[:,2]
    vert["nx"], vert["ny"], vert["nz"] = normals[:,0], normals[:,1], normals[:,2]

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


# ── main conversion ───────────────────────────────────────────────────────────

def convert_scene(
    scene_dir:     Path,
    output_path:   Path,
    target_radius: float = DEFAULT_TARGET_RADIUS,
    max_sh_degree: int  = 3,
    verbose:       bool = True,
) -> bool:
    """
    Load scene, apply canonical sphere normalization, sample, write PLY.
    
    🎯 KEY: Applies SAME normalization as training pipeline!
    """
    try:
        # ── Load ──────────────────────────────────────────────────────────────
        coord   = np.load(scene_dir / "coord.npy")
        color   = np.load(scene_dir / "color.npy")
        opacity = np.load(scene_dir / "opacity.npy")
        scale   = np.load(scene_dir / "scale.npy")
        quat    = np.load(scene_dir / "quat.npy")

        normals = None
        if (scene_dir / "normal.npy").exists():
            normals = np.load(scene_dir / "normal.npy")

        N_raw = len(coord)

        if verbose:
            print(f"  Scene:      {scene_dir.name}")
            print(f"  Raw N:      {N_raw:,} Gaussians")

        # ── 🎯 CANONICAL SPHERE NORMALIZATION ─────────────────────────────────
        # CRITICAL: Must match training pipeline!
        coord, scale = normalize_to_canonical_sphere(
            coord, scale,
            target_radius=target_radius
        )
        
        if verbose:
            max_dist = np.linalg.norm(coord, axis=1).max()
            utilization = (max_dist / target_radius) * 100
            print(f"  Normalized: Canonical sphere (radius={target_radius}m)")
            print(f"    Max distance: {max_dist:.2f}m")
            print(f"    Utilization:  {utilization:.1f}%")
            print(f"    Coord range:  [{coord.min():.2f}, {coord.max():.2f}]m")
        # ─────────────────────────────────────────────────────────────────────

        # ── Sample down to 40k ────────────────────────────────────────────────
        coord, color, scale, quat, opacity, normals = sample_like_training(
            coord, color, opacity, scale, quat, normals,
            target=TARGET_POINTS,
        )

        if verbose:
            print(f"  Sampled N:  {len(coord):,} Gaussians (top-opacity)")

        # ── Convert colour → f_dc ─────────────────────────────────────────────
        f_dc = decode_f_dc_from_color(color, mode="255")

        # ── Invert activations ────────────────────────────────────────────────
        raw_opacity = logit(opacity)
        raw_scales  = np.log(np.maximum(scale, EPS))
        raw_scales  = raw_scales.astype(np.float32)

        if verbose:
            print(f"  Scale range: [{scale.min():.4f}, {scale.max():.4f}]m")
            print(f"  Splat size:  [{np.exp(raw_scales.min()):.4f}, {np.exp(raw_scales.max()):.4f}]m")

        # ── Build PLY ─────────────────────────────────────────────────────────
        vertex = make_vertex_struct(
            coord       = coord.astype(np.float32),
            normals     = normals.astype(np.float32),
            f_dc        = f_dc,
            raw_opacity = raw_opacity,
            raw_scales  = raw_scales,
            quat        = quat.astype(np.float32),
            max_sh_degree = max_sh_degree,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(output_path))

        file_mb = output_path.stat().st_size / 1e6
        if verbose:
            print(f"  ✓ Saved: {output_path.name}  ({file_mb:.1f} MB)\n")

        return True

    except Exception as e:
        print(f"  ⚠  Error: {e}\n")
        import traceback; traceback.print_exc()
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Convert dataset scenes → PLY with canonical sphere normalization"
    )
    ap.add_argument("--dataset-dir", type=Path, required=True,
                    help="Dataset directory (train/val)")
    ap.add_argument("--output-dir",  type=Path, default=Path("./input_scenes_ply"),
                    help="Output directory for PLY files")
    ap.add_argument("--num-scenes",  type=int,  default=5,
                    help="Number of scenes to convert")
    ap.add_argument("--target-radius", type=float, default=DEFAULT_TARGET_RADIUS,
                    help=f"Canonical sphere radius in meters (default: {DEFAULT_TARGET_RADIUS})")
    ap.add_argument("--max-sh-degree", type=int, default=3,
                    help="Max SH degree")
    args = ap.parse_args()

    if not args.dataset_dir.exists():
        print(f"[error] Not found: {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    # Find scene dirs
    scene_dirs = sorted([
        d for d in args.dataset_dir.iterdir()
        if d.is_dir() and (d / "coord.npy").exists()
    ])

    if not scene_dirs:
        print(f"[error] No scene folders found in {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    n = min(args.num_scenes, len(scene_dirs))

    print(f"\n{'='*70}")
    print(f"INPUT 3DGS → PLY with Canonical Sphere Normalization")
    print(f"{'='*70}")
    print(f"  Dataset:        {args.dataset_dir}")
    print(f"  Output:         {args.output_dir}")
    print(f"  Scenes:         {n} of {len(scene_dirs)} available")
    print(f"  Target radius:  {args.target_radius}m  (canonical sphere)")
    print(f"  Sampling:       Deterministic top-40k by opacity")
    print(f"🎯 Normalization: SAME as training pipeline!")
    print(f"  → Input and reconstruction at SAME SCALE (both in {args.target_radius}m sphere)")
    print(f"  → Easy comparison in SuperSplat!")
    print(f"{'='*70}\n")

    saved = 0
    for i, scene_dir in enumerate(scene_dirs[:n]):
        out_path = args.output_dir / f"input_scene_{i:03d}_{scene_dir.name}.ply"
        print(f"[{i+1}/{n}] {scene_dir.name}")
        if convert_scene(
            scene_dir, out_path,
            target_radius=args.target_radius,
            max_sh_degree=args.max_sh_degree
        ):
            saved += 1

    print(f"{'='*70}")
    print(f"✓ Converted {saved}/{n} scenes  →  {args.output_dir}")
    print()
    print(f"How to compare in SuperSplat:")
    print(f"  1. Go to https://supersplat.at/editor")
    print(f"  2. Load: input_scene_000_<name>.ply      (input in {args.target_radius}m sphere)")
    print(f"  3. Load: scene_000_epoch_090.ply         (reconstruction in {args.target_radius}m sphere)")
    print(f"  4. Both at SAME SCALE → easy comparison! ✓")
    print(f"  5. Toggle 👁 in Scene Manager to compare")
    print(f"  6. Or offset: Transform → Position X = 15  (side-by-side)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()