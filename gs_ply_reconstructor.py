"""
3DGS PLY Reconstructor - DEBUG VERSION
=======================================
Comprehensive diagnostics to identify training issues

ACTIVATION PIPELINE (model → PLY):
  Model applies exp() internally → output is already in METRES
  Reconstructor applies log()   → stores log(metres) in PLY (3DGS standard)
  SuperSplat applies exp()      → recovers metres for rendering

  scale flow:  raw_logit → exp() [in model] → metres → log() [here] → PLY
  opacity flow: raw_logit → sigmoid() [in model] → [0,1] → logit() [here] → PLY
"""

import numpy as np
from pathlib import Path
from typing import Optional
from plyfile import PlyData, PlyElement

# ── constants ─────────────────────────────────────────────────────────────────

C0  = 0.28209479177387814   # SH DC constant
EPS = 1e-7

# ── parameter slices ──────────────────────────────────────────────────────────

COORD_SLICE   = slice(0,  3)
COLOR_SLICE   = slice(3,  6)
OPACITY_SLICE = slice(6,  7)
SCALE_SLICE   = slice(7,  10)
QUAT_SLICE    = slice(10, 14)


# ── activation inversions ─────────────────────────────────────────────────────

def logit(p: np.ndarray) -> np.ndarray:
    """Inverse of sigmoid. Model outputs sigmoid(raw) → we store raw in PLY."""
    p = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p)).astype(np.float32)


def safe_log(s: np.ndarray) -> np.ndarray:
    """Inverse of exp. Model outputs exp(raw) metres → we store log(metres) in PLY."""
    return np.log(np.maximum(s.astype(np.float64), EPS)).astype(np.float32)


# ── colour conversion ─────────────────────────────────────────────────────────

def rgb_to_f_dc(rgb: np.ndarray) -> np.ndarray:
    """RGB [0,1] → SH DC coefficients."""
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    return ((rgb - 0.5) / C0).astype(np.float32)


# ── quaternion normalisation ──────────────────────────────────────────────────

def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    norm = np.where(norm > EPS, norm, 1.0)
    return (quat / norm).astype(np.float32)


# ── PLY vertex struct ─────────────────────────────────────────────────────────

def build_vertex_struct(
    coord:      np.ndarray,
    f_dc:       np.ndarray,
    ply_opacity: np.ndarray,
    ply_scales:  np.ndarray,
    quat:       np.ndarray,
    max_sh_degree: int = 3,
    normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    N = coord.shape[0]

    quat        = normalize_quaternion(quat.reshape(N, 4))
    normals     = normals.astype(np.float32) if normals is not None \
                  else np.zeros((N, 3), dtype=np.float32)
    ply_opacity = ply_opacity.reshape(N).astype(np.float32)
    ply_scales  = ply_scales.reshape(N, 3).astype(np.float32)
    f_dc        = f_dc.reshape(N, 3).astype(np.float32)

    num_f_rest = 3 * ((max_sh_degree + 1) ** 2 - 1)

    dtype_list = (
        [("x","f4"), ("y","f4"), ("z","f4")]
      + [("nx","f4"), ("ny","f4"), ("nz","f4")]
      + [(f"f_dc_{i}",   "f4") for i in range(3)]
      + [(f"f_rest_{i}", "f4") for i in range(num_f_rest)]
      + [("opacity", "f4")]
      + [(f"scale_{i}", "f4") for i in range(3)]
      + [(f"rot_{i}",   "f4") for i in range(4)]
    )

    vert = np.empty(N, dtype=dtype_list)

    vert["x"], vert["y"], vert["z"]    = coord[:,0],   coord[:,1],   coord[:,2]
    vert["nx"], vert["ny"], vert["nz"] = normals[:,0], normals[:,1], normals[:,2]

    for i in range(3):
        vert[f"f_dc_{i}"] = f_dc[:, i]

    for i in range(num_f_rest):
        vert[f"f_rest_{i}"] = 0.0

    vert["opacity"] = ply_opacity

    for i in range(3):
        vert[f"scale_{i}"] = ply_scales[:, i]

    for i in range(4):
        vert[f"rot_{i}"] = quat[:, i]

    return vert


# ── 🔍 COMPREHENSIVE DIAGNOSTICS ──────────────────────────────────────────────

def diagnose_scene(coord, rgb, opacity, scale, quat, ply_scales, ply_opacity):
    """
    Print comprehensive diagnostic information.

    Parameter conventions at this point in the pipeline:
      coord   : metres (absolute positions, post-model)
      rgb     : [0, 1] colours (post-clamp, pre-SH conversion)
      opacity : [0, 1] (post-sigmoid from model)
      scale   : metres (post-exp from model) ← NOT log-space here
      quat    : unit quaternions (post-normalise from model)
      ply_scales  : log(metres) written to PLY  (= safe_log(scale))
      ply_opacity : logit([0,1]) written to PLY (= logit(opacity))
    """

    print(f"\n{'='*70}")
    print(f"🔍 COMPREHENSIVE SCENE DIAGNOSTICS")
    print(f"{'='*70}")

    # ── POSITION ANALYSIS ─────────────────────────────────────────────────────
    print(f"\n📍 POSITION ANALYSIS:")
    print(f"  Range:     [{coord.min():.3f}, {coord.max():.3f}]m")
    print(f"  Mean:      {coord.mean(axis=0)}")
    print(f"  Std:       {coord.std(axis=0)}")
    print(f"  Spread:    {coord.max() - coord.min():.3f}m")

    if coord.min() > -8.0 or coord.max() < 8.0:
        print(f"  ⚠️  COMPRESSED! Expected ±9m, got [{coord.min():.1f}, {coord.max():.1f}]")
        print(f"  → Model output range is smaller than target")
        print(f"  → Position loss will plateau at ~5.0")
    else:
        print(f"  ✓  Good range (matches canonical sphere)")

    # ── COLOR ANALYSIS ────────────────────────────────────────────────────────
    print(f"\n🎨 COLOR ANALYSIS:")
    print(f"  Range:     [{rgb.min():.3f}, {rgb.max():.3f}]")
    print(f"  Mean:      {rgb.mean():.3f}")
    print(f"  Std:       {rgb.std():.3f}")

    rgb_spread = rgb.max() - rgb.min()
    print(f"  Spread:    {rgb_spread:.3f}")

    if rgb.std() < 0.1:
        print(f"  ⚠️  ALL GRAY! (std < 0.1)")
        print(f"  → Colors clustered around {rgb.mean():.2f}")
        print(f"  → Model not learning per-scene colors")
    elif rgb.std() < 0.15:
        print(f"  ⚠️  Limited color variation (std < 0.15)")
    else:
        print(f"  ✓  Good color variation")

    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    hist, _ = np.histogram(rgb.flatten(), bins=bins)
    print(f"  Distribution:")
    print(f"    [0.0-0.3]: {hist[0]/len(rgb.flatten())*100:.1f}%")
    print(f"    [0.3-0.4]: {hist[1]/len(rgb.flatten())*100:.1f}%")
    print(f"    [0.4-0.5]: {hist[2]/len(rgb.flatten())*100:.1f}%  ← dark gray")
    print(f"    [0.5-0.6]: {hist[3]/len(rgb.flatten())*100:.1f}%  ← mid gray")
    print(f"    [0.6-0.7]: {hist[4]/len(rgb.flatten())*100:.1f}%  ← light gray")
    print(f"    [0.7-1.0]: {hist[5]/len(rgb.flatten())*100:.1f}%")

    # ── OPACITY ANALYSIS ──────────────────────────────────────────────────────
    print(f"\n👁️  OPACITY ANALYSIS:")
    # opacity here is already [0,1] (post-sigmoid from model)
    print(f"  Model output (post-sigmoid): [{opacity.min():.3f}, {opacity.max():.3f}]")

    num_clamped = np.sum(opacity > 1.0)
    if num_clamped > 0:
        print(f"  ⚠️  {num_clamped} values > 1.0 (will be clamped)")

    # What SuperSplat will see: sigmoid(ply_opacity) = original opacity
    opacity_rendered = 1.0 / (1.0 + np.exp(-ply_opacity.astype(np.float64)))
    print(f"  Rendered (SuperSplat):       [{opacity_rendered.min():.3f}, {opacity_rendered.max():.3f}]")

    avg_opacity = opacity_rendered.mean()
    print(f"  Mean opacity:  {avg_opacity:.3f}")

    if avg_opacity < 0.5:
        print(f"  ⚠️  LOW! (< 0.5) → Splats too transparent → cloudy appearance")
    elif avg_opacity < 0.8:
        print(f"  ⚠️  Medium (0.5-0.8) → Partially transparent")
    else:
        print(f"  ✓  High (> 0.8) → Splats are opaque")

    # ── SCALE ANALYSIS ────────────────────────────────────────────────────────
    # 'scale' is already in METRES here (model applied exp() internally).
    # 'ply_scales' is log(metres) = what gets written to disk.
    # SuperSplat reads log(metres) and applies exp() to render.
    print(f"\n📏 SCALE ANALYSIS:")
    print(f"  Model output (metres):       [{scale.min():.4f}, {scale.max():.4f}]m")
    print(f"  Written to PLY (log-metres): [{ply_scales.min():.4f}, {ply_scales.max():.4f}]")
    print(f"  Mean scale:                  {scale.mean():.4f}m = {scale.mean()*100:.1f}cm")
    print(f"  Median scale:                {np.median(scale):.4f}m = {np.median(scale)*100:.1f}cm")

    if scale.mean() > 0.5:
        print(f"  ⚠️  LARGE! (> 0.5m = 50cm) → Splats blooming into each other")
        print(f"  → Gray cloud appearance")
        print(f"  → Expected: ~0.057m (5.7cm) at init, convergence below 0.10m")
    elif scale.mean() < 0.01:
        print(f"  ⚠️  TOO SMALL! (< 0.01m = 1cm) → Splats collapsing")
    else:
        print(f"  ✓  Reasonable scale (1-50cm)")

    small_scales  = (scale < 0.05).sum()
    medium_scales = ((scale >= 0.05) & (scale < 0.10)).sum()
    large_scales  = ((scale >= 0.10) & (scale < 0.20)).sum()
    huge_scales   = (scale >= 0.20).sum()
    total = scale.size

    print(f"  Distribution:")
    print(f"    < 5cm:   {small_scales/total*100:.1f}%   (target: ~65% for indoor scenes)")
    print(f"    5-10cm:  {medium_scales/total*100:.1f}%   (target: ~25%)")
    print(f"    10-20cm: {large_scales/total*100:.1f}%")
    print(f"    > 20cm:  {huge_scales/total*100:.1f}%")

    # ── ROTATION ANALYSIS ─────────────────────────────────────────────────────
    print(f"\n🔄 ROTATION ANALYSIS:")
    quat_norm = np.linalg.norm(quat, axis=1)
    print(f"  Quaternion norm: [{quat_norm.min():.3f}, {quat_norm.max():.3f}]")

    if np.abs(quat_norm.mean() - 1.0) > 0.1:
        print(f"  ⚠️  Not normalized! (mean norm = {quat_norm.mean():.3f})")
    else:
        print(f"  ✓  Properly normalized")

    print(f"{'='*70}\n")


# ── single-scene reconstruction ───────────────────────────────────────────────

def reconstruct_single_scene(
    prediction:    np.ndarray,
    output_path:   Path,
    max_sh_degree: int = 3,
    verbose:       bool = True,
    color_mode:    str = "1",
) -> Optional[str]:
    """
    Convert model output to 3DGS PLY.

    Model output layout [N, 14]:
      [0:3]   positions  (metres,       raw from model)
      [3:6]   colors     ([0,1],        post-clamp from model)
      [6]     opacity    ([0,1],        post-sigmoid from model)
      [7:10]  scale      (metres,       post-exp from model)   ← KEY: already metres
      [10:14] quaternion (unit,         post-normalise from model)

    PLY format expected by SuperSplat / standard 3DGS viewers:
      positions  : metres  (written as-is)
      f_dc       : (rgb - 0.5) / C0
      opacity    : logit(opacity)  so that sigmoid(stored) = original
      scale      : log(metres)     so that exp(stored)     = original metres
      rotation   : unit quaternion (written as-is)
    """
    try:
        N = prediction.shape[0]

        coord   = prediction[:, COORD_SLICE  ].astype(np.float32)
        rgb     = prediction[:, COLOR_SLICE  ].astype(np.float32)
        opacity = prediction[:, OPACITY_SLICE].astype(np.float32)
        scale   = prediction[:, SCALE_SLICE  ].astype(np.float32)  # already metres
        quat    = prediction[:, QUAT_SLICE   ].astype(np.float32)

        # ── Invert activations before writing to PLY ──────────────────────────
        # opacity: model output is post-sigmoid [0,1] → store logit so viewer
        #          can recover the value with sigmoid(stored)
        ply_opacity = logit(opacity)

        # scale: model output is post-exp metres → store log so viewer
        #        can recover metres with exp(stored)
        ply_scales = safe_log(scale)

        # ── Run diagnostics ───────────────────────────────────────────────────
        if verbose:
            diagnose_scene(coord, rgb, opacity, scale, quat, ply_scales, ply_opacity)

        # ── Build PLY ─────────────────────────────────────────────────────────
        f_dc = rgb_to_f_dc(rgb)

        vertex = build_vertex_struct(
            coord=coord,
            f_dc=f_dc,
            ply_opacity=ply_opacity,
            ply_scales=ply_scales,
            quat=quat,
            max_sh_degree=max_sh_degree,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(output_path))

        print(f"✓ Saved: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"⚠️  Error: {e}")
        import traceback; traceback.print_exc()
        return None


# ── batch reconstruction ──────────────────────────────────────────────────────

def save_reconstructed_gaussians(
    predictions:   np.ndarray,
    output_dir:    Path,
    epoch:         int,
    num_scenes:    int = 2,
    max_sh_degree: int = 3,
    color_mode:    str = "1",
    prefix:        str = "scene",
) -> dict:
    """Save reconstructed scenes with diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_save = min(num_scenes, predictions.shape[0])
    saved  = {}

    print(f"\n{'='*70}")
    print(f"3DGS RECONSTRUCTION - Epoch {epoch}")
    print(f"{'='*70}")

    for i in range(n_save):
        print(f"\n📦 Scene {i}/{n_save-1}:")
        out_path = output_dir / f"{prefix}_{i:03d}_epoch_{epoch:03d}.ply"

        path = reconstruct_single_scene(
            prediction=predictions[i],
            output_path=out_path,
            max_sh_degree=max_sh_degree,
            verbose=True,
        )

        if path:
            saved[f"scene_{i:03d}"] = path

    print(f"\n{'='*70}")
    print(f"✓ Saved {len(saved)}/{n_save} scenes")
    print(f"  Location: {output_dir}")
    print(f"  View at: https://supersplat.at")
    print(f"{'='*70}\n")

    return saved