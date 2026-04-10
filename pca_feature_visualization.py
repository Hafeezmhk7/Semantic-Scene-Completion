"""
PCA Feature Visualization for Can3Tok Semantic Diagnostics
============================================================
No Open3D required — uses plyfile or manual PLY writing.

TWO VISUALIZATION MODES:

1. Per-Gaussian semantic features (per_gaussian_features from decoder)
   ─ visualize_semantic_features(coords, features, output_path)
   ─ N points (40k Gaussians), D-dim features → PCA colors
   ─ Diagnostic: do same-category Gaussians cluster in feature space?

2. Scene-level z_s space (z_s projections from SemanticTokenInfoNCEHead)
   ─ visualize_z_s_space(z_s_proj, label_dists, output_path)
   ─ M points (one per eval scene), 128-dim projections → PCA positions
   ─ Colors: dominant ScanNet72 category per scene
   ─ Diagnostic: do same-category scenes cluster in z_s?

Usage:
    from pca_feature_visualization import (
        visualize_semantic_features,
        visualize_z_s_space,
    )
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List

try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False
    print("Warning: plyfile not found — falling back to manual PLY writer")


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def get_pca_color_torch(feat, brightness=1.25, center=True, q=6, niter=5):
    """
    Low-rank PCA colorization.  Returns [N, 3] float32 in [0,1].
    Blends first 3 and next 3 principal components for richer colors.
    """
    n, d     = feat.shape
    q_actual = min(q, d, n)

    u, s, v = torch.pca_lowrank(feat, center=center, q=q_actual, niter=niter)
    proj    = feat @ v                  # [N, q_actual]

    if q_actual >= 6:
        mix = proj[:, :3] * 0.6 + proj[:, 3:6] * 0.4
    elif q_actual >= 3:
        mix = proj[:, :3]
    else:
        mix = torch.zeros((n, 3), dtype=feat.dtype, device=feat.device)
        mix[:, :q_actual] = proj

    mn  = mix.amin(dim=0, keepdim=True)
    mx  = mix.amax(dim=0, keepdim=True)
    mix = (mix - mn) / (mx - mn + 1e-6)
    return (mix * brightness).clamp(0.0, 1.0)


def build_valid_mask(feat, norm_thresh=0.0):
    """Return (valid_mask, norms).  Filters NaN/Inf and zero-norm rows."""
    finite  = np.isfinite(feat).all(axis=1)
    n, c    = feat.shape
    chunk   = max(1, 1_000_000 // max(1, c))
    norms   = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        norms[sl] = np.linalg.norm(feat[sl].astype(np.float32, copy=False), axis=1)
    return finite & (norms > norm_thresh), norms


def write_ply_with_colors(coords, colors, output_path):
    """Write PLY with float positions and uint8 colors (plyfile or manual)."""
    n          = len(coords)
    colors_u8  = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)

    if HAS_PLYFILE:
        vdtype  = [('x','f4'),('y','f4'),('z','f4'),
                   ('red','u1'),('green','u1'),('blue','u1')]
        verts   = np.empty(n, dtype=vdtype)
        verts['x'], verts['y'], verts['z'] = coords[:,0], coords[:,1], coords[:,2]
        verts['red'], verts['green'], verts['blue'] = \
            colors_u8[:,0], colors_u8[:,1], colors_u8[:,2]
        PlyData([PlyElement.describe(verts, 'vertex')], text=False).write(str(output_path))
    else:
        with open(output_path, 'wb') as f:
            header = (f"ply\nformat binary_little_endian 1.0\n"
                      f"element vertex {n}\n"
                      f"property float x\nproperty float y\nproperty float z\n"
                      f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                      f"end_header\n")
            f.write(header.encode('ascii'))
            for i in range(n):
                f.write(coords[i].astype(np.float32).tobytes())
                f.write(colors_u8[i].tobytes())


# ============================================================================
# MODE 1 — PER-GAUSSIAN SEMANTIC FEATURES (InfoNCE decoder output)
# ============================================================================

def visualize_semantic_features(coords, features, output_path, brightness=1.25,
                                 pca_q=6, pca_niter=5, device='cpu', verbose=True):
    """
    Visualize per-Gaussian semantic features using PCA-based coloring.

    Args:
        coords:      [N, 3]  numpy — 3D Gaussian positions
        features:    [N, D]  numpy — semantic projection features (e.g. [N, 32])
        output_path: str     — output PLY path
        brightness:  float   — brightness multiplier (default 1.25)

    Returns:
        output_path str if successful, None otherwise.

    Diagnostic use:
        If per-Gaussian InfoNCE is working, Gaussians of the same ScanNet72
        category should show similar PCA-derived colors across the scene.
        Load alongside original-color PLY in SuperSplat and compare region colors.
    """
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"PCA FEATURE VISUALIZATION — per-Gaussian ({features.shape})")
            print(f"{'='*60}")

        valid, _ = build_valid_mask(features, norm_thresh=0.0)
        n_valid  = int(valid.sum())
        if verbose:
            print(f"  Valid features: {n_valid}/{len(features)}")
        if n_valid == 0:
            print("  No valid features — skipping.")
            return None

        feat_v   = torch.from_numpy(features[valid]).float().to(device)
        with torch.no_grad():
            color_t = get_pca_color_torch(feat_v, brightness=brightness,
                                          q=pca_q, niter=pca_niter)
        colors = color_t.cpu().numpy()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write_ply_with_colors(coords[valid], colors, str(output_path))

        if verbose:
            print(f"  Saved: {output_path}  ({n_valid} pts)")
        return str(output_path)

    except Exception as e:
        print(f"  PCA visualization error: {e}")
        import traceback; traceback.print_exc()
        return None


# ============================================================================
# MODE 2 — SCENE-LEVEL Z_S SPACE (z_s InfoNCE projections)
# ============================================================================

# Heuristic RGB colors for the 8 most common ScanNet72 super-categories.
# Used to color-code scenes in the z_s PLY by dominant category.
_CATEGORY_COLORS = {
    # floor/wall/ceiling  (structural)
    0:  [0.55, 0.55, 0.55],   # wall
    1:  [0.70, 0.70, 0.65],   # floor
    2:  [0.80, 0.78, 0.72],   # cabinet
    # furniture           (warm)
    3:  [0.90, 0.55, 0.20],   # bed
    4:  [0.85, 0.35, 0.20],   # chair
    5:  [0.80, 0.40, 0.25],   # sofa
    6:  [0.75, 0.50, 0.15],   # table
    7:  [0.70, 0.45, 0.10],   # door
    # display / tech      (blue)
    8:  [0.20, 0.40, 0.80],   # window
    9:  [0.15, 0.35, 0.75],   # bookshelf
    10: [0.10, 0.30, 0.70],   # picture
    11: [0.25, 0.55, 0.85],   # counter
    # bathroom            (cyan)
    12: [0.20, 0.70, 0.70],   # blinds
    14: [0.30, 0.75, 0.75],   # sink
    15: [0.25, 0.65, 0.65],   # bathtub
    # misc / default      (purple)
}
_DEFAULT_COLOR = [0.60, 0.25, 0.70]


def _dominant_category_color(label_dist_row):
    """Return RGB color for the dominant ScanNet72 category in label_dist."""
    dom = int(np.argmax(label_dist_row))
    return _CATEGORY_COLORS.get(dom, _DEFAULT_COLOR)


def visualize_z_s_space(z_s_proj, label_dists, output_path,
                         brightness=1.0, device='cpu', verbose=True):
    """
    Visualize the z_s latent space as a scene-scatter PLY.

    Each point = one scene.  Position is the PCA projection of z_s_proj to 3D.
    Color is derived from the dominant ScanNet72 category in label_dist.

    If z_s InfoNCE is working:
      — scenes with similar dominant categories should cluster spatially.
      — the PLY will show colored clusters in SuperSplat when scaled up.

    Args:
        z_s_proj:    [M, D_proj]  numpy — L2-normalized z_s projections
                                  (from SemanticTokenInfoNCEHead, D_proj=128)
        label_dists: [M, 72]      numpy — per-scene label distributions
        output_path: str          — output PLY path
        brightness:  float        — brightness multiplier (default 1.0)

    Returns:
        output_path if successful, None otherwise.

    Diagnostic:
        Open z_s_space_epoch_NNN.ply in SuperSplat.  Use a large splat scale
        (e.g. 0.5m) to make the 50 scene-points visible.  Same-category scenes
        should cluster — e.g. all apartment scenes in one corner.
        Compare epochs to see if clustering improves with training.
    """
    try:
        M = z_s_proj.shape[0]
        if M < 3:
            print(f"  z_s visualization: need ≥3 scenes (got {M}) — skipping.")
            return None

        if verbose:
            print(f"\n{'='*60}")
            print(f"Z_S SPACE VISUALIZATION — {M} scenes, "
                  f"proj_dim={z_s_proj.shape[1]}")
            print(f"{'='*60}")

        # ── PCA: project z_s_proj [M, D] → 3D positions ───────────────────
        feat_t = torch.from_numpy(z_s_proj.astype(np.float32)).to(device)
        with torch.no_grad():
            q_actual = min(6, feat_t.shape[0] - 1, feat_t.shape[1])
            u, s, v  = torch.pca_lowrank(feat_t, q=q_actual, center=True, niter=10)
            proj3d   = (feat_t @ v[:, :3]).cpu().numpy()          # [M, 3]

        # Scale to ≈10m range so it's visible at normal scene scale
        rng  = proj3d.max(0) - proj3d.min(0)
        rng  = np.where(rng < 1e-6, 1.0, rng)
        proj3d = (proj3d - proj3d.min(0)) / rng * 8.0 - 4.0       # [−4, +4]^3

        # ── Colors from dominant ScanNet72 category ────────────────────────
        colors = np.array(
            [_dominant_category_color(label_dists[i]) for i in range(M)],
            dtype=np.float32)                                      # [M, 3]
        colors = np.clip(colors * brightness, 0.0, 1.0)

        # ── Each scene rendered as a "fat Gaussian" (uniform scale) ────────
        # We write the scene point with scale=0.3m so it's visible in SuperSplat
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write_ply_with_colors(proj3d, colors, str(output_path))

        if verbose:
            dom_cats = np.argmax(label_dists, axis=1)
            unique_c = len(np.unique(dom_cats))
            print(f"  Saved: {output_path}")
            print(f"  Scenes: {M}  |  Unique dominant cats: {unique_c}")
            print(f"  Position range: [{proj3d.min():.2f}, {proj3d.max():.2f}]m")
            print(f"  To view: open in SuperSplat, increase splat scale to ≈0.5m")
        return str(output_path)

    except Exception as e:
        print(f"  z_s visualization error: {e}")
        import traceback; traceback.print_exc()
        return None


# ============================================================================
# COMBINED COMPARISON UTILITY
# ============================================================================

def visualize_comparison(coords, semantic_features, positions, colors,
                          output_dir, scene_name='scene', brightness=1.25):
    """
    Write 3 PLY files for visual comparison (no Open3D):
      1. PCA of semantic projection features
      2. PCA of position features (baseline)
      3. Original colors

    Args:
        coords:            [N, 3]  3D positions
        semantic_features: [N, D]  semantic projection features
        positions:         [N, 3]  position features (baseline)
        colors:            [N, 3]  original RGB [0,1]
        output_dir:        Path
        scene_name:        str
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results    = {}

    print("Generating semantic feature visualization...")
    sp = visualize_semantic_features(
        coords=coords, features=semantic_features,
        output_path=output_dir / f"{scene_name}_semantic_pca.ply",
        brightness=brightness)
    results['semantic'] = sp

    print("Generating position baseline visualization...")
    pp = visualize_semantic_features(
        coords=coords, features=positions,
        output_path=output_dir / f"{scene_name}_position_pca.ply",
        brightness=brightness)
    results['position'] = pp

    print("Saving original colors...")
    op = output_dir / f"{scene_name}_original_colors.ply"
    write_ply_with_colors(coords, colors, str(op))
    results['original'] = str(op)
    print(f"  Saved: {op}")

    return results


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing PCA visualization...")

    # Per-Gaussian test
    N      = 5000
    coords = np.random.randn(N, 3).astype(np.float32)
    feats  = np.random.randn(N, 32).astype(np.float32)
    for i in range(3):
        feats[i * N//3 : (i+1) * N//3] += np.random.randn(32) * 5
    path = visualize_semantic_features(coords, feats, "/tmp/test_per_gaussian.ply")
    print(f"Per-Gaussian PLY: {path}")

    # z_s space test
    M       = 30
    z_s     = np.random.randn(M, 128).astype(np.float32)
    z_s    /= np.linalg.norm(z_s, axis=1, keepdims=True) + 1e-8
    ld      = np.zeros((M, 72), dtype=np.float32)
    for i in range(M):
        cat = i % 5
        ld[i, cat * 5:(cat + 1) * 5] = np.random.dirichlet(np.ones(5))
    path = visualize_z_s_space(z_s, ld, "/tmp/test_z_s_space.ply")
    print(f"z_s space PLY:    {path}")
    print("All tests passed.")