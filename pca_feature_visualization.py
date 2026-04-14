"""
PCA Feature Visualization for Can3Tok Semantic Diagnostics
============================================================

THREE VISUALIZATION MODES — all produce PLY files viewable in SuperSplat:

1. Per-Gaussian semantic features  (decoder hidden state → SemanticProjectionHead)
   → visualize_semantic_features(coords, features, output_path)
   → N points (40k Gaussians per scene), D-dim features → PCA colors
   → Diagnostic: do same-category Gaussians cluster in feature space?
   → Output: scene{i}_semantic_infonce.ply

2. z_s token space  (z_s projections from SemanticTokenInfoNCEHead)
   → visualize_z_s_space(z_s_proj, label_dists, output_path)
   → M points (one per eval scene), 128-dim projections → PCA positions
   → Colors: dominant ScanNet72 category per scene
   → Diagnostic: do same-category scenes cluster in z_s?
   → Output: z_s_space_epoch_NNN.ply

3. z_s token visualization  (NEW — same style as #1 but for the 16 z_s tokens)
   → visualize_zs_tokens(zs_tokens, label_dists, output_path)
   → B×16 points (one per token per scene), D-dim raw token features → PCA positions
   → Colors: dominant ScanNet72 category of the scene each token belongs to
   → Diagnostic: if z_s token InfoNCE is working, same-category tokens cluster.
                 Directly analogous to per-Gaussian PCA — compare the two PLYs.
   → Output: zs_tokens_epoch_NNN.ply

COMPARISON:
  Per-Gaussian PLY:  40,000 points, per-Gaussian category labels (fine-grained)
  z_s token PLY:         B×16 points, scene dominant-category labels (coarse)
  Both use the same PCA coloring pipeline — visually comparable side by side.
"""

import numpy as np
import torch
from pathlib import Path

try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False
    print("Warning: plyfile not found — using manual PLY writer")


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def get_pca_color_torch(feat, brightness=1.25, center=True, q=6, niter=5):
    """PCA-based colorization. Returns [N, 3] float32 in [0, 1]."""
    n, d     = feat.shape
    q_actual = min(q, d, n)
    u, s, v  = torch.pca_lowrank(feat, center=center, q=q_actual, niter=niter)
    proj     = feat @ v
    mix = proj[:, :3] * 0.6 + proj[:, 3:6] * 0.4 if q_actual >= 6 else (
          proj[:, :3] if q_actual >= 3 else
          torch.zeros((n, 3), dtype=feat.dtype, device=feat.device))
    mn  = mix.amin(0, keepdim=True)
    mx  = mix.amax(0, keepdim=True)
    mix = (mix - mn) / (mx - mn + 1e-6)
    return (mix * brightness).clamp(0.0, 1.0)


def get_pca_positions_torch(feat, center=True, q=3, niter=10):
    """PCA → 3D positions. Returns [N, 3] float32 normalized to [−4, +4]^3."""
    n, d     = feat.shape
    q_actual = min(q, d, n - 1)
    if q_actual < 3:
        return torch.zeros(n, 3, dtype=torch.float32, device=feat.device)
    u, s, v  = torch.pca_lowrank(feat, center=center, q=q_actual, niter=niter)
    proj3d   = (feat @ v[:, :3]).cpu().float().numpy()
    rng      = proj3d.max(0) - proj3d.min(0)
    rng      = np.where(rng < 1e-6, 1.0, rng)
    proj3d   = (proj3d - proj3d.min(0)) / rng * 8.0 - 4.0
    return proj3d


def build_valid_mask(feat, norm_thresh=0.0):
    finite = np.isfinite(feat).all(axis=1)
    chunk  = max(1, 1_000_000 // max(1, feat.shape[1]))
    norms  = np.empty(feat.shape[0], dtype=np.float32)
    for i in range(0, feat.shape[0], chunk):
        sl = slice(i, min(i + chunk, feat.shape[0]))
        norms[sl] = np.linalg.norm(feat[sl].astype(np.float32, copy=False), axis=1)
    return finite & (norms > norm_thresh), norms


def write_ply_with_colors(coords, colors, output_path):
    n         = len(coords)
    colors_u8 = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    if HAS_PLYFILE:
        vdtype = [('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')]
        verts  = np.empty(n, dtype=vdtype)
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
# SHARED CATEGORY COLOR PALETTE (ScanNet72 super-categories)
# ============================================================================

_CATEGORY_COLORS = {
    0:  [0.55, 0.55, 0.55],   # wall
    1:  [0.70, 0.70, 0.65],   # floor
    2:  [0.80, 0.78, 0.72],   # cabinet
    3:  [0.90, 0.55, 0.20],   # bed
    4:  [0.85, 0.35, 0.20],   # chair
    5:  [0.80, 0.40, 0.25],   # sofa
    6:  [0.75, 0.50, 0.15],   # table
    7:  [0.70, 0.45, 0.10],   # door
    8:  [0.20, 0.40, 0.80],   # window
    9:  [0.15, 0.35, 0.75],   # bookshelf
    10: [0.10, 0.30, 0.70],   # picture
    11: [0.25, 0.55, 0.85],   # counter
    12: [0.20, 0.70, 0.70],   # blinds
    14: [0.30, 0.75, 0.75],   # sink
    15: [0.25, 0.65, 0.65],   # bathtub
}
_DEFAULT_COLOR = [0.60, 0.25, 0.70]

def _category_color(cat_id):
    return _CATEGORY_COLORS.get(int(cat_id), _DEFAULT_COLOR)


# ============================================================================
# MODE 1 — PER-GAUSSIAN SEMANTIC FEATURES
# ============================================================================

def visualize_semantic_features(coords, features, output_path, brightness=1.25,
                                 pca_q=6, pca_niter=5, device='cpu', verbose=True):
    """
    PCA-based coloring of per-Gaussian semantic features.

    coords:   [N, 3]  numpy — Gaussian 3D positions
    features: [N, D]  numpy — per-Gaussian projection features (e.g. D=32)
    """
    try:
        if verbose:
            print(f"\nPCA VISUALIZATION — per-Gaussian ({features.shape})")
        valid, _ = build_valid_mask(features)
        n_valid  = int(valid.sum())
        if n_valid == 0:
            if verbose: print("  No valid features — skipping.")
            return None
        feat_t  = torch.from_numpy(features[valid]).float().to(device)
        with torch.no_grad():
            color_t = get_pca_color_torch(feat_t, brightness=brightness,
                                          q=pca_q, niter=pca_niter)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write_ply_with_colors(coords[valid], color_t.cpu().numpy(), str(output_path))
        if verbose: print(f"  Saved: {output_path}  ({n_valid} pts)")
        return str(output_path)
    except Exception as e:
        print(f"  PCA vis error: {e}")
        return None


# ============================================================================
# MODE 2 — SCENE-LEVEL Z_S SPACE (z_s projection head output)
# ============================================================================

def visualize_z_s_space(z_s_proj, label_dists, output_path, brightness=1.0,
                         device='cpu', verbose=True):
    """
    Scene-scatter PLY: one point per eval scene, PCA positions, dominant-category colors.

    z_s_proj:    [M, D_proj]  L2-normalised z_s projections (SemanticTokenInfoNCEHead)
    label_dists: [M, 72]      per-scene label distributions
    """
    try:
        M = z_s_proj.shape[0]
        if M < 3:
            if verbose: print(f"  z_s vis: need ≥3 scenes (got {M}) — skipping.")
            return None
        if verbose:
            print(f"\nZ_S SPACE VIS — {M} scenes, proj_dim={z_s_proj.shape[1]}")
        feat_t  = torch.from_numpy(z_s_proj.astype(np.float32)).to(device)
        with torch.no_grad():
            proj3d = get_pca_positions_torch(feat_t)
        colors = np.array([_category_color(np.argmax(label_dists[i])) for i in range(M)],
                          dtype=np.float32)
        colors = np.clip(colors * brightness, 0.0, 1.0)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write_ply_with_colors(proj3d, colors, str(output_path))
        if verbose:
            print(f"  Saved: {output_path}  ({M} scenes)")
            print(f"  Open in SuperSplat with splat scale ~0.5m to see the {M} points")
        return str(output_path)
    except Exception as e:
        print(f"  z_s vis error: {e}")
        return None


# ============================================================================
# MODE 3 — Z_S TOKEN VISUALIZATION  (NEW — same style as per-Gaussian)
# ============================================================================

def visualize_zs_tokens(zs_tokens, label_dists, output_path,
                          brightness=1.25, device='cpu', verbose=True):
    """
    PCA visualization of the 16 z_s tokens — directly analogous to per-Gaussian PLY.

    Each point = one token from one scene.
    Total points = B × 16.
    Position = PCA of the B×16 token feature vectors (L2-normalised) → 3D.
    Color    = dominant ScanNet72 category of the scene that token belongs to.

    HOW TO READ:
      If z_s token InfoNCE is working correctly:
        — tokens from bedroom scenes should cluster in one region of the PCA space
        — tokens from office scenes should cluster in another region
        — each cluster has 16 points (one per token), tightly grouped
      Compare with per-Gaussian PCA:
        — per-Gaussian: 40k points per scene, fine-grained category colors
        — z_s token:    16 points per scene, coarse scene-type colors
      Both should show clustering if the respective InfoNCE losses are converging.

    Args:
        zs_tokens:   [B, T, D]  raw z_s tokens from Z[:, :T, :] (L2-norm applied inside)
        label_dists: [B, 72]    per-scene category distributions
        output_path: str        output PLY path
    """
    try:
        B, T, D = zs_tokens.shape
        if B < 2:
            if verbose: print(f"  z_s token vis: need ≥2 scenes — skipping.")
            return None

        if verbose:
            print(f"\nZ_S TOKEN VIS — {B} scenes × {T} tokens = {B*T} points, D={D}")

        # ── L2-normalise and flatten ──────────────────────────────────────────
        feat_np = zs_tokens.reshape(B * T, D).astype(np.float32)
        valid, _= build_valid_mask(feat_np)
        n_valid  = int(valid.sum())
        if n_valid == 0:
            if verbose: print("  No valid token features — skipping.")
            return None

        # ── PCA → 3D positions ────────────────────────────────────────────────
        feat_t = torch.from_numpy(feat_np[valid]).float().to(device)
        feat_t = torch.nn.functional.normalize(feat_t, p=2, dim=-1)
        with torch.no_grad():
            proj3d = get_pca_positions_torch(feat_t, niter=15)   # [n_valid, 3]

        # ── Colors by dominant category of parent scene ───────────────────────
        # Each scene b contributes T consecutive points; all get scene b's color
        dom_cats = np.argmax(label_dists, axis=1)   # [B]
        colors_all = np.array(
            [_category_color(dom_cats[b]) for b in range(B) for _ in range(T)],
            dtype=np.float32)                        # [B*T, 3]
        colors_valid = colors_all[valid]
        colors_valid = np.clip(colors_valid * brightness, 0.0, 1.0)

        # ── Write PLY ─────────────────────────────────────────────────────────
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write_ply_with_colors(proj3d, colors_valid, str(output_path))

        if verbose:
            dom_unique = len(np.unique(dom_cats))
            print(f"  Saved: {output_path}")
            print(f"  {n_valid} points ({B} scenes × {T} tokens)")
            print(f"  Unique dominant categories: {dom_unique}")
            print(f"  If InfoNCE working: same-color points should cluster spatially")
            print(f"  Compare with scene{{i}}_semantic_infonce.ply for per-Gaussian analogue")

        return str(output_path)

    except Exception as e:
        print(f"  z_s token vis error: {e}")
        import traceback; traceback.print_exc()
        return None


# ============================================================================
# COMPARISON UTILITY
# ============================================================================

def visualize_comparison(coords, semantic_features, positions, colors,
                          output_dir, scene_name='scene', brightness=1.25):
    """Write 3 PLYs for visual comparison: semantic PCA, position PCA, original colors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
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
    op = output_dir / f"{scene_name}_original_colors.ply"
    write_ply_with_colors(coords, colors, str(op))
    results['original'] = str(op)
    return results


# ============================================================================
# QUICK TEST
# ============================================================================
if __name__ == "__main__":
    print("Testing all three visualization modes...")

    # Mode 1: per-Gaussian
    N      = 5000
    coords = np.random.randn(N, 3).astype(np.float32)
    feats  = np.random.randn(N, 32).astype(np.float32)
    for i in range(3):
        feats[i*N//3:(i+1)*N//3] += np.random.randn(32) * 5
    p = visualize_semantic_features(coords, feats, "/tmp/test_per_gaussian.ply")
    print(f"Mode 1 (per-Gaussian): {p}")

    # Mode 2: z_s space
    M   = 30
    z_s = np.random.randn(M, 128).astype(np.float32)
    z_s /= np.linalg.norm(z_s, axis=1, keepdims=True) + 1e-8
    ld  = np.zeros((M, 72), dtype=np.float32)
    for i in range(M):
        cat = i % 5
        ld[i, cat*5:(cat+1)*5] = np.random.dirichlet(np.ones(5))
    p = visualize_z_s_space(z_s, ld, "/tmp/test_z_s_space.ply")
    print(f"Mode 2 (z_s space):    {p}")

    # Mode 3: z_s tokens  (NEW)
    B, T, D = 20, 16, 32
    tokens  = np.random.randn(B, T, D).astype(np.float32)
    # Cluster: scenes 0-9 are "bedroom", scenes 10-19 are "office"
    tokens[:10]  += np.random.randn(D) * 3
    tokens[10:]  += np.random.randn(D) * 3
    ld2 = np.zeros((B, 72), dtype=np.float32)
    ld2[:10, 3] = 0.8   # bed dominant (cat 3)
    ld2[10:, 6] = 0.8   # table dominant (cat 6)
    p = visualize_zs_tokens(tokens, ld2, "/tmp/test_zs_tokens.ply")
    print(f"Mode 3 (z_s tokens):   {p}")

    print("\nAll tests passed. If clustering is present, it will be visible in SuperSplat.")