"""
PCA Feature Visualization for Semantic Awareness (NO Open3D!)
=============================================================

Simplified version using only numpy and plyfile (already installed).
No Open3D dependency needed!

Usage:
    from pca_feature_visualization import visualize_semantic_features
    
    visualize_semantic_features(
        coords=coords,           # [N, 3] positions
        features=semantic_feats, # [N, 32] semantic features
        output_path="semantic_vis.ply",
        brightness=1.25
    )
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple

# Try to import plyfile (you should have this already)
try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False
    print("⚠️  Warning: plyfile not found, will write PLY manually")


def get_pca_color_torch(
    feat: torch.Tensor,
    brightness: float = 1.25,
    center: bool = True,
    q: int = 6,
    niter: int = 5
) -> torch.Tensor:
    """
    Torch low-rank PCA colorization.
    
    Args:
        feat: (N, D) float32 tensor of features
        brightness: Brightness multiplier (default 1.25)
        center: Whether to center features before PCA
        q: Number of principal components to compute
        niter: Number of iterations for power iteration
    
    Returns:
        color: (N, 3) float32 in [0,1] - RGB colors
    """
    # Adjust q if feature dimension is too small
    n, d = feat.shape
    q_actual = min(q, d, n)  # Can't have more PCs than dimensions or samples
    
    # u: (N, q_actual), s: (q_actual,), v: (D, q_actual)
    u, s, v = torch.pca_lowrank(feat, center=center, q=q_actual, niter=niter)
    
    # Project features onto principal components
    projection = feat @ v  # (N, q_actual)
    
    # Handle different cases based on available PCs
    if q_actual >= 6:
        # Standard case: blend first 3 and next 3 PCs
        mix = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4  # (N, 3)
    elif q_actual >= 3:
        # Only have 3-5 PCs: use first 3 only
        mix = projection[:, :3]  # (N, 3)
    else:
        # Less than 3 PCs: pad with zeros
        mix = torch.zeros((n, 3), dtype=feat.dtype, device=feat.device)
        mix[:, :q_actual] = projection
    
    # Per-channel min-max normalization to [0,1]
    min_val = mix.amin(dim=0, keepdim=True)
    max_val = mix.amax(dim=0, keepdim=True)
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (mix - min_val) / div
    
    # Apply brightness and clamp
    color = (color * brightness).clamp_(0.0, 1.0)
    
    return color


def build_valid_mask(feat: np.ndarray, norm_thresh: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build mask for valid feature vectors.
    
    Args:
        feat: (N, D) feature array
        norm_thresh: Minimum L2 norm threshold (default 0.0)
    
    Returns:
        valid_mask: (N,) boolean mask
        norms: (N,) L2 norms
    """
    # Finite mask (no NaN/Inf)
    finite_mask = np.isfinite(feat).all(axis=1)
    
    # L2 norm in chunks (memory efficient)
    n, c = feat.shape
    chunk = max(1, 1_000_000 // max(1, c))
    norms = np.empty(n, dtype=np.float32)
    
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        norms[sl] = np.linalg.norm(feat[sl].astype(np.float32, copy=False), axis=1)
    
    # Norm mask (above threshold)
    norm_mask = norms > norm_thresh
    
    valid = finite_mask & norm_mask
    return valid, norms


def write_ply_with_colors(coords: np.ndarray, colors: np.ndarray, output_path: str):
    """
    Write PLY file with positions and colors.
    
    Uses plyfile if available, otherwise writes manually.
    
    Args:
        coords: (N, 3) positions
        colors: (N, 3) RGB colors in [0, 1]
        output_path: Path to save PLY
    """
    n = len(coords)
    
    # Convert colors to 0-255 range
    colors_255 = (colors * 255).astype(np.uint8)
    
    if HAS_PLYFILE:
        # Use plyfile library (preferred)
        vertex_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
        
        vertices = np.empty(n, dtype=vertex_dtype)
        vertices['x'] = coords[:, 0]
        vertices['y'] = coords[:, 1]
        vertices['z'] = coords[:, 2]
        vertices['red'] = colors_255[:, 0]
        vertices['green'] = colors_255[:, 1]
        vertices['blue'] = colors_255[:, 2]
        
        el = PlyElement.describe(vertices, 'vertex')
        PlyData([el], text=False).write(output_path)
        
    else:
        # Write PLY manually (fallback)
        with open(output_path, 'wb') as f:
            # Header
            header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
            f.write(header.encode('ascii'))
            
            # Vertex data
            for i in range(n):
                # Write position (3 floats)
                f.write(coords[i].astype(np.float32).tobytes())
                # Write color (3 unsigned bytes)
                f.write(colors_255[i].tobytes())


def visualize_semantic_features(
    coords: np.ndarray,
    features: np.ndarray,
    output_path: str,
    brightness: float = 1.25,
    pca_q: int = 6,
    pca_niter: int = 5,
    device: str = "cpu",
    verbose: bool = True
) -> Optional[str]:
    """
    Visualize semantic features using PCA-based coloring.
    
    NO Open3D required! Uses plyfile or manual PLY writing.
    
    Args:
        coords: (N, 3) numpy array of 3D positions
        features: (N, D) numpy array of semantic features
        output_path: Path to save PLY file
        brightness: Brightness multiplier for colors
        pca_q: Number of principal components
        pca_niter: Number of iterations for PCA
        device: Torch device ('cpu' or 'cuda')
        verbose: Print progress messages
    
    Returns:
        output_path if successful, None if failed
    """
    try:
        if verbose:
            print(f"\n{'='*70}")
            print("PCA FEATURE VISUALIZATION")
            print(f"{'='*70}")
            print(f"Input: {features.shape[0]} points, {features.shape[1]}-dim features")
        
        # Filter invalid features
        valid_mask, norms = build_valid_mask(features, norm_thresh=0.0)
        n_valid = int(valid_mask.sum())
        n_total = features.shape[0]
        
        if verbose:
            print(f"Valid features: {n_valid}/{n_total} ({n_valid/n_total*100:.2f}%)")
        
        if n_valid == 0:
            print("⚠️  No valid features! Skipping visualization.")
            return None
        
        # Filter features and coords
        feat_valid = features[valid_mask]
        coords_valid = coords[valid_mask]
        
        # Convert to torch
        if verbose:
            print("Computing PCA colors...")
        
        feat_t = torch.from_numpy(feat_valid).to(torch.float32).to(device)
        
        with torch.no_grad():
            color_t = get_pca_color_torch(
                feat_t,
                brightness=brightness,
                center=True,
                q=pca_q,
                niter=pca_niter
            )
        
        colors = color_t.cpu().numpy().astype(np.float32)  # (N, 3) in [0,1]
        
        # Write PLY file (NO Open3D!)
        if verbose:
            print("Writing PLY file...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        write_ply_with_colors(coords_valid, colors, str(output_path))
        
        if verbose:
            print(f"✓ Saved PCA visualization: {output_path}")
            print(f"  Points: {len(coords_valid)}")
            print(f"  Color range: R[{colors[:,0].min():.3f}, {colors[:,0].max():.3f}] "
                  f"G[{colors[:,1].min():.3f}, {colors[:,1].max():.3f}] "
                  f"B[{colors[:,2].min():.3f}, {colors[:,2].max():.3f}]")
            print(f"{'='*70}\n")
        
        return str(output_path)
        
    except Exception as e:
        print(f"⚠️  Error during PCA visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_comparison(
    coords: np.ndarray,
    semantic_features: np.ndarray,
    positions: np.ndarray,
    colors: np.ndarray,
    output_dir: Path,
    scene_name: str = "scene",
    brightness: float = 1.25
) -> dict:
    """
    Create comparison visualizations (NO Open3D!):
    1. Semantic features (PCA colored)
    2. Position features (PCA colored)
    3. Original colors
    
    Args:
        coords: (N, 3) 3D positions
        semantic_features: (N, D) semantic features
        positions: (N, 3) position features (for baseline)
        colors: (N, 3) original RGB colors [0,1]
        output_dir: Directory to save visualizations
        scene_name: Name for output files
        brightness: Brightness multiplier
    
    Returns:
        dict with paths to saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Semantic features
    print("Generating semantic feature visualization...")
    semantic_path = visualize_semantic_features(
        coords=coords,
        features=semantic_features,
        output_path=output_dir / f"{scene_name}_semantic_pca.ply",
        brightness=brightness,
        verbose=True
    )
    results['semantic'] = semantic_path
    
    # 2. Position baseline
    print("Generating position baseline visualization...")
    position_path = visualize_semantic_features(
        coords=coords,
        features=positions,
        output_path=output_dir / f"{scene_name}_position_pca.ply",
        brightness=brightness,
        verbose=True
    )
    results['position'] = position_path
    
    # 3. Original colors
    print("Saving original colors...")
    original_path = output_dir / f"{scene_name}_original_colors.ply"
    write_ply_with_colors(coords, colors, str(original_path))
    results['original'] = str(original_path)
    print(f"✓ Saved original colors: {original_path}\n")
    
    return results


# Test function
if __name__ == "__main__":
    print("Testing PCA feature visualization (NO Open3D!)...")
    
    # Create synthetic data
    n_points = 10000
    coords = np.random.randn(n_points, 3).astype(np.float32)
    
    # Create features with some structure (3 clusters)
    features = np.random.randn(n_points, 32).astype(np.float32) * 0.5
    cluster_ids = np.random.randint(0, 3, n_points)
    for i in range(3):
        mask = cluster_ids == i
        features[mask] += np.random.randn(32) * 5  # Add cluster offset
    
    # Visualize
    output_path = visualize_semantic_features(
        coords=coords,
        features=features,
        output_path="test_pca_vis.ply",
        brightness=1.25
    )
    
    if output_path:
        print(f"✓ Test successful! Saved to: {output_path}")
        print("\n✓ NO Open3D needed! Uses plyfile or manual PLY writing.")
    else:
        print("✗ Test failed!")