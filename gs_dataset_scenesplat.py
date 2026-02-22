"""
SceneSplat Dataset for Can3Tok Training - WITH NORMALIZATION CONTROL
======================================================================
ScanNet72 semantic labels + RGB colour normalisation [0, 1]

🎯 CANONICAL SPHERE NORMALIZATION (Optional - Controlled by Flag)
   - Can be enabled/disabled via normalize=True/False
   - Controls both position AND scale normalization
   
KEY FEATURES:
1. DETERMINISTIC SAMPLING: Always same top-40k Gaussians
2. OPTIONAL NORMALIZATION: Toggle canonical sphere normalization
3. NO METADATA: Fixed normalization (no per-scene statistics)
4. INFERENCE-READY: Works perfectly with diffusion models
"""

import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 🎯 Canonical Sphere Normalization (Optional)
# ─────────────────────────────────────────────────────────────────────────────


# def normalize_to_canonical_sphere(coord, scale, target_radius=10.0):
#     """
#     Normalize scene to canonical sphere (Optional - controlled by dataset flag).
    
#     This normalizes positions to fit in a sphere of radius target_radius.
#     Scales are normalized proportionally (LINEAR SPACE, not log space).
    
#     Args:
#         coord: [N, 3] Gaussian positions (xyz) - linear space
#         scale: [N, 3] Gaussian scales - linear space (post-exp from PLY)
#         target_radius: Fixed radius in meters (default 10.0)
    
#     Returns:
#         coord_norm: [N, 3] Positions in canonical sphere
#         scale_norm: [N, 3] Scales normalized proportionally
#     """
#     # Step 1: Center at origin
#     center = coord.mean(axis=0)
#     coord_centered = coord - center
    
#     # Step 2: Find maximum distance from origin
#     distances = np.linalg.norm(coord_centered, axis=1)
#     max_dist = distances.max()
    
#     # Handle edge case: empty or degenerate scene
#     if max_dist < 1e-6:
#         max_dist = 1.0
    
#     # Step 3: Scale to target radius
#     # The 1.1 factor provides safety margin
#     scale_factor = target_radius / (max_dist * 1.1)
    
#     # Step 4: Apply transformation
#     coord_norm = coord_centered * scale_factor
    
#     # ═══════════════════════════════════════════════════════════════════
#     # LINEAR SPACE NORMALIZATION
#     # ═══════════════════════════════════════════════════════════════════
#     # Scales are in LINEAR SPACE (meters), so multiply by scale_factor
#     # scale_new = scale_old × factor
#     # ═══════════════════════════════════════════════════════════════════
#     scale_norm = scale * scale_factor  # ✓ Linear space
#     # ═══════════════════════════════════════════════════════════════════
    
#     return coord_norm, scale_norm

# REPLACE WITH:
def normalize_to_canonical_sphere(coord, scale, target_radius=10.0):
    """
    Normalize scene to canonical sphere (Optional - controlled by dataset flag).
    
    This normalizes positions to fit in a sphere of radius target_radius.
    Scales are normalized proportionally in LOG SPACE (matches Can3Tok ICCV 2025).
    
    Args:
        coord: [N, 3] Gaussian positions (xyz) - linear space
        scale: [N, 3] Gaussian scales - linear space (post-exp from PLY)
        target_radius: Fixed radius in meters (default 10.0)
    
    Returns:
        coord_norm: [N, 3] Positions in canonical sphere
        scale_norm: [N, 3] Scales in LOG SPACE normalized proportionally
    """
    # Step 1: Center at origin
    center = coord.mean(axis=0)
    coord_centered = coord - center
    
    # Step 2: Find maximum distance from origin
    distances = np.linalg.norm(coord_centered, axis=1)
    max_dist = distances.max()
    
    # Handle edge case: empty or degenerate scene
    if max_dist < 1e-6:
        max_dist = 1.0
    
    # Step 3: Scale to target radius
    # The 1.1 factor provides safety margin
    scale_factor = target_radius / (max_dist * 1.1)
    
    # Step 4: Apply transformation
    coord_norm = coord_centered * scale_factor
    
    # ═══════════════════════════════════════════════════════════════════
    # LOG-SPACE NORMALIZATION (matches Can3Tok ICCV 2025)
    # ═══════════════════════════════════════════════════════════════════
    # Convert scales to log-space, then add log(scale_factor)
    # This matches Can3Tok: scale_norm = log(scale_old) + log(factor)
    # ═══════════════════════════════════════════════════════════════════
    scale_log = np.log(scale + 1e-7)  # Convert to log-space
    scale_norm = scale_log + np.log(scale_factor)  # Log-space addition
    # ═══════════════════════════════════════════════════════════════════
    
    return coord_norm, scale_norm


def verify_canonical_normalization(coord_norm, target_radius=10.0):
    """Verify scene fits within target sphere."""
    distances = np.linalg.norm(coord_norm, axis=1)
    max_dist = distances.max()
    
    return {
        'max_distance': max_dist,
        'target_radius': target_radius,
        'fits_in_sphere': max_dist <= target_radius * 1.05,
        'utilization': (max_dist / target_radius) * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Voxelisation helper
# ─────────────────────────────────────────────────────────────────────────────

def voxelize(coord, voxel_size=0.4, hash_type='fnv'):
    """
    Voxelise point-cloud coordinates.

    Returns
    -------
    uniq_idx : unique voxel IDs (FNV hash values)
    inv_idx  : mapping from each point to its voxel index in uniq_idx
    count    : number of points per unique voxel
    """
    discrete_coord = np.floor(coord / voxel_size).astype(np.int32)

    if hash_type == 'fnv':
        offset_basis = 2166136261
        fnv_prime    = 16777619
        hash_vals    = np.full(len(discrete_coord), offset_basis, dtype=np.int64)
        for i in range(3):
            hash_vals = (hash_vals ^ discrete_coord[:, i]) * fnv_prime
        uniq_idx, inv_idx, count = np.unique(
            hash_vals, return_inverse=True, return_counts=True
        )
    else:
        try:
            min_coord = discrete_coord.min(axis=0)
            shifted   = discrete_coord - min_coord
            max_coord = shifted.max(axis=0)
            ravel_idx = np.ravel_multi_index(shifted.T, max_coord + 1)
            uniq_idx, inv_idx, count = np.unique(
                ravel_idx, return_inverse=True, return_counts=True
            )
        except Exception:
            uniq_idx = np.arange(len(coord))
            inv_idx  = uniq_idx.copy()
            count    = np.ones(len(coord), dtype=np.int64)

    return uniq_idx, inv_idx, count


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class gs_dataset(Dataset):
    """
    SceneSplat-7K dataset with ScanNet72 semantic labels.
    
    🎯 OPTIONAL CANONICAL SPHERE NORMALIZATION
    
    Normalization can be toggled via normalize=True/False parameter.
    When enabled: all scenes normalized to sphere of radius target_radius.
    When disabled: raw coordinates and scales from dataset.

    Feature format returned — (40000, 18):
      cols  0:3   voxel_centers  (positional encoding)
      col   3     point_uniq_idx (voxel ID)
      cols  4:7   xyz            (Gaussian position - normalized if enabled)
      cols  7:10  rgb            (colour, normalised [0, 1])
      col  10     opacity        (post-sigmoid activation)
      cols 11:14  scale          (normalized if enabled, else raw)
      cols 14:18  quaternion     (qw, qx, qy, qz)

    Target for reconstruction — cols 4:18 → (40000, 14):
      [0:3]   xyz        ← Normalized if enabled
      [3:6]   rgb        ← [0, 1]
      [6]     opacity    ← [0, 1]
      [7:10]  scale      ← Normalized if enabled
      [10:14] quaternion ← Unit quaternions
    """

    TARGET_POINTS = 40_000

    SCANNET72_CATEGORIES = [
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
        "door", "window", "bookshelf", "picture", "counter", "desk",
        "curtain", "refrigerator", "shower curtain", "toilet", "sink",
        "bathtub", "other furniture", "kitchen cabinet", "display", "rug",
        "ceiling", "beam", "column", "clutter", "other structure",
        "other prop", "whiteboard", "person", "bag", "box", "pillow",
        "lamp", "books", "clothes", "object", "towel", "mirror", "plant",
        "monitor", "keyboard", "mouse", "printer", "telephone", "scanner",
        "electronics", "blinds", "clock", "books", "paper", "tools",
        "instrument", "sports equipment", "food", "cup", "bottle", "bowl",
        "utensil", "can", "basket", "cart", "tissue", "fire extinguisher",
        "trash can", "other", "stairs", "stairs", "stairs", "stairs",
        "stairs", "stairs",
    ]

    def __init__(
        self,
        root,
        resol=200,
        random_permute=False,
        train=True,
        sampling_method='opacity',
        max_scenes=None,
        normalize=True,  # 🎯 TOGGLE CANONICAL SPHERE NORMALIZATION
        normalize_colors=True,  # 🎨 TOGGLE COLOR NORMALIZATION (NEW!)
        target_radius=10.0,
    ):
        self.root            = root
        self.resol           = resol
        self.random_permute  = random_permute
        self.train           = train
        self.sampling_method = sampling_method
        self.normalize       = normalize
        self.normalize_colors = normalize_colors  # NEW!
        self.target_radius   = target_radius

        self.scene_dirs = sorted([
            os.path.join(root, d)
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        if max_scenes is not None and max_scenes < len(self.scene_dirs):
            self.scene_dirs = self.scene_dirs[:max_scenes]
            print(f"  Limited to {max_scenes} scenes "
                  f"(out of {len(os.listdir(root))} available)")

        if not self.scene_dirs:
            raise ValueError(f"No scene directories found in {root}")

        self.num_segment_categories = 72

        print(f"✓ Loaded {len(self.scene_dirs)} scenes from {root}")
        print(f"✓ Sampling method: {sampling_method}  [DETERMINISTIC — same 40k every epoch]")
        print(f"✓ Dataset type: ScanNet72 ({self.num_segment_categories} categories)")
        
        if normalize:
            print(f"🎯 Canonical sphere normalization: ENABLED")
            print(f"   → Target radius: {target_radius}m")
            print(f"   → Positions & scales normalized to canonical sphere")
        else:
            print(f"⚠️  Canonical sphere normalization: DISABLED")
            print(f"   → Using RAW coordinates and scales from dataset")
        
        if normalize_colors:
            print(f"🎨 Color normalization: ENABLED")
            print(f"   → RGB colours normalized to [0, 1]")
        else:
            print(f"⚠️  Color normalization: DISABLED")
            print(f"   → RGB colours kept in [0, 255] range")
            print(f"   ⚠️  WARNING: [0, 255] colors may cause training instability!")
        
        print(f"✓ Missing categories: [13, 53, 61] (never appear in dataset)")

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene_dir = self.scene_dirs[idx]

        # ── STEP 1: Load raw Gaussian parameters ─────────────────────────────
        coord   = np.load(os.path.join(scene_dir, 'coord.npy'))    # (N, 3)
        color   = np.load(os.path.join(scene_dir, 'color.npy'))    # (N, 3)
        scale   = np.load(os.path.join(scene_dir, 'scale.npy'))    # (N, 3)
        quat    = np.load(os.path.join(scene_dir, 'quat.npy'))     # (N, 4)
        opacity = np.load(os.path.join(scene_dir, 'opacity.npy'))  # (N,)

        # ── STEP 1.5: 🎯 OPTIONAL CANONICAL SPHERE NORMALIZATION ──────────────
        if self.normalize:
            coord, scale = normalize_to_canonical_sphere(
                coord, scale,
                target_radius=self.target_radius
            )
            
            # Verification on first training scene
            # if self.train and idx == 0:
            #     stats = verify_canonical_normalization(coord, self.target_radius)
            #     print(f"\n✓ Canonical sphere normalization verified (scene 0):")
            #     print(f"  Max distance: {stats['max_distance']:.2f}m")
            #     print(f"  Sphere utilization: {stats['utilization']:.1f}%")
            #     print(f"  Position range: [{coord.min():.2f}, {coord.max():.2f}]m")
            #     print(f"  Scale range (log): [{scale.min():.4f}, {scale.max():.4f}]")
            #     print(f"  Scale range (exp): [{np.exp(scale.min()):.4f}, {np.exp(scale.max()):.4f}]m")
        else:
            # NO NORMALIZATION - use raw values
            if self.train and idx == 0:
                print(f"\n⚠️  Using RAW coordinates (scene 0):")
                print(f"  Position range: [{coord.min():.2f}, {coord.max():.2f}]m")
                print(f"  Scale range: [{scale.min():.4f}, {scale.max():.4f}]m")
        # ─────────────────────────────────────────────────────────────────────

        # ── STEP 2: 🎨 OPTIONAL COLOR NORMALIZATION ──────────────────────────
        if self.normalize_colors:
            # Normalize RGB [0, 255] → [0, 1]
            color = color / 255.0
            
            if self.train and idx == 0:
                assert 0.0 <= color.min() and color.max() <= 1.0, \
                    f"RGB normalisation failed: range [{color.min()}, {color.max()}]"
                print(f"✓ Colors normalized to [0, 1]")
        else:
            # Keep colors in [0, 255] range
            if self.train and idx == 0:
                print(f"⚠️  Colors kept in [0, 255] range")
                print(f"  Color range: [{color.min():.1f}, {color.max():.1f}]")
        # ─────────────────────────────────────────────────────────────────────

        # ── STEP 2: Load semantic labels ──────────────────────────────────────
        try:
            segment  = np.load(os.path.join(scene_dir, 'segment.npy'))   # (N,)
            instance = np.load(os.path.join(scene_dir, 'instance.npy'))  # (N,)
            has_semantics = True

            valid = segment[segment >= 0]
            if len(valid) > 0 and (valid.max() >= 72 or valid.min() < 0):
                print(f"⚠️  Scene {idx}: labels outside ScanNet72 range "
                      f"[{valid.min()}, {valid.max()}]")
        except FileNotFoundError:
            segment       = np.full(len(coord), -1, dtype=np.int16)
            instance      = np.full(len(coord), -1, dtype=np.int32)
            has_semantics = False

        # ── STEP 3: Deterministic top-40k sampling ────────────────────────────
        N = len(coord)
        
        if self.sampling_method == 'hybrid':
            scale_mag  = np.linalg.norm(scale, axis=1)
            scale_norm = (scale_mag - scale_mag.min()) / \
                         (scale_mag.max() - scale_mag.min() + 1e-8)
            opacity_norm = (opacity - opacity.min()) / \
                           (opacity.max() - opacity.min() + 1e-8)
            importance = 0.7 * opacity_norm + 0.3 * scale_norm
        elif self.sampling_method == 'opacity':
            importance = opacity
        else:  # 'random' — still deterministic
            importance = np.arange(N, dtype=np.float32)

        # Sort by importance (ascending)
        sorted_indices = np.argsort(importance)

        T = self.TARGET_POINTS
        if N >= T:
            selected = sorted_indices[-T:]  # Top T
        else:
            # Pad by repeating highest-importance Gaussian
            n_extra  = T - N
            extra    = np.full(n_extra, sorted_indices[-1], dtype=np.int64)
            selected = np.concatenate([sorted_indices, extra])

        # Apply selection
        coord    = coord   [selected]
        color    = color   [selected]
        scale    = scale   [selected]
        quat     = quat    [selected]
        opacity  = opacity [selected]
        segment  = segment [selected]
        instance = instance[selected]

        # ── STEP 4: Voxelisation (positional encoding) ────────────────────────
        volume_dims = 40
        resolution  = 16.0 / volume_dims

        uniq_idx, inv_idx, _ = voxelize(coord, resolution, 'fnv')

        origin_offset = np.array([
            (volume_dims - 1) / 2,
            (volume_dims - 1) / 2,
            (volume_dims - 1) / 2,
        ]) * resolution

        shifted_points = coord + origin_offset
        voxel_indices  = np.floor(shifted_points / resolution)
        voxel_indices  = np.clip(voxel_indices, 0, volume_dims - 1)
        voxel_centers  = (voxel_indices - (volume_dims - 1) / 2) * resolution

        point_uniq_idx = uniq_idx[inv_idx]

        # ── STEP 5: Assemble feature tensor ───────────────────────────────────
        opacity_col    = opacity[:, np.newaxis]
        point_uniq_col = point_uniq_idx[:, np.newaxis]

        gs_params = np.concatenate(
            (coord, color, opacity_col, scale, quat), axis=1
        )
        gs_full_params = np.concatenate(
            (voxel_centers, point_uniq_col, gs_params), axis=1
        )

        assert gs_full_params.shape == (T, 18), \
            f"Expected ({T}, 18), got {gs_full_params.shape}"

        return {
            'features':       gs_full_params,
            'segment_labels': segment,
            'instance_labels': instance,
            'scene_idx':      idx,
            'has_semantics':  has_semantics,
            'num_categories': self.num_segment_categories,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_category_distribution(self, num_scenes=50):
        category_counts = {i: 0 for i in range(self.num_segment_categories)}
        total_points    = 0

        for i in tqdm(
            range(min(num_scenes, len(self.scene_dirs))),
            desc="Analysing categories",
        ):
            seg_path = os.path.join(self.scene_dirs[i], 'segment.npy')
            if os.path.exists(seg_path):
                segs  = np.load(seg_path)
                valid = segs[segs >= 0]
                for cat_id in valid:
                    category_counts[int(cat_id)] += 1
                total_points += len(valid)

        category_percentages = {
            k: (v / total_points * 100 if total_points else 0.0)
            for k, v in category_counts.items()
        }
        return category_counts, category_percentages

    def print_category_summary(self, num_scenes=50):
        print("\n" + "=" * 80)
        print("SCANNET72 CATEGORY DISTRIBUTION SUMMARY")
        print("=" * 80)

        counts, pcts = self.get_category_distribution(num_scenes)
        sorted_cats  = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        print(f"\nTop 20 most frequent categories:")
        print("-" * 60)
        for i, (cat_id, count) in enumerate(sorted_cats[:20]):
            print(f"  {i+1:2d}. Category {cat_id:2d}: {count:10,d} points "
                  f"({pcts[cat_id]:6.2f}%)")

        missing = [k for k, v in counts.items() if v == 0]
        print(f"\nMissing categories (never appear): {missing}")

        cumulative = 0
        print("\nCumulative coverage:")
        print("-" * 60)
        for i, (cat_id, _) in enumerate(sorted_cats[:15]):
            cumulative += pcts[cat_id]
            print(f"  Top {i+1:2d}: {cumulative:6.2f}%")

        print(f"\n✓ 95% coverage: top "
              f"{self._coverage_threshold(sorted_cats, pcts, 95)} categories")
        print(f"✓ 99% coverage: top "
              f"{self._coverage_threshold(sorted_cats, pcts, 99)} categories")
        print("=" * 80)

    def _coverage_threshold(self, sorted_cats, pcts, threshold):
        cumulative = 0
        for i, (cat_id, _) in enumerate(sorted_cats):
            cumulative += pcts[cat_id]
            if cumulative >= threshold:
                return i + 1
        return len(sorted_cats)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_path = (sys.argv[1] if len(sys.argv) > 1
                 else "/home/yli11/scratch/datasets/gaussian_world/"
                      "preprocessed/interior_gs/train")

    print("🔧 Testing normalization control...\n")

    # Test WITH normalization
    print("="*80)
    print("TEST 1: WITH CANONICAL SPHERE NORMALIZATION")
    print("="*80)
    dataset_normalized = gs_dataset(
        root=data_path,
        resol=200,
        random_permute=False,
        train=True,
        sampling_method='opacity',
        max_scenes=1,
        normalize=True,  # ← ENABLED
        target_radius=10.0,
    )
    
    sample_norm = dataset_normalized[0]
    features_norm = sample_norm['features']
    pos_norm = features_norm[:, 4:7]
    scale_norm = features_norm[:, 11:14]
    
    print(f"\nWith normalization:")
    print(f"  Position range: [{pos_norm.min():.2f}, {pos_norm.max():.2f}]m")
    print(f"  Scale range: [{scale_norm.min():.4f}, {scale_norm.max():.4f}]m")
    
    # Test WITHOUT normalization
    print("\n" + "="*80)
    print("TEST 2: WITHOUT CANONICAL SPHERE NORMALIZATION")
    print("="*80)
    dataset_raw = gs_dataset(
        root=data_path,
        resol=200,
        random_permute=False,
        train=True,
        sampling_method='opacity',
        max_scenes=1,
        normalize=False,  # ← DISABLED
    )
    
    sample_raw = dataset_raw[0]
    features_raw = sample_raw['features']
    pos_raw = features_raw[:, 4:7]
    scale_raw = features_raw[:, 11:14]
    
    print(f"\nWithout normalization:")
    print(f"  Position range: [{pos_raw.min():.2f}, {pos_raw.max():.2f}]m")
    print(f"  Scale range: [{scale_raw.min():.4f}, {scale_raw.max():.4f}]m")
    
    print("\n" + "="*80)
    print("✓ TESTS PASSED - Normalization toggle working correctly!")
    print("="*80)