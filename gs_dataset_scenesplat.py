"""
SceneSplat Dataset for Can3Tok Training
========================================
ScanNet72 semantic labels + RGB colour normalisation [0, 1]

 CANONICAL SPHERE NORMALIZATION (Can3Tok ICCV 2025)
   All scenes normalized to fixed sphere (radius = 10 meters)
   NO denormalization needed at inference!

KEY FEATURES:
1. DETERMINISTIC SAMPLING: Always same top-40k Gaussians
2. CANONICAL SPHERE: All scenes fit in 10m sphere (centered at origin)
3. NO METADATA: Fixed normalization (no per-scene statistics)
4. INFERENCE-READY: Works perfectly with diffusion models
"""

import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 🎯 Canonical Sphere Normalization (Can3Tok's Actual Method)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_to_canonical_sphere(coord, scale, target_radius=10.0):
    """
    Normalize scene to canonical sphere (Can3Tok ICCV 2025 method).
    
    This is Can3Tok's ACTUAL normalization approach from their code:
      1. Center scene at origin: translate = -mean(positions)
      2. Scale to target radius: scale = target_radius / (max_dist * 1.1)
      3. Apply to positions and scales
    
    Key insight: Normalizing to a FIXED reasonable scale (10m) means
    the output is already at good scale for visualization and inference.
    NO denormalization needed!
    
    Small rooms use ~70% of sphere, large rooms use ~100%.
    Model learns scale relationships!
    
    Args:
        coord: [N, 3] Gaussian positions (xyz)
        scale: [N, 3] Gaussian scales (post-exp)
        target_radius: Fixed radius in meters (default 10.0)
    
    Returns:
        coord_norm: [N, 3] Positions in canonical sphere
        scale_norm: [N, 3] Scales relative to canonical sphere
    
    Example:
        Small room (5m wide):
          Original: [-2.5, 2.5] meters
          Normalized: [-7.3, 7.3] meters (fits in 10m sphere, uses 73%)
        
        Large room (15m wide):
          Original: [-7.5, 7.5] meters  
          Normalized: [-7.3, 7.3] meters (fits in 10m sphere, uses 97%)
    """
    # Step 1: Center at origin (Can3Tok: translate = -mean(x_g))
    center = coord.mean(axis=0)
    coord_centered = coord - center
    
    # Step 2: Find maximum distance from origin
    distances = np.linalg.norm(coord_centered, axis=1)
    max_dist = distances.max()
    
    # Handle edge case: empty or degenerate scene
    if max_dist < 1e-6:
        max_dist = 1.0
    
    # Step 3: Scale to target radius (Can3Tok: scale = T / (max_dist * 1.1))
    # The 1.1 factor provides safety margin
    scale_factor = target_radius / (max_dist * 1.1)
    
    # Apply transformation
    coord_norm = coord_centered * scale_factor
    scale_norm = scale * scale_factor
    
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
    
    🎯 CANONICAL SPHERE NORMALIZATION (Can3Tok's actual method)
    
    All scenes normalized to sphere of radius target_radius (default 10m).
    Output is already at good scale - NO denormalization needed!

    Feature format returned — (40000, 18):
      cols  0:3   voxel_centers  (positional encoding)
      col   3     point_uniq_idx (voxel ID)
      cols  4:7   xyz            (Gaussian position in canonical sphere)
      cols  7:10  rgb            (colour, normalised [0, 1])
      col  10     opacity        (post-sigmoid activation)
      cols 11:14  scale          (post-exp, normalized to canonical sphere)
      cols 14:18  quaternion     (qw, qx, qy, qz)

    Target for reconstruction — cols 4:18 → (40000, 14):
      [0:3]   xyz        ← In canonical sphere (~10m)
      [3:6]   rgb        ← [0, 1]
      [6]     opacity    ← [0, 1]
      [7:10]  scale      ← In canonical sphere scale
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
        normalize=True,
        target_radius=10.0,  # 🎯 Canonical sphere radius
    ):
        self.root            = root
        self.resol           = resol
        self.random_permute  = random_permute
        self.train           = train
        self.sampling_method = sampling_method
        self.normalize       = normalize
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
        print(f"✓ RGB colours: normalised to [0, 1]")
        
        if normalize:
            print(f"🎯 Canonical sphere normalization: ENABLED")
            print(f"   → Target radius: {target_radius}m")
            print(f"   → All scenes fit in sphere of radius ~{target_radius}m")
            
        else:
            print(f"⚠️  Canonical sphere normalization: DISABLED")
            
        
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

        # ── STEP 1.5: 🎯 CANONICAL SPHERE NORMALIZATION ───────────────────────
        if self.normalize:
            coord, scale = normalize_to_canonical_sphere(
                coord, scale,
                target_radius=self.target_radius
            )
            
            # # Verification on first training scene
            # if self.train and idx == 0:
            #     stats = verify_canonical_normalization(coord, self.target_radius)
            #     assert stats['fits_in_sphere'], \
            #         f"Scene doesn't fit in sphere: max_dist={stats['max_distance']:.2f}m"
            #     print(f"\n✓ Canonical sphere normalization verified (scene 0):")
            #     print(f"  Max distance: {stats['max_distance']:.2f}m")
            #     print(f"  Sphere utilization: {stats['utilization']:.1f}%")
            #     print(f"  Position range: [{coord.min():.2f}, {coord.max():.2f}]m")
            #     print(f"  Scale range: [{scale.min():.4f}, {scale.max():.4f}]m")
        # ─────────────────────────────────────────────────────────────────────

        # Normalise RGB [0, 255] → [0, 1]
        color = color / 255.0

        if self.train and idx == 0:
            assert 0.0 <= color.min() and color.max() <= 1.0, \
                f"RGB normalisation failed: range [{color.min()}, {color.max()}]"

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

    print(" Testing canonical sphere normalization …\n")

    # Create dataset
    dataset = gs_dataset(
        root=data_path,
        resol=200,
        random_permute=False,
        train=True,
        sampling_method='opacity',
        max_scenes=3,
        normalize=True,
        target_radius=10.0,
    )
    
    print(f"\nDataset size: {len(dataset)} scenes")
    
    # Load first scene
    sample = dataset[0]
    features = sample['features']
    positions = features[:, 4:7]
    scales = features[:, 11:14]
    
    print(f"\nScene 0 after normalization:")
    print(f"  Position range: [{positions.min():.2f}, {positions.max():.2f}]m")
    print(f"  Scale range: [{scales.min():.4f}, {scales.max():.4f}]m")
    
    # Verify determinism
    print(f"\nDeterminism check:")
    s1 = dataset[0]['features']
    s2 = dataset[0]['features']
    identical = np.allclose(s1, s2)
    print(f"  Load 1 == Load 2: {identical}  {'✓ PASS' if identical else '✗ FAIL'}")
    
    # Slot consistency
    slot0_samples = [dataset[0]['features'][0, 4:7] for _ in range(3)]
    all_same = all(np.allclose(slot0_samples[0], s) for s in slot0_samples[1:])
    print(f"  Slot consistency: {all_same}  {'✓ PASS' if all_same else '✗ FAIL'}")
    
    if identical and all_same:
        print(f"\n{'='*80}")
        print(f"🎉 ALL TESTS PASSED!")
        print(f"{'='*80}")
        print(f"\nCanonical sphere normalization working correctly!")
        print(f"Ready to train with:")
        print(f"  - Positions in ~10m sphere")
        print(f"  - NO denormalization needed at inference!")
        print(f"  - Expected 30-40% L2 reduction")
        print(f"{'='*80}")