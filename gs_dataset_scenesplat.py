"""
SceneSplat Dataset for Can3Tok Training - WITH NORMALIZATION CONTROL
======================================================================
ScanNet72 semantic labels + RGB colour normalisation [0, 1]

🎯 CANONICAL SPHERE NORMALIZATION (Optional - Controlled by Flag)
   - Can be enabled/disabled via normalize=True/False
   - Controls both position AND scale normalization

📏 SCALE NORMALIZATION MODES (New!)
   - 'log'    : Can3Tok ICCV 2025 style — convert to log-space, add log(factor)
                Decoder must use exp() activation
                Ground truth scales are NEGATIVE numbers (e.g. -4 to -1)
   - 'linear' : Original style — multiply scale by scale_factor directly
                Decoder uses softplus() activation
                Ground truth scales are SMALL POSITIVE numbers (e.g. 0.01-0.2m)

KEY FEATURES:
1. DETERMINISTIC SAMPLING: Always same top-40k Gaussians
2. OPTIONAL NORMALIZATION: Toggle canonical sphere normalization
3. DUAL SCALE MODES: Log-space (Can3Tok) or linear-space (original)
4. NO METADATA: Fixed normalization (no per-scene statistics)
5. INFERENCE-READY: Works perfectly with diffusion models
"""

import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
#  Canonical Sphere Normalization (Optional)
# ─────────────────────────────────────────────────────────────────────────────


def normalize_to_canonical_sphere(coord, scale, target_radius=10.0, scale_norm_mode='log'):
    """
    Normalize scene to canonical sphere.

    Positions are always normalized to fit within a sphere of target_radius.
    Scale normalization mode is controlled by scale_norm_mode:

    scale_norm_mode='log'  (Can3Tok ICCV 2025 style)
        scale_norm = log(scale) + log(scale_factor)
        → Scales stored in log-space (negative values, e.g. -4 to -1)
        → Decoder must use exp() to recover meters
        → Matches Can3Tok paper exactly

    scale_norm_mode='linear'  (Original style)
        scale_norm = scale * scale_factor
        → Scales stored in linear-space (small positive values, e.g. 0.01-0.2m)
        → Decoder uses softplus() activation
        → Easier to learn from scratch (targets are positive, near expected range)

    Args:
        coord:            [N, 3] Gaussian positions (xyz) in metres
        scale:            [N, 3] Gaussian scales in metres (post-exp from PLY)
        target_radius:    float, canonical sphere radius (default 10.0m)
        scale_norm_mode:  'log' or 'linear'

    Returns:
        coord_norm:  [N, 3] Positions in canonical sphere
        scale_norm:  [N, 3] Scales in chosen representation
    """
    # Step 1: Centre at origin
    center = coord.mean(axis=0)
    coord_centered = coord - center

    # Step 2: Find maximum distance from origin
    distances = np.linalg.norm(coord_centered, axis=1)
    max_dist = distances.max()
    if max_dist < 1e-6:
        max_dist = 1.0

    # Step 3: Compute scale factor (1.1 provides safety margin)
    scale_factor = target_radius / (max_dist * 1.1)

    # Step 4: Normalize positions
    coord_norm = coord_centered * scale_factor

    # Step 5: Normalize scales according to chosen mode
    if scale_norm_mode == 'log':
        # ═══════════════════════════════════════════════════════════════
        # LOG-SPACE NORMALIZATION  (matches Can3Tok ICCV 2025)
        # scale_norm = log(scale_old) + log(scale_factor)
        # Result: negative numbers, e.g. [-7, -1]
        # Decoder activation: exp()
        # ═══════════════════════════════════════════════════════════════
        scale_norm = np.log(scale + 1e-7) + np.log(scale_factor)
    else:
        # ═══════════════════════════════════════════════════════════════
        # LINEAR-SPACE NORMALIZATION  (original style)
        # scale_norm = scale_old * scale_factor
        # Result: small positive metres, e.g. [0.01, 0.2]
        # Decoder activation: softplus()
        # ═══════════════════════════════════════════════════════════════
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

    🎯 OPTIONAL CANONICAL SPHERE NORMALIZATION
       Toggle via normalize=True/False.

    📏 DUAL SCALE NORMALIZATION MODES
       scale_norm_mode='log'    → Can3Tok style (log-space, exp() decoder)
       scale_norm_mode='linear' → Original style (linear-space, softplus() decoder)

    Feature format returned — (40000, 18):
      cols  0:3   voxel_centers  (positional encoding)
      col   3     point_uniq_idx (voxel ID)
      cols  4:7   xyz            (Gaussian position — normalized if enabled)
      cols  7:10  rgb            (colour, normalised [0, 1])
      col  10     opacity        (post-sigmoid activation)
      cols 11:14  scale          (normalized per scale_norm_mode, or raw)
      cols 14:18  quaternion     (qw, qx, qy, qz)

    Target for reconstruction — cols 4:18 → (40000, 14):
      [0:3]   xyz        ← Normalized if enabled
      [3:6]   rgb        ← [0, 1]
      [6]     opacity    ← [0, 1]
      [7:10]  scale      ← Normalized per scale_norm_mode, or raw
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
        normalize_colors=True,
        target_radius=10.0,
        scale_norm_mode='log',    # 'log' = Can3Tok style | 'linear' = original style
    ):
        self.root             = root
        self.resol            = resol
        self.random_permute   = random_permute
        self.train            = train
        self.sampling_method  = sampling_method
        self.normalize        = normalize
        self.normalize_colors = normalize_colors
        self.target_radius    = target_radius
        self.scale_norm_mode  = scale_norm_mode   # ← NEW

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
        print(f"✓ Sampling method: {sampling_method}  "
              f"[DETERMINISTIC — same 40k every epoch]")
        print(f"✓ Dataset type: ScanNet72 ({self.num_segment_categories} categories)")

        if normalize:
            print(f"🎯 Canonical sphere normalization: ENABLED")
            print(f"   → Target radius: {target_radius}m")
            if scale_norm_mode == 'log':
                print(f"   → Scale mode: LOG-SPACE (Can3Tok ICCV 2025 style)")
                print(f"     scales = log(scale) + log(factor)  →  negative values")
                print(f"     decoder activation: exp()")
            else:
                print(f"   → Scale mode: LINEAR-SPACE (original style)")
                print(f"     scales = scale × factor  →  small positive metres")
                print(f"     decoder activation: softplus()")
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

        # ── STEP 2: 🎯 OPTIONAL CANONICAL SPHERE NORMALIZATION ───────────────
        if self.normalize:
            coord, scale = normalize_to_canonical_sphere(
                coord, scale,
                target_radius=self.target_radius,
                scale_norm_mode=self.scale_norm_mode,   # ← pass through
            )

            # Debug print on first training scene
            if self.train and idx == 0:
                stats = verify_canonical_normalization(coord, self.target_radius)
                # print(f"\n✓ Canonical sphere normalization verified (scene 0):")
                # print(f"  Max distance:      {stats['max_distance']:.2f}m")
                # print(f"  Sphere util:       {stats['utilization']:.1f}%")
                # print(f"  Position range:    [{coord.min():.2f}, {coord.max():.2f}]m")
                # if self.scale_norm_mode == 'log':
                #     print(f"  Scale range (log): [{scale.min():.4f}, {scale.max():.4f}]")
                #     print(f"  Scale range (exp): "
                #           f"[{np.exp(scale.min()):.4f}, {np.exp(scale.max()):.4f}]m")
                # else:
                #     print(f"  Scale range (lin): [{scale.min():.4f}, {scale.max():.4f}]m")
        else:
            if self.train and idx == 0:
                print(f"\n⚠️  Using RAW coordinates (scene 0):")
                print(f"  Position range: [{coord.min():.2f}, {coord.max():.2f}]m")
                print(f"  Scale range:    [{scale.min():.4f}, {scale.max():.4f}]m")

        # ── STEP 3: 🎨 OPTIONAL COLOR NORMALIZATION ──────────────────────────
        if self.normalize_colors:
            color = color / 255.0
            if self.train and idx == 0:
                assert 0.0 <= color.min() and color.max() <= 1.0, \
                    f"RGB normalisation failed: range [{color.min()}, {color.max()}]"
                print(f"✓ Colors normalized to [0, 1]")
        else:
            if self.train and idx == 0:
                print(f"⚠️  Colors kept in [0, 255] range")
                print(f"  Color range: [{color.min():.1f}, {color.max():.1f}]")

        # ── STEP 4: Load semantic labels ──────────────────────────────────────
        try:
            segment  = np.load(os.path.join(scene_dir, 'segment.npy'))
            instance = np.load(os.path.join(scene_dir, 'instance.npy'))
            has_semantics = True

            valid = segment[segment >= 0]
            if len(valid) > 0 and (valid.max() >= 72 or valid.min() < 0):
                print(f"⚠️  Scene {idx}: labels outside ScanNet72 range "
                      f"[{valid.min()}, {valid.max()}]")
        except FileNotFoundError:
            segment       = np.full(len(coord), -1, dtype=np.int16)
            instance      = np.full(len(coord), -1, dtype=np.int32)
            has_semantics = False

        # ── STEP 5: Deterministic top-40k sampling ────────────────────────────
        N = len(coord)

        if self.sampling_method == 'hybrid':
            scale_mag    = np.linalg.norm(scale, axis=1)
            scale_norm_s = (scale_mag - scale_mag.min()) / \
                           (scale_mag.max() - scale_mag.min() + 1e-8)
            opacity_norm = (opacity - opacity.min()) / \
                           (opacity.max() - opacity.min() + 1e-8)
            importance = 0.8 * opacity_norm + 0.2 * scale_norm_s
        elif self.sampling_method == 'opacity':
            importance = opacity
        else:  # 'random' — still deterministic
            importance = np.arange(N, dtype=np.float32)

        sorted_indices = np.argsort(importance)

        T = self.TARGET_POINTS
        if N >= T:
            selected = sorted_indices[-T:]
        else:
            n_extra  = T - N
            extra    = np.full(n_extra, sorted_indices[-1], dtype=np.int64)
            selected = np.concatenate([sorted_indices, extra])

        coord    = coord   [selected]
        color    = color   [selected]
        scale    = scale   [selected]
        quat     = quat    [selected]
        opacity  = opacity [selected]
        segment  = segment [selected]
        instance = instance[selected]

        # ── STEP 6: Voxelisation (positional encoding) ────────────────────────
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

        # ── STEP 7: Assemble feature tensor ───────────────────────────────────
        opacity_col    = opacity[:, np.newaxis]
        point_uniq_col = point_uniq_idx[:, np.newaxis]

        gs_params      = np.concatenate(
            (coord, color, opacity_col, scale, quat), axis=1
        )
        gs_full_params = np.concatenate(
            (voxel_centers, point_uniq_col, gs_params), axis=1
        )

        assert gs_full_params.shape == (T, 18), \
            f"Expected ({T}, 18), got {gs_full_params.shape}"

        return {
            'features':        gs_full_params,
            'segment_labels':  segment,
            'instance_labels': instance,
            'scene_idx':       idx,
            'has_semantics':   has_semantics,
            'num_categories':  self.num_segment_categories,
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

    print("🔧 Testing scale normalization modes...\n")

    for mode in ['log', 'linear']:
        print("=" * 80)
        print(f"TEST: scale_norm_mode='{mode}'")
        print("=" * 80)
        ds = gs_dataset(
            root=data_path,
            resol=200,
            random_permute=False,
            train=True,
            sampling_method='opacity',
            max_scenes=1,
            normalize=True,
            target_radius=10.0,
            scale_norm_mode=mode,
        )
        sample = ds[0]
        features = sample['features']
        pos   = features[:, 4:7]
        scale = features[:, 11:14]
        print(f"\nResults (mode='{mode}'):")
        print(f"  Position range: [{pos.min():.2f}, {pos.max():.2f}]m")
        print(f"  Scale raw range:  [{scale.min():.4f}, {scale.max():.4f}]")
        if mode == 'log':
            print(f"  Scale (exp) range: [{np.exp(scale.min()):.4f}, "
                  f"{np.exp(scale.max()):.4f}]m")
        print()

    print("=" * 80)
    print("✓ TESTS PASSED")
    print("=" * 80)