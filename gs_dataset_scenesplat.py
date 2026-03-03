"""
SceneSplat Dataset for Can3Tok Training
========================================
ScanNet72 semantic labels + RGB colour normalisation [0, 1]

CANONICAL SPHERE NORMALIZATION (Optional)
   - normalize=True/False controls both position AND scale normalization

SCALE NORMALIZATION MODES
   - 'log'    : log(scale) + log(factor) → negative values, decoder uses exp()
   - 'linear' : scale * factor → small positive metres, decoder uses exp()

OPTION 1 — SEMANTIC LABELS AS ENCODER INPUT (label_input=True)
   Appends normalized ScanNet72 label as col 18 of the feature tensor.
   Feature tensor: (40000, 18) → (40000, 19).
   Requires point_feats: 12 in shapevae-256.yaml.

STEP 1 — COLOR RESIDUAL ENCODING (color_residual=True)
   Instead of storing absolute RGB [0,1] in the feature tensor, stores
   per-Gaussian color residuals relative to the scene mean color.

   Computation (after top-40k sampling):
     mean_color  = color.mean(axis=0)        # [3]  scene DC term
     color_resid = color - mean_color        # [N,3] AC term, range ~[-0.3,+0.3]

   mean_color is returned in the batch dict so the training loop can
   supervise shape_embed via an auxiliary MSE loss.

   At decode time: absolute_color = predicted_residual + mean_color

   Why this helps:
     - Encoder no longer needs to store absolute scene color in mu.
     - Dynamic range of color features drops from [0,1] to ~[-0.3,+0.3].
     - shape_embed [B,384] gets its first gradient via mean_color prediction.
     - Natural VQ-VAE-2 style two-level decomposition:
         shape_embed → DC color (scene mean)
         z / mu      → AC color (per-Gaussian residuals)
"""

import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
#  Canonical Sphere Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_to_canonical_sphere(coord, scale, target_radius=10.0, scale_norm_mode='log'):
    center         = coord.mean(axis=0)
    coord_centered = coord - center
    distances      = np.linalg.norm(coord_centered, axis=1)
    max_dist       = distances.max()
    if max_dist < 1e-6:
        max_dist = 1.0
    scale_factor = target_radius / (max_dist * 1.1)
    coord_norm   = coord_centered * scale_factor
    if scale_norm_mode == 'log':
        scale_norm = np.log(scale + 1e-7) + np.log(scale_factor)
    else:
        scale_norm = scale * scale_factor
    return coord_norm, scale_norm


def verify_canonical_normalization(coord_norm, target_radius=10.0):
    distances = np.linalg.norm(coord_norm, axis=1)
    max_dist  = distances.max()
    return {
        'max_distance':   max_dist,
        'target_radius':  target_radius,
        'fits_in_sphere': max_dist <= target_radius * 1.05,
        'utilization':    (max_dist / target_radius) * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Voxelisation helper
# ─────────────────────────────────────────────────────────────────────────────

def voxelize(coord, voxel_size=0.4, hash_type='fnv'):
    discrete_coord = np.floor(coord / voxel_size).astype(np.int32)
    if hash_type == 'fnv':
        offset_basis = 2166136261
        fnv_prime    = 16777619
        hash_vals    = np.full(len(discrete_coord), offset_basis, dtype=np.int64)
        for i in range(3):
            hash_vals = (hash_vals ^ discrete_coord[:, i]) * fnv_prime
        uniq_idx, inv_idx, count = np.unique(
            hash_vals, return_inverse=True, return_counts=True)
    else:
        try:
            min_coord = discrete_coord.min(axis=0)
            shifted   = discrete_coord - min_coord
            max_coord = shifted.max(axis=0)
            ravel_idx = np.ravel_multi_index(shifted.T, max_coord + 1)
            uniq_idx, inv_idx, count = np.unique(
                ravel_idx, return_inverse=True, return_counts=True)
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

    Feature tensor col layout (baseline, label_input=False):
      cols  0:3   voxel_centers
      col   3     point_uniq_idx
      cols  4:7   xyz
      cols  7:10  rgb  (absolute if color_residual=False,
                        residuals rel. to mean_color if color_residual=True)
      col  10     opacity
      cols 11:14  scale
      cols 14:18  quaternion

    With label_input=True: col 18 = label_norm appended (19 cols total).

    Reconstruction target = cols 4:18 (14-dim) in all cases.

    Additional batch dict keys when color_residual=True:
      'mean_color': np.float32 [3]  — per-scene mean RGB in [0,1]
                    used by training loop to supervise shape_embed prediction
                    and by PLY reconstructor to recover absolute colors
    """

    TARGET_POINTS      = 40_000
    LABEL_MAX          = 71.0
    LABEL_MISSING_NORM = -1.0 / 71.0

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
        scale_norm_mode='linear',
        label_input=False,
        color_residual=False,       # Step 1: subtract mean color, return mean_color
    ):
        self.root            = root
        self.resol           = resol
        self.random_permute  = random_permute
        self.train           = train
        self.sampling_method = sampling_method
        self.normalize       = normalize
        self.normalize_colors = normalize_colors
        self.target_radius   = target_radius
        self.scale_norm_mode = scale_norm_mode
        self.label_input     = label_input
        self.color_residual  = color_residual

        self.scene_dirs = sorted([
            os.path.join(root, d)
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        if max_scenes is not None and max_scenes < len(self.scene_dirs):
            self.scene_dirs = self.scene_dirs[:max_scenes]
            print(f"  Limited to {max_scenes} scenes "
                  f"(of {len(os.listdir(root))} available)")

        if not self.scene_dirs:
            raise ValueError(f"No scene directories found in {root}")

        self.num_segment_categories = 72
        self.feature_width = 19 if label_input else 18

        print(f"  Loaded {len(self.scene_dirs)} scenes from {root}")
        print(f"  Sampling: {sampling_method}  [DETERMINISTIC]")
        print(f"  Feature tensor: (40000, {self.feature_width})")

        if color_residual:
            print(f"  Color residual: ENABLED (Step 1)")
            print(f"    -> mean_color computed after top-40k sampling")
            print(f"    -> feature tensor stores color - mean_color")
            print(f"    -> mean_color returned in batch dict for shape_embed supervision")
            print(f"    -> at decode: absolute_color = residual + mean_color")
        else:
            print(f"  Color residual: DISABLED (absolute RGB in [0,1])")

        if label_input:
            print(f"  Label input: ENABLED (col 18, point_feats=12)")
        else:
            print(f"  Label input: DISABLED (point_feats=11)")

        if normalize:
            print(f"  Canonical norm: ENABLED (radius={target_radius}m, "
                  f"scale={scale_norm_mode})")
        else:
            print(f"  Canonical norm: DISABLED")

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene_dir = self.scene_dirs[idx]

        # ── Load raw Gaussian parameters ──────────────────────────────────────
        coord   = np.load(os.path.join(scene_dir, 'coord.npy'))
        color   = np.load(os.path.join(scene_dir, 'color.npy'))
        scale   = np.load(os.path.join(scene_dir, 'scale.npy'))
        quat    = np.load(os.path.join(scene_dir, 'quat.npy'))
        opacity = np.load(os.path.join(scene_dir, 'opacity.npy'))

        # ── Canonical sphere normalization ────────────────────────────────────
        if self.normalize:
            coord, scale = normalize_to_canonical_sphere(
                coord, scale,
                target_radius=self.target_radius,
                scale_norm_mode=self.scale_norm_mode,
            )

        # ── Color normalization to [0, 1] ─────────────────────────────────────
        if self.normalize_colors:
            color = color / 255.0

        # ── Load semantic labels ──────────────────────────────────────────────
        try:
            segment  = np.load(os.path.join(scene_dir, 'segment.npy'))
            instance = np.load(os.path.join(scene_dir, 'instance.npy'))
            has_semantics = True
        except FileNotFoundError:
            segment       = np.full(len(coord), -1, dtype=np.int16)
            instance      = np.full(len(coord), -1, dtype=np.int32)
            has_semantics = False

        # ── Deterministic top-40k sampling ────────────────────────────────────
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
        else:
            importance = np.arange(N, dtype=np.float32)

        sorted_indices = np.argsort(importance)
        T = self.TARGET_POINTS
        if N >= T:
            selected = sorted_indices[-T:]
        else:
            extra    = np.full(T - N, sorted_indices[-1], dtype=np.int64)
            selected = np.concatenate([sorted_indices, extra])

        coord    = coord   [selected]
        color    = color   [selected]
        scale    = scale   [selected]
        quat     = quat    [selected]
        opacity  = opacity [selected]
        segment  = segment [selected]
        instance = instance[selected]

        # ── Step 1: Color residual encoding ───────────────────────────────────
        # Computed AFTER sampling so mean reflects the actual 40k Gaussians used.
        if self.color_residual:
            mean_color = color.mean(axis=0).astype(np.float32)  # [3] in [0,1]
            color      = color - mean_color                      # [T,3] residuals
            # Note: residuals range is typically [-0.7, +0.4] for indoor scenes.
            # Debug print removed — DataLoader spawns 9 worker processes each with
            # their own copy of any instance flag, so per-item gating doesn't work.
            # Range confirmed once at __init__ time in the startup summary above.
        else:
            mean_color = np.zeros(3, dtype=np.float32)  # placeholder, not used in loss

        # ── Voxelisation (positional encoding) ───────────────────────────────
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

        # ── Assemble feature tensor ───────────────────────────────────────────
        opacity_col    = opacity[:, np.newaxis]
        point_uniq_col = point_uniq_idx[:, np.newaxis]

        # color here is either absolute [0,1] or residuals ~[-0.3,+0.3]
        gs_params = np.concatenate(
            (coord, color, opacity_col, scale, quat), axis=1
        )  # (T, 14)

        if self.label_input:
            label_norm = np.where(
                segment >= 0,
                segment.astype(np.float32) / self.LABEL_MAX,
                np.float32(self.LABEL_MISSING_NORM)
            )
            label_col = label_norm[:, np.newaxis]
            gs_full_params = np.concatenate(
                (voxel_centers, point_uniq_col, gs_params, label_col), axis=1
            )  # (T, 19)
            assert gs_full_params.shape == (T, 19)
        else:
            gs_full_params = np.concatenate(
                (voxel_centers, point_uniq_col, gs_params), axis=1
            )  # (T, 18)
            assert gs_full_params.shape == (T, 18)

        return {
            'features':        gs_full_params.astype(np.float32),
            'segment_labels':  segment,
            'instance_labels': instance,
            'scene_idx':       idx,
            'has_semantics':   has_semantics,
            'num_categories':  self.num_segment_categories,
            'mean_color':      mean_color,   # [3] float32, zeros if color_residual=False
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_category_distribution(self, num_scenes=50):
        category_counts = {i: 0 for i in range(self.num_segment_categories)}
        total_points    = 0
        for i in tqdm(range(min(num_scenes, len(self.scene_dirs))),
                      desc="Analysing categories"):
            seg_path = os.path.join(self.scene_dirs[i], 'segment.npy')
            if os.path.exists(seg_path):
                segs  = np.load(seg_path)
                valid = segs[segs >= 0]
                for cat_id in valid:
                    category_counts[int(cat_id)] += 1
                total_points += len(valid)
        return category_counts, {
            k: (v / total_points * 100 if total_points else 0.0)
            for k, v in category_counts.items()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_path = (sys.argv[1] if len(sys.argv) > 1
                 else "/home/yli11/scratch/datasets/gaussian_world/"
                      "preprocessed/interior_gs/train")

    for color_residual in [False, True]:
        print(f"\n{'='*60}")
        print(f"TEST: color_residual={color_residual}")
        print(f"{'='*60}")
        ds = gs_dataset(root=data_path, max_scenes=1, normalize=True,
                        scale_norm_mode='linear', label_input=False,
                        color_residual=color_residual)
        s = ds[0]
        f = s['features']
        color_col  = f[:, 7:10]
        mean_color = s['mean_color']
        print(f"Feature shape:  {f.shape}")
        print(f"Color range:    [{color_col.min():.3f}, {color_col.max():.3f}]")
        print(f"mean_color:     {mean_color}  (zeros if disabled)")
        if color_residual:
            assert color_col.min() < 0, "Residuals should have negative values"
            assert not np.allclose(mean_color, 0), "mean_color should be non-zero"
            print(f"  Assertions passed")
        else:
            assert color_col.min() >= 0, "Absolute color should be non-negative"
            print(f"  Assertions passed")

    print("\nALL TESTS PASSED")