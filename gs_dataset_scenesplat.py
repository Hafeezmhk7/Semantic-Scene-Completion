"""
SceneSplat Dataset for Can3Tok Training
========================================
STEP 1 — COLOR RESIDUAL (color_residual=True)
   mean_color  = color.mean()   -> shape_embed via MeanColorHead
   color_resid = color - mean   -> AC residuals in feature tensor
   Batch dict: 'mean_color' [3]

POSITION SCAFFOLD (position_scaffold=True)
   8^3=512 super-voxels. Batch keys: scaffold_anchors, scaffold_token_ids, position_offsets

SCENE LAYOUT HEAD (scene_layout_head=True)
   per-category centroids: category_centroids[72,3], category_valid[72]

NEW: POSITION LAYOUT RESIDUAL (position_layout_residual=True)
   Requires scene_layout_head=True.
   Closes the DC/AC arithmetic loop for position:

   DC  = per-Gaussian category centroid  (dc_position[i] = centroid[segment[i]])
   AC  = coord - dc_position             (position_residuals, range ~+-0.5m)

   Feature tensor cols 4:7 STILL contain ABSOLUTE coords (encoder needs this).
   Separate batch keys for the arithmetic:
     'dc_position'        [40000, 3]  — added back at PLY save
     'position_residuals' [40000, 3]  — used as reconstruction TARGET

   Invertibility guaranteed: dc_position + position_residuals == coord exactly.

JEPA IDEA 1 (jepa_idea1=True, requires position_scaffold=True)
   voxel_label_dists[512,72], voxel_valid[512]
"""

import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def normalize_to_canonical_sphere(coord, scale, target_radius=10.0,
                                   scale_norm_mode='log'):
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


def compute_position_scaffold(coord, scaffold_dims=8, domain_size=16.0):
    num_tokens  = scaffold_dims ** 3
    cell_size   = domain_size / scaffold_dims
    half_domain = domain_size / 2.0
    shifted = coord + half_domain
    sv_idx  = np.floor(shifted / cell_size).astype(np.int32)
    sv_idx  = np.clip(sv_idx, 0, scaffold_dims - 1)
    scaffold_token_ids = (
        sv_idx[:, 0] * scaffold_dims ** 2 +
        sv_idx[:, 1] * scaffold_dims +
        sv_idx[:, 2]
    ).astype(np.int32)
    anchor_counts = np.bincount(scaffold_token_ids,
                                minlength=num_tokens).astype(np.float64)
    anchor_sum    = np.zeros((num_tokens, 3), dtype=np.float64)
    for dim in range(3):
        anchor_sum[:, dim] = np.bincount(
            scaffold_token_ids, weights=coord[:, dim].astype(np.float64),
            minlength=num_tokens)
    scaffold_anchors = np.zeros((num_tokens, 3), dtype=np.float64)
    occupied = anchor_counts > 0
    scaffold_anchors[occupied] = anchor_sum[occupied] / anchor_counts[occupied, np.newaxis]
    empty_idx = np.where(~occupied)[0]
    if len(empty_idx) > 0:
        ix = empty_idx // (scaffold_dims ** 2)
        iy = (empty_idx // scaffold_dims) % scaffold_dims
        iz = empty_idx % scaffold_dims
        cell_size_f = float(cell_size)
        scaffold_anchors[empty_idx, 0] = ix * cell_size_f + cell_size_f/2.0 - half_domain
        scaffold_anchors[empty_idx, 1] = iy * cell_size_f + cell_size_f/2.0 - half_domain
        scaffold_anchors[empty_idx, 2] = iz * cell_size_f + cell_size_f/2.0 - half_domain
    scaffold_anchors   = scaffold_anchors.astype(np.float32)
    position_offsets   = (coord - scaffold_anchors[scaffold_token_ids]).astype(np.float32)
    return scaffold_anchors, scaffold_token_ids, position_offsets


def compute_category_centroids(coord, segment, num_cats=72):
    """Per-ScanNet72-category spatial centroids."""
    category_centroids = np.zeros((num_cats, 3), dtype=np.float32)
    category_valid     = np.zeros(num_cats, dtype=np.float32)
    valid_mask = segment >= 0
    if valid_mask.sum() == 0:
        return category_centroids, category_valid
    valid_coord   = coord[valid_mask]
    valid_segment = segment[valid_mask].astype(np.int64)
    counts = np.bincount(valid_segment, minlength=num_cats)
    for dim in range(3):
        sums = np.bincount(valid_segment,
                           weights=valid_coord[:, dim].astype(np.float64),
                           minlength=num_cats)
        present = counts > 0
        category_centroids[present, dim] = (sums[present] / counts[present]).astype(np.float32)
    category_valid = (counts > 0).astype(np.float32)
    return category_centroids, category_valid


def compute_position_layout_residuals(coord, segment, category_centroids,
                                       category_valid):
    """
    NEW — DC/AC position decomposition using per-category centroids.

    Mirrors Step 1 (color residual) but for xyz position:

    COLOR STEP 1:                     POSITION (this function):
    mean_color = color.mean()         dc_position[i] = centroid[label[i]]
    color_resid = color - mean_color  pos_resid[i] = coord[i] - dc_position[i]
    stored in feature tensor (AC)     stored in batch dict (AC)
    added back at PLY save            added back at PLY save

    Unlabelled Gaussians (segment==-1): fallback to scene mean position.

    Guarantee: dc_position + position_residuals == coord  (exact, no rounding)
    """
    N          = len(coord)
    dc_position = np.zeros((N, 3), dtype=np.float32)
    scene_mean  = coord.mean(axis=0).astype(np.float32)

    valid_mask   = segment >= 0
    invalid_mask = ~valid_mask

    # Labelled Gaussians: DC = their category centroid
    if valid_mask.sum() > 0:
        valid_segs = segment[valid_mask].astype(np.int64)
        dc_position[valid_mask] = category_centroids[valid_segs]

        # Edge case: Gaussian labelled with a category absent from this scene
        # (category_valid==0 means centroid was never computed -> stays at 0)
        # Fall back to scene mean for these
        absent_cat_mask = valid_mask.copy()
        absent_cat_mask[valid_mask] = (category_valid[valid_segs] == 0)
        if absent_cat_mask.sum() > 0:
            dc_position[absent_cat_mask] = scene_mean

    # Unlabelled Gaussians: DC = scene mean
    if invalid_mask.sum() > 0:
        dc_position[invalid_mask] = scene_mean

    position_residuals = (coord - dc_position).astype(np.float32)
    return dc_position, position_residuals


def compute_voxel_label_dists(scaffold_token_ids, segment, num_tokens=512, num_cats=72):
    """Per-super-voxel label distributions (JEPA Idea 1)."""
    voxel_label_dists = np.zeros((num_tokens, num_cats), dtype=np.float32)
    voxel_valid       = np.zeros(num_tokens, dtype=np.float32)
    valid_mask = segment >= 0
    if valid_mask.sum() == 0:
        return voxel_label_dists, voxel_valid
    valid_tids = scaffold_token_ids[valid_mask].astype(np.int64)
    valid_segs = segment[valid_mask].astype(np.int64)
    combined = valid_tids * num_cats + valid_segs
    counts   = np.bincount(combined, minlength=num_tokens * num_cats).reshape(
        num_tokens, num_cats)
    row_sums = counts.sum(axis=1)
    occupied = row_sums > 0
    voxel_valid[occupied] = 1.0
    safe_sums = np.maximum(row_sums, 1)[:, np.newaxis]
    voxel_label_dists = (counts / safe_sums).astype(np.float32)
    return voxel_label_dists, voxel_valid


class gs_dataset(Dataset):
    """
    SceneSplat-7K dataset with ScanNet72 semantic labels.

    Feature tensor col layout (label_input=False, 18 cols):
      0:3   voxel_centers
      3     point_uniq_idx
      4:7   xyz  (ALWAYS ABSOLUTE — encoder Fourier embedder needs this)
      7:10  rgb  (absolute or residuals if color_residual=True)
      10    opacity
      11:14 scale
      14:18 quaternion

    When position_layout_residual=True:
      Cols 4:7 STILL hold ABSOLUTE xyz (encoder unchanged).
      The training loop swaps the position TARGET to 'position_residuals'.
      PLY save adds 'dc_position' back to get absolute coordinates.
    """

    TARGET_POINTS      = 40_000
    LABEL_MAX          = 71.0
    LABEL_MISSING_NORM = -1.0 / 71.0
    SCAFFOLD_DIMS      = 8
    SCAFFOLD_DOMAIN    = 16.0
    SCAFFOLD_TOKENS    = 512
    NUM_CATS           = 72

    def __init__(self, root, resol=200, random_permute=False, train=True,
                 sampling_method='opacity', max_scenes=None, normalize=True,
                 normalize_colors=True, target_radius=10.0, scale_norm_mode='linear',
                 label_input=False, color_residual=False, position_scaffold=False,
                 scene_layout_head=False, jepa_idea1=False,
                 position_layout_residual=False):

        self.root                     = root
        self.resol                    = resol
        self.random_permute           = random_permute
        self.train                    = train
        self.sampling_method          = sampling_method
        self.normalize                = normalize
        self.normalize_colors         = normalize_colors
        self.target_radius            = target_radius
        self.scale_norm_mode          = scale_norm_mode
        self.label_input              = label_input
        self.color_residual           = color_residual
        self.position_scaffold        = position_scaffold
        self.scene_layout_head        = scene_layout_head
        self.jepa_idea1               = jepa_idea1
        self.position_layout_residual = position_layout_residual

        if position_layout_residual and not scene_layout_head:
            print("  [INFO] position_layout_residual=True requires scene_layout_head=True. "
                  "Enabling automatically.")
            self.scene_layout_head = True

        if jepa_idea1 and not position_scaffold:
            print("  [INFO] jepa_idea1=True requires position_scaffold=True. Enabling.")
            self.position_scaffold = True

        self.scene_dirs = sorted([
            os.path.join(root, d)
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        if max_scenes is not None and max_scenes < len(self.scene_dirs):
            self.scene_dirs = self.scene_dirs[:max_scenes]
            print(f"  Limited to {max_scenes} scenes")
        if not self.scene_dirs:
            raise ValueError(f"No scene directories found in {root}")

        self.num_segment_categories = self.NUM_CATS
        self.feature_width = 19 if label_input else 18

        print(f"  Loaded {len(self.scene_dirs)} scenes from {root}")
        print(f"  color_residual={color_residual} | position_scaffold={self.position_scaffold}")
        print(f"  scene_layout_head={self.scene_layout_head} | jepa_idea1={jepa_idea1}")
        print(f"  position_layout_residual={position_layout_residual}")
        if position_layout_residual:
            print(f"  POSITION DC/AC ENABLED:")
            print(f"    DC = category centroid per Gaussian")
            print(f"    AC = coord - centroid  (~+-0.5m vs +-10m absolute)")
            print(f"    Encoder: absolute coords unchanged")
            print(f"    Target: position_residuals | PLY: adds dc_position back")

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene_dir = self.scene_dirs[idx]

        coord   = np.load(os.path.join(scene_dir, 'coord.npy'))
        color   = np.load(os.path.join(scene_dir, 'color.npy'))
        scale   = np.load(os.path.join(scene_dir, 'scale.npy'))
        quat    = np.load(os.path.join(scene_dir, 'quat.npy'))
        opacity = np.load(os.path.join(scene_dir, 'opacity.npy'))

        if self.normalize:
            coord, scale = normalize_to_canonical_sphere(
                coord, scale, target_radius=self.target_radius,
                scale_norm_mode=self.scale_norm_mode)
        if self.normalize_colors:
            color = color / 255.0

        try:
            segment  = np.load(os.path.join(scene_dir, 'segment.npy'))
            instance = np.load(os.path.join(scene_dir, 'instance.npy'))
            has_semantics = True
        except FileNotFoundError:
            segment       = np.full(len(coord), -1, dtype=np.int16)
            instance      = np.full(len(coord), -1, dtype=np.int32)
            has_semantics = False

        # Deterministic top-40k sampling
        N = len(coord)
        if self.sampling_method == 'hybrid':
            scale_mag    = np.linalg.norm(scale, axis=1)
            scale_norm_s = ((scale_mag - scale_mag.min()) /
                            (scale_mag.max() - scale_mag.min() + 1e-8))
            opacity_norm = ((opacity - opacity.min()) /
                            (opacity.max() - opacity.min() + 1e-8))
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

        # Step 1: Color residual
        if self.color_residual:
            mean_color = color.mean(axis=0).astype(np.float32)
            color      = color - mean_color
        else:
            mean_color = np.zeros(3, dtype=np.float32)

        # Position scaffold
        if self.position_scaffold:
            scaffold_anchors, scaffold_token_ids, position_offsets = \
                compute_position_scaffold(coord, scaffold_dims=self.SCAFFOLD_DIMS,
                                          domain_size=self.SCAFFOLD_DOMAIN)
        else:
            scaffold_anchors   = np.zeros((self.SCAFFOLD_TOKENS, 3), dtype=np.float32)
            scaffold_token_ids = np.zeros(T, dtype=np.int32)
            position_offsets   = np.zeros((T, 3), dtype=np.float32)

        # Scene layout head: per-category centroids
        # Must run BEFORE position_layout_residual (needs category_centroids)
        if self.scene_layout_head:
            category_centroids, category_valid = compute_category_centroids(
                coord, segment, num_cats=self.NUM_CATS)
        else:
            category_centroids = np.zeros((self.NUM_CATS, 3), dtype=np.float32)
            category_valid     = np.zeros(self.NUM_CATS, dtype=np.float32)

        # NEW: Position layout residual
        # Feature tensor cols 4:7 keep ABSOLUTE coord.
        # Training loop will use position_residuals as the target.
        # PLY save will add dc_position back.
        if self.position_layout_residual:
            dc_position, position_residuals = compute_position_layout_residuals(
                coord, segment, category_centroids, category_valid)
        else:
            dc_position        = np.zeros((T, 3), dtype=np.float32)
            position_residuals = np.zeros((T, 3), dtype=np.float32)

        # JEPA Idea 1
        if self.jepa_idea1 and self.position_scaffold:
            voxel_label_dists, voxel_valid = compute_voxel_label_dists(
                scaffold_token_ids, segment,
                num_tokens=self.SCAFFOLD_TOKENS, num_cats=self.NUM_CATS)
        else:
            voxel_label_dists = np.zeros((self.SCAFFOLD_TOKENS, self.NUM_CATS),
                                         dtype=np.float32)
            voxel_valid       = np.zeros(self.SCAFFOLD_TOKENS, dtype=np.float32)

        # Encoder voxelisation (always uses absolute coord)
        volume_dims = 40
        resolution  = 16.0 / volume_dims
        uniq_idx, inv_idx, _ = voxelize(coord, resolution, 'fnv')
        origin_offset  = np.array([(volume_dims - 1) / 2] * 3) * resolution
        shifted_pts    = coord + origin_offset
        voxel_idx      = np.floor(shifted_pts / resolution)
        voxel_idx      = np.clip(voxel_idx, 0, volume_dims - 1)
        voxel_centers  = (voxel_idx - (volume_dims - 1) / 2) * resolution
        point_uniq_idx = uniq_idx[inv_idx]

        # Feature tensor [T, 18]
        # Cols 4:7 = ABSOLUTE xyz regardless of position_layout_residual
        opacity_col    = opacity[:, np.newaxis]
        point_uniq_col = point_uniq_idx[:, np.newaxis]
        gs_params      = np.concatenate((coord, color, opacity_col, scale, quat), axis=1)

        if self.label_input:
            label_norm = np.where(
                segment >= 0,
                segment.astype(np.float32) / self.LABEL_MAX,
                np.float32(self.LABEL_MISSING_NORM))
            gs_full_params = np.concatenate(
                (voxel_centers, point_uniq_col, gs_params,
                 label_norm[:, np.newaxis]), axis=1)
        else:
            gs_full_params = np.concatenate(
                (voxel_centers, point_uniq_col, gs_params), axis=1)

        # Scene-level label distribution (Move 1)
        label_dist = np.zeros(self.NUM_CATS, dtype=np.float32)
        valid_seg  = segment[segment >= 0]
        if len(valid_seg) > 0:
            for k in range(self.NUM_CATS):
                label_dist[k] = (valid_seg == k).sum()
            label_dist /= label_dist.sum()

        return {
            'features':           gs_full_params.astype(np.float32),
            'segment_labels':     segment,
            'instance_labels':    instance,
            'scene_idx':          idx,
            'has_semantics':      has_semantics,
            'num_categories':     self.num_segment_categories,
            'mean_color':         mean_color,
            'label_dist':         label_dist,
            'scaffold_anchors':   scaffold_anchors,
            'scaffold_token_ids': scaffold_token_ids,
            'position_offsets':   position_offsets,
            'category_centroids': category_centroids,  # [72, 3]
            'category_valid':     category_valid,       # [72]
            # NEW — position DC/AC
            'dc_position':        dc_position,          # [40000, 3]
            'position_residuals': position_residuals,   # [40000, 3]
            'voxel_label_dists':  voxel_label_dists,
            'voxel_valid':        voxel_valid,
        }

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


if __name__ == "__main__":
    import sys
    data_path = (sys.argv[1] if len(sys.argv) > 1
                 else "/home/yli11/scratch/datasets/gaussian_world/"
                      "preprocessed/interior_gs/train")

    print("Testing position_layout_residual...")
    ds = gs_dataset(root=data_path, max_scenes=1, normalize=True,
                    scale_norm_mode='linear', color_residual=True,
                    scene_layout_head=True, position_layout_residual=True)
    s = ds[0]

    coord_abs = s['features'][:, 4:7]
    dc        = s['dc_position']
    resid     = s['position_residuals']

    # Verify exact invertibility
    err = np.abs((dc + resid) - coord_abs).max()
    print(f"  Invertibility error: {err:.2e}  (must be < 1e-5)")
    assert err < 1e-5, f"FAIL: {err}"

    print(f"  Absolute range:  [{coord_abs.min():.3f}, {coord_abs.max():.3f}]m")
    print(f"  DC range:        [{dc.min():.3f}, {dc.max():.3f}]m")
    print(f"  Residual range:  [{resid.min():.3f}, {resid.max():.3f}]m")
    reduction = coord_abs.std() / (resid.std() + 1e-8)
    print(f"  Dynamic range reduction: {reduction:.1f}x  (want > 5x)")
    print("\nPASSED")