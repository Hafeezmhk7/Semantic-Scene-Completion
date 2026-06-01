"""
SceneSplat Dataset for Can3Tok Training
========================================
NORMALIZATION:
  normalize_with_norm_factor() is used in _load_and_process().
  - When norm_factor.npy is present (grid chunks after precomputation):
      Uses the global parent-scene coordinate frame. All chunks from the
      same room share one consistent coordinate system.
  - When norm_factor.npy is absent (full scenes in train/):
      Falls back to per-scene normalization — identical to original Can3Tok.

  This means the same dataset class works correctly for BOTH:
    - train_grid1.0cm_chunk8x8_stride6x6/  (chunks → uses norm_factor.npy)
    - train/                                (full scenes → per-scene fallback)
    - val/                                  (full scenes → per-scene fallback)

SKIP_SCENES PARAMETER:
  skip_scenes=N causes the first N sorted scene directories to be skipped
  before applying max_scenes. This lets you carve out a held-out validation
  split from the same directory without any file-system changes.

  Example — training on the first 3800 chunks and evaluating on the rest:
    train_ds = gs_dataset(root=chunk_dir, max_scenes=3800, ...)
    val_ds   = gs_dataset(root=chunk_dir, skip_scenes=3800, ...)

  Because all chunks share the same norm_factor.npy (computed by
  precompute_norm_from_chunks.py from the union of all chunks per scene),
  the held-out chunks use exactly the same global coordinate frame as the
  training chunks — normalization is fully consistent.

PRELOADING (preload=True, default):
  All scenes preprocessed once at __init__ and stored in RAM.
  Workers inherit via copy-on-write fork — zero per-batch overhead.
"""

import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def normalize_to_canonical_sphere(coord, scale, target_radius=10.0,
                                   scale_norm_mode='linear'):
    """Per-scene normalization (fallback when norm_factor.npy absent)."""
    center       = coord.mean(axis=0)
    coord_c      = coord - center
    max_dist     = np.linalg.norm(coord_c, axis=1).max()
    if max_dist < 1e-6:
        max_dist = 1.0
    scale_factor = target_radius / (max_dist * 1.1)
    coord_norm   = coord_c * scale_factor
    if scale_norm_mode == 'log':
        scale_norm = np.log(scale + 1e-7) + np.log(scale_factor)
    else:
        scale_norm = scale * scale_factor
    return coord_norm, scale_norm


def normalize_with_norm_factor(coord, scale, scene_dir,
                                target_radius=10.0, scale_norm_mode='linear'):
    """
    Normalize using precomputed global norm_factor when available,
    otherwise fall back to per-scene normalization.

    norm_factor.npy = [cx, cy, cz, scale_factor]
    Produced by precompute_norm_from_chunks.py, which combines ALL chunks of
    a scene → computes center+scale from the union → global scene frame.

    WHY THIS MATTERS FOR CHUNKS:
      Without: each chunk normalized into its own local 10m sphere.
        Chunk A (left side of room): positions centered on left wall.
        Chunk B (right side of room): positions centered on right wall.
        z_A ≈ z_B (encoder sees similar content) but targets differ
        → gradient cancellation → position loss stuck at ~2300.

      With: all chunks of room share one coordinate frame.
        Position [2, 0, 1] always refers to the same physical location.
        Decoder can learn generalizable position priors → converges.

    FALLBACK (full scenes, val set):
      If norm_factor.npy is absent, identical to normalize_to_canonical_sphere.
    """
    nf_path = os.path.join(scene_dir, 'norm_factor.npy')

    if os.path.exists(nf_path):
        nf           = np.load(nf_path)
        center       = nf[:3]
        scale_factor = float(nf[3])
        coord_norm   = (coord - center) * scale_factor
    else:
        center       = coord.mean(axis=0)
        coord_c      = coord - center
        max_dist     = np.linalg.norm(coord_c, axis=1).max()
        if max_dist < 1e-6:
            max_dist = 1.0
        scale_factor = target_radius / (max_dist * 1.1)
        coord_norm   = coord_c * scale_factor

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
    """Build 8×8×8 scaffold grid with HARD voxel assignment."""
    num_tokens  = scaffold_dims ** 3
    cell_size   = domain_size / scaffold_dims
    half_domain = domain_size / 2.0

    shifted = coord + half_domain
    sv_idx  = np.floor(shifted / cell_size).astype(np.int32)
    sv_idx  = np.clip(sv_idx, 0, scaffold_dims - 1)
    scaffold_token_ids = (
        sv_idx[:, 0] * scaffold_dims ** 2 +
        sv_idx[:, 1] * scaffold_dims +
        sv_idx[:, 2]).astype(np.int32)

    anchor_counts = np.bincount(scaffold_token_ids, minlength=num_tokens).astype(np.float64)
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
        scaffold_anchors[empty_idx, 0] = ix * cell_size + cell_size/2.0 - half_domain
        scaffold_anchors[empty_idx, 1] = iy * cell_size + cell_size/2.0 - half_domain
        scaffold_anchors[empty_idx, 2] = iz * cell_size + cell_size/2.0 - half_domain
    scaffold_anchors = scaffold_anchors.astype(np.float32)

    position_offsets = (coord - scaffold_anchors[scaffold_token_ids]).astype(np.float32)
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


def compute_position_layout_residuals(coord, segment, category_centroids, category_valid):
    """DC/AC position decomposition using per-category centroids."""
    N            = len(coord)
    dc_position  = np.zeros((N, 3), dtype=np.float32)
    scene_mean   = coord.mean(axis=0).astype(np.float32)
    valid_mask   = segment >= 0
    invalid_mask = ~valid_mask
    if valid_mask.sum() > 0:
        valid_segs = segment[valid_mask].astype(np.int64)
        dc_position[valid_mask] = category_centroids[valid_segs]
        absent_cat_mask = valid_mask.copy()
        absent_cat_mask[valid_mask] = (category_valid[valid_segs] == 0)
        if absent_cat_mask.sum() > 0:
            dc_position[absent_cat_mask] = scene_mean
    if invalid_mask.sum() > 0:
        dc_position[invalid_mask] = scene_mean
    position_residuals = (coord - dc_position).astype(np.float32)
    return dc_position, position_residuals


def compute_voxel_label_dists(scaffold_token_ids, segment, num_tokens=512, num_cats=72):
    """Per-super-voxel label distributions."""
    voxel_label_dists = np.zeros((num_tokens, num_cats), dtype=np.float32)
    voxel_valid       = np.zeros(num_tokens, dtype=np.float32)
    valid_mask = segment >= 0
    if valid_mask.sum() == 0:
        return voxel_label_dists, voxel_valid
    valid_tids = scaffold_token_ids[valid_mask].astype(np.int64)
    valid_segs = segment[valid_mask].astype(np.int64)
    combined = valid_tids * num_cats + valid_segs
    counts   = np.bincount(combined, minlength=num_tokens * num_cats).reshape(num_tokens, num_cats)
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
      4:7   xyz  (normalized coordinates)
      7:10  rgb
      10    opacity
      11:14 scale
      14:18 quaternion

    Parameters
    ----------
    skip_scenes : int or None
        Number of sorted scene directories to skip before applying max_scenes.
        Use this to carve out a held-out validation split from the same folder
        as the training data without any file-system changes.
        Example:
            train = gs_dataset(root=chunk_dir, max_scenes=3800, ...)
            val   = gs_dataset(root=chunk_dir, skip_scenes=3800, ...)
        Both use the same global norm_factor.npy — normalization is consistent.
    """

    TARGET_POINTS      = 40_000
    LABEL_MAX          = 71.0
    LABEL_MISSING_NORM = -1.0 / 71.0
    SCAFFOLD_DIMS      = 8
    SCAFFOLD_DOMAIN    = 16.0
    SCAFFOLD_TOKENS    = 512
    NUM_CATS           = 72

    def __init__(self, root, resol=200, random_permute=False, train=True,
                 sampling_method='opacity', max_scenes=None, skip_scenes=None,
                 normalize=True, normalize_colors=True, use_chunk_norm_factor=True, target_radius=10.0,
                 scale_norm_mode='linear', label_input=False, color_residual=False,
                 position_scaffold=False, scene_layout_head=False, jepa_idea1=False,
                 position_layout_residual=False, preload=True,
                 disable_semantics=False):

        self.root                     = root
        self.resol                    = resol
        self.random_permute           = random_permute
        self.train                    = train
        self.sampling_method          = sampling_method
        self.normalize                = normalize
        self.normalize_colors         = normalize_colors
        self.use_chunk_norm_factor    = use_chunk_norm_factor
        self.target_radius            = target_radius
        self.scale_norm_mode          = scale_norm_mode
        self.label_input              = label_input
        self.color_residual           = color_residual
        self.position_scaffold        = position_scaffold
        self.scene_layout_head        = scene_layout_head
        self.jepa_idea1               = jepa_idea1
        self.position_layout_residual = position_layout_residual
        self.disable_semantics        = disable_semantics  # True for non-ScanNet72 datasets

        if position_layout_residual and not scene_layout_head:
            print("  [INFO] position_layout_residual=True requires scene_layout_head=True. Enabling.")
            self.scene_layout_head = True
        if jepa_idea1 and not position_scaffold:
            print("  [INFO] jepa_idea1=True requires position_scaffold=True. Enabling.")
            self.position_scaffold = True

        # Build sorted list of all scene directories
        self.scene_dirs = sorted([
            os.path.join(root, d)
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        total_available = len(self.scene_dirs)

        # ── SKIP: remove the first skip_scenes dirs (used as training data) ──
        # This is the key mechanism for held-out chunk validation:
        # train = gs_dataset(root=chunk_dir, max_scenes=3800, ...)
        # val   = gs_dataset(root=chunk_dir, skip_scenes=3800, ...)
        # Both sets of chunks share the same norm_factor.npy per scene, so
        # the global coordinate frame is identical for training and validation.
        if skip_scenes is not None and skip_scenes > 0:
            self.scene_dirs = self.scene_dirs[skip_scenes:]
            print(f"  Skipped first {skip_scenes} scenes (training split)  "
                  f"→ {len(self.scene_dirs)} remaining for this split")

        # ── CAP at max_scenes ────────────────────────────────────────────────
        if max_scenes is not None and max_scenes < len(self.scene_dirs):
            self.scene_dirs = self.scene_dirs[:max_scenes]
            print(f"  Limited to {max_scenes} scenes")

        if not self.scene_dirs:
            raise ValueError(
                f"No scene directories found in {root} "
                f"(total={total_available}, skip={skip_scenes}, max={max_scenes})")

        self.num_segment_categories = self.NUM_CATS
        self.feature_width = 19 if label_input else 18

        # ── Normalization mode report ─────────────────────────────────────────
        _is_chunks   = ('chunk' in root or 'grid' in root)
        _sample_size = min(20, len(self.scene_dirs))
        _nf_count    = sum(
            1 for d in self.scene_dirs[:_sample_size]
            if os.path.exists(os.path.join(d, 'norm_factor.npy'))
        )

        print(f"  Loaded {len(self.scene_dirs)} scenes from {os.path.basename(root)}")
        if _is_chunks:
            _all_ok = (_nf_count == _sample_size)
            if not use_chunk_norm_factor:
                _status = 'DISABLED by --no_chunk_norm_factor (per-scene fallback)'
            elif _all_ok:
                _status = 'GLOBAL frame ✓'
            else:
                _status = 'MISSING — position will NOT converge ✗'
            print(f"  norm_factor.npy : {_nf_count}/{_sample_size} sampled  ({_status})")
            if _all_ok and _sample_size > 0:
                # Spot-check: show one example norm_factor value
                _ex = self.scene_dirs[0]
                _nf = np.load(os.path.join(_ex, 'norm_factor.npy'))
                print(f"  Example         : {os.path.basename(_ex)} → "
                      f"center=({_nf[0]:.2f},{_nf[1]:.2f},{_nf[2]:.2f}) "
                      f"scale={_nf[3]:.4f}")
        else:
            _has_nf = os.path.exists(os.path.join(self.scene_dirs[0], 'norm_factor.npy'))
            print(f"  norm_factor.npy : {'present → global frame' if _has_nf else 'absent → per-scene fallback ✓'}")

        print(f"  color_residual={color_residual} | position_scaffold={self.position_scaffold}")
        print(f"  scene_layout_head={self.scene_layout_head}")

        self._preloaded = None
        if preload:
            self._preload_all()

    def _preload_all(self):
        n = len(self.scene_dirs)
        est_gb = n * (40000 * 18 * 4 + 40000 * 6 + 512 * 3 * 4) / 1e9
        print(f"  Preloading {n} scenes into RAM (~{est_gb:.1f} GB)...")
        self._preloaded = [None] * n
        failed = 0
        for idx in tqdm(range(n), desc="  Preloading", ncols=80, leave=False):
            try:
                self._preloaded[idx] = self._load_and_process(idx)
            except Exception as e:
                print(f"\n  [WARNING] Failed scene {idx} ({self.scene_dirs[idx]}): {e}")
                failed += 1
                if idx > 0 and self._preloaded[0] is not None:
                    self._preloaded[idx] = self._preloaded[0]
        print(f"  Preloaded {n - failed}/{n} scenes. "
              f"{'All OK.' if failed == 0 else f'{failed} used fallback.'}")

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        if self._preloaded is not None:
            return self._preloaded[idx]
        return self._load_and_process(idx)

    def _load_and_process(self, idx):
        scene_dir = self.scene_dirs[idx]

        coord   = np.load(os.path.join(scene_dir, 'coord.npy'))
        color   = np.load(os.path.join(scene_dir, 'color.npy'))
        scale   = np.load(os.path.join(scene_dir, 'scale.npy'))
        quat    = np.load(os.path.join(scene_dir, 'quat.npy'))
        opacity = np.load(os.path.join(scene_dir, 'opacity.npy'))

        if self.normalize:
            # use_chunk_norm_factor=True  → use norm_factor.npy global frame for
            #   chunks (all chunks of a room share one coordinate system).
            #   This is REQUIRED for position loss convergence on chunks.
            # use_chunk_norm_factor=False → ignore norm_factor.npy even if present,
            #   forcing per-scene normalisation. Use this to ablate the global frame
            #   and verify that it is the cause of convergence difference.
            # Full scenes: norm_factor.npy is absent so both modes give the same
            #   per-scene fallback regardless of this flag.
            _is_chunk_data = ('chunk' in scene_dir or 'grid' in scene_dir)
            _use_nf = self.use_chunk_norm_factor or not _is_chunk_data
            if _use_nf:
                coord, scale = normalize_with_norm_factor(
                    coord, scale,
                    scene_dir=scene_dir,
                    target_radius=self.target_radius,
                    scale_norm_mode=self.scale_norm_mode)
            else:
                # Per-scene fallback: ignore norm_factor.npy even if present.
                coord, scale = normalize_to_canonical_sphere(
                    coord, scale,
                    target_radius=self.target_radius,
                    scale_norm_mode=self.scale_norm_mode)

        if self.normalize_colors:
            color = color / 255.0

        if self.disable_semantics:
            # Extra datasets (ArkitScenes, ScanNet++, etc.) use different label
            # spaces incompatible with ScanNet72. Disable semantics so InfoNCE
            # losses don't receive mixed-space labels. Reconstruction is unaffected.
            segment       = np.full(len(coord), -1, dtype=np.int16)
            instance      = np.full(len(coord), -1, dtype=np.int32)
            has_semantics = False
        else:
            try:
                segment  = np.load(os.path.join(scene_dir, 'segment.npy'))
                instance = np.load(os.path.join(scene_dir, 'instance.npy'))
                has_semantics = True
            except FileNotFoundError:
                segment       = np.full(len(coord), -1, dtype=np.int16)
                instance      = np.full(len(coord), -1, dtype=np.int32)
                has_semantics = False

        # Top-40k sampling by opacity (deterministic)
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

        if self.color_residual:
            mean_color = color.mean(axis=0).astype(np.float32)
            color      = color - mean_color
        else:
            mean_color = np.zeros(3, dtype=np.float32)

        T_pts = len(coord)
        if self.position_scaffold:
            (scaffold_anchors,
             scaffold_token_ids,
             position_offsets) = compute_position_scaffold(
                coord, scaffold_dims=self.SCAFFOLD_DIMS, domain_size=self.SCAFFOLD_DOMAIN)
        else:
            scaffold_anchors   = np.zeros((self.SCAFFOLD_TOKENS, 3), dtype=np.float32)
            scaffold_token_ids = np.zeros(T_pts, dtype=np.int32)
            position_offsets   = np.zeros((T_pts, 3), dtype=np.float32)

        if self.scene_layout_head:
            category_centroids, category_valid = compute_category_centroids(
                coord, segment, num_cats=self.NUM_CATS)
        else:
            category_centroids = np.zeros((self.NUM_CATS, 3), dtype=np.float32)
            category_valid     = np.zeros(self.NUM_CATS, dtype=np.float32)

        if self.position_layout_residual:
            dc_position, position_residuals = compute_position_layout_residuals(
                coord, segment, category_centroids, category_valid)
        else:
            dc_position        = np.zeros((T_pts, 3), dtype=np.float32)
            position_residuals = np.zeros((T_pts, 3), dtype=np.float32)

        if self.jepa_idea1 and self.position_scaffold:
            voxel_label_dists, voxel_valid = compute_voxel_label_dists(
                scaffold_token_ids, segment,
                num_tokens=self.SCAFFOLD_TOKENS, num_cats=self.NUM_CATS)
        else:
            voxel_label_dists = np.zeros((self.SCAFFOLD_TOKENS, self.NUM_CATS), dtype=np.float32)
            voxel_valid       = np.zeros(self.SCAFFOLD_TOKENS, dtype=np.float32)

        volume_dims    = 40
        resolution     = 16.0 / volume_dims
        uniq_idx, inv_idx, _ = voxelize(coord, resolution, 'fnv')
        origin_offset  = np.array([(volume_dims - 1) / 2] * 3) * resolution
        shifted_pts    = coord + origin_offset
        voxel_idx      = np.floor(shifted_pts / resolution)
        voxel_idx      = np.clip(voxel_idx, 0, volume_dims - 1)
        voxel_centers  = (voxel_idx - (volume_dims - 1) / 2) * resolution
        point_uniq_idx = uniq_idx[inv_idx]

        opacity_col    = opacity[:, np.newaxis]
        point_uniq_col = point_uniq_idx[:, np.newaxis]
        gs_params      = np.concatenate((coord, color, opacity_col, scale, quat), axis=1)

        if self.label_input:
            label_norm = np.where(
                segment >= 0,
                segment.astype(np.float32) / self.LABEL_MAX,
                np.float32(self.LABEL_MISSING_NORM))
            gs_full_params = np.concatenate(
                (voxel_centers, point_uniq_col, gs_params, label_norm[:, np.newaxis]), axis=1)
        else:
            gs_full_params = np.concatenate(
                (voxel_centers, point_uniq_col, gs_params), axis=1)

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
            'category_centroids': category_centroids,
            'category_valid':     category_valid,
            'dc_position':        dc_position,
            'position_residuals': position_residuals,
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
    import sys, time
    data_path = (sys.argv[1] if len(sys.argv) > 1
                 else "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs/train_grid1.0cm_chunk8x8_stride6x6")

    print(f"Testing dataset: {data_path}")

    # Test train/val split via skip_scenes
    ds_train = gs_dataset(root=data_path, max_scenes=5, normalize=True,
                          scale_norm_mode='linear', color_residual=True,
                          position_scaffold=True, preload=True)
    ds_val   = gs_dataset(root=data_path, skip_scenes=5, max_scenes=3,
                          normalize=True, scale_norm_mode='linear',
                          color_residual=True, position_scaffold=True, preload=True)

    print(f"\n  Train: {len(ds_train)} scenes  |  Val: {len(ds_val)} scenes")
    assert ds_train.scene_dirs[-1] != ds_val.scene_dirs[0], \
        "ERROR: train and val share the same scene dirs!"
    print(f"  Train last : {ds_train.scene_dirs[-1]}")
    print(f"  Val first  : {ds_val.scene_dirs[0]}")
    print(f"  No overlap: ✓")

    t0 = time.time()
    s            = ds_train[0]
    coord_abs    = s['features'][:, 4:7]
    scaf_anchors = s['scaffold_anchors']
    token_ids    = s['scaffold_token_ids']
    pos_offsets  = s['position_offsets']
    dc  = scaf_anchors[token_ids]
    err = np.abs((dc + pos_offsets) - coord_abs).max()
    print(f"\n  Scaffold invertibility error: {err:.2e}  (must be < 1e-5)")
    assert err < 1e-5, f"FAIL: {err}"
    print(f"PASSED")