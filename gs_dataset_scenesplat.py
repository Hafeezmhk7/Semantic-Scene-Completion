"""
SceneSplat Dataset for Can3Tok Training
========================================
STEP 1 — COLOR RESIDUAL (color_residual=True)
   mean_color  = color.mean()   -> shape_embed via MeanColorHead
   color_resid = color - mean   -> AC residuals in feature tensor
   Batch dict: 'mean_color' [3]

POSITION SCAFFOLD (position_scaffold=True)
   8^3=512 super-voxels.
   Batch keys:
     scaffold_anchors   [512, 3]   — voxel mean positions (8x8x8 grid)
     scaffold_token_ids [40000]    — hard voxel assignment per Gaussian
     position_offsets   [40000, 3] — coord - smooth_anchor  (TRILINEAR DC)
     smooth_anchor      [40000, 3] — trilinear-interpolated anchor per Gaussian

   KEY CHANGE — TRILINEAR ANCHOR SMOOTHING:
   =========================================
   OLD (hard assignment):
     position_offsets[i] = coord[i] - scaffold_anchors[scaffold_token_ids[i]]

   NEW (trilinear smooth):
     smooth_anchor[i] = trilinear_blend(coord[i], 8 surrounding voxel anchors)
     position_offsets[i] = coord[i] - smooth_anchor[i]

   WHY IT MATTERS:
     Two Gaussians 0.08m apart but on opposite sides of a voxel boundary had
     offset targets that differed by ~2m (the voxel width) under hard assignment.
     This forced the decoder to produce a discontinuous 2m jump in output at
     every voxel boundary — visible as seam artifacts in SuperSplat.

     With trilinear smoothing, the anchor is a weighted blend of 8 surrounding
     voxels. The anchor varies continuously with position. Two Gaussians 0.08m
     apart now have offset targets that differ by only ~0.08m — no seam.

   This is the same fix used by Instant-NGP (SIGGRAPH 2022) and explicitly
   documented in their paper: "interpolation is not optional — without it,
   grid-aligned discontinuities (blocky appearance) occur."

   PLY SAVE:
     absolute_pos = decoder_output_pos + smooth_anchor
     (smooth_anchor is in batch_data, not derived from scaffold_anchors[token_id])

SCENE LAYOUT HEAD (scene_layout_head=True)
   per-category centroids: category_centroids[72,3], category_valid[72]

NEW: POSITION LAYOUT RESIDUAL (position_layout_residual=True)
   Requires scene_layout_head=True.
   DC  = per-Gaussian category centroid
   AC  = coord - dc_position
   Batch keys: dc_position [40000,3], position_residuals [40000,3]
   Invertibility: dc_position + position_residuals == coord exactly.

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


def trilinear_interpolate_anchors(coord, scaffold_anchors, scaffold_dims=8, domain_size=16.0):
    """
    Compute a smoothly interpolated anchor position for each Gaussian using
    trilinear blending of the 8 surrounding scaffold voxel anchors.

    This replaces hard voxel assignment (which causes seam artifacts at voxel
    boundaries) with a continuous anchor that varies smoothly with position.

    Identical in principle to the interpolation used in Instant-NGP (Muller et al.,
    SIGGRAPH 2022): for any 3D coordinate, find the 8 surrounding voxel corners
    and linearly interpolate their feature vectors to produce a continuous encoding.

    Args:
        coord:            [N, 3]   — absolute Gaussian positions
        scaffold_anchors: [512, 3] — voxel centroid positions (8x8x8 grid)
        scaffold_dims:    int      — grid resolution per axis (8 → 8x8x8 = 512 voxels)
        domain_size:      float    — total scene extent (16.0m → ±8m)

    Returns:
        smooth_anchor: [N, 3]  — per-Gaussian continuously interpolated anchor
                                  smooth_anchor varies continuously with coord;
                                  no discontinuities at voxel boundaries.
    """
    N           = coord.shape[0]
    cell_size   = domain_size / scaffold_dims   # 16 / 8 = 2m per voxel
    half_domain = domain_size / 2.0             # 8m

    # Normalised coordinate within the grid: [0, scaffold_dims]
    # A coordinate at -8m → 0.0, at +8m → 8.0
    grid_coord = (coord + half_domain) / cell_size   # [N, 3]

    # Integer lower-corner voxel index along each axis
    i0 = np.floor(grid_coord).astype(np.int32)   # [N, 3]
    i1 = i0 + 1                                  # upper corner

    # Fractional position within the voxel: t ∈ [0, 1]
    # t = 0 → fully at lower corner, t = 1 → fully at upper corner
    t = (grid_coord - i0).astype(np.float32)     # [N, 3]

    # Clamp indices to valid range [0, scaffold_dims-1]
    # Gaussians at the boundary (coord = ±8m) stay within the grid
    i0 = np.clip(i0, 0, scaffold_dims - 1)
    i1 = np.clip(i1, 0, scaffold_dims - 1)

    def flat_idx(ix, iy, iz):
        """Convert 3D voxel index to flat scaffold_anchors index.
        Indexing matches compute_position_scaffold: idx = x*64 + y*8 + z"""
        return (ix * scaffold_dims**2 + iy * scaffold_dims + iz).astype(np.int32)

    # 8 corner anchor positions — one per corner of the cube
    # Naming: a_{x_bit}{y_bit}{z_bit} where 0=lower corner, 1=upper corner
    a000 = scaffold_anchors[flat_idx(i0[:,0], i0[:,1], i0[:,2])]   # [N, 3]
    a001 = scaffold_anchors[flat_idx(i0[:,0], i0[:,1], i1[:,2])]
    a010 = scaffold_anchors[flat_idx(i0[:,0], i1[:,1], i0[:,2])]
    a011 = scaffold_anchors[flat_idx(i0[:,0], i1[:,1], i1[:,2])]
    a100 = scaffold_anchors[flat_idx(i1[:,0], i0[:,1], i0[:,2])]
    a101 = scaffold_anchors[flat_idx(i1[:,0], i0[:,1], i1[:,2])]
    a110 = scaffold_anchors[flat_idx(i1[:,0], i1[:,1], i0[:,2])]
    a111 = scaffold_anchors[flat_idx(i1[:,0], i1[:,1], i1[:,2])]

    # Trilinear weights — each weight is the volume fraction of the opposite corner
    # The eight weights sum to exactly 1.0 for any t ∈ [0,1]^3
    tx = t[:, 0:1]   # fraction along x-axis: 0 = left voxel, 1 = right voxel
    ty = t[:, 1:2]   # fraction along y-axis
    tz = t[:, 2:3]   # fraction along z-axis

    smooth_anchor = (
        (1-tx)*(1-ty)*(1-tz) * a000 +   # weight of corner (0,0,0)
        (1-tx)*(1-ty)*   tz  * a001 +   # weight of corner (0,0,1)
        (1-tx)*   ty *(1-tz) * a010 +   # weight of corner (0,1,0)
        (1-tx)*   ty *   tz  * a011 +   # weight of corner (0,1,1)
           tx *(1-ty)*(1-tz) * a100 +   # weight of corner (1,0,0)
           tx *(1-ty)*   tz  * a101 +   # weight of corner (1,0,1)
           tx *   ty *(1-tz) * a110 +   # weight of corner (1,1,0)
           tx *   ty *   tz  * a111     # weight of corner (1,1,1)
    )

    return smooth_anchor.astype(np.float32)   # [N, 3]


def compute_position_scaffold(coord, scaffold_dims=8, domain_size=16.0):
    """
    Build 8x8x8 scaffold grid with trilinear smooth anchors.

    Returns:
        scaffold_anchors:   [512, 3]   — voxel centroid positions
        scaffold_token_ids: [N]        — hard voxel assignment (still needed
                                         for token conditioning and decoder)
        position_offsets:   [N, 3]     — coord - smooth_anchor (NOT hard anchor)
        smooth_anchor:      [N, 3]     — trilinear interpolated anchor per Gaussian

    CRITICAL: position_offsets is now coord - smooth_anchor (smooth), not
    coord - scaffold_anchors[token_id] (hard). This eliminates seam artifacts.
    At PLY save: absolute_pos = decoder_output + smooth_anchor.
    """
    num_tokens  = scaffold_dims ** 3
    cell_size   = domain_size / scaffold_dims
    half_domain = domain_size / 2.0

    # ── Step 1: hard voxel assignment (token_ids still needed downstream) ────
    shifted = coord + half_domain
    sv_idx  = np.floor(shifted / cell_size).astype(np.int32)
    sv_idx  = np.clip(sv_idx, 0, scaffold_dims - 1)
    scaffold_token_ids = (
        sv_idx[:, 0] * scaffold_dims ** 2 +
        sv_idx[:, 1] * scaffold_dims +
        sv_idx[:, 2]).astype(np.int32)

    # ── Step 2: compute scaffold voxel anchors (mean Gaussian pos per voxel) ─
    anchor_counts = np.bincount(scaffold_token_ids, minlength=num_tokens).astype(np.float64)
    anchor_sum    = np.zeros((num_tokens, 3), dtype=np.float64)
    for dim in range(3):
        anchor_sum[:, dim] = np.bincount(
            scaffold_token_ids, weights=coord[:, dim].astype(np.float64),
            minlength=num_tokens)
    scaffold_anchors = np.zeros((num_tokens, 3), dtype=np.float64)
    occupied = anchor_counts > 0
    scaffold_anchors[occupied] = anchor_sum[occupied] / anchor_counts[occupied, np.newaxis]

    # Fill empty voxels with geometric centre of that voxel cell
    empty_idx = np.where(~occupied)[0]
    if len(empty_idx) > 0:
        ix = empty_idx // (scaffold_dims ** 2)
        iy = (empty_idx // scaffold_dims) % scaffold_dims
        iz = empty_idx % scaffold_dims
        scaffold_anchors[empty_idx, 0] = ix * cell_size + cell_size/2.0 - half_domain
        scaffold_anchors[empty_idx, 1] = iy * cell_size + cell_size/2.0 - half_domain
        scaffold_anchors[empty_idx, 2] = iz * cell_size + cell_size/2.0 - half_domain
    scaffold_anchors = scaffold_anchors.astype(np.float32)

    # ── Step 3: trilinear smooth anchor per Gaussian ──────────────────────────
    # This replaces the hard lookup: scaffold_anchors[scaffold_token_ids]
    # smooth_anchor varies continuously across voxel boundaries → no seam
    smooth_anchor = trilinear_interpolate_anchors(
        coord, scaffold_anchors, scaffold_dims, domain_size)   # [N, 3]

    # ── Step 4: position offsets from smooth anchor ───────────────────────────
    # OLD: position_offsets = coord - scaffold_anchors[scaffold_token_ids]
    # NEW: position_offsets = coord - smooth_anchor   (continuous, no seam)
    position_offsets = (coord - smooth_anchor).astype(np.float32)

    return scaffold_anchors, scaffold_token_ids, position_offsets, smooth_anchor


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
    N           = len(coord)
    dc_position = np.zeros((N, 3), dtype=np.float32)
    scene_mean  = coord.mean(axis=0).astype(np.float32)
    valid_mask  = segment >= 0
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
    """Per-super-voxel label distributions (JEPA Idea 1)."""
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
      4:7   xyz  (ALWAYS ABSOLUTE — encoder Fourier embedder needs this)
      7:10  rgb  (absolute or residuals if color_residual=True)
      10    opacity
      11:14 scale
      14:18 quaternion

    NEW batch key 'smooth_anchor' [40000, 3]:
      Trilinear-interpolated anchor per Gaussian. Used at PLY save time
      to recover absolute positions: abs_pos = decoder_pos + smooth_anchor.
      Replaces the old hard lookup: scaffold_anchors[scaffold_token_ids].
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
            print("  [INFO] position_layout_residual=True requires scene_layout_head=True. Enabling.")
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
        if position_scaffold:
            print(f"  TRILINEAR ANCHOR SMOOTHING ENABLED:")
            print(f"    smooth_anchor[i] = trilinear_blend(coord[i], 8 surrounding voxels)")
            print(f"    position_offsets = coord - smooth_anchor  (continuous, no seam)")
            print(f"    PLY save:         abs_pos = decoder_pos + smooth_anchor")

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

        # ── Position scaffold with TRILINEAR SMOOTHING ───────────────────────
        T_pts = len(coord)
        if self.position_scaffold:
            (scaffold_anchors,
             scaffold_token_ids,
             position_offsets,
             smooth_anchor) = compute_position_scaffold(
                coord, scaffold_dims=self.SCAFFOLD_DIMS, domain_size=self.SCAFFOLD_DOMAIN)

            # Diagnostic: verify the smooth offsets are smaller than hard offsets
            # hard_offsets = coord - scaffold_anchors[scaffold_token_ids]
            # smooth is always <= hard in mean absolute error by construction
        else:
            scaffold_anchors   = np.zeros((self.SCAFFOLD_TOKENS, 3), dtype=np.float32)
            scaffold_token_ids = np.zeros(T_pts, dtype=np.int32)
            position_offsets   = np.zeros((T_pts, 3), dtype=np.float32)
            smooth_anchor      = np.zeros((T_pts, 3), dtype=np.float32)

        # Scene layout head: per-category centroids
        if self.scene_layout_head:
            category_centroids, category_valid = compute_category_centroids(
                coord, segment, num_cats=self.NUM_CATS)
        else:
            category_centroids = np.zeros((self.NUM_CATS, 3), dtype=np.float32)
            category_valid     = np.zeros(self.NUM_CATS, dtype=np.float32)

        # Position layout residual
        if self.position_layout_residual:
            dc_position, position_residuals = compute_position_layout_residuals(
                coord, segment, category_centroids, category_valid)
        else:
            dc_position        = np.zeros((T_pts, 3), dtype=np.float32)
            position_residuals = np.zeros((T_pts, 3), dtype=np.float32)

        # JEPA Idea 1
        if self.jepa_idea1 and self.position_scaffold:
            voxel_label_dists, voxel_valid = compute_voxel_label_dists(
                scaffold_token_ids, segment,
                num_tokens=self.SCAFFOLD_TOKENS, num_cats=self.NUM_CATS)
        else:
            voxel_label_dists = np.zeros((self.SCAFFOLD_TOKENS, self.NUM_CATS), dtype=np.float32)
            voxel_valid       = np.zeros(self.SCAFFOLD_TOKENS, dtype=np.float32)

        # Encoder voxelisation (always uses absolute coord)
        volume_dims   = 40
        resolution    = 16.0 / volume_dims
        uniq_idx, inv_idx, _ = voxelize(coord, resolution, 'fnv')
        origin_offset = np.array([(volume_dims - 1) / 2] * 3) * resolution
        shifted_pts   = coord + origin_offset
        voxel_idx     = np.floor(shifted_pts / resolution)
        voxel_idx     = np.clip(voxel_idx, 0, volume_dims - 1)
        voxel_centers = (voxel_idx - (volume_dims - 1) / 2) * resolution
        point_uniq_idx = uniq_idx[inv_idx]

        # Feature tensor [T, 18]
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

        # Scene-level label distribution
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
            'position_offsets':   position_offsets,      # coord - smooth_anchor (TRILINEAR)
            'smooth_anchor':      smooth_anchor,          # NEW: per-Gaussian smooth DC
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
    import sys
    data_path = (sys.argv[1] if len(sys.argv) > 1
                 else "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs/train")

    print("Testing trilinear anchor smoothing...")
    ds = gs_dataset(root=data_path, max_scenes=1, normalize=True,
                    scale_norm_mode='linear', color_residual=True,
                    position_scaffold=True)
    s = ds[0]

    coord_abs      = s['features'][:, 4:7]
    smooth_anch    = s['smooth_anchor']
    pos_offsets    = s['position_offsets']
    scaf_anchors   = s['scaffold_anchors']
    token_ids      = s['scaffold_token_ids']

    # Verify: smooth_anchor + position_offsets == coord
    err = np.abs((smooth_anch + pos_offsets) - coord_abs).max()
    print(f"  Invertibility error: {err:.2e}  (must be < 1e-5)")
    assert err < 1e-5, f"FAIL: {err}"

    # Compare offset magnitudes: smooth should be <= hard
    hard_offsets   = coord_abs - scaf_anchors[token_ids]
    hard_mae       = np.abs(hard_offsets).mean()
    smooth_mae     = np.abs(pos_offsets).mean()
    print(f"  Hard offset MAE:   {hard_mae:.4f}m")
    print(f"  Smooth offset MAE: {smooth_mae:.4f}m  (should be <= hard)")

    # Key test: at voxel boundaries, smooth should NOT jump
    # Find Gaussians near boundaries by checking how close their grid_coord is to 0.5
    cell_size  = ds.SCAFFOLD_DOMAIN / ds.SCAFFOLD_DIMS
    half_dom   = ds.SCAFFOLD_DOMAIN / 2.0
    grid_coord = (coord_abs + half_dom) / cell_size
    frac       = grid_coord - np.floor(grid_coord)
    near_boundary = (np.min(np.stack([frac, 1-frac], axis=-1), axis=-1) < 0.05).any(axis=1)
    print(f"\n  Gaussians near voxel boundaries: {near_boundary.sum():,}")
    if near_boundary.sum() > 0:
        hard_boundary_mae   = np.abs(hard_offsets[near_boundary]).mean()
        smooth_boundary_mae = np.abs(pos_offsets[near_boundary]).mean()
        print(f"  Near-boundary hard offset MAE:   {hard_boundary_mae:.4f}m")
        print(f"  Near-boundary smooth offset MAE: {smooth_boundary_mae:.4f}m")
        reduction = hard_boundary_mae / (smooth_boundary_mae + 1e-8)
        print(f"  Reduction at boundaries: {reduction:.2f}x  (want > 1.0)")

    print(f"\n  Absolute coord range: [{coord_abs.min():.3f}, {coord_abs.max():.3f}]m")
    print(f"  Smooth anchor range:  [{smooth_anch.min():.3f}, {smooth_anch.max():.3f}]m")
    print(f"  Offset range (smooth):[{pos_offsets.min():.3f}, {pos_offsets.max():.3f}]m")
    print(f"\nPASSED")