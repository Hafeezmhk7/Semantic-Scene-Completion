"""
SceneSplat Dataset for Can3Tok Training
========================================

STEP 1 — COLOR RESIDUAL (color_residual=True)
   mean_color  = color.mean()   → shape_embed supervised via MeanColorHead
   color_resid = color - mean   → AC residuals stored in feature tensor
   Batch dict: 'mean_color' [3]

POSITION SCAFFOLD (position_scaffold=True) — Scaffold-GS inspired
   Divides the scene into 8×8×8 = 512 super-voxels (matching the 512 latent
   tokens). Each super-voxel k has an anchor â_k = mean position of its
   Gaussians. Position offsets δp_i = p_i - â_{k(i)} are the reconstruction
   target instead of absolute coordinates.

   Why this helps (Scaffold-GS, Lu et al. CVPR 2024; LION, Zeng et al. 2022):
     - Offset range ~[-1,+1]m vs absolute range ~[-10,+10]m → 10× variance reduction
     - Decoder position head has a much easier regression target
     - shape_embed → AnchorPositionHead → [512,3] gets a new gradient path

   IMPORTANT: The encoder feature tensor always stores ABSOLUTE positions (cols 4:7).
   The scaffold data is returned as SEPARATE batch dict keys for the training loop.
   Batch dict: 'scaffold_anchors' [512,3], 'scaffold_token_ids' [40000],
               'position_offsets' [40000,3]

MOVE 1 — SCENE SEMANTIC (always computed)
   Label distribution precomputed in DataLoader workers → zero GPU cost.
   Batch dict: 'label_dist' [72]
"""

import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
#  Canonical Sphere Normalization
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Voxelisation (for encoder positional encoding — unchanged from baseline)
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
# Position Scaffold — 8×8×8 super-voxel anchor computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_position_scaffold(coord, scaffold_dims=8, domain_size=16.0):
    """
    Compute scaffold anchors and per-Gaussian token assignments for position
    residual encoding. Implements the two-level position decomposition from
    Scaffold-GS (Lu et al., CVPR 2024) adapted to the Can3Tok latent structure.

    Architecture mapping:
      Scaffold-GS anchor  â_k  ↔  Can3Tok super-voxel centre
      Scaffold-GS child Gaussians ↔  Can3Tok Gaussians assigned to token k
      Scaffold-GS offset  δp_j ↔  position_offsets[i] = coord[i] - â_{k(i)}

    The 8^3=512 super-voxels match exactly the 512 effective latent tokens
    produced by the Can3Tok encoder's KL projection.

    Args:
        coord:         [N, 3]  Gaussian positions in canonical space (±10m)
        scaffold_dims: int     Grid resolution per axis (8 → 8³=512 tokens)
        domain_size:   float   Domain width in metres (16m covers the ±8m scene)

    Returns:
        scaffold_anchors:   [512, 3] float32  per-token anchor positions (gt)
        scaffold_token_ids: [N]      int32    token assignment per Gaussian (0-511)
        position_offsets:   [N, 3]   float32  δp_i = coord_i - anchor_{k(i)}
    """
    num_tokens  = scaffold_dims ** 3       # 512
    cell_size   = domain_size / scaffold_dims  # 2.0m per cell
    half_domain = domain_size / 2.0        # 8.0m

    # ── Assign each Gaussian to a super-voxel token ──────────────────────────
    # Shift from canonical [-8,8]m to [0,16]m, then discretise.
    shifted = coord + half_domain
    sv_idx  = np.floor(shifted / cell_size).astype(np.int32)
    sv_idx  = np.clip(sv_idx, 0, scaffold_dims - 1)

    # Linearise 3D cell index → scalar token ID in [0, 511]
    scaffold_token_ids = (
        sv_idx[:, 0] * scaffold_dims ** 2 +
        sv_idx[:, 1] * scaffold_dims +
        sv_idx[:, 2]
    ).astype(np.int32)

    # ── Compute anchor = mean position of Gaussians in each token region ──────
    # Fully vectorised via np.bincount — no Python loop over tokens.
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

    # Empty super-voxels → use geometric grid centre as fallback
    empty_idx = np.where(~occupied)[0]
    if len(empty_idx) > 0:
        ix = empty_idx // (scaffold_dims ** 2)
        iy = (empty_idx // scaffold_dims) % scaffold_dims
        iz = empty_idx % scaffold_dims
        scaffold_anchors[empty_idx, 0] = ix * cell_size + cell_size / 2.0 - half_domain
        scaffold_anchors[empty_idx, 1] = iy * cell_size + cell_size / 2.0 - half_domain
        scaffold_anchors[empty_idx, 2] = iz * cell_size + cell_size / 2.0 - half_domain

    scaffold_anchors = scaffold_anchors.astype(np.float32)

    # ── Per-Gaussian position offsets ─────────────────────────────────────────
    # δp_i = p_i - â_{k(i)}.
    # Bounded by ±cell_size/2 = ±1.0m (vs absolute ±10m) → 10x variance drop.
    position_offsets = (coord - scaffold_anchors[scaffold_token_ids]).astype(np.float32)

    return scaffold_anchors, scaffold_token_ids, position_offsets


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class gs_dataset(Dataset):
    """
    SceneSplat-7K dataset with ScanNet72 semantic labels.

    Feature tensor col layout (label_input=False, 18 cols):
      0:3   voxel_centers  (encoder coarse positional encoding, 40^3 grid)
      3     point_uniq_idx (voxel ID for each Gaussian)
      4:7   xyz            ← ALWAYS ABSOLUTE — encoder Fourier embedder needs this
      7:10  rgb            absolute [0,1] or residuals ~[-0.3,+0.3] (color_residual)
      10    opacity
      11:14 scale
      14:18 quaternion

    Reconstruction target = cols 4:18 (14-dim). Position part (first 3 of 14):
      - Absolute xyz when position_scaffold=False
      - Replaced by position_offsets when position_scaffold=True (training loop
        swaps in the offset tensor from the batch dict)
    """

    TARGET_POINTS   = 40_000
    LABEL_MAX       = 71.0
    LABEL_MISSING_NORM = -1.0 / 71.0
    SCAFFOLD_DIMS   = 8        # 8^3 = 512 super-voxels = num latent tokens
    SCAFFOLD_DOMAIN = 16.0     # metres
    SCAFFOLD_TOKENS = 512

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
        color_residual=False,
        position_scaffold=False,    # Scaffold-GS inspired position residual
    ):
        self.root              = root
        self.resol             = resol
        self.random_permute    = random_permute
        self.train             = train
        self.sampling_method   = sampling_method
        self.normalize         = normalize
        self.normalize_colors  = normalize_colors
        self.target_radius     = target_radius
        self.scale_norm_mode   = scale_norm_mode
        self.label_input       = label_input
        self.color_residual    = color_residual
        self.position_scaffold = position_scaffold

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

        if position_scaffold:
            print(f"  Position scaffold: ENABLED (Scaffold-GS inspired)")
            cell = self.SCAFFOLD_DOMAIN / self.SCAFFOLD_DIMS
            print(f"    -> {self.SCAFFOLD_DIMS}^3={self.SCAFFOLD_TOKENS} super-voxels "
                  f"({cell:.1f}m per cell, domain={self.SCAFFOLD_DOMAIN}m)")
            print(f"    -> batch keys: scaffold_anchors[512,3], "
                  f"scaffold_token_ids[40000], position_offsets[40000,3]")
            print(f"    -> encoder receives ABSOLUTE positions (feature tensor unchanged)")
            print(f"    -> training loop replaces position target with offsets")
        else:
            print(f"  Position scaffold: DISABLED (absolute xyz as reconstruction target)")

        if label_input:
            print(f"  Label input: ENABLED (col 18, point_feats=12)")
        else:
            print(f"  Label input: DISABLED (point_feats=11)")
        if normalize:
            print(f"  Canonical norm: ENABLED (radius={target_radius}m, "
                  f"scale={scale_norm_mode})")

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene_dir = self.scene_dirs[idx]

        # ── Load raw Gaussian attributes ──────────────────────────────────────
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

        if self.normalize_colors:
            color = color / 255.0

        # ── Semantic labels ───────────────────────────────────────────────────
        try:
            segment  = np.load(os.path.join(scene_dir, 'segment.npy'))
            instance = np.load(os.path.join(scene_dir, 'instance.npy'))
            has_semantics = True
        except FileNotFoundError:
            segment       = np.full(len(coord), -1, dtype=np.int16)
            instance      = np.full(len(coord), -1, dtype=np.int32)
            has_semantics = False

        # ── Deterministic top-40k sampling by opacity ─────────────────────────
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

        # ── Step 1: Color residual (after sampling — mean of actual 40k used) ─
        if self.color_residual:
            mean_color = color.mean(axis=0).astype(np.float32)   # [3] DC component
            color      = color - mean_color                       # [T,3] AC residuals
        else:
            mean_color = np.zeros(3, dtype=np.float32)

        # ── Position scaffold (after sampling — anchors from actual 40k) ──────
        # The feature tensor cols 4:7 still contain ABSOLUTE coord.
        # Scaffold data is returned separately and used by the training loop
        # to replace the position reconstruction target with offset targets.
        if self.position_scaffold:
            scaffold_anchors, scaffold_token_ids, position_offsets = \
                compute_position_scaffold(
                    coord,
                    scaffold_dims=self.SCAFFOLD_DIMS,
                    domain_size=self.SCAFFOLD_DOMAIN,
                )
        else:
            # Zero placeholders — collation works whether flag is on or off
            scaffold_anchors   = np.zeros((self.SCAFFOLD_TOKENS, 3), dtype=np.float32)
            scaffold_token_ids = np.zeros(T, dtype=np.int32)
            position_offsets   = np.zeros((T, 3), dtype=np.float32)

        # ── Encoder voxelisation (positional encoding, 40^3 fine grid) ───────
        volume_dims = 40
        resolution  = 16.0 / volume_dims    # 0.4m

        uniq_idx, inv_idx, _ = voxelize(coord, resolution, 'fnv')
        origin_offset = np.array([(volume_dims - 1) / 2] * 3) * resolution
        shifted_pts   = coord + origin_offset
        voxel_idx     = np.floor(shifted_pts / resolution)
        voxel_idx     = np.clip(voxel_idx, 0, volume_dims - 1)
        voxel_centers = (voxel_idx - (volume_dims - 1) / 2) * resolution
        point_uniq_idx = uniq_idx[inv_idx]

        # ── Assemble feature tensor [T, 18] ──────────────────────────────────
        # Cols 4:7 = ABSOLUTE xyz (always, even when position_scaffold=True).
        # The scaffold offset data is in the separate 'position_offsets' key.
        opacity_col    = opacity[:, np.newaxis]
        point_uniq_col = point_uniq_idx[:, np.newaxis]
        gs_params      = np.concatenate(
            (coord, color, opacity_col, scale, quat), axis=1)  # [T, 14]

        if self.label_input:
            label_norm = np.where(
                segment >= 0,
                segment.astype(np.float32) / self.LABEL_MAX,
                np.float32(self.LABEL_MISSING_NORM))
            gs_full_params = np.concatenate(
                (voxel_centers, point_uniq_col, gs_params,
                 label_norm[:, np.newaxis]), axis=1)  # [T, 19]
        else:
            gs_full_params = np.concatenate(
                (voxel_centers, point_uniq_col, gs_params), axis=1)  # [T, 18]

        # ── Scene-level ScanNet72 label distribution (Move 1) ─────────────────
        label_dist = np.zeros(72, dtype=np.float32)
        valid_seg  = segment[segment >= 0]
        if len(valid_seg) > 0:
            for k in range(72):
                label_dist[k] = (valid_seg == k).sum()
            label_dist /= label_dist.sum()

        return {
            # Core encoder input
            'features':        gs_full_params.astype(np.float32),
            'segment_labels':  segment,
            'instance_labels': instance,
            'scene_idx':       idx,
            'has_semantics':   has_semantics,
            'num_categories':  self.num_segment_categories,

            # Step 1: color supervision target for MeanColorHead
            'mean_color':      mean_color,

            # Move 1: semantic supervision target for SceneSemanticHead
            'label_dist':      label_dist,

            # Position scaffold: supervision targets for AnchorPositionHead and
            # as replacement for the absolute position reconstruction target.
            # All three are zero-filled when position_scaffold=False.
            'scaffold_anchors':   scaffold_anchors,    # [512, 3] gt anchor positions
            'scaffold_token_ids': scaffold_token_ids,  # [40000]  token per Gaussian
            'position_offsets':   position_offsets,    # [40000, 3] δp = p - anchor_k
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
    for cr, ps in [(False, False), (True, False), (True, True)]:
        print(f"\n{'='*60}\nTEST: color_residual={cr}, position_scaffold={ps}\n{'='*60}")
        ds = gs_dataset(root=data_path, max_scenes=1, normalize=True,
                        scale_norm_mode='linear', color_residual=cr,
                        position_scaffold=ps)
        s  = ds[0]
        f  = s['features']
        print(f"Feature shape: {f.shape}")
        print(f"Position (abs) range: [{f[:,4:7].min():.2f}, {f[:,4:7].max():.2f}]")
        if ps:
            off = s['position_offsets']
            anc = s['scaffold_anchors']
            tid = s['scaffold_token_ids']
            print(f"Offset range: [{off.min():.3f}, {off.max():.3f}]  (expect ~[-1,+1])")
            recon = off + anc[tid]
            err   = np.abs(recon - f[:, 4:7]).max()
            print(f"Recon error:  {err:.2e}  (expect ~0)")
            assert err < 1e-4
            print("  ✓ scaffold verified")
    print("\nALL TESTS PASSED")