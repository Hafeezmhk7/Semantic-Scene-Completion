"""
eval_architecture.py
====================
Evaluates Can3Tok VAE architecture variants on four targeted experiments.
Compares three decoder strategies:

  Strategy A: latent_disentangle=True
              Z = [z_layout (0:16) | z_geo (16:512)] → all 512 in decoder sequence
  Strategy B: decoder_layout_cross_attn=True / decoder_layout_additive=True
              Z = 512 geometry tokens; z_layout from Layout16Projector → decoder conditioning
  Strategy C: baseline — 512 geometry tokens, no layout conditioning

EXPERIMENTS
-----------
  1. Cosine distance clustering
       Intra vs inter-class distance in z_layout (semantic) and z_geo (geometry).
       z_layout should cluster by scene category; z_geo should not.
       Key metric: ratio = inter_dist / intra_dist  (> 1 = semantic organisation)

  2. Semantic swap visualisation + geometry preservation
       Decode z_AB = [z_layout_A | z_geo_B] and z_BA = [z_layout_B | z_geo_A].
       Save 4 PLYs. Measure geometry preservation (L2 of positions).
       Strategy A: swap first 16 tokens in Z.
       Strategy B: swap z_layout (from projector) while keeping Z unchanged.
       Strategy C: swap first 16 geometry token positions (control experiment).

  3. Partial observation robustness
       Encode partial scenes (20/40/60/80/100% of Gaussians, spatial crop).
       Measure: cosine similarity of z_layout vs full-scene z_layout.
       Tests whether z_layout remains stable from partial observations.
       Critical for scene completion: if z_layout degrades at 30%, completion fails.

  4. Prior sampling quality
       Sample z from N(0,I), decode, measure geometric coherence.
       Fraction of samples with valid positions, reasonable scales, non-degenerate
       structure. High fraction = latent is well-regularised → DiT-trainable.

SCENE SELECTION (from provided metadata)
-----------------------------------------
  Intra/inter distance:  apartment(5), coffee_shop(3), spa_pool(3)
  Swap pair 1:           apartment(0221) vs convenience_store(0203)  ← primary
  Swap pair 2:           coffee_shop(0218) vs spa_pool(0211)         ← secondary
  Partial obs:           apartment scenes (0207,0221,0222,0223,0224) — 5 scenes

USAGE
-----
  python eval_architecture.py \\
      --checkpoint /path/to/checkpoint.pth \\
      --data_root /path/to/train_grid1.0cm_chunk8x8_stride6x6 \\
      --output_dir eval_results/strategy_A \\
      --latent_disentangle --semantic_token_heads \\
      --color_residual --decoder_fourier_pe \\
      --scene_semantic_head --scene_layout_head

  Run once per strategy checkpoint. Compare results.json across runs.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations
# Import preprocessing functions directly from dataset to guarantee
# identical preprocessing to training. Any deviation causes garbage latents.
from gs_dataset_scenesplat import normalize_to_canonical_sphere, voxelize


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def get_args():
    p = argparse.ArgumentParser()
    # Core
    p.add_argument('--checkpoint',     required=True)
    p.add_argument('--data_root',      required=True)
    p.add_argument('--output_dir',     default='eval_results')
    p.add_argument('--device',         default='cuda')
    p.add_argument('--max_sh_degree',  type=int, default=3)
    p.add_argument('--n_samples_prior',type=int, default=200,
                   help='Number of N(0,I) samples for Experiment 4')

    # ── Model flags — must match checkpoint exactly ────────────────────────
    # Strategy A
    p.add_argument('--latent_disentangle',   action='store_true', default=False)
    p.add_argument('--semantic_dims',        type=int, default=512)
    p.add_argument('--semantic_token_heads', action='store_true', default=False)
    p.add_argument('--cross_recon_weight',   type=float, default=0.3)
    p.add_argument('--ortho_weight',         type=float, default=0.1)
    # Strategy B
    p.add_argument('--decoder_layout_cross_attn', action='store_true', default=False)
    p.add_argument('--decoder_layout_additive',   action='store_true', default=False)
    p.add_argument('--structured_layout_tokens',  action='store_true', default=False)
    # Shared
    p.add_argument('--color_residual',       action='store_true', default=False)
    p.add_argument('--scene_semantic_head',  action='store_true', default=False)
    p.add_argument('--scene_layout_head',    action='store_true', default=False)
    p.add_argument('--decoder_fourier_pe',   action='store_true', default=False)
    p.add_argument('--decoder_pos_enc',      action='store_true', default=False)
    p.add_argument('--token_cond',           action='store_true', default=False)
    p.add_argument('--token_cond_approach',  type=str, default='B')
    p.add_argument('--token_cond_adaln',     action='store_true', default=False)
    p.add_argument('--position_scaffold',    action='store_true', default=False)

    # ── Experiments to run ─────────────────────────────────────────────────
    p.add_argument('--skip_exp1', action='store_true', help='Skip cosine clustering')
    p.add_argument('--skip_exp2', action='store_true', help='Skip swap experiment')
    p.add_argument('--skip_exp3', action='store_true', help='Skip partial obs')
    p.add_argument('--skip_exp4', action='store_true', help='Skip prior sampling')

    return p.parse_args()


args = get_args()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
out    = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)

# Determine active strategy
_any_B    = args.decoder_layout_cross_attn or args.decoder_layout_additive
if args.latent_disentangle and not _any_B:
    STRATEGY = 'A'
elif _any_B and not args.latent_disentangle:
    STRATEGY = 'B1' if args.decoder_layout_cross_attn and not args.decoder_layout_additive \
               else ('B2' if args.decoder_layout_additive and not args.decoder_layout_cross_attn
               else 'B3')
elif not args.latent_disentangle and not _any_B:
    STRATEGY = 'C'
else:
    STRATEGY = 'mixed'

print(f"\n{'='*70}")
print(f"  Can3Tok — Architecture Evaluation  (Strategy {STRATEGY})")
print(f"{'='*70}")
print(f"  Checkpoint: {args.checkpoint}")
print(f"  Output:     {out}")
print(f"  Strategy:   {STRATEGY}")
print(f"  Flags:      latent_disentangle={args.latent_disentangle}  "
      f"decoder_layout_cross_attn={args.decoder_layout_cross_attn}  "
      f"decoder_layout_additive={args.decoder_layout_additive}")
print(f"              color_residual={args.color_residual}  "
      f"structured_layout_tokens={args.structured_layout_tokens}")
print(f"{'='*70}\n")


# ============================================================================
# SCENE METADATA
# ============================================================================

# Curated scene set — selected for category diversity and intra-class variety.
# Primary swap pair: apartment (0221) vs convenience_store (0203)
#   — different from previous eval (which used 0207 apartment + 0218 coffee_shop)
#   — convenience_store has very different geometry (shelves, counter, open floor)
# Secondary swap pair: coffee_shop (0218) vs spa_pool (0211)
#   — maximum visual contrast in color palette and spatial layout
# Partial obs: all 5 apartment scenes — richest category for stability test

SCENE_METADATA = [
    {"scene_id": "0202_840156", "category": "spa_pool"},
    {"scene_id": "0203_840160", "category": "convenience_store"},
    {"scene_id": "0204_840158", "category": "futuristic_pod"},
    {"scene_id": "0205_840168", "category": "library"},
    {"scene_id": "0206_840163", "category": "gym"},
    {"scene_id": "0207_840167", "category": "apartment"},
    {"scene_id": "0208_840166", "category": "go_kart"},
    {"scene_id": "0209_840159", "category": "museum"},
    {"scene_id": "0210_840153", "category": "concert_hall"},
    {"scene_id": "0211_840172", "category": "spa_pool"},
    {"scene_id": "0212_840152", "category": "spa_pool"},
    {"scene_id": "0213_840169", "category": "lobby"},
    {"scene_id": "0214_840176", "category": "convenience_store"},
    {"scene_id": "0215_840179", "category": "washroom"},
    {"scene_id": "0216_840180", "category": "club"},
    {"scene_id": "0217_840181", "category": "club"},
    {"scene_id": "0218_840182", "category": "coffee_shop"},
    {"scene_id": "0219_840183", "category": "coffee_shop"},
    {"scene_id": "0220_840185", "category": "coffee_shop"},
    {"scene_id": "0221_840242", "category": "apartment"},
    {"scene_id": "0222_840246", "category": "apartment"},
    {"scene_id": "0223_840262", "category": "apartment"},
    {"scene_id": "0224_840270", "category": "apartment"},
]

# Categories with ≥ 2 scenes for valid intra-class distance
EVAL_CATEGORIES   = ['apartment', 'coffee_shop', 'spa_pool', 'convenience_store', 'club']

# Swap experiment: primary and secondary pairs
# Primary: visually very different room types, different color palette
SWAP_PAIRS = [
    ('0221_840242', 'apartment',          '0203_840160', 'convenience_store'),
    ('0218_840182', 'coffee_shop',        '0211_840172', 'spa_pool'),
]

# Partial observation experiment: use all 5 apartment scenes
PARTIAL_OBS_SCENES = [
    '0207_840167', '0221_840242', '0222_840246', '0223_840262', '0224_840270'
]

cat_index = defaultdict(list)
for entry in SCENE_METADATA:
    cat_index[entry['category']].append(entry['scene_id'])


# ============================================================================
# MODEL LOADING
# ============================================================================

print("Loading model...")
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file

config_path  = './model/configs/aligned_shape_latents/shapevae-256.yaml'
model_config = get_config_from_file(config_path).model
p_cfg = model_config.params.shape_module_cfg.params

p_cfg.color_residual             = args.color_residual
p_cfg.scene_semantic_head        = args.scene_semantic_head
p_cfg.scene_layout_head          = args.scene_layout_head
p_cfg.latent_disentangle         = args.latent_disentangle
p_cfg.semantic_dims              = args.semantic_dims
p_cfg.decoder_pos_enc            = args.decoder_pos_enc
p_cfg.decoder_fourier_pe         = args.decoder_fourier_pe
p_cfg.token_cond                 = args.token_cond
p_cfg.token_cond_approach        = args.token_cond_approach
p_cfg.token_cond_adaln           = args.token_cond_adaln
p_cfg.semantic_token_heads       = args.semantic_token_heads
p_cfg.position_scaffold          = args.position_scaffold
p_cfg.decoder_layout_cross_attn  = args.decoder_layout_cross_attn
p_cfg.decoder_layout_additive    = args.decoder_layout_additive
p_cfg.structured_layout_tokens   = args.structured_layout_tokens
# Unused at eval time
p_cfg.semantic_mode              = 'none'
p_cfg.decoder_zs_cross_attn      = False
p_cfg.jepa_idea1                 = False
p_cfg.predict_seg_labels         = False
p_cfg.query_decoder              = False
p_cfg.position_layout_residual   = False

model = instantiate_from_config(model_config)
ckpt  = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.to(device)
model.eval()
sm = model.shape_model   # shorthand

epoch_str = ckpt.get('epoch', '?')
val_str   = ckpt.get('val_l2_error', ckpt.get('final_val_l2', '?'))
print(f"  Loaded: epoch={epoch_str}  val_L2={val_str}")
print(f"  Missing keys: {len(missing)}  Unexpected: {len(unexpected)}")
if missing:
    print(f"    Missing (first 5): {missing[:5]}")
print()


# ============================================================================
# SCENE LOADING
# ============================================================================

TARGET_POINTS = 40_000

def find_scene_dir(scene_id):
    data_root = Path(args.data_root)
    # Try exact suffix match
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and scene_id in d.name:
            return d
    return None


def load_scene(scene_dir, coverage=1.0, spatial_crop=True):
    """
    Load and preprocess a scene to the 18-channel encoder input.
    Uses normalize_to_canonical_sphere and voxelize imported directly from
    gs_dataset_scenesplat.py — guaranteed identical to training preprocessing.

    Previous implementation had two bugs:
      1. Normalization: did not center the scene (coord - mean) before scaling.
         The dataset centers first, then scales. Missing centering causes the
         encoder to receive positions at a completely different location.
      2. Voxelization: used vox_idx[:,0] (x-dimension only) as voxel ID
         instead of the FNV hash from voxelize(). The encoder queries use
         voxel IDs for grid-based position encoding; wrong IDs break this.

    coverage: float in (0,1] — fraction of Gaussians to keep.
              Spatial crop (z-axis) when spatial_crop=True, else top-k opacity.
    Returns: feat [40000,18], mean_color [3], label_dist [72], coord [N,3]
    """
    d = Path(scene_dir)
    coord   = np.load(d / 'coord.npy').astype(np.float32)
    color   = np.load(d / 'color.npy').astype(np.float32)
    scale   = np.load(d / 'scale.npy').astype(np.float32)
    quat    = np.load(d / 'quat.npy').astype(np.float32)
    opacity = np.load(d / 'opacity.npy').astype(np.float32)

    # ── Normalisation — MUST use dataset function exactly ─────────────────
    # normalize_to_canonical_sphere:
    #   1. center = coord.mean(axis=0)
    #   2. coord_centered = coord - center
    #   3. scale_factor = target_radius / (max_dist_from_center * 1.1)
    #   4. coord_norm = coord_centered * scale_factor
    #   5. scale_norm = scale * scale_factor  (linear mode)
    coord, scale = normalize_to_canonical_sphere(
        coord, scale, target_radius=10.0, scale_norm_mode='linear')
    color = color / 255.0

    # Segment labels
    try:
        segment = np.load(d / 'segment.npy')
    except FileNotFoundError:
        segment = np.full(len(coord), -1, dtype=np.int16)

    N = len(coord)

    # ── Partial coverage ──────────────────────────────────────────────────
    if coverage < 1.0:
        n_keep = max(1, int(N * coverage))
        if spatial_crop:
            # Spatial crop by z-coordinate: simulates scanning from one end
            z_thresh = np.percentile(coord[:, 2], coverage * 100)
            idx      = np.where(coord[:, 2] <= z_thresh)[0]
            if len(idx) == 0:
                idx = np.argsort(opacity)[-n_keep:]
            elif len(idx) > n_keep:
                idx = idx[np.argsort(opacity[idx])[-n_keep:]]
        else:
            idx = np.argsort(opacity)[-n_keep:]

        # Pad back to N with last selected index
        if len(idx) < N:
            pad = np.full(N - len(idx), idx[-1] if len(idx) > 0 else 0, dtype=np.int64)
            idx = np.concatenate([idx, pad])
        coord   = coord  [idx]
        color   = color  [idx]
        scale   = scale  [idx]
        quat    = quat   [idx]
        opacity = opacity[idx]
        segment = segment[idx]
        N       = len(coord)

    # ── Top-40k by opacity — mirrors gs_dataset exactly ───────────────────
    sorted_indices = np.argsort(opacity)
    if N >= TARGET_POINTS:
        sel = sorted_indices[-TARGET_POINTS:]
    else:
        pad = np.full(TARGET_POINTS - N, sorted_indices[-1], dtype=np.int64)
        sel = np.concatenate([sorted_indices, pad])

    coord   = coord  [sel]
    color   = color  [sel]
    scale   = scale  [sel]
    quat    = quat   [sel]
    opacity = opacity[sel]
    segment = segment[sel]

    # ── Color residual ────────────────────────────────────────────────────
    mean_color = color.mean(axis=0).astype(np.float32)
    if args.color_residual:
        color = color - mean_color

    # ── Encoder voxelisation — MUST use voxelize() from dataset ───────────
    # Uses FNV hash to assign each Gaussian to a voxel bucket.
    # point_uniq_idx = uniq_idx[inv_idx]  — unique bucket ID per Gaussian.
    # Previous code used vox_idx[:,0] (x-dim only) — completely wrong.
    volume_dims = 40
    resolution  = 16.0 / volume_dims
    uniq_idx, inv_idx, _ = voxelize(coord, resolution, 'fnv')
    origin_offset  = np.array([(volume_dims - 1) / 2] * 3) * resolution
    shifted_pts    = coord + origin_offset
    voxel_idx      = np.floor(shifted_pts / resolution)
    voxel_idx      = np.clip(voxel_idx, 0, volume_dims - 1)
    voxel_centers  = (voxel_idx - (volume_dims - 1) / 2) * resolution
    point_uniq_idx = uniq_idx[inv_idx]   # FNV hash bucket per Gaussian

    gs_params = np.concatenate(
        [coord, color, opacity[:, None], scale, quat], axis=1)  # [N, 14]
    features  = np.concatenate(
        [voxel_centers, point_uniq_idx[:, None], gs_params],
        axis=1).astype(np.float32)  # [N, 18]

    # ── Label distribution ────────────────────────────────────────────────
    label_dist = np.zeros(72, dtype=np.float32)
    valid_seg  = segment[segment >= 0].astype(np.int32)
    if len(valid_seg) > 0:
        for k in range(72):
            label_dist[k] = (valid_seg == k).sum()
        s = label_dist.sum()
        if s > 0:
            label_dist /= s

    return features, mean_color, label_dist, coord


@torch.no_grad()
def encode_scene(features_np):
    """
    Full forward pass. Returns dict with all latent quantities.
    """
    feat = torch.from_numpy(features_np).unsqueeze(0).to(device)

    (shape_embed, mu, log_var, z, UV_gs_recover, _) = model(
        feat, feat, feat, feat[:, :, :3])

    # z_layout extraction
    if args.latent_disentangle:
        # Strategy A: z_layout = first 16 tokens of reshaped Z
        Z_reshaped = z.reshape(1, 512, 32)
        z_layout_t = Z_reshaped[:, :16, :]                    # [1, 16, 32]
        z_layout   = z_layout_t.squeeze(0).cpu().numpy()      # [16, 32]
        mu_s = sm._mu_s_cache.squeeze(0).cpu().numpy()        # [512]
        mu_g = sm._mu_g_cache.squeeze(0).cpu().numpy()        # [15872]
    elif _any_B and sm.last_z_layout is not None:
        # Strategy B: z_layout from Layout16Projector(shape_embed)
        z_layout_t = sm.last_z_layout                         # [1, 16, 32]
        z_layout   = z_layout_t.squeeze(0).cpu().numpy()
        mu_s = z_layout.flatten()                             # [512] — use as proxy
        mu_g = mu.squeeze(0)[args.semantic_dims:].cpu().numpy()
    else:
        # Strategy C: no z_layout; use first 16 geometry token positions as proxy
        Z_reshaped = z.reshape(1, 512, 32)
        z_layout_t = Z_reshaped[:, :16, :]
        z_layout   = z_layout_t.squeeze(0).cpu().numpy()
        mu_s = mu.squeeze(0)[:args.semantic_dims].cpu().numpy()
        mu_g = mu.squeeze(0)[args.semantic_dims:].cpu().numpy()

    pred = UV_gs_recover.squeeze(0).reshape(40000, 14).cpu().numpy()

    return {
        'z':         z.squeeze(0).cpu(),                      # [16384]
        'z_layout':  z_layout,                                # [16, 32]
        'z_layout_t':z_layout_t.cpu(),                        # [1, 16, 32]
        'mu_s':      mu_s,                                     # [512]
        'mu_g':      mu_g,                                     # [15872]
        'mu':        mu.squeeze(0).cpu().numpy(),              # [16384]
        'log_var':   log_var.squeeze(0).cpu().numpy(),
        'pred':      pred,
        'scene_sem': (sm.last_scene_semantic_pred.squeeze(0).cpu().numpy()
                      if sm.last_scene_semantic_pred is not None else None),
    }


@torch.no_grad()
def decode_z(z_flat, mean_color_np, z_layout_override=None):
    """
    Decode a flat z [16384] → predictions [40000, 14].

    z_layout_override: [1, 16, 32] tensor — used for Strategy B swap.
    """
    Z = z_flat.unsqueeze(0).to(device).reshape(1, 512, 32)

    # Strategy B: inject z_layout override before decode
    if z_layout_override is not None and _any_B:
        sm.last_z_layout = z_layout_override.to(device)

    UV, _ = sm.decode(
        Z, volume_queries=None,
        return_semantic_features=False,
        z_layout=sm.last_z_layout if _any_B else None)

    pred = UV.squeeze(0).reshape(40000, 14).cpu().numpy()
    if args.color_residual and mean_color_np is not None:
        pred[:, 3:6] = np.clip(pred[:, 3:6] + mean_color_np, 0.0, 1.0)
    return pred


def save_ply(pred, path, desc=''):
    """Save prediction array [40000,14] as PLY via gs_ply_reconstructor.
    Uses the actual function signature: (predictions, output_dir, epoch,
    num_scenes, max_sh_degree, color_mode). The file is saved as
    scene_000_epoch_000.ply in a temp dir, then renamed to the target path.
    """
    import shutil, tempfile
    from gs_ply_reconstructor import save_reconstructed_gaussians
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save to a temp dir so we can rename to the desired filename
    with tempfile.TemporaryDirectory() as tmp:
        save_reconstructed_gaussians(
            predictions=np.array([pred]),
            output_dir=tmp,
            epoch=0,
            num_scenes=1,
            max_sh_degree=args.max_sh_degree,
            color_mode="1")
        # The function writes: scene_000_epoch_000.ply
        tmp_ply = next(Path(tmp).glob("*.ply"), None)
        if tmp_ply is None:
            print(f"    [WARN] save_ply: no .ply written to tmp dir — check gs_ply_reconstructor")
            return
        shutil.copy(str(tmp_ply), str(path))

    if desc:
        print(f"    Saved [{desc}]: {path.name}")


# ============================================================================
# SCENE LOADING CACHE
# ============================================================================

print("Encoding scenes needed for all experiments...")
all_needed = set()
for cat in EVAL_CATEGORIES:
    all_needed.update(cat_index[cat])
for sid_a, _, sid_b, _ in SWAP_PAIRS:
    all_needed.update([sid_a, sid_b])
all_needed.update(PARTIAL_OBS_SCENES)

# Cap to scenes that actually exist
encoded = {}
for sid in sorted(all_needed):
    sdir = find_scene_dir(sid)
    if sdir is None:
        print(f"  [SKIP] {sid} — directory not found")
        continue
    # Find category
    cat = next((e['category'] for e in SCENE_METADATA if e['scene_id'] == sid), 'unknown')
    print(f"  Encoding {sid} ({cat})...")
    feat, mean_color, label_dist, coord = load_scene(sdir)
    enc = encode_scene(feat)
    encoded[sid] = {**enc,
                    'mean_color': mean_color,
                    'label_dist': label_dist,
                    'coord':      coord,
                    'category':   cat,
                    'scene_id':   sid,
                    'feat':       feat}

print(f"\nEncoded {len(encoded)} scenes.\n")
if len(encoded) < 2:
    print("ERROR: Need at least 2 encoded scenes. Check data_root and scene IDs.")
    sys.exit(1)


# ============================================================================
# EXPERIMENT 1 — COSINE DISTANCE CLUSTERING
# ============================================================================

if not args.skip_exp1:
    print("=" * 70)
    print("EXPERIMENT 1 — Cosine Distance Clustering (z_layout vs z_geo)")
    print("=" * 70)
    print()
    print("Theory:")
    print("  z_layout should cluster by scene type (intra < inter).")
    print("  z_geo should NOT show the same clustering (ratio ≈ 1.0).")
    print("  Ratio = inter_dist / intra_dist.  Target: z_layout ratio >> z_geo ratio.")
    print()

    def cos_dist(a, b):
        """Cosine distance ∈ [0,2]. 0=identical."""
        n = np.linalg.norm
        return float(1.0 - np.dot(a, b) / (n(a) * n(b) + 1e-8))

    # Build per-category id lists (only scenes we encoded)
    cat_ids = defaultdict(list)
    for sid, data in encoded.items():
        if data['category'] in EVAL_CATEGORIES:
            cat_ids[data['category']].append(sid)

    hdr = f"{'Category':<22}  {'N':>2}  {'Intra z_lay':>11}  {'Intra z_geo':>11}  {'Pairs':>5}"
    print(hdr)
    print("─" * 65)

    intra_lay = {}
    intra_geo = {}
    for cat in EVAL_CATEGORIES:
        ids = cat_ids[cat]
        if len(ids) < 2:
            print(f"  {cat:<20}   {len(ids):2d}   (need ≥2 scenes)")
            continue
        pairs = list(combinations(ids, 2))
        dl = [cos_dist(encoded[a]['mu_s'], encoded[b]['mu_s']) for a, b in pairs]
        dg = [cos_dist(encoded[a]['mu_g'], encoded[b]['mu_g']) for a, b in pairs]
        intra_lay[cat] = float(np.mean(dl))
        intra_geo[cat] = float(np.mean(dg))
        print(f"  {cat:<20}   {len(ids):2d}   {intra_lay[cat]:>11.4f}   {intra_geo[cat]:>11.4f}   {len(pairs):>5}")

    print()
    print(f"{'Category pair':<38}  {'Inter z_lay':>11}  {'Inter z_geo':>11}  {'Pairs':>5}")
    print("─" * 75)

    inter_lay_all = []
    inter_geo_all = []
    inter_details = {}
    for cat_a, cat_b in combinations(EVAL_CATEGORIES, 2):
        ids_a = cat_ids[cat_a]
        ids_b = cat_ids[cat_b]
        if not ids_a or not ids_b:
            continue
        pairs = [(a, b) for a in ids_a for b in ids_b]
        dl = [cos_dist(encoded[a]['mu_s'], encoded[b]['mu_s']) for a, b in pairs]
        dg = [cos_dist(encoded[a]['mu_g'], encoded[b]['mu_g']) for a, b in pairs]
        mdl, mdg = float(np.mean(dl)), float(np.mean(dg))
        inter_lay_all.extend(dl)
        inter_geo_all.extend(dg)
        inter_details[f"{cat_a}|{cat_b}"] = {'z_lay': mdl, 'z_geo': mdg, 'n': len(pairs)}
        lbl = f"{cat_a} vs {cat_b}"
        print(f"  {lbl:<36}   {mdl:>11.4f}   {mdg:>11.4f}   {len(pairs):>5}")

    print()
    exp1_results = {}
    if intra_lay and inter_lay_all:
        m_intra_lay = float(np.mean(list(intra_lay.values())))
        m_intra_geo = float(np.mean(list(intra_geo.values())))
        m_inter_lay = float(np.mean(inter_lay_all))
        m_inter_geo = float(np.mean(inter_geo_all))
        r_lay = m_inter_lay / (m_intra_lay + 1e-8)
        r_geo = m_inter_geo / (m_intra_geo + 1e-8)

        print("SUMMARY:")
        print(f"  Mean intra-class dist:  z_layout={m_intra_lay:.4f}   z_geo={m_intra_geo:.4f}")
        print(f"  Mean inter-class dist:  z_layout={m_inter_lay:.4f}   z_geo={m_inter_geo:.4f}")
        print(f"  Ratio inter/intra:      z_layout={r_lay:.3f}   z_geo={r_geo:.3f}")
        print()
        if r_lay > 1.3 and r_lay > r_geo * 1.2:
            verdict = "STRONG: z_layout clusters by scene type; z_geo does not"
        elif r_lay > 1.1 and r_lay > r_geo:
            verdict = "WEAK: z_layout slightly more structured than z_geo"
        elif r_lay > r_geo:
            verdict = "MARGINAL: small difference between z_layout and z_geo"
        else:
            verdict = "NONE: z_layout not more structured than z_geo"
        print(f"  Verdict: {verdict}")
        print()

        exp1_results = {
            'intra_lay': {k: float(v) for k, v in intra_lay.items()},
            'intra_geo': {k: float(v) for k, v in intra_geo.items()},
            'inter_details': inter_details,
            'mean_intra_lay': m_intra_lay, 'mean_intra_geo': m_intra_geo,
            'mean_inter_lay': m_inter_lay, 'mean_inter_geo': m_inter_geo,
            'ratio_lay': r_lay, 'ratio_geo': r_geo,
            'verdict': verdict,
        }
    else:
        exp1_results = {'error': 'insufficient scenes for intra-class comparison'}


# ============================================================================
# EXPERIMENT 2 — SWAP VISUALISATION + GEOMETRY PRESERVATION
# ============================================================================

if not args.skip_exp2:
    print("=" * 70)
    print("EXPERIMENT 2 — Semantic Swap + Geometry Preservation")
    print("=" * 70)
    print()
    print("Swap procedure by strategy:")
    if STRATEGY == 'A':
        print("  Strategy A: Z_AB = [z_layout_A (tokens 0:16) | z_geo_B (tokens 16:512)]")
        print("  Decoded using standard decoder — both are in the sequence.")
    elif STRATEGY.startswith('B'):
        print(f"  Strategy {STRATEGY}: Z stays as z_geo_B unchanged.")
        print("  z_layout_A passed as conditioning override to decoder.")
    else:
        print("  Strategy C: swap first 16 geometry token positions (control).")
        print("  No z_layout exists — this tests whether the swap does anything at all.")
    print()

    swap_dir = out / 'swap_visualisation'
    swap_dir.mkdir(exist_ok=True)
    exp2_results = {'pairs': []}

    for sid_a, cat_a, sid_b, cat_b in SWAP_PAIRS:
        if sid_a not in encoded or sid_b not in encoded:
            print(f"  [SKIP] {sid_a} or {sid_b} not encoded")
            exp2_results['pairs'].append({'error': f'missing {sid_a} or {sid_b}'})
            continue

        da = encoded[sid_a]
        db = encoded[sid_b]
        print(f"  Pair: {sid_a} ({cat_a})  ↔  {sid_b} ({cat_b})")

        mu_a = torch.from_numpy(da['mu']).float()
        mu_b = torch.from_numpy(db['mu']).float()
        D_s  = args.semantic_dims

        # ── Build swap latents ───────────────────────────────────────────
        if STRATEGY == 'A':
            # z_AB: semantic tokens from A (dim 0:D_s), geometry from B (dim D_s:)
            z_AB = torch.cat([mu_a[:D_s], mu_b[D_s:]])
            z_BA = torch.cat([mu_b[:D_s], mu_a[D_s:]])
            z_layout_override_A = None   # not needed for Strategy A
            z_layout_override_B = None
        elif STRATEGY.startswith('B'):
            # Z stays geometry; swap z_layout (from projector)
            z_AB = mu_b.clone()          # geometry stays B
            z_BA = mu_a.clone()          # geometry stays A
            z_layout_override_A = da['z_layout_t']  # [1,16,32] — layout from A
            z_layout_override_B = db['z_layout_t']  # [1,16,32] — layout from B
        else:
            # Strategy C: swap first 16 token positions in the geometry Z
            z_AB = torch.cat([mu_a[:D_s], mu_b[D_s:]])
            z_BA = torch.cat([mu_b[:D_s], mu_a[D_s:]])
            z_layout_override_A = None
            z_layout_override_B = None

        # ── Decode all four variants ─────────────────────────────────────
        # When using mu (posterior mean) not sampled z — avoids random noise
        recon_A  = decode_z(mu_a, da['mean_color'])
        recon_B  = decode_z(mu_b, db['mean_color'])

        if STRATEGY.startswith('B'):
            # AB: z_geo from B, z_layout from A
            recon_AB = decode_z(z_AB, da['mean_color'],
                                z_layout_override=z_layout_override_A)
            # BA: z_geo from A, z_layout from B
            recon_BA = decode_z(z_BA, db['mean_color'],
                                z_layout_override=z_layout_override_B)
        else:
            recon_AB = decode_z(z_AB, da['mean_color'])
            recon_BA = decode_z(z_BA, db['mean_color'])

        # ── Save PLYs ────────────────────────────────────────────────────
        pair_dir = swap_dir / f"{cat_a}_vs_{cat_b}"
        pair_dir.mkdir(exist_ok=True)

        variants = [
            (recon_A,  f'A_self_{sid_a}_{cat_a}',
             f"Original A — {cat_a}"),
            (recon_B,  f'B_self_{sid_b}_{cat_b}',
             f"Original B — {cat_b}"),
            (recon_AB, f'AB_layout{cat_a}_geo{cat_b}',
             f"z_AB: layout={cat_a}, geometry={cat_b}  ← should look like {cat_b} room in {cat_a} colors"),
            (recon_BA, f'BA_layout{cat_b}_geo{cat_a}',
             f"z_BA: layout={cat_b}, geometry={cat_a}  ← should look like {cat_a} room in {cat_b} colors"),
        ]
        for pred, fname, desc in variants:
            save_ply(pred, pair_dir / f"{fname}.ply", desc)

        # ── Geometry preservation metric ──────────────────────────────────
        def pos_l2(a, b):
            """Mean per-point L2 distance in position space."""
            return float(np.mean(np.linalg.norm(a[:, :3] - b[:, :3], axis=1)))

        def color_shift(a, b):
            """Mean per-point L2 distance in color space."""
            return float(np.mean(np.linalg.norm(a[:, 3:6] - b[:, 3:6], axis=1)))

        # Sanity: re-decoding A from mu_a should give same as encode_scene pred
        l2_A_self    = pos_l2(recon_A,  da['pred'])   # ~0
        l2_B_self    = pos_l2(recon_B,  db['pred'])   # ~0
        # Key: does z_AB preserve B's geometry?
        l2_AB_vs_B   = pos_l2(recon_AB, recon_B)
        l2_BA_vs_A   = pos_l2(recon_BA, recon_A)
        # Color shift after swap
        col_AB_vs_A  = color_shift(recon_AB, recon_A)  # AB should shift color toward A
        col_BA_vs_B  = color_shift(recon_BA, recon_B)

        print()
        print(f"  Geometry preservation (mean per-point position L2):")
        print(f"    Sanity: re-decode A vs forward-pass A  = {l2_A_self:.4f}  (should be ~0)")
        print(f"    Sanity: re-decode B vs forward-pass B  = {l2_B_self:.4f}  (should be ~0)")
        print(f"    z_AB position vs original B            = {l2_AB_vs_B:.4f}")
        print(f"    z_BA position vs original A            = {l2_BA_vs_A:.4f}")
        print(f"    Baseline (A pos vs B pos):             = {pos_l2(recon_A, recon_B):.4f}")
        print()
        print(f"  Color consistency after swap:")
        print(f"    z_AB color vs A color                  = {col_AB_vs_A:.4f}")
        print(f"      (lower = z_layout_A transferred color palette to hybrid)")
        print()

        # Interpretation
        baseline_pos = pos_l2(recon_A, recon_B)
        if l2_AB_vs_B < baseline_pos * 0.7:
            geo_verdict = f"GEOMETRY PRESERVED: z_AB is {l2_AB_vs_B/l2_B_self:.1f}x self-recon L2 — close to B's geometry"
        elif l2_AB_vs_B < baseline_pos:
            geo_verdict = "PARTIAL: z_AB closer to B geometry than random cross-scene, but not fully preserved"
        else:
            geo_verdict = "NOT PRESERVED: z_AB geometry as different from B as A is from B — disentanglement failed"
        print(f"  Geometry verdict: {geo_verdict}")
        print()

        exp2_results['pairs'].append({
            'sid_a': sid_a, 'cat_a': cat_a,
            'sid_b': sid_b, 'cat_b': cat_b,
            'l2_A_self': l2_A_self, 'l2_B_self': l2_B_self,
            'l2_AB_vs_B': l2_AB_vs_B, 'l2_BA_vs_A': l2_BA_vs_A,
            'baseline_pos': baseline_pos,
            'col_AB_vs_A': col_AB_vs_A, 'col_BA_vs_B': col_BA_vs_B,
            'geo_verdict': geo_verdict,
        })

    print(f"  PLY files saved to: {swap_dir}")
    print()
    print("  How to interpret in SuperSplat:")
    print("    Open all 4 PLYs for a pair simultaneously.")
    if STRATEGY == 'A':
        print("    z_AB should look like a coffee shop / convenience store in shape,")
        print("    but with the apartment's mean color palette.")
        print("    If z_g carries geometry cleanly, AB ≈ B in structure.")
    elif STRATEGY.startswith('B'):
        print("    z_AB uses B's full Z (geometry) + A's z_layout (conditioning).")
        print("    AB should be identical to B if z_layout has no effect → z_layout useless.")
        print("    AB should differ from B in color/layout if z_layout conditions the decoder.")
    else:
        print("    Strategy C: swapping first 16 geometry positions should disrupt geometry.")
        print("    If AB ≈ B despite the swap, the first 16 tokens carry no distinctive info.")
    print()


# ============================================================================
# EXPERIMENT 3 — PARTIAL OBSERVATION ROBUSTNESS
# ============================================================================

if not args.skip_exp3:
    print("=" * 70)
    print("EXPERIMENT 3 — Partial Observation Robustness of z_layout")
    print("=" * 70)
    print()
    print("Theory:")
    print("  For scene completion to work, z_layout must remain semantically")
    print("  stable even when only part of the scene is observed.")
    print("  We progressively reduce Gaussian coverage and measure:")
    print("    (a) Cosine similarity of z_layout vs full-scene z_layout")
    print("    (b) KL divergence of scene_semantic_head prediction vs GT label_dist")
    print("  Target: cosine sim > 0.7 at 40% coverage for completion to be viable.")
    print()

    coverages = [0.20, 0.40, 0.60, 0.80, 1.00]

    exp3_results = {'scenes': {}}

    hdr = (f"  {'Scene':<15}  {'Coverage':>8}  "
           f"{'CosSim z_lay':>13}  {'CosSim z_geo':>13}  {'KL sem pred':>11}")
    print(hdr)
    print("  " + "─" * 68)

    for sid in PARTIAL_OBS_SCENES:
        if sid not in encoded:
            print(f"  [SKIP] {sid} not encoded")
            continue

        sdir = find_scene_dir(sid)
        if sdir is None:
            continue

        data_full = encoded[sid]
        z_lay_full = data_full['z_layout'].flatten()  # [512]
        z_geo_full = data_full['mu_g']               # [15872]
        ld_gt      = data_full['label_dist']

        scene_rows = []
        for cov in coverages:
            feat_p, _, ld_p, _ = load_scene(sdir, coverage=cov, spatial_crop=True)
            enc_p = encode_scene(feat_p)

            z_lay_p = enc_p['z_layout'].flatten()
            z_geo_p = enc_p['mu_g']

            def cos_sim(a, b):
                return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

            cs_lay = cos_sim(z_lay_full, z_lay_p)
            cs_geo = cos_sim(z_geo_full, z_geo_p)

            # KL of semantic head output vs GT label dist
            kl_sem = float('nan')
            if enc_p['scene_sem'] is not None:
                pred = np.clip(enc_p['scene_sem'], 1e-8, 1.0)
                pred /= pred.sum()
                gt   = np.clip(ld_gt, 1e-8, 1.0)
                gt  /= gt.sum()
                kl_sem = float(np.sum(gt * np.log(gt / pred + 1e-8)))

            print(f"  {sid:<15}  {cov*100:>7.0f}%  {cs_lay:>13.4f}  {cs_geo:>13.4f}  {kl_sem:>11.4f}")
            scene_rows.append({
                'coverage': cov, 'cos_sim_zlayout': cs_lay,
                'cos_sim_zgeo': cs_geo, 'kl_sem': kl_sem,
            })

        exp3_results['scenes'][sid] = scene_rows
        print()

    # Summary: average cosine similarity at each coverage across all apartment scenes
    print("  SUMMARY — mean cosine similarity of z_layout across all scenes:")
    print(f"  {'Coverage':>8}  {'Mean CosSim z_layout':>21}  {'Mean CosSim z_geo':>18}")
    print("  " + "─" * 52)
    for i, cov in enumerate(coverages):
        cs_lay_vals = [exp3_results['scenes'][sid][i]['cos_sim_zlayout']
                       for sid in PARTIAL_OBS_SCENES if sid in exp3_results['scenes']]
        cs_geo_vals = [exp3_results['scenes'][sid][i]['cos_sim_zgeo']
                       for sid in PARTIAL_OBS_SCENES if sid in exp3_results['scenes']]
        if cs_lay_vals:
            print(f"  {cov*100:>7.0f}%  {np.mean(cs_lay_vals):>21.4f}  {np.mean(cs_geo_vals):>18.4f}")

    print()
    # Viability verdict for completion
    cov40_vals = [exp3_results['scenes'][sid][1]['cos_sim_zlayout']
                  for sid in PARTIAL_OBS_SCENES if sid in exp3_results['scenes']]
    if cov40_vals:
        mean_40 = float(np.mean(cov40_vals))
        if mean_40 > 0.80:
            comp_verdict = f"VIABLE: z_layout very stable at 40% (cos_sim={mean_40:.3f})"
        elif mean_40 > 0.60:
            comp_verdict = f"MARGINAL: z_layout partially stable at 40% (cos_sim={mean_40:.3f})"
        else:
            comp_verdict = f"NOT VIABLE: z_layout degrades too much at 40% (cos_sim={mean_40:.3f})"
        print(f"  Scene completion viability at 40% coverage: {comp_verdict}")
        exp3_results['completion_verdict_40pct'] = comp_verdict
    print()


# ============================================================================
# EXPERIMENT 4 — PRIOR SAMPLING QUALITY
# ============================================================================

if not args.skip_exp4:
    print("=" * 70)
    print("EXPERIMENT 4 — Prior Sampling Quality (latent space regularity)")
    print("=" * 70)
    print()
    print("Theory:")
    print("  Sample z from N(0,I). Decode. Measure geometric coherence.")
    print("  High fraction of valid samples = latent is well-regularised")
    print("  = DiT can learn P(z) from N(0,I) as starting distribution.")
    print()
    print(f"  Sampling {args.n_samples_prior} z vectors from N(0,I)...")
    print()

    @torch.no_grad()
    def sample_and_decode():
        """Sample z ~ N(0,I), decode, measure basic geometric validity."""
        z_s = torch.randn(1, args.semantic_dims)         # z_layout part
        z_g = torch.randn(1, 16384 - args.semantic_dims) # z_geo part
        z   = torch.cat([z_s, z_g], dim=1).squeeze(0)    # [16384]

        pred = decode_z(z, mean_color_np=None)
        pos  = pred[:, 0:3]
        sc   = pred[:, 7:10]
        op   = pred[:, 6]

        return {
            'pos_in_bounds':  bool((np.abs(pos) < 15.0).all()),         # loose bound
            'scale_ok':       bool((sc > 0.0001).all() and (sc < 5.0).all()),
            'opacity_ok':     bool((op > 0.0).all() and (op < 1.0).all()),
            'pos_std':        float(pos.std()),
            'mean_opacity':   float(op.mean()),
            'max_scale':      float(sc.max()),
        }

    results_s = []
    for i in range(args.n_samples_prior):
        try:
            r = sample_and_decode()
            results_s.append(r)
        except Exception as e:
            results_s.append({'pos_in_bounds': False, 'scale_ok': False,
                               'opacity_ok': False, 'error': str(e)})

    # Aggregate
    n = len(results_s)
    frac_pos    = float(np.mean([r['pos_in_bounds'] for r in results_s]))
    frac_scale  = float(np.mean([r['scale_ok']      for r in results_s]))
    frac_opac   = float(np.mean([r['opacity_ok']    for r in results_s]))
    frac_all    = float(np.mean([r.get('pos_in_bounds', False) and
                                  r.get('scale_ok', False) and
                                  r.get('opacity_ok', False)
                                  for r in results_s]))
    mean_pos_std = float(np.mean([r.get('pos_std', 0) for r in results_s]))
    mean_op      = float(np.mean([r.get('mean_opacity', 0) for r in results_s]))
    mean_max_sc  = float(np.mean([r.get('max_scale', 0)  for r in results_s]))

    print(f"  Results over {n} samples from N(0,I):")
    print(f"    Positions in bounds (|pos| < 15m):  {frac_pos*100:5.1f}%")
    print(f"    Scales reasonable (0.0001–5.0):     {frac_scale*100:5.1f}%")
    print(f"    Opacities in [0,1]:                 {frac_opac*100:5.1f}%")
    print(f"    All three valid:                    {frac_all*100:5.1f}%  ← primary metric")
    print(f"    Mean position std:                  {mean_pos_std:.4f}  (higher = more diverse)")
    print(f"    Mean opacity:                       {mean_op:.4f}")
    print(f"    Mean max scale:                     {mean_max_sc:.4f}")
    print()

    if frac_all > 0.60:
        samp_verdict = f"GOOD: {frac_all*100:.0f}% valid samples — latent well-regularised for DiT"
    elif frac_all > 0.30:
        samp_verdict = f"MODERATE: {frac_all*100:.0f}% valid — some regularisation, DiT may struggle"
    else:
        samp_verdict = f"POOR: only {frac_all*100:.0f}% valid — latent far from N(0,I), DiT training difficult"
    print(f"  Sampling verdict: {samp_verdict}")
    print()

    exp4_results = {
        'n_samples': n,
        'frac_pos_bounds': frac_pos,
        'frac_scale_ok':   frac_scale,
        'frac_opacity_ok': frac_opac,
        'frac_all_valid':  frac_all,
        'mean_pos_std':    mean_pos_std,
        'verdict':         samp_verdict,
    }


# ============================================================================
# SAVE RESULTS
# ============================================================================

report = {
    'checkpoint':    args.checkpoint,
    'strategy':      STRATEGY,
    'flags': {
        'latent_disentangle':        args.latent_disentangle,
        'decoder_layout_cross_attn': args.decoder_layout_cross_attn,
        'decoder_layout_additive':   args.decoder_layout_additive,
        'structured_layout_tokens':  args.structured_layout_tokens,
        'color_residual':            args.color_residual,
        'semantic_token_heads':      args.semantic_token_heads,
        'scene_semantic_head':       args.scene_semantic_head,
        'scene_layout_head':         args.scene_layout_head,
    },
    'n_scenes_encoded': len(encoded),
    'exp1_clustering':    exp1_results  if not args.skip_exp1 else 'skipped',
    'exp2_swap':          exp2_results  if not args.skip_exp2 else 'skipped',
    'exp3_partial_obs':   exp3_results  if not args.skip_exp3 else 'skipped',
    'exp4_prior_sample':  exp4_results  if not args.skip_exp4 else 'skipped',
}

rpath = out / 'results.json'
with open(rpath, 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f"Full results → {rpath}")

# ── Quick summary for grep ─────────────────────────────────────────────────
print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  Strategy:   {STRATEGY}  (checkpoint: {Path(args.checkpoint).name})")
if not args.skip_exp1 and 'ratio_lay' in exp1_results:
    print(f"  Exp 1 — Clustering ratio:  z_layout={exp1_results['ratio_lay']:.3f}  "
          f"z_geo={exp1_results['ratio_geo']:.3f}  "
          f"(target: z_layout >> z_geo)")
    print(f"           Verdict: {exp1_results['verdict']}")
if not args.skip_exp2 and exp2_results['pairs']:
    for pair in exp2_results['pairs']:
        if 'l2_AB_vs_B' in pair:
            print(f"  Exp 2 — Swap {pair['cat_a']} ↔ {pair['cat_b']}:  "
                  f"geo_L2={pair['l2_AB_vs_B']:.4f}  "
                  f"baseline={pair['baseline_pos']:.4f}  "
                  f"self={pair['l2_B_self']:.4f}")
            print(f"           Verdict: {pair['geo_verdict']}")
if not args.skip_exp3 and 'completion_verdict_40pct' in exp3_results:
    print(f"  Exp 3 — {exp3_results['completion_verdict_40pct']}")
if not args.skip_exp4 and 'frac_all_valid' in exp4_results:
    print(f"  Exp 4 — Prior sampling:  {exp4_results['frac_all_valid']*100:.0f}% valid  "
          f"Verdict: {exp4_results['verdict']}")
print()
print(f"  PLY files:  {out / 'swap_visualisation'}")
print(f"  Full JSON:  {rpath}")
print()