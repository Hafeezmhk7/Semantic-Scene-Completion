"""
eval_disentanglement.py
=======================
Tests latent space disentanglement of Can3Tok VAE using curated scene metadata.

THREE EXPERIMENTS:
  1. Intra vs Inter-class cosine distance in z_s and z_g
     — Tests whether z_s clusters by semantic category (apartment near apartment,
       far from coffee_shop). z_g is used as control: it should NOT show the same
       clustering if geometry is truly independent of semantics.

  2. SceneSemanticHead prediction consistency within categories
     — Measures JSD between predicted label distributions for same-category scenes.
       Same-category scenes should have small JSD; cross-category scenes large JSD.
       Directly validates that z_s encodes interpretable semantic composition.

  3. Qualitative swap visualisation (PLY output)
     — Takes one apartment and one coffee_shop, forms cross-latents
       z_AB = [z_s^apt, z_g^coffee] and z_BA = [z_s^coffee, z_g^apt].
       Decodes all four and saves PLYs for visual inspection in SuperSplat.
       Tests whether the decoder respects the semantic/geometric split.

USAGE:
  python eval_disentanglement.py \
      --checkpoint /path/to/best_model.pth \
      --metadata scene_metadata.json \
      --data_root /path/to/train_grid1.0cm_chunk8x8_stride6x6 \
      --output_dir eval_disentanglement_results \
      [flags matching the checkpoint: --color_residual --latent_disentangle ...]
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import sys

# Suppress noisy prints from model init
import io, contextlib

from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
from gs_dataset_scenesplat import (
    normalize_to_canonical_sphere, voxelize,
    compute_category_centroids
)
from gs_ply_reconstructor import save_reconstructed_gaussians


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Can3Tok Disentanglement Evaluation')
parser.add_argument('--checkpoint',    type=str, required=True)
parser.add_argument('--metadata',      type=str, required=True,
                    help='JSON file with scene_id and category fields')
parser.add_argument('--data_root',     type=str,
                    default='/home/yli11/scratch/datasets/gaussian_world/preprocessed/'
                            'interior_gs/train_grid1.0cm_chunk8x8_stride6x6')
parser.add_argument('--output_dir',    type=str, default='eval_disentanglement_results')

# Model flags — must match the checkpoint
parser.add_argument('--color_residual',      action='store_true', default=False)
parser.add_argument('--scene_semantic_head', action='store_true', default=False)
parser.add_argument('--scene_layout_head',   action='store_true', default=False)
parser.add_argument('--latent_disentangle',  action='store_true', default=False)
parser.add_argument('--semantic_dims',       type=int, default=512)
parser.add_argument('--decoder_pos_enc',     action='store_true', default=False)
parser.add_argument('--decoder_fourier_pe',  action='store_true', default=False)
parser.add_argument('--token_cond',          action='store_true', default=False)
parser.add_argument('--token_cond_approach', type=str, default='B')
parser.add_argument('--token_cond_adaln',    action='store_true', default=False)
parser.add_argument('--semantic_token_heads',action='store_true', default=False)
parser.add_argument('--position_scaffold',   action='store_true', default=False)

# Eval settings
parser.add_argument('--categories',    type=str, nargs='+',
                    default=['apartment', 'coffee_shop', 'spa_pool'],
                    help='Categories to include in distance experiment')
parser.add_argument('--swap_cat_a',    type=str, default='apartment')
parser.add_argument('--swap_cat_b',    type=str, default='coffee_shop')
parser.add_argument('--swap_scene_idx_a', type=int, default=0,
                    help='Index within swap_cat_a scenes to use for swap')
parser.add_argument('--swap_scene_idx_b', type=int, default=0,
                    help='Index within swap_cat_b scenes to use for swap')
parser.add_argument('--max_sh_degree', type=int, default=3)
parser.add_argument('--device',        type=str, default='cuda')

args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*70}")
print(f"  Can3Tok — Disentanglement Evaluation")
print(f"{'='*70}")
print(f"  Checkpoint: {args.checkpoint}")
print(f"  Output:     {output_dir}")
print(f"  Categories: {args.categories}")
print(f"  Swap pair:  {args.swap_cat_a}  ↔  {args.swap_cat_b}")
print(f"{'='*70}\n")


# ============================================================================
# LOAD MODEL
# ============================================================================

print("Loading model...")
config_path  = './model/configs/aligned_shape_latents/shapevae-256.yaml'
model_config = get_config_from_file(config_path).model
p = model_config.params.shape_module_cfg.params

p.color_residual          = args.color_residual
p.scene_semantic_head     = args.scene_semantic_head
p.scene_layout_head       = args.scene_layout_head
p.latent_disentangle      = args.latent_disentangle
p.semantic_dims           = args.semantic_dims
p.decoder_pos_enc         = args.decoder_pos_enc
p.decoder_fourier_pe      = args.decoder_fourier_pe
p.token_cond              = args.token_cond
p.token_cond_approach     = args.token_cond_approach
p.token_cond_adaln        = args.token_cond_adaln
p.semantic_token_heads    = args.semantic_token_heads
p.position_scaffold       = args.position_scaffold

# Flags not relevant for evaluation — set to safe defaults
p.semantic_mode           = 'none'
p.decoder_shape_prepend   = False
p.decoder_shape_cross_attn= False
p.decoder_cross_attn_layers= 4
p.jepa_idea1              = False
p.predict_seg_labels      = False
p.query_decoder           = False

model = instantiate_from_config(model_config)
ckpt  = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.to(device)
model.eval()
print(f"  Loaded checkpoint (epoch {ckpt.get('epoch', '?')},"
      f" val L2={ckpt.get('val_l2_error', ckpt.get('final_val_l2', '?'))})\n")


# ============================================================================
# SCENE LOADING HELPERS
# ============================================================================

TARGET_POINTS = 40_000

def find_scene_dir(data_root, scene_id):
    """Find scene directory matching scene_id anywhere in the tree."""
    data_root = Path(data_root)
    # Try direct match first
    for d in data_root.iterdir():
        if d.is_dir() and scene_id in d.name:
            return d
    return None


def load_scene(scene_dir):
    """
    Load and preprocess a single scene to the 18-channel feature tensor.
    Mirrors gs_dataset.__getitem__ exactly so encoder receives identical input.
    Returns: features [40000, 18], mean_color [3], label_dist [72], coord [40000, 3]
    """
    scene_dir = Path(scene_dir)
    coord   = np.load(scene_dir / 'coord.npy')
    color   = np.load(scene_dir / 'color.npy')
    scale   = np.load(scene_dir / 'scale.npy')
    quat    = np.load(scene_dir / 'quat.npy')
    opacity = np.load(scene_dir / 'opacity.npy')

    # Normalise to canonical sphere (linear scale)
    coord, scale = normalize_to_canonical_sphere(coord, scale, target_radius=10.0,
                                                  scale_norm_mode='linear')
    color = color / 255.0

    # Load segment labels
    try:
        segment = np.load(scene_dir / 'segment.npy')
    except FileNotFoundError:
        segment = np.full(len(coord), -1, dtype=np.int16)

    # Top-40k by opacity (deterministic)
    sorted_idx = np.argsort(opacity)
    N = len(coord)
    if N >= TARGET_POINTS:
        selected = sorted_idx[-TARGET_POINTS:]
    else:
        extra    = np.full(TARGET_POINTS - N, sorted_idx[-1], dtype=np.int64)
        selected = np.concatenate([sorted_idx, extra])

    coord   = coord  [selected]
    color   = color  [selected]
    scale   = scale  [selected]
    quat    = quat   [selected]
    opacity = opacity[selected]
    segment = segment[selected]

    # Color residual
    mean_color = color.mean(axis=0).astype(np.float32)
    if args.color_residual:
        color = color - mean_color

    # Encoder voxelisation
    volume_dims   = 40
    resolution    = 16.0 / volume_dims
    uniq_idx, inv_idx, _ = voxelize(coord, resolution, 'fnv')
    origin_offset = np.array([(volume_dims - 1) / 2] * 3) * resolution
    shifted_pts   = coord + origin_offset
    voxel_idx     = np.floor(shifted_pts / resolution)
    voxel_idx     = np.clip(voxel_idx, 0, volume_dims - 1)
    voxel_centers = (voxel_idx - (volume_dims - 1) / 2) * resolution
    point_uniq_idx = uniq_idx[inv_idx]

    gs_params = np.concatenate(
        (coord, color, opacity[:, None], scale, quat), axis=1)
    features = np.concatenate(
        (voxel_centers, point_uniq_idx[:, None], gs_params), axis=1).astype(np.float32)

    # Scene-level label distribution
    label_dist = np.zeros(72, dtype=np.float32)
    valid_seg  = segment[segment >= 0]
    if len(valid_seg) > 0:
        for k in range(72):
            label_dist[k] = (valid_seg == k).sum()
        label_dist /= label_dist.sum()

    return features, mean_color, label_dist, coord, segment


@torch.no_grad()
def encode_scene(features_np):
    """
    Encode a single scene. Returns mu_s, mu_g, z, scene_sem_pred, scene_layout_pred.
    """
    feat = torch.from_numpy(features_np).unsqueeze(0).to(device)  # [1, 40000, 18]

    (shape_embed, mu, log_var, z,
     UV_gs_recover, _) = model(
        feat, feat, feat, feat[:, :, :3])

    raw = model.shape_model

    # mu_s and mu_g
    if args.latent_disentangle:
        mu_s = raw._mu_s_cache.squeeze(0).cpu().numpy()   # [semantic_dims]
        mu_g = raw._mu_g_cache.squeeze(0).cpu().numpy()   # [geom_dims]
    else:
        # Without disentanglement, use the full mu split manually for comparison
        mu_s = mu.squeeze(0)[:args.semantic_dims].cpu().numpy()
        mu_g = mu.squeeze(0)[args.semantic_dims:].cpu().numpy()

    z_full  = z.squeeze(0).cpu()
    rec     = UV_gs_recover.squeeze(0).reshape(40000, 14).cpu().numpy()

    scene_sem_pred    = (raw.last_scene_semantic_pred.squeeze(0).cpu().numpy()
                         if raw.last_scene_semantic_pred is not None else None)
    scene_layout_pred = (raw.last_scene_layout_pred.squeeze(0).cpu().numpy()
                         if raw.last_scene_layout_pred is not None else None)
    mean_color_pred   = (raw.last_mean_color_pred.squeeze(0).cpu().numpy()
                         if raw.last_mean_color_pred is not None else None)

    return {
        'mu_s': mu_s,
        'mu_g': mu_g,
        'z':    z_full,
        'mu':   mu.squeeze(0).cpu().numpy(),
        'log_var': log_var.squeeze(0).cpu().numpy(),
        'reconstruction': rec,
        'scene_sem_pred':    scene_sem_pred,
        'scene_layout_pred': scene_layout_pred,
        'mean_color_pred':   mean_color_pred,
    }


@torch.no_grad()
def decode_latent(z_tensor, mean_color_np):
    """Decode a latent tensor [16384] -> PLY-ready numpy [40000, 14] with color added back."""
    z_in   = z_tensor.unsqueeze(0).to(device)     # [1, 16384]
    lat    = z_in.reshape(1, 512, 32)
    UV, _  = model.shape_model.decode(lat, volume_queries=None,
                                       return_semantic_features=False)
    pred   = UV.squeeze(0).reshape(40000, 14).cpu().numpy()

    if args.color_residual and mean_color_np is not None:
        pred[:, 3:6] += mean_color_np
        pred[:, 3:6]  = np.clip(pred[:, 3:6], 0.0, 1.0)
    return pred


# ============================================================================
# LOAD METADATA AND BUILD SCENE INDEX
# ============================================================================

with open(args.metadata) as f:
    metadata = json.load(f)

# Build category → list of scene info
cat_scenes = defaultdict(list)
for entry in metadata:
    cat_scenes[entry['category']].append(entry)

print("Scene inventory:")
for cat, scenes in sorted(cat_scenes.items()):
    ids = [s['scene_id'] for s in scenes]
    print(f"  {cat:25s}: {len(scenes)} scenes  {ids}")
print()

# Filter to requested categories
eval_categories = args.categories
print(f"Evaluating categories: {eval_categories}\n")


# ============================================================================
# ENCODE ALL CURATED SCENES
# ============================================================================

print("Encoding curated scenes...")
encoded = {}   # scene_id -> {'mu_s', 'mu_g', 'z', 'label_dist', 'category', 'mean_color'}

for cat in eval_categories:
    for entry in cat_scenes.get(cat, []):
        sid  = entry['scene_id']
        sdir = find_scene_dir(args.data_root, sid)
        if sdir is None:
            print(f"  [WARN] Scene directory not found for {sid} — skipping")
            continue

        print(f"  Encoding {sid} ({cat})...")
        features, mean_color, label_dist, coord, segment = load_scene(sdir)
        enc = encode_scene(features)

        encoded[sid] = {
            **enc,
            'label_dist': label_dist,
            'category':   cat,
            'scene_id':   sid,
            'mean_color': mean_color,
            'coord':      coord,
            'segment':    segment,
        }

print(f"\nSuccessfully encoded {len(encoded)} scenes.\n")

if len(encoded) < 2:
    print("ERROR: Need at least 2 scenes. Check data_root and scene IDs.")
    sys.exit(1)


# ============================================================================
# EXPERIMENT 1 — INTRA vs INTER-CLASS COSINE DISTANCE IN z_s and z_g
# ============================================================================

print("=" * 70)
print("EXPERIMENT 1 — Intra vs Inter-Class Cosine Distance")
print("=" * 70)
print()
print("Hypothesis:")
print("  z_s (semantic subspace): intra-class distance << inter-class distance")
print("  z_g (geometry subspace): distances NOT organised by semantic category")
print("  This asymmetry is the key signature of successful disentanglement.")
print()

def cosine_dist(a, b):
    """Cosine distance in [0, 2]. 0 = identical, 2 = opposite."""
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - np.dot(a_n, b_n))

# Build per-category lists
cat_to_ids = defaultdict(list)
for sid, data in encoded.items():
    if data['category'] in eval_categories:
        cat_to_ids[data['category']].append(sid)

# Compute per-category intra distances for z_s and z_g
print("─" * 60)
print(f"{'Category':<25}  {'N':>3}  {'Intra z_s':>10}  {'Intra z_g':>10}  {'Pairs':>5}")
print("─" * 60)

intra_zs = {}
intra_zg = {}
for cat in eval_categories:
    ids = cat_to_ids[cat]
    if len(ids) < 2:
        print(f"  {cat:<23}   {len(ids):3d}   (need ≥2 scenes for intra distance)")
        continue
    pairs = list(combinations(ids, 2))
    ds_list = [cosine_dist(encoded[a]['mu_s'], encoded[b]['mu_s']) for a, b in pairs]
    dg_list = [cosine_dist(encoded[a]['mu_g'], encoded[b]['mu_g']) for a, b in pairs]
    intra_zs[cat] = np.mean(ds_list)
    intra_zg[cat] = np.mean(dg_list)
    print(f"  {cat:<23}   {len(ids):3d}   {intra_zs[cat]:10.4f}   {intra_zg[cat]:10.4f}   {len(pairs):5d}")

print()
print("─" * 60)
print(f"{'Category pair':<40}  {'Inter z_s':>10}  {'Inter z_g':>10}  {'Pairs':>5}")
print("─" * 60)

inter_zs_all = []
inter_zg_all = []
for cat_a, cat_b in combinations(eval_categories, 2):
    ids_a = cat_to_ids[cat_a]
    ids_b = cat_to_ids[cat_b]
    if not ids_a or not ids_b:
        continue
    pairs = [(a, b) for a in ids_a for b in ids_b]
    ds_list = [cosine_dist(encoded[a]['mu_s'], encoded[b]['mu_s']) for a, b in pairs]
    dg_list = [cosine_dist(encoded[a]['mu_g'], encoded[b]['mu_g']) for a, b in pairs]
    mean_ds = np.mean(ds_list)
    mean_dg = np.mean(dg_list)
    inter_zs_all.extend(ds_list)
    inter_zg_all.extend(dg_list)
    label = f"{cat_a} vs {cat_b}"
    print(f"  {label:<38}   {mean_ds:10.4f}   {mean_dg:10.4f}   {len(pairs):5d}")

print()

# Summary ratios
if intra_zs and inter_zs_all:
    mean_intra_zs = np.mean(list(intra_zs.values()))
    mean_intra_zg = np.mean(list(intra_zg.values()))
    mean_inter_zs = np.mean(inter_zs_all)
    mean_inter_zg = np.mean(inter_zg_all)
    ratio_zs = mean_inter_zs / (mean_intra_zs + 1e-8)
    ratio_zg = mean_inter_zg / (mean_intra_zg + 1e-8)

    print("─" * 60)
    print(f"SUMMARY:")
    print(f"  Mean intra-class cosine dist:  z_s = {mean_intra_zs:.4f}   z_g = {mean_intra_zg:.4f}")
    print(f"  Mean inter-class cosine dist:  z_s = {mean_inter_zs:.4f}   z_g = {mean_inter_zg:.4f}")
    print(f"  Disentanglement ratio (inter/intra):")
    print(f"    z_s: {ratio_zs:.2f}  (> 1.0 = semantic clustering in z_s)")
    print(f"    z_g: {ratio_zg:.2f}  (≈ 1.0 = geometry NOT clustered by category)")
    print()
    if ratio_zs > ratio_zg and ratio_zs > 1.2:
        print("  ✓ DISENTANGLEMENT SIGNAL: z_s shows stronger semantic clustering than z_g")
    elif ratio_zs > ratio_zg:
        print("  ~ WEAK SIGNAL: z_s slightly more clustered than z_g, but small margin")
    else:
        print("  ✗ NO SIGNAL: z_s and z_g show similar clustering — z_s not semantic")
    print()

# Save results
exp1_results = {
    'intra_zs': {k: float(v) for k, v in intra_zs.items()},
    'intra_zg': {k: float(v) for k, v in intra_zg.items()},
    'inter_zs_mean': float(np.mean(inter_zs_all)) if inter_zs_all else None,
    'inter_zg_mean': float(np.mean(inter_zg_all)) if inter_zg_all else None,
    'ratio_zs': float(ratio_zs) if intra_zs else None,
    'ratio_zg': float(ratio_zg) if intra_zs else None,
}


# ============================================================================
# EXPERIMENT 2 — SCENE SEMANTIC HEAD CONSISTENCY (JSD within/across categories)
# ============================================================================

print("=" * 70)
print("EXPERIMENT 2 — Semantic Head Prediction Consistency (JSD)")
print("=" * 70)
print()
print("Hypothesis:")
print("  Same-category scenes → small JSD between predicted label distributions")
print("  Cross-category scenes → large JSD between predicted label distributions")
print("  If scene_semantic_head is active and z_s encodes semantics,")
print("  the head outputs should reflect category identity.")
print()

def jsd(p, q, eps=1e-8):
    """Jensen-Shannon divergence. Symmetric, in [0, 1] when using log base 2."""
    p = np.clip(p, eps, 1.0);  p = p / p.sum()
    q = np.clip(q, eps, 1.0);  q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m + eps))
    kl_qm = np.sum(q * np.log2(q / m + eps))
    return float(0.5 * kl_pm + 0.5 * kl_qm)

# Use scene_sem_pred if available (from head); fall back to GT label_dist
def get_sem_dist(data):
    if data['scene_sem_pred'] is not None:
        return data['scene_sem_pred']
    return data['label_dist']   # GT fallback

has_sem_pred = any(d['scene_sem_pred'] is not None for d in encoded.values())
sem_source   = "SceneSemanticHead prediction" if has_sem_pred else "GT label distribution (head not active)"
print(f"  Using: {sem_source}\n")

print("─" * 60)
print(f"{'Pair':<40}  {'Same cat':>8}  {'JSD':>8}")
print("─" * 60)

jsd_same  = []
jsd_cross = []
all_ids = list(encoded.keys())
for sid_a, sid_b in combinations(all_ids, 2):
    da = encoded[sid_a]
    db = encoded[sid_b]
    same_cat = da['category'] == db['category']
    if da['category'] not in eval_categories or db['category'] not in eval_categories:
        continue
    d = jsd(get_sem_dist(da), get_sem_dist(db))
    label = f"{sid_a[-10:]} ({da['category'][:6]}) vs {sid_b[-10:]} ({db['category'][:6]})"
    print(f"  {label:<38}  {'same' if same_cat else 'diff':>8}  {d:8.4f}")
    if same_cat:
        jsd_same.append(d)
    else:
        jsd_cross.append(d)

print()
if jsd_same:
    print(f"  Mean JSD within same category:  {np.mean(jsd_same):.4f}  (n={len(jsd_same)})")
if jsd_cross:
    print(f"  Mean JSD across categories:     {np.mean(jsd_cross):.4f}  (n={len(jsd_cross)})")
if jsd_same and jsd_cross:
    ratio = np.mean(jsd_cross) / (np.mean(jsd_same) + 1e-8)
    print(f"  Cross/within JSD ratio:         {ratio:.2f}  (> 1.0 = category-discriminative)")
    print()
    if ratio > 1.5:
        print("  ✓ SEMANTIC SIGNAL: Cross-category scenes have meaningfully different predicted distributions")
    elif ratio > 1.1:
        print("  ~ WEAK SIGNAL: Slight difference between within and cross-category distributions")
    else:
        print("  ✗ NO SIGNAL: z_s does not produce category-discriminative semantic predictions")
print()

exp2_results = {
    'jsd_same_mean':  float(np.mean(jsd_same))  if jsd_same  else None,
    'jsd_cross_mean': float(np.mean(jsd_cross)) if jsd_cross else None,
    'sem_source':     sem_source,
}


# ============================================================================
# EXPERIMENT 3 — QUALITATIVE SWAP VISUALISATION
# ============================================================================

print("=" * 70)
print("EXPERIMENT 3 — Qualitative Latent Swap Visualisation")
print("=" * 70)
print()
print(f"  Swap pair: {args.swap_cat_a}  ↔  {args.swap_cat_b}")
print()

swap_ids_a = cat_to_ids[args.swap_cat_a]
swap_ids_b = cat_to_ids[args.swap_cat_b]

if not swap_ids_a or not swap_ids_b:
    print(f"  [SKIP] One or both swap categories not found in encoded scenes.")
else:
    # Select scenes
    idx_a = min(args.swap_scene_idx_a, len(swap_ids_a) - 1)
    idx_b = min(args.swap_scene_idx_b, len(swap_ids_b) - 1)
    sid_a = swap_ids_a[idx_a]
    sid_b = swap_ids_b[idx_b]

    data_a = encoded[sid_a]
    data_b = encoded[sid_b]

    print(f"  Scene A: {sid_a} ({args.swap_cat_a})")
    print(f"  Scene B: {sid_b} ({args.swap_cat_b})")
    print()

    D_s = args.semantic_dims

    # Build swap latents from mu (posterior means, not sampled z — avoids noise)
    mu_a  = torch.from_numpy(data_a['mu']).float()   # [16384]
    mu_b  = torch.from_numpy(data_b['mu']).float()

    # z_AB: semantic from A, geometry from B
    z_AB = torch.cat([mu_a[:D_s], mu_b[D_s:]], dim=0)
    # z_BA: semantic from B, geometry from A
    z_BA = torch.cat([mu_b[:D_s], mu_a[D_s:]], dim=0)

    print("  Decoding 4 variants...")

    # Update conditioning signal for decoder before each decode
    # Reconstruction A (self)
    if args.scene_layout_head and args.token_cond and 'B' in args.token_cond_approach.upper():
        if args.semantic_token_heads:
            _ed = model.shape_model.embed_dim
            z_sem_a = mu_a[_ed:D_s]
            with torch.no_grad():
                model.shape_model.last_scene_layout_pred = \
                    model.shape_model.scene_layout_module(
                        z_sem_a.unsqueeze(0).to(device)).squeeze(0).unsqueeze(0)
        else:
            # Encode scene A to get shape_embed for conditioning
            feat_a = torch.from_numpy(
                load_scene(find_scene_dir(args.data_root, sid_a))[0]
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                model.shape_model.last_scene_layout_pred = \
                    model.shape_model.scene_layout_module(
                        model.shape_model.encode_latents(feat_a, feat_a)[0].unsqueeze(0)
                    )

    recon_a = decode_latent(mu_a, data_a['mean_color'] if args.color_residual else None)

    # Reconstruction B (self)
    if args.scene_layout_head and args.token_cond and 'B' in args.token_cond_approach.upper():
        if args.semantic_token_heads:
            z_sem_b = mu_b[_ed:D_s]
            with torch.no_grad():
                model.shape_model.last_scene_layout_pred = \
                    model.shape_model.scene_layout_module(
                        z_sem_b.unsqueeze(0).to(device)).squeeze(0).unsqueeze(0)

    recon_b = decode_latent(mu_b, data_b['mean_color'] if args.color_residual else None)

    # Cross-recon z_AB: semantics A, geometry B
    # Layout conditioning should use A's semantic tokens (we swapped z_s^A in)
    if args.scene_layout_head and args.token_cond and 'B' in args.token_cond_approach.upper():
        if args.semantic_token_heads:
            z_sem_a = mu_a[_ed:D_s]
            with torch.no_grad():
                model.shape_model.last_scene_layout_pred = \
                    model.shape_model.scene_layout_module(
                        z_sem_a.unsqueeze(0).to(device)).squeeze(0).unsqueeze(0)

    recon_AB = decode_latent(z_AB,
                              data_a['mean_color'] if args.color_residual else None)

    # Cross-recon z_BA: semantics B, geometry A
    if args.scene_layout_head and args.token_cond and 'B' in args.token_cond_approach.upper():
        if args.semantic_token_heads:
            z_sem_b = mu_b[_ed:D_s]
            with torch.no_grad():
                model.shape_model.last_scene_layout_pred = \
                    model.shape_model.scene_layout_module(
                        z_sem_b.unsqueeze(0).to(device)).squeeze(0).unsqueeze(0)

    recon_BA = decode_latent(z_BA,
                              data_b['mean_color'] if args.color_residual else None)

    # Save all four
    swap_dir = output_dir / 'swap_visualisation'
    swap_dir.mkdir(parents=True, exist_ok=True)

    from gs_ply_reconstructor import reconstruct_single_scene
    variants = [
        (recon_a,  f'A_self_{sid_a}_{args.swap_cat_a}.ply',
         f"Original A ({args.swap_cat_a})"),
        (recon_b,  f'B_self_{sid_b}_{args.swap_cat_b}.ply',
         f"Original B ({args.swap_cat_b})"),
        (recon_AB, f'AB_semA_geoB_{args.swap_cat_a}sem_{args.swap_cat_b}geo.ply',
         f"z_AB: semantics={args.swap_cat_a}, geometry={args.swap_cat_b}"),
        (recon_BA, f'BA_semB_geoA_{args.swap_cat_b}sem_{args.swap_cat_a}geo.ply',
         f"z_BA: semantics={args.swap_cat_b}, geometry={args.swap_cat_a}"),
    ]

    print()
    for pred, fname, desc in variants:
        path = swap_dir / fname
        reconstruct_single_scene(pred, path, max_sh_degree=args.max_sh_degree, verbose=False)
        print(f"  Saved: {path.name}")
        print(f"    → {desc}")

    # Quantitative geometry preservation check
    print()
    print("  Geometry preservation check (position L2):")
    print("  Goal: recon_AB position should be close to original B (z_g^B preserved)")
    print()

    def pos_l2(pred_a, pred_b):
        return float(np.linalg.norm(pred_a[:, :3] - pred_b[:, :3]) / len(pred_a))

    l2_a_self  = pos_l2(recon_a, data_a['reconstruction'])
    l2_b_self  = pos_l2(recon_b, data_b['reconstruction'])
    l2_AB_vs_B = pos_l2(recon_AB, recon_b)   # cross-AB vs original B
    l2_BA_vs_A = pos_l2(recon_BA, recon_a)   # cross-BA vs original A

    print(f"    Self-recon A vs re-decode A:      {l2_a_self:.4f}  (should be ~0 — sanity check)")
    print(f"    Self-recon B vs re-decode B:      {l2_b_self:.4f}  (sanity check)")
    print(f"    z_AB reconstruction vs B:          {l2_AB_vs_B:.4f}  (lower = geometry of B preserved)")
    print(f"    z_BA reconstruction vs A:          {l2_BA_vs_A:.4f}  (lower = geometry of A preserved)")
    print()
    print("  Interpretation: if z_AB is close to B in position space, z_g^B was")
    print("  decoded correctly despite replacing z_s with scene A's semantic token.")
    print("  This would confirm z_g carries the geometry and z_s carries the semantics.")
    print()

    exp3_results = {
        'sid_a': sid_a,  'cat_a': args.swap_cat_a,
        'sid_b': sid_b,  'cat_b': args.swap_cat_b,
        'l2_a_self':   l2_a_self,
        'l2_b_self':   l2_b_self,
        'l2_AB_vs_B':  l2_AB_vs_B,
        'l2_BA_vs_A':  l2_BA_vs_A,
    }


# ============================================================================
# SAVE SUMMARY REPORT
# ============================================================================

results = {
    'checkpoint':     args.checkpoint,
    'categories':     eval_categories,
    'num_scenes':     len(encoded),
    'experiment_1':   exp1_results,
    'experiment_2':   exp2_results,
    'experiment_3':   exp3_results if swap_ids_a and swap_ids_b else 'skipped',
}

report_path = output_dir / 'results.json'
with open(report_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved → {report_path}")

# Print final summary for quick reading
print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  Exp 1 — Disentanglement ratio:  z_s={exp1_results.get('ratio_zs', 'N/A'):.2f}  "
      f"z_g={exp1_results.get('ratio_zg', 'N/A'):.2f}")
if exp2_results['jsd_same_mean'] is not None:
    print(f"  Exp 2 — JSD ratio (cross/within):  "
          f"{exp2_results['jsd_cross_mean']:.4f} / {exp2_results['jsd_same_mean']:.4f} = "
          f"{exp2_results['jsd_cross_mean'] / (exp2_results['jsd_same_mean'] + 1e-8):.2f}")
if isinstance(exp3_results, dict):
    print(f"  Exp 3 — Geometry preservation:  "
          f"z_AB vs B = {exp3_results['l2_AB_vs_B']:.4f}  "
          f"(self-recon baseline = {exp3_results['l2_b_self']:.4f})")
print()
print(f"  PLY files for visual inspection: {swap_dir}")
print(f"  Load all 4 PLYs in SuperSplat and compare.")
print()