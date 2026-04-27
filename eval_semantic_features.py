"""
eval_semantic_features.py
=========================
Quantitative evaluation of semantic feature quality in Can3Tok VAE.

PURPOSE
-------
Quantifies the semantic structure enforced by different InfoNCE losses.
Addresses the supervisor's suggestion: instead of only visual PCA,
measure the actual intra-class vs inter-class distances in feature space.

THREE EXPERIMENTS
-----------------
Exp A — Per-Gaussian Feature Quality  [requires --semantic_mode hidden]
    Extracts the [B, 40000, 32] L2-normalised per-Gaussian features from
    SemanticProjectionHead (decoder hidden → projection → per-Gaussian features).
    These are the exact features that pgNCE and Pool+pgNCE supervise.
    Three metrics:
      A1. Fisher Ratio  = mean_inter_prototype_dist / mean_intra_dist
          Same formula as architecture Exp1 but at Gaussian granularity.
      A2. Silhouette Score  (per-Gaussian, subsampled for speed)
          s(i) = (b(i) - a(i)) / max(a(i), b(i)) in [-1, 1]
      A3. Linear Probe Accuracy
          Logistic regression on frozen [N_total, 32] features predicting
          ScanNet72 category. Standard evaluation from PointContrast (ECCV 2020).

Exp B — Scene-Level Projection Head Discrimination  [always runs]
    Measures how discriminative the projection head OUTPUTS are, not mu_s.
    The architecture eval (Exp1) measured clustering on raw mu_s [512].
    This experiment measures clustering on the InfoNCE-supervised projections:
      - z_layout_proj [B, 128]   from LayNCE head
      - pool_hidden   [B, 1024]  from PoolNCE intermediate
      - mu_s          [B, 512]   raw latent (reference)
    Comparison shows whether InfoNCE heads produce more discriminative
    representations than the underlying latent space.

Exp C — Per-Category Feature Quality Breakdown  [requires Exp A]
    Reports Fisher Ratio and Silhouette per ScanNet72 category, sorted
    from best to worst. Shows which semantic categories (furniture, objects)
    cluster well vs which do not (floor, wall, ceiling).

USAGE EXAMPLES
--------------
# pgNCE model:
python eval_semantic_features.py \\
    --checkpoint /path/to/pgNCE_best_model.pth \\
    --data_root /path/to/scenes \\
    --output_dir eval_results/pgNCE \\
    --semantic_mode hidden \\
    --latent_disentangle --semantic_token_heads --structured_layout_tokens \\
    --color_residual --decoder_fourier_pe \\
    --scene_semantic_head --scene_layout_head

# Pool+pgNCE model:
python eval_semantic_features.py \\
    --checkpoint /path/to/pool_pgNCE_best_model.pth \\
    --data_root /path/to/scenes \\
    --output_dir eval_results/pool_pgNCE \\
    --semantic_mode hidden --zs_pool_infonce_weight 0.1 \\
    --latent_disentangle --semantic_token_heads --structured_layout_tokens \\
    --color_residual --decoder_fourier_pe \\
    --scene_semantic_head --scene_layout_head

# LayNCE model (Exp B only, no per-Gaussian features):
python eval_semantic_features.py \\
    --checkpoint /path/to/layNCE_best_model.pth \\
    --data_root /path/to/scenes \\
    --output_dir eval_results/layNCE \\
    --semantic_mode none --zs_layout_infonce_weight 0.1 \\
    --latent_disentangle --semantic_token_heads --structured_layout_tokens \\
    --color_residual --decoder_fourier_pe \\
    --scene_semantic_head --scene_layout_head
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not available. "
          "Install: pip install scikit-learn --break-system-packages")
    print("       Linear probe (Exp A3) will be skipped.\n")

# ============================================================================
#  ARGUMENT PARSING
# ============================================================================
ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint',    required=True)
ap.add_argument('--data_root',     required=True)
ap.add_argument('--output_dir',    default='eval_results/semantic_features')
ap.add_argument('--device',        default='cuda')
ap.add_argument('--n_scenes',      type=int, default=15)
ap.add_argument('--sil_subsample', type=int, default=2000,
                help='Gaussians per scene for silhouette (reduces O(N^2) cost)')
ap.add_argument('--lp_n_folds',    type=int, default=5,
                help='K-fold cross-validation folds for linear probe')
ap.add_argument('--lp_per_scene',  type=int, default=2000,
                help='Gaussians per scene collected for linear probe dataset')
# Model flags — must match checkpoint
ap.add_argument('--semantic_mode',             type=str,  default='none')
ap.add_argument('--latent_disentangle',        action='store_true', default=False)
ap.add_argument('--semantic_dims',             type=int,  default=512)
ap.add_argument('--semantic_token_heads',      action='store_true', default=False)
ap.add_argument('--structured_layout_tokens',  action='store_true', default=False)
ap.add_argument('--color_residual',            action='store_true', default=False)
ap.add_argument('--scene_semantic_head',       action='store_true', default=False)
ap.add_argument('--scene_layout_head',         action='store_true', default=False)
ap.add_argument('--decoder_fourier_pe',        action='store_true', default=False)
ap.add_argument('--decoder_pos_enc',           action='store_true', default=False)
ap.add_argument('--decoder_layout_cross_attn', action='store_true', default=False)
ap.add_argument('--decoder_layout_additive',   action='store_true', default=False)
ap.add_argument('--token_cond',                action='store_true', default=False)
ap.add_argument('--token_cond_approach',       type=str,  default='B')
ap.add_argument('--token_cond_adaln',          action='store_true', default=False)
ap.add_argument('--position_scaffold',         action='store_true', default=False)
ap.add_argument('--zs_layout_infonce_weight',  type=float, default=0.0)
ap.add_argument('--zs_pool_infonce_weight',    type=float, default=0.0)
args = ap.parse_args()

RUN_EXP_A = (args.semantic_mode == 'hidden')
device    = torch.device(args.device if torch.cuda.is_available() else 'cpu')
out_dir   = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*68}")
print(f"  Can3Tok — Semantic Feature Quality Evaluation")
print(f"{'='*68}")
print(f"  Checkpoint:    {args.checkpoint}")
print(f"  semantic_mode: {args.semantic_mode}")
print(f"  Exp A (per-Gaussian): {'ENABLED' if RUN_EXP_A else 'DISABLED — add --semantic_mode hidden'}")
print(f"  Exp B (scene-level):  ENABLED")
print(f"  Exp C (per-category): {'ENABLED' if RUN_EXP_A else 'DISABLED'}")
print(f"{'='*68}\n")

# ============================================================================
#  MODEL LOADING
# ============================================================================
print("Loading model...")
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file

config_path  = './model/configs/aligned_shape_latents/shapevae-256.yaml'
model_config = get_config_from_file(config_path).model
p = model_config.params.shape_module_cfg.params

p.semantic_mode              = args.semantic_mode
p.color_residual             = args.color_residual
p.scene_semantic_head        = args.scene_semantic_head
p.scene_layout_head          = args.scene_layout_head
p.latent_disentangle         = args.latent_disentangle
p.semantic_dims              = args.semantic_dims
p.decoder_pos_enc            = args.decoder_pos_enc
p.decoder_fourier_pe         = args.decoder_fourier_pe
p.token_cond                 = args.token_cond
p.token_cond_approach        = args.token_cond_approach
p.token_cond_adaln           = args.token_cond_adaln
p.semantic_token_heads       = args.semantic_token_heads
p.position_scaffold          = args.position_scaffold
p.decoder_layout_cross_attn  = args.decoder_layout_cross_attn
p.decoder_layout_additive    = args.decoder_layout_additive
p.structured_layout_tokens   = args.structured_layout_tokens
p.decoder_zs_cross_attn     = False
p.jepa_idea1                 = False
p.predict_seg_labels         = False
p.query_decoder              = False
p.position_layout_residual   = False

model = instantiate_from_config(model_config)

# Pre-filter shape mismatches so strict=False actually works
ckpt     = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
ckpt_sd  = ckpt['model_state_dict']
model_sd = model.state_dict()
filtered, skipped = {}, []
for k, v in ckpt_sd.items():
    if k in model_sd and model_sd[k].shape != v.shape:
        skipped.append(k)
    else:
        filtered[k] = v
if skipped:
    print(f"  [WARN] {len(skipped)} shape-mismatched keys skipped (version diff)")
    for k in skipped[:4]:
        print(f"    {k}")

missing, _ = model.load_state_dict(filtered, strict=False)
model.to(device)
model.eval()
sm = model.shape_model

print(f"  epoch={ckpt.get('epoch','?')}  val_L2={ckpt.get('val_l2_error','?')}")
print(f"  SemanticProjectionHead (pg features): {sm.semantic_projection_hidden is not None}")
print(f"  z_layout_infonce_head  (LayNCE):      {sm.z_layout_infonce_head is not None}")
print(f"  z_s_infonce_head       (z_s pool):    {sm.z_s_infonce_head is not None}")
print(f"  zs_pool_proj_head      (PoolNCE):     {sm.zs_pool_proj_head is not None}")
print()

_any_B = args.decoder_layout_cross_attn or args.decoder_layout_additive

# ============================================================================
#  DATA LOADING  (exact mirror of gs_dataset)
# ============================================================================
SCENE_META = [
    ("0202_840156", "spa_pool"),         ("0203_840160", "convenience_store"),
    ("0207_840167", "apartment"),        ("0211_840172", "spa_pool"),
    ("0212_840152", "spa_pool"),         ("0214_840176", "convenience_store"),
    ("0216_840180", "club"),             ("0217_840181", "club"),
    ("0218_840182", "coffee_shop"),      ("0219_840183", "coffee_shop"),
    ("0220_840185", "coffee_shop"),      ("0221_840242", "apartment"),
    ("0222_840246", "apartment"),        ("0223_840262", "apartment"),
    ("0224_840270", "apartment"),
][:args.n_scenes]

TARGET_N = 40_000
from gs_dataset_scenesplat import normalize_to_canonical_sphere, voxelize


def _find_scene_dir(sid):
    for d in sorted(Path(args.data_root).iterdir()):
        if d.is_dir() and sid in d.name:
            return d
    return None


def load_scene(scene_dir):
    """
    Exact mirror of gs_dataset.__getitem__.
    Returns: features [40000, 18], mean_color [3], label_dist [72], segment [40000].
    """
    d = Path(scene_dir)
    coord   = np.load(d / 'coord.npy').astype(np.float32)
    color   = np.load(d / 'color.npy').astype(np.float32)
    scale   = np.load(d / 'scale.npy').astype(np.float32)
    quat    = np.load(d / 'quat.npy').astype(np.float32)
    opacity = np.load(d / 'opacity.npy').astype(np.float32)

    coord, scale = normalize_to_canonical_sphere(
        coord, scale, target_radius=10.0, scale_norm_mode='linear')
    color = color / 255.0

    try:
        segment = np.load(d / 'segment.npy').astype(np.int32)
    except FileNotFoundError:
        segment = np.full(len(coord), -1, dtype=np.int32)

    N = len(coord)
    sorted_idx = np.argsort(opacity)
    if N >= TARGET_N:
        sel = sorted_idx[-TARGET_N:]
    else:
        pad = np.full(TARGET_N - N, sorted_idx[-1])
        sel = np.concatenate([sorted_idx, pad])

    coord   = coord  [sel]; color   = color  [sel]
    scale   = scale  [sel]; quat    = quat   [sel]
    opacity = opacity[sel]; segment = segment[sel]

    mean_color = color.mean(axis=0).astype(np.float32)
    if args.color_residual:
        color = color - mean_color

    # Voxelisation — FNV hash (identical to training)
    volume_dims = 40
    res         = 16.0 / volume_dims
    uniq_idx, inv_idx, _ = voxelize(coord, res, 'fnv')
    origin_off  = np.array([(volume_dims - 1) / 2] * 3) * res
    vox_idx     = np.clip(np.floor((coord + origin_off) / res), 0, volume_dims - 1)
    vox_centers = (vox_idx - (volume_dims - 1) / 2) * res
    pt_idx      = uniq_idx[inv_idx]

    gs_params = np.concatenate([coord, color, opacity[:, None], scale, quat], axis=1)
    features  = np.concatenate(
        [vox_centers, pt_idx[:, None], gs_params], axis=1).astype(np.float32)

    label_dist = np.zeros(72, dtype=np.float32)
    valid_seg  = segment[segment >= 0]
    if len(valid_seg) > 0:
        for k in range(72):
            label_dist[k] = (valid_seg == k).sum()
        label_dist /= label_dist.sum()

    return features, mean_color, label_dist, segment


# ============================================================================
#  FORWARD PASS
# ============================================================================
@torch.no_grad()
def run_forward(features_np):
    """
    Full Can3Tok forward pass.

    Sets return_semantic_features=True so the decoder runs
    SemanticProjectionHead if semantic_mode='hidden'.

    Returns a dict with:
      pg_features    [40000, 32] or None — per-Gaussian L2-norm features
      mu_s           [512]              — raw semantic latent mean
      z_layout_proj  [128]  or None     — LayNCE projection head output
      pool_hidden    [1024] or None     — PoolNCE intermediate state
    """
    feat = torch.from_numpy(features_np).unsqueeze(0).to(device)

    # AlignedShapeLatentPerceiver.forward signature:
    #   (pc, feats, volume_queries, scaffold_anchors=None, ...,
    #    return_semantic_features=None)
    # Returns: shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features
    (shape_embed, mu, log_var, z,
     UV_gs_recover, pg_features) = model(
        feat, feat, feat, feat[:, :, :3],
        return_semantic_features=True)

    result = {}

    # ── Per-Gaussian features ──────────────────────────────────────────────
    # Non-None only when semantic_mode='hidden' AND return_semantic_features=True
    result['pg_features'] = (
        pg_features.squeeze(0).cpu().numpy()   # [40000, 32]  L2-normalised
        if pg_features is not None else None)

    # ── mu_s — raw semantic latent mean ───────────────────────────────────
    # For Strategy A: sm._mu_s_cache is set by the encoder split
    # For Strategy B/C: use first semantic_dims of mu
    if args.latent_disentangle and sm._mu_s_cache is not None:
        result['mu_s'] = sm._mu_s_cache.squeeze(0).cpu().numpy()  # [512]
    else:
        result['mu_s'] = mu.squeeze(0)[:args.semantic_dims].cpu().numpy()

    # ── z_layout_proj [128] — LayNCE head output ──────────────────────────
    # For Strategy A with latent_disentangle:
    #   forward() routes last_z_s_infonce_proj → last_z_layout_proj
    # For Strategy B:
    #   forward() runs z_layout_infonce_head on flatten(z_layout)
    result['z_layout_proj'] = (
        sm.last_z_layout_proj.squeeze(0).cpu().numpy()
        if sm.last_z_layout_proj is not None else None)

    # ── pool_hidden [1024] — PoolNCE intermediate ─────────────────────────
    # Strategy A: last_zs_pool_hidden set by zs_pool_proj_head
    # Strategy B: last_z_layout_pool_hidden set by z_layout_pool_head
    ph = getattr(sm, 'last_zs_pool_hidden', None)
    if ph is None:
        ph = getattr(sm, 'last_z_layout_pool_hidden', None)
    result['pool_hidden'] = (
        ph.squeeze(0).cpu().numpy() if ph is not None else None)

    return result


# ============================================================================
#  ENCODE ALL SCENES
# ============================================================================
print("Encoding scenes...")
print()
encoded = {}

for sid, cat in SCENE_META:
    sdir = _find_scene_dir(sid)
    if sdir is None:
        print(f"  [SKIP] {sid} — not found under {args.data_root}")
        continue
    feat, mean_color, label_dist, segment = load_scene(sdir)
    fwd = run_forward(feat)
    encoded[sid] = {
        **fwd,
        'category':   cat,
        'label_dist': label_dist,
        'segment':    segment,   # [40000] ScanNet72 labels, -1 = unlabelled
    }
    pg = '✓' if fwd['pg_features']   is not None else '—'
    zl = '✓' if fwd['z_layout_proj'] is not None else '—'
    ph = '✓' if fwd['pool_hidden']   is not None else '—'
    print(f"  {sid}  {cat:<22}  pg={pg}  z_lay={zl}  pool={ph}")

print(f"\n  {len(encoded)} scenes encoded.\n")

# ============================================================================
#  SHARED METRIC HELPERS
# ============================================================================

# Partial ScanNet72 name map (indices 0-19 cover most indoor scenes)
CATNAMES = {
    0:"wall",        1:"floor",        2:"cabinet",     3:"bed",
    4:"chair",       5:"sofa",         6:"table",       7:"door",
    8:"window",      9:"bookshelf",   10:"picture",    11:"counter",
   12:"desk",       13:"curtain",     14:"refrigerator", 15:"shower_curtain",
   16:"toilet",     17:"sink",        18:"bathtub",    19:"otherfurniture",
   20:"person",     21:"desk2",       22:"curtain2",
}


def l2_norm_rows(M):
    """L2-normalise each row of matrix M [N, D]."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(norms < 1e-8, 1.0, norms)


def compute_fisher_gaussians(features, labels):
    """
    Fisher Ratio = mean_inter_prototype_dist / mean_intra_dist
    on per-Gaussian L2-normalised features.

    features : [N, D]  float32 — already L2-normalised
    labels   : [N]     int32   — ScanNet72 label, -1 = unlabelled

    Returns
    -------
    ratio      : float | None
    mean_intra : float | None
    mean_inter : float | None
    per_cat    : dict {cat_id: {'intra': float, 'n': int}}
    """
    mask  = labels >= 0
    F, L  = features[mask], labels[mask]
    ucat  = np.unique(L)

    if len(ucat) < 2 or len(F) == 0:
        return None, None, None, {}

    protos     = []
    intra_vals = {}

    for c in ucat:
        cf    = F[L == c]
        proto = l2_norm_rows(cf.mean(axis=0, keepdims=True))[0]
        protos.append(proto)
        # Mean cosine distance from all same-class Gaussians to their prototype
        intra_vals[int(c)] = {
            'intra': float(np.mean(1.0 - cf @ proto)),
            'n':     int(len(cf))
        }

    P = np.stack(protos)                              # [K, D]
    K = len(P)
    # Pairwise inter-prototype cosine distances (upper triangle only)
    inter = [1.0 - float(P[i] @ P[j]) for i in range(K) for j in range(i+1, K)]

    mean_intra = float(np.mean([v['intra'] for v in intra_vals.values()]))
    mean_inter = float(np.mean(inter)) if inter else 0.0
    ratio      = mean_inter / max(mean_intra, 1e-8)
    return ratio, mean_intra, mean_inter, intra_vals


def compute_silhouette_gaussians(features, labels, subsample):
    """
    Approximate silhouette score on per-Gaussian features.
    Subsamples balanced across categories to keep O(N^2) tractable.

    Returns
    -------
    mean_sil : float | None  in [-1, 1]
    per_cat  : dict {cat_id: float}
    """
    mask  = labels >= 0
    F, L  = features[mask], labels[mask]
    ucat  = np.unique(L)
    if len(ucat) < 2:
        return None, {}

    # Balanced subsample — at most (subsample // n_cats) per category
    per_k   = max(10, subsample // len(ucat))
    sel_idx = []
    for c in ucat:
        ci = np.where(L == c)[0]
        if len(ci) > per_k:
            ci = np.random.choice(ci, per_k, replace=False)
        sel_idx.append(ci)
    sel = np.concatenate(sel_idx)
    F   = F[sel]
    L   = L[sel]

    if len(np.unique(L)) < 2:
        return None, {}

    # Full cosine distance matrix on the subsample  [N_sub, N_sub]
    D = (1.0 - F @ F.T).astype(np.float64)
    np.fill_diagonal(D, 0.0)

    N     = len(F)
    s_arr = np.zeros(N)
    for i in range(N):
        same_mask    = L == L[i]
        same_mask[i] = False
        other_mask   = ~same_mask
        other_mask  &= (L >= 0)

        if same_mask.sum() == 0:
            s_arr[i] = 0.0
            continue

        a = D[i, same_mask].mean()

        # b = smallest mean distance to any other category
        b_vals = [D[i, L == c].mean() for c in np.unique(L[other_mask])]
        b = min(b_vals) if b_vals else 0.0

        denom    = max(a, b)
        s_arr[i] = (b - a) / denom if denom > 1e-8 else 0.0

    mean_sil = float(s_arr.mean())
    per_cat  = {int(c): float(s_arr[L == c].mean()) for c in np.unique(L)}
    return mean_sil, per_cat


def compute_fisher_scenes(vecs_dict, cats_dict):
    """
    Fisher Ratio on scene-level vectors (one vector per scene).

    vecs_dict : {scene_id: np.array [D]}
    cats_dict : {scene_id: str category name}

    Returns ratio, mean_intra, mean_inter, {cat_name: intra}
    """
    sids = list(vecs_dict.keys())
    if len(sids) < 2:
        return None, None, None, {}

    V    = l2_norm_rows(np.stack([vecs_dict[s] for s in sids]))  # [N, D]
    cats = np.array([cats_dict[s] for s in sids])
    ucat = np.unique(cats)
    if len(ucat) < 2:
        return None, None, None, {}

    protos, intras = [], {}
    for c in ucat:
        m     = cats == c
        proto = l2_norm_rows(V[m].mean(axis=0, keepdims=True))[0]
        protos.append(proto)
        intras[c] = float(np.mean(1.0 - V[m] @ proto))

    P     = np.stack(protos)
    K     = len(P)
    inter = [1.0 - float(P[i] @ P[j]) for i in range(K) for j in range(i+1, K)]

    mean_i = float(np.mean(list(intras.values())))
    mean_e = float(np.mean(inter)) if inter else 0.0
    ratio  = mean_e / max(mean_i, 1e-8)
    return ratio, mean_i, mean_e, intras


# ============================================================================
#  EXPERIMENT A — PER-GAUSSIAN FEATURE QUALITY
# ============================================================================
exp_a = {'enabled': RUN_EXP_A, 'scenes': {}}

if not RUN_EXP_A:
    print("EXPERIMENT A: SKIPPED  (add --semantic_mode hidden to enable)\n")
else:
    print("=" * 68)
    print("EXPERIMENT A  —  Per-Gaussian Feature Quality")
    print("=" * 68)
    print()
    print("  Features: SemanticProjectionHead output [40000, 32] L2-normalised.")
    print("  Labels:   Ground-truth ScanNet72 per-Gaussian segment labels.")
    print("  Metrics:  Fisher Ratio (A1), Silhouette Score (A2), Linear Probe (A3).")
    print()

    hdr = (f"  {'Scene':<15} {'Category':<22} "
           f"{'Fisher':>7} {'Intra':>7} {'Inter':>7} {'Sil':>7} {'N_cats':>7}")
    print(hdr)
    print("  " + "─" * 71)

    all_fisher, all_intra, all_inter, all_sil = [], [], [], []
    per_cat_intra = defaultdict(list)   # cat_id → [per-scene intra distances]
    per_cat_sil   = defaultdict(list)
    lp_X, lp_y   = [], []              # collected for linear probe

    for sid, data in encoded.items():
        feats = data['pg_features']
        segs  = data['segment']

        if feats is None:
            print(f"  {sid:<15} {data['category']:<22}  (no per-Gaussian features)")
            continue

        # A1 — Fisher Ratio
        ratio, intra, inter, per_cat = compute_fisher_gaussians(feats, segs)
        # A2 — Silhouette
        sil, per_cat_s = compute_silhouette_gaussians(feats, segs, args.sil_subsample)

        if ratio is not None:
            all_fisher.append(ratio)
            all_intra.append(intra)
            all_inter.append(inter)
            for c, cv in per_cat.items():
                per_cat_intra[c].append(cv['intra'])
        if sil is not None:
            all_sil.append(sil)
            for c, sv in per_cat_s.items():
                per_cat_sil[c].append(sv)

        # Collect subsampled features for linear probe (A3)
        valid = segs >= 0
        if valid.sum() > 0:
            idx = np.where(valid)[0]
            if len(idx) > args.lp_per_scene:
                idx = np.random.choice(idx, args.lp_per_scene, replace=False)
            lp_X.append(feats[idx])
            lp_y.append(segs[idx])

        ncats = int(len(np.unique(segs[segs >= 0]))) if (segs >= 0).any() else 0
        r_s = f"{ratio:.3f}" if ratio is not None else "  n/a"
        i_s = f"{intra:.3f}" if intra is not None else "  n/a"
        e_s = f"{inter:.3f}" if inter is not None else "  n/a"
        s_s = f"{sil:.3f}"   if sil   is not None else "  n/a"
        print(f"  {sid:<15} {data['category']:<22} "
              f"{r_s:>7} {i_s:>7} {e_s:>7} {s_s:>7} {ncats:>7}")

        exp_a['scenes'][sid] = {
            'fisher_ratio': ratio, 'mean_intra': intra,
            'mean_inter': inter, 'silhouette': sil,
        }

    print("  " + "─" * 71)
    mf = float(np.mean(all_fisher)) if all_fisher else None
    mi = float(np.mean(all_intra))  if all_intra  else None
    me = float(np.mean(all_inter))  if all_inter  else None
    ms = float(np.mean(all_sil))    if all_sil    else None

    def _fmt(v): return f"{v:.3f}" if v is not None else "  n/a"
    print(f"  {'MEAN':<15} {'':<22} {_fmt(mf):>7} {_fmt(mi):>7} {_fmt(me):>7} {_fmt(ms):>7}")
    print()

    exp_a.update({
        'mean_fisher_ratio': mf,
        'mean_intra': mi, 'mean_inter': me,
        'mean_silhouette': ms,
    })

    # ── A3: Linear Probe ──────────────────────────────────────────────────
    print("  A3 — Linear Probe")
    print("  " + "─" * 50)
    lp_acc = lp_std = None

    if not HAS_SKLEARN:
        print("  Skipped — install scikit-learn to enable\n")
    elif not lp_X:
        print("  Skipped — no valid features collected\n")
    else:
        X = np.concatenate(lp_X)
        y = np.concatenate(lp_y)

        # Keep labelled, categories with >= 5 examples
        mask = y >= 0
        X, y = X[mask], y[mask]
        cats_present, cnts = np.unique(y, return_counts=True)
        keep = cats_present[cnts >= 5]
        mask = np.isin(y, keep)
        X, y = X[mask], y[mask]
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]

        print(f"  Total points: {len(X):,}  |  Categories: {len(keep)}")
        print(f"  Random baseline: ~{100/len(keep):.1f}%")
        print()

        # LinearSVC is much faster and more memory-efficient than
        # LogisticRegression(lbfgs) for large multiclass problems.
        # lbfgs on 30k points × 42 classes runs out of memory on cluster nodes.
        # LinearSVC uses a dual coordinate descent solver that scales well.
        # Cap per-fold train to 10k points to keep runtime under 2 minutes.
        MAX_TRAIN = 10000
        fold_n, accs = len(X) // args.lp_n_folds, []
        for fold in range(args.lp_n_folds):
            v0, v1    = fold * fold_n, (fold + 1) * fold_n
            X_val     = X[v0:v1];  y_val = y[v0:v1]
            X_tr      = np.concatenate([X[:v0], X[v1:]])
            y_tr      = np.concatenate([y[:v0], y[v1:]])
            # Subsample training set if large
            if len(X_tr) > MAX_TRAIN:
                idx  = np.random.choice(len(X_tr), MAX_TRAIN, replace=False)
                X_tr = X_tr[idx]; y_tr = y_tr[idx]
            clf = LinearSVC(
                max_iter=2000, C=0.1, dual=True)
            clf.fit(X_tr, y_tr)
            a = float((clf.predict(X_val) == y_val).mean())
            accs.append(a)
            print(f"    Fold {fold+1}/{args.lp_n_folds}: {a*100:.1f}%")

        lp_acc = float(np.mean(accs))
        lp_std = float(np.std(accs))
        print(f"\n  Linear Probe Accuracy: {lp_acc*100:.2f}% ± {lp_std*100:.2f}%")
        print()
        print("  Interpretation:")
        print("    If features are random: accuracy ≈ random baseline above.")
        print("    pgNCE and Pool+pgNCE should both score well above random.")
        print("    Pool+pgNCE > pgNCE accuracy confirms PCA observation.")
        print()

    exp_a['linear_probe_acc'] = lp_acc
    exp_a['linear_probe_std'] = lp_std


# ============================================================================
#  EXPERIMENT B — SCENE-LEVEL PROJECTION HEAD DISCRIMINATION
# ============================================================================
print("=" * 68)
print("EXPERIMENT B  —  Scene-Level Projection Head Discrimination")
print("=" * 68)
print()
print("  Compares Fisher Ratio on three scene-level representations.")
print("  mu_s [512] is the same measurement as architecture Exp1.")
print("  z_layout_proj [128] and pool_hidden [1024] are the InfoNCE head outputs.")
print("  Higher ratio on head output vs mu_s = head learned extra discrimination.")
print()

cats_d = {sid: data['category'] for sid, data in encoded.items()}

mu_s_vecs = {sid: d['mu_s']         for sid, d in encoded.items() if d.get('mu_s')         is not None}
zl_vecs   = {sid: d['z_layout_proj'] for sid, d in encoded.items() if d.get('z_layout_proj') is not None}
ph_vecs   = {sid: d['pool_hidden']   for sid, d in encoded.items() if d.get('pool_hidden')   is not None}

exp_b = {}
for rep_name, vecs in [
    ("mu_s  [raw latent, 512d]     ", mu_s_vecs),
    ("z_layout_proj [LayNCE, 128d] ", zl_vecs),
    ("pool_hidden   [PoolNCE,1024d]", ph_vecs),
]:
    if not vecs:
        print(f"  {rep_name}: — (head not active)")
        continue
    cats_sub = {sid: cats_d[sid] for sid in vecs}
    ratio, intra, inter, intras = compute_fisher_scenes(vecs, cats_sub)
    if ratio is None:
        print(f"  {rep_name}: — (< 2 categories)")
        continue
    print(f"  {rep_name}:  ratio={ratio:.3f}  intra={intra:.4f}  inter={inter:.4f}  N={len(vecs)}")
    exp_b[rep_name.strip()] = {
        'ratio': ratio, 'mean_intra': intra, 'mean_inter': inter,
        'n_scenes': len(vecs),
        'per_cat_intra': {str(k): float(v) for k, v in intras.items()}
    }

print()
print("  Note: ratio > 1.0 means same-category scenes cluster closer than")
print("  different-category scenes. This is the same criterion as architecture Exp1")
print("  but now measured on the projection head output rather than raw mu_s.")
print()

# ============================================================================
#  EXPERIMENT C — PER-CATEGORY FEATURE QUALITY BREAKDOWN
# ============================================================================
exp_c = {'enabled': RUN_EXP_A}

if RUN_EXP_A and per_cat_intra:
    print("=" * 68)
    print("EXPERIMENT C  —  Per-Category Feature Quality Breakdown")
    print("=" * 68)
    print()
    print("  Mean intra-class cosine distance per ScanNet72 category.")
    print("  Lower intra = tighter cluster in feature space = better.")
    print("  Sorted from tightest (best) to most spread (worst).")
    print("  Categories appearing in < 2 scenes are excluded.")
    print()

    hdr = (f"  {'Category':<26} {'ID':>4} {'Mean intra':>11} "
           f"{'Std':>6} {'Silhouette':>10} {'N_scenes':>9}")
    print(hdr)
    print("  " + "─" * 70)

    cat_summary = {}
    for cat_id in sorted(per_cat_intra.keys()):
        vals = per_cat_intra[cat_id]
        if len(vals) < 2: continue
        sil_vals = per_cat_sil.get(cat_id, [])
        cat_summary[cat_id] = {
            'name':       CATNAMES.get(cat_id, f"cat_{cat_id}"),
            'mean_intra': float(np.mean(vals)),
            'std_intra':  float(np.std(vals)),
            'mean_sil':   float(np.mean(sil_vals)) if sil_vals else None,
            'n_scenes':   len(vals),
        }

    for cat_id, cs in sorted(cat_summary.items(), key=lambda x: x[1]['mean_intra']):
        sil_s = f"{cs['mean_sil']:.3f}" if cs['mean_sil'] is not None else "       n/a"
        print(f"  {cs['name']:<26} {cat_id:>4}  {cs['mean_intra']:>10.4f} "
              f" {cs['std_intra']:>6.4f}  {sil_s:>9}  {cs['n_scenes']:>8}")

    print()
    print("  Expected if InfoNCE is working well:")
    print("    Functionally distinct objects (toilet, bathtub, keyboard, monitor)")
    print("    should have LOW intra (tight clusters).")
    print("    Spatially diffuse categories (wall, floor, ceiling) will always")
    print("    have HIGH intra because they span the full scene in 3D space.")
    print()

    exp_c['per_category'] = {str(k): v for k, v in
                              sorted(cat_summary.items(), key=lambda x: x[1]['mean_intra'])}

# ============================================================================
#  SAVE RESULTS
# ============================================================================
report = {
    'checkpoint':       args.checkpoint,
    'semantic_mode':    args.semantic_mode,
    'flags': {
        'latent_disentangle':       args.latent_disentangle,
        'structured_layout_tokens': args.structured_layout_tokens,
        'zs_layout_infonce_weight': args.zs_layout_infonce_weight,
        'zs_pool_infonce_weight':   args.zs_pool_infonce_weight,
    },
    'n_scenes_encoded': len(encoded),
    'exp_a': exp_a,
    'exp_b': exp_b,
    'exp_c': exp_c,
}

rpath = out_dir / 'semantic_feature_results.json'
with open(rpath, 'w') as f:
    json.dump(report, f, indent=2, default=str)

print("=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print(f"  {Path(args.checkpoint).name}")
if RUN_EXP_A:
    if exp_a.get('mean_fisher_ratio') is not None:
        print(f"  Exp A Fisher Ratio:     {exp_a['mean_fisher_ratio']:.4f}")
    if exp_a.get('mean_silhouette') is not None:
        print(f"  Exp A Silhouette:       {exp_a['mean_silhouette']:.4f}")
    if exp_a.get('linear_probe_acc') is not None:
        print(f"  Exp A Linear Probe:     {exp_a['linear_probe_acc']*100:.2f}%")
for name, res in exp_b.items():
    print(f"  Exp B {name:<32}: ratio={res['ratio']:.3f}")
print()
print(f"  Results → {rpath}")