"""
can3tok_tsne.py
===============
Deterministic t-SNE latent space analysis comparing four JEPA Idea 1 experiments:

  Run E: JEPA Idea 1 only
  Run F: JEPA Idea 1 + Scene Semantic Head
  Run G: JEPA Idea 1 + InfoNCE
  Run H: JEPA Idea 1 + Scene Semantic Head + InfoNCE

WHY THESE FOUR EXPERIMENTS
--------------------------
JEPA Idea 1 (SpatialSemanticHead) forces shape_embed to answer
"what category is at position (x,y,z)?" — spatially resolved supervision.

Scene Semantic Head (SceneSemanticHead) forces shape_embed to answer
"what categories exist globally in this scene?" — coarser but cheaper.

InfoNCE on the decoder hidden state pushes Gaussians of the same ScanNet72
category to cluster in the per-Gaussian feature space, with gradients
flowing back through the decoder into mu.

These experiments test:
  E vs A (baseline, add separately if desired): does JEPA Idea 1 alone help?
  F vs E: does adding global semantic supervision on top of spatial help?
  G vs E: does InfoNCE (mu path) and JEPA Idea 1 (shape_embed path) synergise?
  H vs G: marginal value of SceneSemanticHead when both JEPA+InfoNCE are present?

EXPERIMENT A — Inter-category separation
  Encode scenes from different room types. t-SNE coloured by category.
  Metric: Silhouette score (higher = better)

EXPERIMENT B — Intra-scene chunk consistency
  All spatial chunks of one scene should cluster tightly in shape_embed
  because the label distribution is translation-invariant.
  Metric: Mean pairwise distance among chunks (lower = better)

WHAT WE VISUALISE: shape_embed [B, 384] (LATENT_MODE='shape_embed')
  This is the exact token that JEPA Idea 1 and SceneSemanticHead directly
  supervise. Using this isolates the effect of our contributions.
  Set LATENT_MODE='full' to use mu [B, 16384] instead.
"""

# ============================================================================
# USER CONFIGURATION
# ============================================================================

SCENES = {
    'apartment':    ['0207_840167', '0221_840242', '0222_840246', '0223_840262'],
    'coffee_shop':  ['0218_840182', '0219_840183', '0220_840185'],
    'gym':          ['0206_840163'],
    'concert_hall': ['0210_840153'],
    'museum':       ['0209_840159'],
}

CHUNK_SCENE_ID = '0210_840153'

# ── Four JEPA Idea 1 ablation checkpoints ────────────────────────────────────
MODEL_CHECKPOINTS = {
    'Run E\nJEPA Idea 1 only': (
        '/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/'
        'RGB_job_20604720_none_colorresidual_jepa1/best_model.pth'
    ),
    'Run F\nJEPA + SceneSemantic': (
        '/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/'
        'RGB_job_20604419_none_colorresidual_scenesemantic_jepa1/best_model.pth'
    ),
    'Run G\nJEPA + InfoNCE': (
        '/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/'
        'RGB_job_20604828_hidden_colorresidual_jepa1_beta0.3/best_model.pth'
    ),
    'Run H\nJEPA + SceneSemantic + InfoNCE': (
        '/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/'
        'RGB_job_20604904_hidden_colorresidual_scenesemantic_jepa1_beta0.3/best_model.pth'
    ),
}

DATASET_ROOT = '/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs'
SPLIT        = 'train_grid1.0cm_chunk8x8_stride6x6'

# 'shape_embed' → [384]   directly supervised by JEPA Idea 1 / SceneSemanticHead
# 'full'        → [16384] all post-KL tokens — matches the before-JEPA analysis
LATENT_MODE = 'full'

CACHE_DIR  = 'tsne_cache'
TSNE_SEED  = 42
TSNE_N_ITER = 2000
RESOL       = 200

# ============================================================================
# CATEGORY COLOURS
# ============================================================================

CATEGORY_COLORS = {
    'apartment':         '#E63946',
    'coffee_shop':       '#F4A261',
    'convenience_store': '#2A9D8F',
    'club':              '#457B9D',
    'spa_pool':          '#8338EC',
    'cinema':            '#06D6A0',
    'concert_hall':      '#FFB703',
    'go_kart':           '#FB5607',
    'library':           '#3A86FF',
    'gym':               '#FF006E',
    'office':            '#8D99AE',
    'hotel':             '#EF8D2F',
    'museum':            '#56CBF9',
    'salon':             '#C77DFF',
    'lobby':             '#4CC9F0',
    'washroom':          '#B5E48C',
    'restaurant':        '#7B2D8B',
    'wedding_hall':      '#F9C74F',
    'futuristic_pod':    '#F72585',
}
DEFAULT_COLOR    = '#AAAAAA'
BACKGROUND_COLOR = '#DDDDDD'

# ============================================================================
# IMPORTS
# ============================================================================

import argparse
import hashlib
import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

np.random.seed(TSNE_SEED)
torch.manual_seed(TSNE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(TSNE_SEED)

sys.path.insert(0, os.path.dirname(__file__))
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
from gs_dataset_scenesplat import gs_dataset


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(ckpt_path: str, device: torch.device):
    """
    Load a Can3Tok checkpoint, reading all architecture flags from the
    checkpoint metadata so the model exactly matches what was saved.

    Flags that affect model architecture (must match the state dict):
      semantic_mode        — which InfoNCE head is built
      color_residual       — whether MeanColorHead is built
      scene_semantic_head  — whether SceneSemanticHead is built
      jepa_idea1           — whether SpatialSemanticHead is built  ← NEW

    Failing to set any of these causes a state_dict key mismatch.
    """
    print(f"  Loading: {Path(ckpt_path).parent.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    flags = {
        'semantic_mode':       ckpt.get('semantic_mode',       'none'),
        'color_residual':      ckpt.get('color_residual',      False),
        'scene_semantic_head': ckpt.get('scene_semantic_head', False),
        'jepa_idea1':          ckpt.get('jepa_idea1',          False),   # ← NEW
        'scale_norm_mode':     ckpt.get('scale_norm_mode',     'linear'),
        'use_canonical_norm':  ckpt.get('use_canonical_norm',  True),
    }

    config_path  = './model/configs/aligned_shape_latents/shapevae-256.yaml'
    model_config = get_config_from_file(config_path).model
    p = model_config.params.shape_module_cfg.params

    p.semantic_mode       = flags['semantic_mode']
    p.color_residual      = flags['color_residual']
    p.scene_semantic_head = flags['scene_semantic_head']
    p.jepa_idea1          = flags['jepa_idea1']            # ← NEW

    model = instantiate_from_config(model_config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    epoch  = ckpt.get('epoch', '?')
    val_l2 = ckpt.get('val_l2_error', ckpt.get('final_val_l2', '?'))
    print(f"    semantic_mode={flags['semantic_mode']}  "
          f"color_residual={flags['color_residual']}  "
          f"scene_semantic_head={flags['scene_semantic_head']}  "
          f"jepa_idea1={flags['jepa_idea1']}  "
          f"epoch={epoch}  val_L2={val_l2}")
    return model, flags


# ============================================================================
# DATASET / ENCODING
# ============================================================================

def find_chunk_dirs(scene_id: str, split_dir: Path) -> list:
    return sorted([d for d in split_dir.iterdir()
                   if d.is_dir() and d.name.startswith(scene_id)])


def load_features_from_dir(scene_dir: Path, flags: dict) -> np.ndarray:
    if not scene_dir.is_dir():
        return None
    try:
        ds = gs_dataset(
            root             = str(scene_dir.parent),
            resol            = RESOL,
            random_permute   = False,
            train            = False,
            sampling_method  = 'opacity',
            max_scenes       = None,
            normalize        = flags['use_canonical_norm'],
            normalize_colors = True,
            target_radius    = 10.0,
            scale_norm_mode  = flags['scale_norm_mode'],
            color_residual   = flags['color_residual'],
            # jepa_idea1 not needed for inference — we only read features/mean_color
        )
        scene_names = [os.path.basename(d) for d in ds.scene_dirs]
        target_name = scene_dir.name
        if target_name in scene_names:
            idx = scene_names.index(target_name)
        else:
            matches = [n for n in scene_names if n.startswith(target_name)]
            if not matches:
                return None
            idx = scene_names.index(matches[0])
        return ds[idx]['features']
    except Exception as e:
        print(f"    [ERROR] Loading {scene_dir.name}: {e}")
        return None


@torch.no_grad()
def encode_features(model, features: np.ndarray, device: torch.device,
                    latent_mode: str = 'shape_embed') -> np.ndarray:
    """
    Encode a [40000, 18] feature array → latent vector.

    latent_mode='shape_embed' → shape_embed [384]
      The exact token supervised by JEPA Idea 1 and SceneSemanticHead.
      Best for measuring the effect of our contributions.

    latent_mode='full' → mu [16384]
      All post-KL tokens. Captures per-region geometry as well.

    Uses model.shape_model.encode() directly — no decoder pass needed.
    sample_posterior=False → uses mu (mode), not random z, for determinism.
    """
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    shape_embed, mu, log_var, z, posterior = model.shape_model.encode(
        x, x, sample_posterior=False)
    if latent_mode == 'shape_embed':
        return shape_embed[0].detach().cpu().numpy()   # [384]
    else:
        return mu[0].detach().cpu().numpy()            # [16384]


# ============================================================================
# CACHING
# ============================================================================

def cache_key(ckpt_path: str, scene_dir_name: str, latent_mode: str) -> str:
    ckpt_hash = hashlib.md5(ckpt_path.encode()).hexdigest()[:8]
    return f"{ckpt_hash}_{scene_dir_name}_{latent_mode}.npy"


def load_or_encode(ckpt_path, scene_dir, model, flags, device,
                   latent_mode, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    fname = os.path.join(CACHE_DIR,
                         cache_key(ckpt_path, scene_dir.name, latent_mode))
    if use_cache and os.path.exists(fname):
        return np.load(fname)
    features = load_features_from_dir(scene_dir, flags)
    if features is None:
        return None
    latent = encode_features(model, features, device, latent_mode)
    np.save(fname, latent)
    return latent


# ============================================================================
# t-SNE
# ============================================================================

def run_tsne(latents: np.ndarray, seed: int = TSNE_SEED,
             n_iter: int = TSNE_N_ITER) -> np.ndarray:
    N          = len(latents)
    perplexity = float(max(2, min(15, N // 3)))
    print(f"  t-SNE: N={N}, perplexity={perplexity:.0f}, "
          f"n_iter={n_iter}, seed={seed}")
    return TSNE(
        n_components=2, perplexity=perplexity, max_iter=n_iter,
        init='pca', random_state=seed, learning_rate='auto',
    ).fit_transform(latents)


# ============================================================================
# METRICS
# ============================================================================

def silhouette(embedding: np.ndarray, labels: list) -> float:
    unique = list(set(labels))
    if len(unique) < 2 or len(labels) < 4:
        return float('nan')
    y = np.array([unique.index(l) for l in labels])
    try:
        return float(silhouette_score(embedding, y))
    except Exception:
        return float('nan')


def mean_pairwise_dist(points: np.ndarray) -> float:
    if len(points) < 2:
        return float('nan')
    dists = [np.linalg.norm(points[i] - points[j])
             for i in range(len(points))
             for j in range(i + 1, len(points))]
    return float(np.mean(dists))


# ============================================================================
# EXPERIMENT A — Inter-category separation
# ============================================================================

def run_experiment_a(split_dir: Path, models_info: dict, use_cache: bool):
    print(f"\n{'='*65}")
    print(f"EXPERIMENT A — Inter-category separation")
    print(f"  Metric: Silhouette score (higher = better cluster separation)")
    print(f"  Expected: spatial supervision → cleaner room-type clusters")
    print(f"{'='*65}")

    scene_list = []
    for category, scene_ids in SCENES.items():
        for sid in scene_ids:
            chunks = find_chunk_dirs(sid, split_dir)
            if not chunks:
                print(f"  [WARN] No dirs found for {sid}")
                continue
            scene_list.append((sid, category, chunks[0]))

    print(f"\n  Scenes found: {len(scene_list)}")
    for sid, cat, d in scene_list:
        print(f"    {cat:<20}  {sid}  →  {d.name}")

    results = {}
    for model_name, (ckpt_path, model, flags, device) in models_info.items():
        print(f"\n  Encoding: {model_name.replace(chr(10), ' ')}")
        latents, labels, scene_ids = [], [], []
        for sid, category, scene_dir in scene_list:
            latent = load_or_encode(ckpt_path, scene_dir, model, flags,
                                    device, LATENT_MODE, use_cache)
            if latent is None:
                print(f"    [SKIP] {sid}")
                continue
            latents.append(latent)
            labels.append(category)
            scene_ids.append(sid)
            print(f"    ✓  {category:<20}  {sid}")

        if len(latents) < 3:
            print(f"  [WARN] Too few scenes ({len(latents)}) — skipping t-SNE")
            results[model_name] = None
            continue

        embedding = run_tsne(np.stack(latents))
        sil       = silhouette(embedding, labels)
        print(f"  Silhouette: {sil:.4f}")

        results[model_name] = {
            'embedding':  embedding,
            'labels':     labels,
            'scene_ids':  scene_ids,
            'silhouette': sil,
        }
    return results


# ============================================================================
# EXPERIMENT B — Intra-scene chunk consistency
# ============================================================================

def run_experiment_b(split_dir: Path, models_info: dict, use_cache: bool,
                     n_background: int = 6):
    print(f"\n{'='*65}")
    print(f"EXPERIMENT B — Intra-scene chunk consistency")
    print(f"  Scene: {CHUNK_SCENE_ID}")
    print(f"  Metric: mean pairwise distance of chunk embeddings (lower = better)")
    print(f"  Why: label distribution is translation-invariant → all chunks")
    print(f"       of the same room should map to nearby shape_embed vectors")
    print(f"{'='*65}")

    chunk_dirs = find_chunk_dirs(CHUNK_SCENE_ID, split_dir)
    if not chunk_dirs:
        print(f"  [ERROR] No chunks found for {CHUNK_SCENE_ID}")
        return {}

    print(f"\n  Chunks ({len(chunk_dirs)}):")
    for d in chunk_dirs:
        print(f"    {d.name}")

    target_category = next(
        (cat for cat, ids in SCENES.items() if CHUNK_SCENE_ID in ids), 'unknown')
    background_list = []
    for cat, scene_ids in SCENES.items():
        if cat == target_category:
            continue
        for sid in scene_ids:
            chunks = find_chunk_dirs(sid, split_dir)
            if chunks:
                background_list.append((sid, cat, chunks[0]))
                break
        if len(background_list) >= n_background:
            break

    results = {}
    for model_name, (ckpt_path, model, flags, device) in models_info.items():
        print(f"\n  Encoding: {model_name.replace(chr(10), ' ')}")
        latents, labels, colors, markers = [], [], [], []

        chunk_latents = []
        for chunk_dir in chunk_dirs:
            latent = load_or_encode(ckpt_path, chunk_dir, model, flags,
                                    device, LATENT_MODE, use_cache)
            if latent is None:
                print(f"    [SKIP chunk] {chunk_dir.name}")
                continue
            chunk_latents.append(latent)
            latents.append(latent)
            labels.append(f'chunk_{chunk_dir.name[-5:]}')
            colors.append('#E63946')
            markers.append('chunk')
            print(f"    ✓ chunk  {chunk_dir.name}")

        if len(chunk_latents) < 2:
            results[model_name] = None
            continue

        compactness = mean_pairwise_dist(np.stack(chunk_latents))

        for sid, cat, scene_dir in background_list:
            latent = load_or_encode(ckpt_path, scene_dir, model, flags,
                                    device, LATENT_MODE, use_cache)
            if latent is None:
                continue
            latents.append(latent)
            labels.append(cat)
            colors.append(CATEGORY_COLORS.get(cat, BACKGROUND_COLOR))
            markers.append('background')
            print(f"    ✓ bg     {cat:<20}  {sid}")

        if len(latents) < 3:
            results[model_name] = None
            continue

        embedding = run_tsne(np.stack(latents))
        print(f"  Compactness: {compactness:.4f}  (lower = tighter chunks)")

        results[model_name] = {
            'embedding':   embedding,
            'labels':      labels,
            'colors':      colors,
            'markers':     markers,
            'n_chunks':    len(chunk_latents),
            'compactness': compactness,
            'chunk_scene': CHUNK_SCENE_ID,
        }
    return results


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

# Short labels for subplot titles (strip the \n run name formatting)
RUN_SHORT = {
    'Run E\nJEPA Idea 1 only':              'E: JEPA only',
    'Run F\nJEPA + SceneSemantic':          'F: JEPA + SceneSem',
    'Run G\nJEPA + InfoNCE':                'G: JEPA + InfoNCE',
    'Run H\nJEPA + SceneSemantic + InfoNCE':'H: JEPA + SceneSem + InfoNCE',
}


def _style_ax(ax):
    ax.set_facecolor('#F7F7F7')
    ax.grid(True, linewidth=0.4, alpha=0.6, color='white')
    ax.tick_params(labelsize=7)
    ax.set_xlabel('t-SNE dim 1', fontsize=8)
    ax.set_ylabel('t-SNE dim 2', fontsize=8)


def plot_exp_a_panel(ax, result, title):
    if result is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='#999999')
        ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
        return

    _style_ax(ax)
    embedding = result['embedding']
    labels    = result['labels']
    scene_ids = result['scene_ids']
    sil       = result['silhouette']
    label_arr = np.array(labels)

    for cat in sorted(set(labels)):
        mask  = label_arr == cat
        pts   = embedding[mask]
        color = CATEGORY_COLORS.get(cat, DEFAULT_COLOR)
        sids  = [s for s, l in zip(scene_ids, labels) if l == cat]

        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=110, alpha=0.88,
                   edgecolors='white', linewidths=0.8, zorder=3,
                   label=f"{cat.replace('_',' ')} (n={mask.sum()})")

        for pt, sid in zip(pts, sids):
            ax.annotate(sid[-7:], xy=pt, fontsize=5.5, color='#444444',
                        ha='center', xytext=(0, -8),
                        textcoords='offset points')

        centroid = pts.mean(axis=0)
        ax.scatter(centroid[0], centroid[1], c=color, marker='*', s=260,
                   zorder=6, edgecolors='#333333', linewidths=0.8, alpha=1.0)

    ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
    sil_text = f"Silhouette: {sil:.3f}" if not np.isnan(sil) else "Silhouette: N/A"
    ax.text(0.03, 0.97, sil_text, transform=ax.transAxes,
            fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.9, edgecolor='#CCCCCC'))


def plot_exp_b_panel(ax, result, title):
    if result is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='#999999')
        ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
        return

    _style_ax(ax)
    embedding   = result['embedding']
    labels      = result['labels']
    colors      = result['colors']
    markers     = result['markers']
    compactness = result['compactness']
    chunk_scene = result['chunk_scene']
    markers_arr = np.array(markers)

    for i in range(len(embedding)):
        if markers[i] == 'background':
            ax.scatter(embedding[i, 0], embedding[i, 1],
                       c=BACKGROUND_COLOR, s=65, alpha=0.5,
                       edgecolors='white', linewidths=0.4, zorder=2)
            ax.annotate(labels[i].replace('_', '\n'), xy=embedding[i],
                        fontsize=5.5, color='#777777', ha='center',
                        xytext=(0, 6), textcoords='offset points')

    chunk_pts = embedding[markers_arr == 'chunk']
    for i, pt in enumerate(chunk_pts):
        ax.scatter(pt[0], pt[1], c='#E63946', s=150, alpha=0.92,
                   edgecolors='white', linewidths=1.2, zorder=5)
        ax.annotate(f'c{i+1}', xy=pt, fontsize=6, color='#E63946',
                    fontweight='bold', ha='center',
                    xytext=(0, 7), textcoords='offset points')

    if len(chunk_pts) >= 3:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(chunk_pts)
            for simplex in hull.simplices:
                ax.plot(chunk_pts[simplex, 0], chunk_pts[simplex, 1],
                        color='#E63946', alpha=0.4, linewidth=1.5,
                        zorder=4, linestyle='--')
        except Exception:
            pass

    ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
    cpt_text = (f"Spread: {compactness:.3f}  ↓ better\n"
                f"n={len(chunk_pts)} chunks")
    ax.text(0.03, 0.97, cpt_text, transform=ax.transAxes,
            fontsize=7.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.9, edgecolor='#CCCCCC'))

    ax.legend(handles=[
        mpatches.Patch(color='#E63946', label=f'{chunk_scene} chunks'),
        mpatches.Patch(color=BACKGROUND_COLOR, label='Other scenes'),
    ], fontsize=7, loc='lower right', framealpha=0.88, edgecolor='#CCCCCC')


# ============================================================================
# FIGURE ASSEMBLY  (2 rows × 4 cols)
# ============================================================================

def make_figure(exp_a_results, exp_b_results, output_dir, skip_a, skip_b):
    model_names = list(MODEL_CHECKPOINTS.keys())
    n_models    = len(model_names)
    rows        = (0 if skip_a else 1) + (0 if skip_b else 1)
    if rows == 0:
        print("[WARN] Nothing to plot.")
        return

    fig, axes = plt.subplots(
        rows, n_models,
        figsize=(5.8 * n_models, 6.0 * rows),
        squeeze=False,
    )
    fig.patch.set_facecolor('white')
    row = 0

    if not skip_a:
        for col, name in enumerate(model_names):
            short = RUN_SHORT.get(name, name.replace('\n', ' — '))
            plot_exp_a_panel(axes[row][col],
                             exp_a_results.get(name),
                             f"Exp A: Category Clustering\n{short}")

        # Shared legend on leftmost panel
        r0 = next((v for v in exp_a_results.values() if v), None)
        if r0:
            cats = sorted(set(r0['labels']))
            axes[row][0].legend(
                handles=[mpatches.Patch(
                    color=CATEGORY_COLORS.get(c, DEFAULT_COLOR),
                    label=c.replace('_', ' ') + '  ★=centroid')
                    for c in cats],
                fontsize=7, loc='upper right',
                title='Room type', title_fontsize=7,
                framealpha=0.88, edgecolor='#CCCCCC',
            )
        row += 1

    if not skip_b:
        for col, name in enumerate(model_names):
            short = RUN_SHORT.get(name, name.replace('\n', ' — '))
            plot_exp_b_panel(axes[row][col],
                             exp_b_results.get(name),
                             f"Exp B: Chunk Consistency\n{short}")
        row += 1

    # ── Global title with silhouette comparison ───────────────────────────────
    if not skip_a:
        sil_parts = []
        for name in model_names:
            short = RUN_SHORT.get(name, name.split('\n')[-1])
            r     = exp_a_results.get(name)
            sil   = r['silhouette'] if r else float('nan')
            sil_parts.append(f"{short}: {sil:.3f}" if not np.isnan(sil)
                              else f"{short}: N/A")
        fig.suptitle(
            f"Can3Tok JEPA Idea 1 Ablation — latent={LATENT_MODE}\n"
            f"Silhouette  |  " + "   |   ".join(sil_parts),
            fontsize=10, fontweight='bold', y=1.01,
        )
    else:
        fig.suptitle(
            f"Can3Tok JEPA Idea 1 Ablation — latent={LATENT_MODE}",
            fontsize=10, fontweight='bold', y=1.01,
        )

    fig.tight_layout(h_pad=3.0, w_pad=2.0)
    os.makedirs(output_dir, exist_ok=True)
    png = os.path.join(output_dir, 'can3tok_jepa_tsne.png')
    pdf = os.path.join(output_dir, 'can3tok_jepa_tsne.pdf')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {png}")
    print(f"  Saved: {pdf}")
    return png


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def print_summary(exp_a_results, exp_b_results):
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY — JEPA Idea 1 Ablation")
    print(f"{'='*70}")

    print(f"\nExp A — Silhouette score  (higher = better inter-category separation):")
    print(f"  {'Model':<45} {'Silhouette':>10}")
    print(f"  {'─'*57}")
    for name, r in exp_a_results.items():
        sil   = r['silhouette'] if r else float('nan')
        short = RUN_SHORT.get(name, name.replace('\n', ' '))
        print(f"  {short:<45} {sil:>10.4f}")

    print(f"\nExp B — Chunk compactness  (lower = tighter = better):")
    print(f"  {'Model':<45} {'Compactness':>12}  {'N chunks':>8}")
    print(f"  {'─'*68}")
    for name, r in exp_b_results.items():
        short = RUN_SHORT.get(name, name.replace('\n', ' '))
        if r:
            print(f"  {short:<45} {r['compactness']:>12.4f}  {r['n_chunks']:>8}")
        else:
            print(f"  {short:<45} {'N/A':>12}")

    print(f"\nKey comparisons:")
    runs = list(exp_a_results.keys())
    e_sil = (exp_a_results.get(runs[0]) or {}).get('silhouette', float('nan'))
    f_sil = (exp_a_results.get(runs[1]) or {}).get('silhouette', float('nan'))
    g_sil = (exp_a_results.get(runs[2]) or {}).get('silhouette', float('nan'))
    h_sil = (exp_a_results.get(runs[3]) or {}).get('silhouette', float('nan'))
    print(f"  F vs E  (+SceneSemantic):        Δsilhouette = {f_sil - e_sil:+.4f}")
    print(f"  G vs E  (+InfoNCE):              Δsilhouette = {g_sil - e_sil:+.4f}")
    print(f"  H vs G  (+SceneSemantic+InfoNCE):Δsilhouette = {h_sil - g_sil:+.4f}")
    print(f"  H vs F  (+InfoNCE):              Δsilhouette = {h_sil - f_sil:+.4f}")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    global LATENT_MODE
    ap = argparse.ArgumentParser(
        description='Can3Tok t-SNE — JEPA Idea 1 ablation')
    ap.add_argument('--output_dir',  default='tsne_results')
    ap.add_argument('--no_cache',    action='store_true')
    ap.add_argument('--skip_exp_a',  action='store_true')
    ap.add_argument('--skip_exp_b',  action='store_true')
    ap.add_argument('--latent_mode', default=LATENT_MODE,
                    choices=['shape_embed', 'full'])
    args = ap.parse_args()

    LATENT_MODE = args.latent_mode
    use_cache   = not args.no_cache
    device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    split_dir   = Path(DATASET_ROOT) / SPLIT

    print(f"\n{'='*65}")
    print(f"Can3Tok t-SNE — JEPA Idea 1 Ablation (4 runs)")
    print(f"{'='*65}")
    print(f"  Device:      {device}")
    print(f"  Split dir:   {split_dir}")
    print(f"  Latent mode: {LATENT_MODE}  "
          f"({'shape_embed [384] — directly supervised' if LATENT_MODE=='shape_embed' else 'full mu [16384]'})")
    print(f"  Cache:       {CACHE_DIR}  (use_cache={use_cache})")
    print(f"  Output:      {args.output_dir}")

    if not split_dir.exists():
        print(f"\n[ERROR] Split directory not found: {split_dir}")
        sys.exit(1)

    # ── Load models ───────────────────────────────────────────────────────────
    models_info = {}
    for name, ckpt_path in MODEL_CHECKPOINTS.items():
        if not Path(ckpt_path).exists():
            print(f"\n[WARN] Checkpoint not found: {ckpt_path}")
            print(f"       Skipping: {name.replace(chr(10), ' ')}")
            continue
        print(f"\nLoading: {name.replace(chr(10), ' ')}")
        model, flags = load_model(ckpt_path, device)
        models_info[name] = (ckpt_path, model, flags, device)

    if not models_info:
        print("\n[ERROR] No valid checkpoints found.")
        sys.exit(1)

    print(f"\nLoaded {len(models_info)}/4 models.")

    # ── Run experiments ───────────────────────────────────────────────────────
    exp_a_results = {}
    exp_b_results = {}

    if not args.skip_exp_a:
        exp_a_results = run_experiment_a(split_dir, models_info, use_cache)

    if not args.skip_exp_b:
        exp_b_results = run_experiment_b(split_dir, models_info, use_cache)

    make_figure(exp_a_results, exp_b_results,
                args.output_dir, args.skip_exp_a, args.skip_exp_b)

    print_summary(exp_a_results, exp_b_results)

    print(f"Collect outputs:")
    print(f"  scp user@snellius:$(pwd)/{args.output_dir}/can3tok_jepa_tsne.png .")
    print(f"  scp user@snellius:$(pwd)/{args.output_dir}/can3tok_jepa_tsne.pdf .")


if __name__ == '__main__':
    main()