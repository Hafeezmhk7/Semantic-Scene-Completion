"""
can3tok_tsne.py
===============
Deterministic t-SNE latent space analysis comparing three Can3Tok models:

  Run A: Color residual only
  Run B: Color residual + InfoNCE hidden
  Run C/D: Color residual + InfoNCE + Label Distribution (Move 1)

WHY THESE TWO EXPERIMENTS
--------------------------
InfoNCE (Run B) enforces: Gaussians of the same ScanNet72 category (floor,
wall, bed...) cluster in the per-Gaussian decoder feature space [B, 40k, 32].
This signal flows back through the decoder into mu. Scene-level clustering
emerges *indirectly*.

Label Distribution (Run C/D) enforces: shape_embed [B, 32] must directly
predict the fraction of each ScanNet72 category in the scene. Two apartments
share similar label distributions (floor/wall/ceiling/furniture). A gym and
a coffee shop are very different. So shape_embed becomes a *direct* room
fingerprint encoder. Same room type → similar shape_embed → nearby in t-SNE.

EXPERIMENT A — Inter-category separation
  Encode scenes from 4-5 semantically different room types.
  t-SNE coloured by category. Same room type should cluster together.
  Expected: Run A (random scatter) < Run B (weak clusters) < Run C/D (clear clusters)
  Metric: Silhouette score (higher = better separation)

EXPERIMENT B — Intra-scene chunk consistency
  The dataset splits each room into overlapping 8×8m chunks.
  Scene 0210_840153 (concert hall) has chunks r0_c0, r0_c1, r1_c0, etc.
  Key insight: the label distribution is translation-invariant — every
  spatial chunk of the same room has roughly the same floor/wall/ceiling
  fractions. So Run C/D's shape_embed should be nearly identical across chunks.
  Expected: Run A (chunks scattered) < Run B (slight clustering) < Run C/D (tight cluster)
  Metric: Mean pairwise distance among chunks (lower = better)

WHAT WE VISUALISE: shape_embed mu only [B, 32]
  This is the global scene token. Both MeanColorHead and SceneSemanticHead
  directly supervise it. Using only shape_embed (not the full 16k latent)
  isolates the exact part of the representation that our contributions target.
  Set LATENT_MODE = 'full' to use the entire mu [B, 511*32] instead.

DETERMINISM GUARANTEE
  - sampling_method='opacity' → fixed argsort, same 40k Gaussians every run
  - torch/numpy seeds fixed at startup
  - TSNE(init='pca', random_state=42) → reproducible embedding
  - Latents cached to {CACHE_DIR}/ → re-runs skip encoding entirely

USAGE
-----
  # 1. Edit SCENES, MODEL_CHECKPOINTS, CHUNK_SCENE_ID below
  # 2. Run:
  python can3tok_tsne.py

  # Re-run without re-encoding (uses cached latents):
  python can3tok_tsne.py   # cache auto-detected

  # Force re-encode (delete cache or run):
  python can3tok_tsne.py --no_cache

  # Only run one experiment:
  python can3tok_tsne.py --skip_exp_b
  python can3tok_tsne.py --skip_exp_a
"""

# ============================================================================
# USER CONFIGURATION — Edit these dicts, then run the script
# ============================================================================

# Scenes to use for Experiment A (inter-category separation).
# Each key is the display name. Values are scene_ids to look for in the dataset.
# Include 3+ scenes per category where possible — 1-2 also works as singletons.
SCENES = {
    'apartment':    ['0207_840167', '0221_840242', '0222_840246', '0223_840262'],
    'coffee_shop':  ['0218_840182', '0219_840183', '0220_840185'],
    'gym':          ['0206_840163'],
    'concert_hall': ['0210_840153'],
    'museum':       ['0209_840159'],
}

# Scene to use for Experiment B (intra-scene chunk consistency).
# Must have multiple chunk directories in the dataset split.
# Supervisor-provided concert hall is a good choice: typically 4-9 chunks.
CHUNK_SCENE_ID = '0210_840153'

# The three models to compare. Keys become subplot titles.
# Fill in your actual checkpoint paths.
MODEL_CHECKPOINTS = {
    'Run A\nColor Residual':              '/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/RGB_job_20207686_none_colorresidual/best_model.pth',
    'Run B\nColor Residual + InfoNCE':    '/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/RGB_job_20231337_hidden_colorresidual_beta0.3/best_model.pth',
    'Run C\nColor Residual + + InfoNCE + LabelDist':  '/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/RGB_job_20242092_hidden_colorresidual_scenesemantic_beta0.3/best_model.pth',
}

# Dataset location
DATASET_ROOT = '/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs'
SPLIT        = 'train_grid1.0cm_chunk8x8_stride6x6'

# Latent to use for t-SNE:
#   'shape_embed' → mu[:, 0, :] = [32]   (directly supervised by our contributions)
#   'full'        → mu.reshape(-1) = [16384]  (all tokens)
LATENT_MODE = 'full'

# Cache directory — latents saved here so re-runs are instantaneous
CACHE_DIR = 'tsne_cache'

# t-SNE settings — fixed for determinism
TSNE_SEED   = 42
TSNE_N_ITER = 2000
RESOL       = 200     # voxel grid resolution for dataset loading

# ============================================================================
# CATEGORY COLOURS (one per room type)
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
DEFAULT_COLOR = '#AAAAAA'
CHUNK_COLORS  = [
    '#E63946', '#2196F3', '#4CAF50', '#FF9800',
    '#9C27B0', '#00BCD4', '#FF5722', '#8BC34A',
]
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

# Fix all seeds right here — before any torch/numpy usage
np.random.seed(TSNE_SEED)
torch.manual_seed(TSNE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(TSNE_SEED)

# Local Can3Tok imports
sys.path.insert(0, os.path.dirname(__file__))
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
from gs_dataset_scenesplat import gs_dataset


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(ckpt_path: str, device: torch.device):
    """
    Load a Can3Tok checkpoint. Reads all three architecture flags from the
    checkpoint and sets them on the config before instantiation so the model
    exactly matches what was saved.

    The three flags that affect model architecture (must match the state dict):
      semantic_mode       -> controls which InfoNCE head is built
      color_residual      -> controls whether MeanColorHead is built
      scene_semantic_head -> controls whether SceneSemanticHead is built

    Failing to set any of these causes a state_dict mismatch: the checkpoint
    has the head weights but the freshly built model does not have those modules.
    """
    print(f"  Loading: {Path(ckpt_path).parent.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    flags = {
        'semantic_mode':       ckpt.get('semantic_mode',       'none'),
        'color_residual':      ckpt.get('color_residual',      False),
        'scene_semantic_head': ckpt.get('scene_semantic_head', False),
        'scale_norm_mode':     ckpt.get('scale_norm_mode',     'linear'),
        'use_canonical_norm':  ckpt.get('use_canonical_norm',  True),
    }

    config_path  = './model/configs/aligned_shape_latents/shapevae-256.yaml'
    model_config = get_config_from_file(config_path).model
    p = model_config.params.shape_module_cfg.params

    # Set ALL three architecture flags — only semantic_mode was set before.
    p.semantic_mode       = flags['semantic_mode']
    p.color_residual      = flags['color_residual']
    p.scene_semantic_head = flags['scene_semantic_head']

    model = instantiate_from_config(model_config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    epoch  = ckpt.get('epoch', '?')
    val_l2 = ckpt.get('val_l2_error', ckpt.get('final_val_l2', '?'))
    print(f"    semantic_mode={flags['semantic_mode']}  "
          f"color_residual={flags['color_residual']}  "
          f"scene_semantic_head={flags['scene_semantic_head']}  "
          f"epoch={epoch}  val_L2={val_l2}")
    return model, flags



# ============================================================================
# DATASET / ENCODING
# ============================================================================

def find_chunk_dirs(scene_id: str, split_dir: Path) -> list:
    """
    Find all chunk directories that belong to scene_id.
    In train_grid1.0cm_chunk8x8_stride6x6, scene 0210_840153 has chunks:
        0210_840153_r0_c0 / 0210_840153_r0_c1 / 0210_840153_r1_c0 / ...
    """
    return sorted([d for d in split_dir.iterdir()
                   if d.is_dir() and d.name.startswith(scene_id)])


def load_features_from_dir(scene_dir: Path, flags: dict) -> np.ndarray:
    """
    Load a single scene directory as a [40000, 18] feature tensor.

    Uses gs_dataset with:
      - sampling_method='opacity'  → deterministic top-40k (same every run)
      - normalize / scale_norm_mode from the model's training flags

    Returns None if the scene directory cannot be found/loaded.
    """
    if not scene_dir.is_dir():
        return None

    try:
        ds = gs_dataset(
            root             = str(scene_dir.parent),
            resol            = RESOL,
            random_permute   = False,
            train            = False,
            sampling_method  = 'opacity',   # DETERMINISTIC — same 40k every run
            max_scenes       = None,
            normalize        = flags['use_canonical_norm'],
            normalize_colors = True,
            target_radius    = 10.0,
            scale_norm_mode  = flags['scale_norm_mode'],
        )

        # Find this specific directory in the dataset's scene list
        scene_names = [os.path.basename(d) for d in ds.scene_dirs]
        target_name = scene_dir.name

        if target_name in scene_names:
            idx = scene_names.index(target_name)
        else:
            matches = [n for n in scene_names if n.startswith(target_name)]
            if not matches:
                return None
            idx = scene_names.index(matches[0])

        return ds[idx]['features']   # [40000, 18]

    except Exception as e:
        print(f"    [ERROR] Loading {scene_dir.name}: {e}")
        return None


@torch.no_grad()
def encode_features(model, features: np.ndarray, device: torch.device,
                    latent_mode: str = 'shape_embed') -> np.ndarray:
    """
    Encode a [40000, 18] feature array → latent vector.

    The model forward() returns:
      (shape_embed, mu, log_var, z, reconstruction, semantic_features)

    Shapes:
      shape_embed [B, width=384]  — pre-KL global scene token (encoder token 0)
                                    directly supervised by MeanColorHead and
                                    SceneSemanticHead. Best for measuring the
                                    effect of our contributions.
      mu          [B, 16384]      — flat post-KL latent (already through
                                    kl_emb_proj_mean, NOT shaped [B, 512, 32])

    latent_mode='shape_embed' → shape_embed[0] = [384]
      Use this: it is exactly what both Step 1 and Move 1 directly supervise.

    latent_mode='full' → mu[0] = [16384]
      All post-KL latent tokens. Captures per-region geometry as well as
      global scene identity.

    Uses model.encode() directly (not forward()) so no reconstruction pass
    is needed — faster and uses mu (not z) for determinism.
    """
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    # Call encode() directly: returns (shape_embed, mu, log_var, z, posterior)
    # This avoids the full decoder forward pass — much faster for inference.
    # sample_posterior=False → use mu (mode), not a random z sample.
    shape_embed, mu, log_var, z, posterior = model.shape_model.encode(
        x, x, sample_posterior=False
    )
    # shape_embed: [1, width=384]
    # mu:          [1, 16384]   (flat — already through kl_emb_proj_mean)

    if latent_mode == 'shape_embed':
        return shape_embed[0].detach().cpu().numpy()   # [384]
    else:
        return mu[0].detach().cpu().numpy()            # [16384]


# ============================================================================
# CACHING
# ============================================================================

def cache_key(ckpt_path: str, scene_dir_name: str, latent_mode: str) -> str:
    """Build a short deterministic cache filename."""
    ckpt_hash = hashlib.md5(ckpt_path.encode()).hexdigest()[:8]
    return f"{ckpt_hash}_{scene_dir_name}_{latent_mode}.npy"


def load_or_encode(ckpt_path: str, scene_dir: Path, model, flags: dict,
                   device: torch.device, latent_mode: str,
                   use_cache: bool = True) -> np.ndarray:
    """
    Return the latent for a scene. Loads from cache if available,
    otherwise encodes and saves to cache.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    fname = os.path.join(CACHE_DIR, cache_key(ckpt_path, scene_dir.name, latent_mode))

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
    """
    Run t-SNE with fixed seed and PCA init for full reproducibility.
    Perplexity auto-scaled to dataset size (must be < N-1).
    """
    N = len(latents)
    perplexity = float(max(2, min(15, N // 3)))

    print(f"  t-SNE: N={N}, perplexity={perplexity:.0f}, n_iter={n_iter}, seed={seed}")

    return TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        init='pca',        # reproducible (not random)
        random_state=seed,
        learning_rate='auto',
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
    """Mean pairwise Euclidean distance among a set of points."""
    if len(points) < 2:
        return float('nan')
    dists = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dists.append(np.linalg.norm(points[i] - points[j]))
    return float(np.mean(dists))


# ============================================================================
# EXPERIMENT A — Inter-category separation
# ============================================================================

def run_experiment_a(split_dir: Path, models_info: dict, use_cache: bool):
    """
    Encode all scenes from SCENES dict, run t-SNE per model.

    Returns: dict model_name → {embedding, labels, scene_ids, silhouette}
    """
    print(f"\n{'='*65}")
    print(f"EXPERIMENT A — Inter-category separation")
    print(f"  What we measure: do same room types cluster in shape_embed?")
    print(f"  Expected: Run A random < Run B partial < Run C clear clusters")
    print(f"{'='*65}")

    # Build a flat list of (scene_id, category, directory) to encode
    scene_list = []
    for category, scene_ids in SCENES.items():
        for sid in scene_ids:
            chunks = find_chunk_dirs(sid, split_dir)
            if not chunks:
                print(f"  [WARN] No directories found for {sid}")
                continue
            # Use the first chunk directory as the scene representative
            # (deterministic: sorted, first = r0_c0 = most central)
            scene_list.append((sid, category, chunks[0]))

    print(f"\n  Scenes found: {len(scene_list)}")
    for sid, cat, d in scene_list:
        print(f"    {cat:<20}  {sid}  →  {d.name}")

    results = {}

    for model_name, (ckpt_path, model, flags, device) in models_info.items():
        print(f"\n  Encoding for: {model_name.replace(chr(10), ' ')}")

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
            print(f"  [WARN] Too few scenes ({len(latents)}) to run t-SNE")
            results[model_name] = None
            continue

        latents_arr = np.stack(latents)
        embedding   = run_tsne(latents_arr)
        sil         = silhouette(embedding, labels)

        print(f"  Silhouette score: {sil:.4f}  (higher = better separation)")

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
    """
    Encode all spatial chunks of CHUNK_SCENE_ID + a set of background scenes.

    Key question: do chunks of the same room cluster together in shape_embed?

    The label distribution is translation-invariant — every 8x8m chunk of the
    same concert hall has roughly the same floor/wall/ceiling/seat fractions.
    So Run C/D should produce nearly identical shape_embed vectors for all chunks.

    Background scenes (from other categories) are shown as grey dots so you can
    see that the chunk cluster is also separated from other room types.
    """
    print(f"\n{'='*65}")
    print(f"EXPERIMENT B — Intra-scene chunk consistency")
    print(f"  Scene: {CHUNK_SCENE_ID}")
    print(f"  Question: do all spatial chunks of this room cluster tightly?")
    print(f"  Why they should: label distribution is translation-invariant")
    print(f"{'='*65}")

    # Find all chunks of the target scene
    chunk_dirs = find_chunk_dirs(CHUNK_SCENE_ID, split_dir)
    if not chunk_dirs:
        print(f"  [ERROR] No chunks found for {CHUNK_SCENE_ID}")
        return {}

    print(f"\n  Chunks found for {CHUNK_SCENE_ID} ({len(chunk_dirs)} total):")
    for d in chunk_dirs:
        print(f"    {d.name}")

    # Collect background scenes: pick one scene from each other category
    background_list = []
    target_category = next(
        (cat for cat, ids in SCENES.items() if CHUNK_SCENE_ID in ids), 'unknown'
    )
    for cat, scene_ids in SCENES.items():
        if cat == target_category:
            continue
        for sid in scene_ids:
            chunks = find_chunk_dirs(sid, split_dir)
            if chunks:
                background_list.append((sid, cat, chunks[0]))
                break   # one per category is enough
        if len(background_list) >= n_background:
            break

    print(f"\n  Background scenes ({len(background_list)}):")
    for sid, cat, d in background_list:
        print(f"    {cat:<20}  {sid}  →  {d.name}")

    results = {}

    for model_name, (ckpt_path, model, flags, device) in models_info.items():
        print(f"\n  Encoding for: {model_name.replace(chr(10), ' ')}")

        latents, labels, colors, markers = [], [], [], []

        # ── Encode target scene chunks ────────────────────────────────────────
        chunk_latents = []
        for chunk_dir in chunk_dirs:
            latent = load_or_encode(ckpt_path, chunk_dir, model, flags,
                                    device, LATENT_MODE, use_cache)
            if latent is None:
                print(f"    [SKIP chunk] {chunk_dir.name}")
                continue
            chunk_latents.append(latent)
            latents.append(latent)
            labels.append(f'chunk_{chunk_dir.name[-5:]}')   # short label
            colors.append('#E63946')   # all chunks same colour
            markers.append('chunk')
            print(f"    ✓ chunk  {chunk_dir.name}")

        if len(chunk_latents) < 2:
            print(f"  [WARN] Only {len(chunk_latents)} chunks — skipping")
            results[model_name] = None
            continue

        chunk_arr = np.stack(chunk_latents)
        compactness = mean_pairwise_dist(chunk_arr)

        # ── Encode background scenes ──────────────────────────────────────────
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

        latents_arr = np.stack(latents)
        embedding   = run_tsne(latents_arr)

        print(f"  Chunk compactness: {compactness:.4f}  (lower = tighter = better)")

        results[model_name] = {
            'embedding':    embedding,
            'labels':       labels,
            'colors':       colors,
            'markers':      markers,
            'n_chunks':     len(chunk_latents),
            'compactness':  compactness,
            'chunk_scene':  CHUNK_SCENE_ID,
        }

    return results


# ============================================================================
# PLOTTING — Experiment A
# ============================================================================

def _style_ax(ax):
    ax.set_facecolor('#F7F7F7')
    ax.grid(True, linewidth=0.4, alpha=0.6, color='white')
    ax.tick_params(labelsize=7)
    ax.set_xlabel('t-SNE dim 1', fontsize=8)
    ax.set_ylabel('t-SNE dim 2', fontsize=8)


def plot_exp_a_panel(ax, result: dict, title: str):
    if result is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='#999999')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
        return

    _style_ax(ax)
    embedding  = result['embedding']
    labels     = result['labels']
    scene_ids  = result['scene_ids']
    sil        = result['silhouette']
    label_arr  = np.array(labels)

    for cat in sorted(set(labels)):
        mask  = label_arr == cat
        pts   = embedding[mask]
        color = CATEGORY_COLORS.get(cat, DEFAULT_COLOR)
        sids  = [s for s, l in zip(scene_ids, labels) if l == cat]

        ax.scatter(pts[:, 0], pts[:, 1],
                   c=color, s=120, alpha=0.88,
                   edgecolors='white', linewidths=0.8, zorder=3,
                   label=f"{cat.replace('_', ' ')} (n={mask.sum()})")

        # Short scene ID label below each point
        for pt, sid in zip(pts, sids):
            ax.annotate(sid[-7:], xy=pt,
                        fontsize=5.5, color='#444444', ha='center',
                        xytext=(0, -8), textcoords='offset points')

        # Category centroid as a star
        centroid = pts.mean(axis=0)
        ax.scatter(centroid[0], centroid[1],
                   c=color, marker='*', s=280, zorder=6,
                   edgecolors='#333333', linewidths=0.8, alpha=1.0)

    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

    # Silhouette annotation
    sil_text = f"Silhouette: {sil:.3f}" if not np.isnan(sil) else "Silhouette: N/A"
    ax.text(0.03, 0.97, sil_text,
            transform=ax.transAxes, fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      alpha=0.9, edgecolor='#CCCCCC'))


# ============================================================================
# PLOTTING — Experiment B
# ============================================================================

def plot_exp_b_panel(ax, result: dict, title: str):
    if result is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='#999999')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
        return

    _style_ax(ax)
    embedding  = result['embedding']
    labels     = result['labels']
    colors     = result['colors']
    markers    = result['markers']
    compactness= result['compactness']
    chunk_scene= result['chunk_scene']

    # Background scenes first (grey, rendered behind)
    for i, (pt, lbl, col, mrk) in enumerate(
            zip(embedding, labels, colors, markers)):
        if mrk == 'background':
            ax.scatter(pt[0], pt[1],
                       c=BACKGROUND_COLOR, s=70, alpha=0.55,
                       edgecolors='white', linewidths=0.4, zorder=2)
            ax.annotate(lbl.replace('_', '\n'),
                        xy=pt, fontsize=5.5, color='#777777',
                        ha='center', xytext=(0, 6),
                        textcoords='offset points')

    # Target scene chunks (red, rendered on top)
    chunk_pts = embedding[np.array(markers) == 'chunk']
    for i, pt in enumerate(chunk_pts):
        ax.scatter(pt[0], pt[1],
                   c='#E63946', s=160, alpha=0.92,
                   edgecolors='white', linewidths=1.2, zorder=5,
                   marker='o')
        ax.annotate(f'chunk {i+1}', xy=pt,
                    fontsize=5.5, color='#E63946', fontweight='bold',
                    ha='center', xytext=(0, 7),
                    textcoords='offset points')

    # Convex hull outline around chunk cluster (if ≥ 3 chunks)
    if len(chunk_pts) >= 3:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(chunk_pts)
            for simplex in hull.simplices:
                ax.plot(chunk_pts[simplex, 0], chunk_pts[simplex, 1],
                        color='#E63946', alpha=0.4, linewidth=1.5, zorder=4,
                        linestyle='--')
        except Exception:
            pass

    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

    # Compactness annotation
    cpt_text = (f"Chunk spread: {compactness:.3f}\n"
                f"(lower = tighter = better)\n"
                f"n={len(chunk_pts)} chunks of {chunk_scene}")
    ax.text(0.03, 0.97, cpt_text,
            transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      alpha=0.9, edgecolor='#CCCCCC'))

    # Legend entries
    handles = [
        mpatches.Patch(color='#E63946',
                       label=f'{chunk_scene} chunks (n={len(chunk_pts)})'),
        mpatches.Patch(color=BACKGROUND_COLOR,
                       label='Background scenes'),
    ]
    ax.legend(handles=handles, fontsize=7, loc='lower right',
              framealpha=0.88, edgecolor='#CCCCCC')


# ============================================================================
# FULL FIGURE ASSEMBLY
# ============================================================================

def make_figure(exp_a_results: dict, exp_b_results: dict,
                output_dir: str, skip_a: bool, skip_b: bool):

    model_names = list(MODEL_CHECKPOINTS.keys())
    n_models    = len(model_names)

    rows = (0 if skip_a else 1) + (0 if skip_b else 1)
    if rows == 0:
        print("[WARN] Nothing to plot.")
        return

    fig, axes = plt.subplots(
        rows, n_models,
        figsize=(6.5 * n_models, 6.5 * rows),
        squeeze=False,
    )
    fig.patch.set_facecolor('white')

    row = 0

    # ── Row 1: Experiment A ───────────────────────────────────────────────────
    if not skip_a:
        for col, name in enumerate(model_names):
            short = name.replace('\n', ' — ')
            title = f"Exp A: Category Clustering\n{short}"
            r = exp_a_results.get(name)
            plot_exp_a_panel(axes[row][col], r, title)

        # Shared category legend on first panel
        r0 = next((v for v in exp_a_results.values() if v is not None), None)
        if r0:
            cats = sorted(set(r0['labels']))
            patches = [
                mpatches.Patch(
                    color=CATEGORY_COLORS.get(c, DEFAULT_COLOR),
                    label=c.replace('_', ' ') + '  ★=centroid'
                )
                for c in cats
            ]
            axes[row][0].legend(
                handles=patches, fontsize=7, loc='upper right',
                title='Room category', title_fontsize=7,
                framealpha=0.88, edgecolor='#CCCCCC',
            )
        row += 1

    # ── Row 2: Experiment B ───────────────────────────────────────────────────
    if not skip_b:
        for col, name in enumerate(model_names):
            short = name.replace('\n', ' — ')
            title = f"Exp B: Chunk Consistency\n{short}"
            r = exp_b_results.get(name)
            plot_exp_b_panel(axes[row][col], r, title)
        row += 1

    # ── Silhouette comparison bar (top of figure) ─────────────────────────────
    if not skip_a:
        sil_vals = [(name.replace('\n', ' '), exp_a_results.get(name, {}) or {})
                    for name in model_names]
        sil_str  = '   |   '.join(
            f"{n.split('—')[-1].strip()}: {d.get('silhouette', float('nan')):.3f}"
            if d else f"{n}: N/A"
            for n, d in sil_vals
        )
        fig.suptitle(
            f"Can3Tok Latent Space — {LATENT_MODE} representation\n"
            f"Exp A silhouette: {sil_str}",
            fontsize=11, fontweight='bold', y=1.01,
        )
    else:
        fig.suptitle(
            f"Can3Tok Latent Space — {LATENT_MODE} representation",
            fontsize=11, fontweight='bold', y=1.01,
        )

    fig.tight_layout(h_pad=3.0, w_pad=2.0)

    os.makedirs(output_dir, exist_ok=True)
    png = os.path.join(output_dir, 'can3tok_tsne.png')
    pdf = os.path.join(output_dir, 'can3tok_tsne.pdf')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {png}")
    print(f"  Saved: {pdf}")
    return png


# ============================================================================
# PRINT SUMMARY TABLE
# ============================================================================

def print_summary(exp_a_results: dict, exp_b_results: dict):
    print(f"\n{'='*65}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"\nExperiment A — Silhouette score (higher = better inter-category separation):")
    print(f"  {'Model':<40} {'Silhouette':>12}")
    print(f"  {'─'*54}")
    for name, r in exp_a_results.items():
        sil = r['silhouette'] if r else float('nan')
        print(f"  {name.replace(chr(10), ' '):<40} {sil:>12.4f}")

    print(f"\nExperiment B — Chunk compactness (lower = tighter chunks = better):")
    print(f"  {'Model':<40} {'Compactness':>12}  {'N chunks':>8}")
    print(f"  {'─'*64}")
    for name, r in exp_b_results.items():
        if r:
            print(f"  {name.replace(chr(10), ' '):<40} "
                  f"{r['compactness']:>12.4f}  {r['n_chunks']:>8}")
        else:
            print(f"  {name.replace(chr(10), ' '):<40} {'N/A':>12}")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    global LATENT_MODE
    ap = argparse.ArgumentParser(
        description='Can3Tok t-SNE — deterministic latent space analysis'
    )
    ap.add_argument('--output_dir',  default='tsne_results')
    ap.add_argument('--no_cache',    action='store_true',
                    help='Re-encode all scenes (ignore cached latents)')
    ap.add_argument('--skip_exp_a',  action='store_true',
                    help='Skip inter-category experiment')
    ap.add_argument('--skip_exp_b',  action='store_true',
                    help='Skip chunk consistency experiment')
    ap.add_argument('--latent_mode', default=LATENT_MODE,
                    choices=['shape_embed', 'full'],
                    help='shape_embed=[32] or full=[16384]')
    args = ap.parse_args()

    use_cache   = not args.no_cache

    LATENT_MODE = args.latent_mode

    device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    split_dir = Path(DATASET_ROOT) / SPLIT

    print(f"\n{'='*65}")
    print(f"Can3Tok t-SNE — Deterministic Latent Space Analysis")
    print(f"{'='*65}")
    print(f"  Device:      {device}")
    print(f"  Split dir:   {split_dir}")
    print(f"  Latent mode: {LATENT_MODE}")
    print(f"  Cache dir:   {CACHE_DIR}  (use_cache={use_cache})")
    print(f"  Output:      {args.output_dir}")

    if not split_dir.exists():
        print(f"\n[ERROR] Split directory not found: {split_dir}")
        sys.exit(1)

    # ── Load all models ───────────────────────────────────────────────────────
    # models_info: name → (ckpt_path, model, flags, device)
    models_info = {}
    for name, ckpt_path in MODEL_CHECKPOINTS.items():
        if not Path(ckpt_path).exists():
            print(f"\n[WARN] Checkpoint not found: {ckpt_path}")
            print(f"       Skipping: {name.replace(chr(10), ' ')}")
            continue
        print(f"\nLoading model: {name.replace(chr(10), ' ')}")
        model, flags = load_model(ckpt_path, device)
        models_info[name] = (ckpt_path, model, flags, device)

    if not models_info:
        print("\n[ERROR] No valid checkpoints found. "
              "Edit MODEL_CHECKPOINTS at the top of the script.")
        sys.exit(1)

    # ── Run experiments ───────────────────────────────────────────────────────
    exp_a_results = {}
    exp_b_results = {}

    if not args.skip_exp_a:
        exp_a_results = run_experiment_a(split_dir, models_info, use_cache)

    if not args.skip_exp_b:
        exp_b_results = run_experiment_b(split_dir, models_info, use_cache)

    # ── Plot ──────────────────────────────────────────────────────────────────
    make_figure(exp_a_results, exp_b_results,
                args.output_dir, args.skip_exp_a, args.skip_exp_b)

    print_summary(exp_a_results, exp_b_results)

    print(f"Done. Collect outputs:")
    print(f"  scp user@snellius:$(pwd)/{args.output_dir}/can3tok_tsne.png .")
    print(f"  scp user@snellius:$(pwd)/{args.output_dir}/can3tok_tsne.pdf .")


if __name__ == '__main__':
    main()