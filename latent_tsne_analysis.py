"""
latent_tsne_analysis.py
========================
Two t-SNE experiments matching the Can3Tok paper:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPERIMENT A — CATEGORY CLUSTERING  (your Figure 8 variant)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Each scene encoded once (deterministic top-40k by opacity).
  Points coloured by room category (apartment, coffee_shop, …).
  Hypothesis: semantic model shows tighter same-category clusters.
  Silhouette score and separation ratio measure cluster quality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPERIMENT B — INTRA-SCENE CONSISTENCY  (paper Figure 8)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Select N_ANCHOR scenes from your labelled set.
  Each anchor is randomly subsampled K times (different 40k Gaussians).
  + M other scenes encoded once as "background".

  Colouring:
    Each anchor scene gets a distinct colour (red, blue, green …).
    All other/background scenes are plotted in grey.

  Expected result:
    K subsamplings of the SAME scene cluster tightly (same colour).
    Different scenes are spread apart (grey dots far from coloured).
    Semantic model → tighter within-scene clusters than baseline.

  This directly matches the paper: "latent spaces from same scenes
  are closer to each other, and otherwise for other scenes."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE:
  python latent_tsne_analysis.py \\
    --scene_config        scene_config.json \\
    --dataset_root        /path/to/interior_gs \\
    --split               train_grid1.0cm_chunk8x8_stride6x6 \\
    --checkpoint_semantic checkpoints/RGB_job_XXXX_hidden_beta0.3/final.pth \\
    --checkpoint_baseline checkpoints/RGB_job_XXXX_none/final.pth \\
    --output_dir          tsne_results/
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
from gs_dataset_scenesplat import gs_dataset


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTES
# ─────────────────────────────────────────────────────────────────────────────

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
    'indoor_pool':       '#48CAE4',
    'futuristic_pod':    '#F72585',
    'restaurant':        '#7B2D8B',
    'wedding_hall':      '#F9C74F',
}
DEFAULT_COLOR = '#888888'

# Anchor colours for Experiment B (distinct, high-contrast)
ANCHOR_COLORS = [
    '#E63946',  # red
    '#2196F3',  # blue
    '#4CAF50',  # green
    '#FF9800',  # orange
    '#9C27B0',  # purple
    '#00BCD4',  # cyan
    '#FF5722',  # deep orange
    '#8BC34A',  # light green
]
OTHER_COLOR = '#BBBBBB'   # grey for non-anchor scenes


# ─────────────────────────────────────────────────────────────────────────────
# SCENE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def load_scene_config(config_path: str, min_per_cat: int = 2, max_per_cat: int = 15):
    with open(config_path) as f:
        entries = json.load(f)

    by_cat = defaultdict(list)
    all_scene_ids = []
    for e in entries:
        cat = e.get('category', '').strip()
        sid = e.get('scene_id', '').strip()
        if not sid or sid == 'unknown':
            continue
        all_scene_ids.append(sid)
        if cat and cat != 'unknown':
            by_cat[cat].append(sid)

    for cat in by_cat:
        by_cat[cat] = by_cat[cat][:max_per_cat]

    valid = [c for c, ids in by_cat.items() if len(ids) >= min_per_cat]
    excluded = [c for c, ids in by_cat.items() if len(ids) < min_per_cat]

    print(f"\nScene config loaded: {config_path}")
    print("─" * 50)
    for cat in sorted(valid):
        print(f"  {cat:<25} {len(by_cat[cat])} scenes")
    if excluded:
        print(f"\n  Excluded (< {min_per_cat} scenes): {excluded}")
    print("─" * 50)

    return dict(by_cat), valid, all_scene_ids


# ─────────────────────────────────────────────────────────────────────────────
# DATASET HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def find_scene_dir(scene_id: str, split_dir: Path):
    """Return the single directory for scene_id (exact or prefix match)."""
    exact = split_dir / scene_id
    if exact.is_dir():
        return exact
    matches = sorted([d for d in split_dir.iterdir()
                      if d.is_dir() and d.name.startswith(scene_id)])
    return matches[0] if matches else None


def find_all_chunks_for_scene(scene_id: str, split_dir: Path) -> list:
    """
    Return ALL chunk directories that belong to scene_id.

    In train_grid1.0cm_chunk8x8_stride6x6 the layout is:
        0210_840153_r0_c0/   0210_840153_r0_c1/   0210_840153_r1_c0/ …

    We match every directory whose name starts with scene_id.
    Falls back to single exact-match directory if no chunks found.
    """
    matches = sorted([d for d in split_dir.iterdir()
                      if d.is_dir() and d.name.startswith(scene_id)])
    return matches   # empty list if scene not found


def load_scene_tensor(scene_id: str,
                      split_dir: Path,
                      training_flags: dict,
                      resol: int = 200,
                      sampling_method: str = 'opacity',
                      random_seed: int = None):
    """
    Load ONE chunk for scene_id as a [40000, 18] tensor.

    sampling_method='opacity'  → deterministic top-40k  (Exp A / single chunk)
    sampling_method='random'   → random 40k subset       (Exp B subsamplings)
    random_seed                → set numpy seed before random sampling
    """
    scene_dir = find_scene_dir(scene_id, split_dir)
    if scene_dir is None:
        return None, "directory not found"

    try:
        if random_seed is not None:
            np.random.seed(random_seed)

        ds = gs_dataset(
            root             = str(scene_dir.parent),
            resol            = resol,
            random_permute   = False,
            train            = False,
            sampling_method  = sampling_method,
            max_scenes       = None,
            normalize        = training_flags['use_canonical_norm'],
            normalize_colors = True,
            target_radius    = 10.0,
            scale_norm_mode  = training_flags['scale_norm_mode'],
        )

        scene_names = [os.path.basename(d) for d in ds.scene_dirs]
        if scene_id in scene_names:
            idx = scene_names.index(scene_id)
        else:
            match = [n for n in scene_names if n.startswith(scene_id)]
            if not match:
                return None, "not in dataset"
            idx = scene_names.index(match[0])

        sample = ds[idx]
        return sample['features'], None

    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    print(f"\nLoading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    flags = {
        'semantic_mode':      ckpt.get('semantic_mode',      'none'),
        'scale_norm_mode':    ckpt.get('scale_norm_mode',    'linear'),
        'use_canonical_norm': ckpt.get('use_canonical_norm', True),
    }

    config_path  = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
    model_config = get_config_from_file(config_path).model
    model_config.params.shape_module_cfg.params.semantic_mode = flags['semantic_mode']

    model = instantiate_from_config(model_config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    print(f"  semantic_mode={flags['semantic_mode']}  "
          f"scale_norm={flags['scale_norm_mode']}  "
          f"epoch={ckpt.get('epoch','?')}  "
          f"val_L2={ckpt.get('val_l2_error', ckpt.get('final_val_l2','?'))}")
    return model, flags


# ─────────────────────────────────────────────────────────────────────────────
# ENCODING  (shared by both experiments)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_tensor(model, features: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Encode a [40000, 18] numpy array → z flat [16384].
    Returns None on failure.
    """
    try:
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        _, mu, log_var, z, _, _ = model(x, x, x, x[:, :, :3])
        return z.squeeze(0).reshape(-1).detach().cpu().numpy()
    except Exception as e:
        return None


@torch.no_grad()
def encode_scene_aggregated(model,
                             scene_id: str,
                             split_dir: Path,
                             training_flags: dict,
                             device: torch.device,
                             resol: int = 200) -> tuple:
    """
    Encode ALL chunks of a scene and return their MEAN latent vector.

    Why average?
      Each 8×8m chunk captures one spatial slice of the room.
      The model encodes each chunk to z ∈ R^16384.
      Averaging across all chunks gives a single vector that represents
      the entire scene, weighted equally across its spatial extent.
      This is far more representative than any single chunk.

    Returns:
        z_mean   np.ndarray [16384]  — mean latent across all chunks
        n_chunks int                 — number of chunks successfully encoded
        error    str | None         — error message if completely failed
    """
    chunk_dirs = find_all_chunks_for_scene(scene_id, split_dir)
    if not chunk_dirs:
        return None, 0, "no chunk directories found"

    chunk_latents = []

    for chunk_dir in chunk_dirs:
        try:
            ds = gs_dataset(
                root             = str(chunk_dir.parent),
                resol            = resol,
                random_permute   = False,
                train            = False,
                sampling_method  = 'opacity',   # deterministic per chunk
                max_scenes       = None,
                normalize        = training_flags['use_canonical_norm'],
                normalize_colors = True,
                target_radius    = 10.0,
                scale_norm_mode  = training_flags['scale_norm_mode'],
            )

            # Find this specific chunk in the dataset
            scene_names = [os.path.basename(d) for d in ds.scene_dirs]
            chunk_name  = chunk_dir.name
            if chunk_name in scene_names:
                idx = scene_names.index(chunk_name)
            else:
                match = [n for n in scene_names if n.startswith(chunk_name)]
                if not match:
                    continue
                idx = scene_names.index(match[0])

            feats = ds[idx]['features']
            z = encode_tensor(model, feats, device)
            if z is not None:
                chunk_latents.append(z)

        except Exception:
            continue

    if not chunk_latents:
        return None, 0, "all chunks failed to encode"

    z_mean = np.mean(chunk_latents, axis=0)   # [16384]
    return z_mean, len(chunk_latents), None


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT A — CATEGORY CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_for_category_experiment(model,
                                    scenes_by_category: dict,
                                    valid_categories: list,
                                    split_dir: Path,
                                    training_flags: dict,
                                    device: torch.device,
                                    resol: int = 200,
                                    aggregate_chunks: bool = False):
    """
    Encode each labelled scene and return one latent vector per scene.

    aggregate_chunks=False (default):
        Encode the single most central chunk only.
        Fast, matches training distribution exactly.

    aggregate_chunks=True:
        Encode ALL chunks of each scene and average their z vectors.
        Produces a holistic scene representation covering the full
        spatial extent of the room, not just one 8×8m slice.
        Better for category clustering because the whole room is captured.

    Returns latents [N, 16384], labels [N], scene_ids [N], skipped list.
    """
    latents, labels, scene_ids, skipped = [], [], [], []

    mode_str = "aggregate ALL chunks (mean z)" if aggregate_chunks else "single central chunk"
    print(f"\n  Mode: {mode_str}")

    for cat in valid_categories:
        ids = scenes_by_category[cat]
        print(f"\n  {cat} ({len(ids)} scenes)...")
        for sid in tqdm(ids, desc=f"    {cat}", leave=False):

            if aggregate_chunks:
                z, n_chunks, err = encode_scene_aggregated(
                    model, sid, split_dir, training_flags, device, resol)
                if z is None:
                    print(f"    [SKIP] {sid}: {err}")
                    skipped.append((sid, err))
                    continue
                print(f"    {sid}  ← mean of {n_chunks} chunks")
            else:
                feats, err = load_scene_tensor(sid, split_dir, training_flags,
                                               resol, sampling_method='opacity')
                if feats is None:
                    print(f"    [SKIP] {sid}: {err}")
                    skipped.append((sid, err))
                    continue
                z = encode_tensor(model, feats, device)
                if z is None:
                    skipped.append((sid, 'forward pass failed'))
                    continue

            latents.append(z)
            labels.append(cat)
            scene_ids.append(sid)

    if not latents:
        return None, None, None, skipped
    return np.stack(latents), labels, scene_ids, skipped


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT B — INTRA-SCENE CONSISTENCY  (paper Figure 8)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_for_consistency_experiment(model,
                                       anchor_scene_ids: list,
                                       other_scene_ids: list,
                                       split_dir: Path,
                                       training_flags: dict,
                                       device: torch.device,
                                       n_subsamplings: int = 5,
                                       resol: int = 200):
    """
    Paper Figure 8:
      - Each anchor scene is randomly subsampled n_subsamplings times
        → each subsampling gets the same colour as its scene
      - Other scenes encoded once → grey

    Returns:
        latents   [N, 16384]
        labels    [N]    scene_id string for anchors, 'other' for others
        colors    [N]    hex colour string per point
        markers   [N]    'anchor' or 'other'
        scene_ids [N]
        skipped   list
    """
    latents, labels, colors, markers, scene_ids = [], [], [], [], []
    skipped = []

    # ── Anchor scenes: K random subsamplings each ────────────────────────────
    print(f"\n  Encoding {len(anchor_scene_ids)} anchor scenes "
          f"× {n_subsamplings} subsamplings each...")

    for anchor_idx, sid in enumerate(anchor_scene_ids):
        color = ANCHOR_COLORS[anchor_idx % len(ANCHOR_COLORS)]
        ok = 0
        for k in range(n_subsamplings):
            feats, err = load_scene_tensor(
                sid, split_dir, training_flags, resol,
                sampling_method = 'random',
                random_seed     = 42 + k * 100 + anchor_idx,
            )
            if feats is None:
                print(f"    [SKIP] {sid} subsample {k}: {err}")
                skipped.append((f"{sid}_sub{k}", err))
                continue

            z = encode_tensor(model, feats, device)
            if z is None:
                skipped.append((f"{sid}_sub{k}", 'forward pass failed'))
                continue

            latents.append(z)
            labels.append(sid)
            colors.append(color)
            markers.append('anchor')
            scene_ids.append(f"{sid}_s{k}")
            ok += 1

        print(f"    {sid}  →  {ok}/{n_subsamplings} subsamplings encoded  "
              f"[{color}]")

    # ── Other scenes: one encoding each ──────────────────────────────────────
    print(f"\n  Encoding {len(other_scene_ids)} background scenes (1× each)...")

    for sid in tqdm(other_scene_ids, desc="    others", leave=False):
        feats, err = load_scene_tensor(sid, split_dir, training_flags,
                                       resol, sampling_method='opacity')
        if feats is None:
            skipped.append((sid, err))
            continue

        z = encode_tensor(model, feats, device)
        if z is None:
            skipped.append((sid, 'forward pass failed'))
            continue

        latents.append(z)
        labels.append('other')
        colors.append(OTHER_COLOR)
        markers.append('other')
        scene_ids.append(sid)

    if not latents:
        return None, None, None, None, None, skipped

    return (np.stack(latents), labels, colors, markers, scene_ids, skipped)


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE
# ─────────────────────────────────────────────────────────────────────────────

def run_tsne(latents: np.ndarray, perplexity=None, n_iter=2000, seed=42):
    N = latents.shape[0]
    if perplexity is None:
        perplexity = float(max(2, min(15, N // 3)))
    perplexity = float(max(2.0, min(perplexity, N - 1)))

    print(f"\n  t-SNE: {N} points, perplexity={perplexity:.0f}, n_iter={n_iter}")
    return TSNE(
        n_components=2, perplexity=perplexity, max_iter=n_iter,
        init='pca', random_state=seed, learning_rate='auto',
    ).fit_transform(latents)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(embedding, labels):
    unique = list(set(labels))
    metrics = {}
    if len(unique) < 2 or len(labels) < 3:
        return metrics
    y = np.array([unique.index(l) for l in labels])
    try:
        metrics['silhouette'] = float(silhouette_score(embedding, y))
    except Exception:
        pass
    intra, inter = [], []
    for cat in unique:
        pts = embedding[np.array(labels) == cat]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                intra.append(np.linalg.norm(pts[i] - pts[j]))
    for i, ci in enumerate(unique):
        for j, cj in enumerate(unique):
            if i >= j:
                continue
            pi = embedding[np.array(labels) == ci]
            pj = embedding[np.array(labels) == cj]
            for a in pi:
                for b in pj:
                    inter.append(np.linalg.norm(a - b))
    if intra:
        metrics['intra_mean'] = float(np.mean(intra))
    if inter:
        metrics['inter_mean'] = float(np.mean(inter))
    if intra and inter and np.mean(intra) > 0:
        metrics['separation_ratio'] = float(np.mean(inter) / np.mean(intra))
    return metrics


def intra_scene_compactness(embedding, labels, markers):
    """
    Experiment B specific metric:
    For each anchor scene compute mean pairwise distance between its subsamplings.
    Lower = tighter = better consistency.
    """
    anchor_ids = [l for l, m in zip(labels, markers) if m == 'anchor']
    unique_anchors = [a for a in dict.fromkeys(anchor_ids)]   # preserve order

    compactness = {}
    for sid in unique_anchors:
        pts = embedding[[i for i, (l, m) in enumerate(zip(labels, markers))
                         if l == sid and m == 'anchor']]
        if len(pts) < 2:
            continue
        dists = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dists.append(np.linalg.norm(pts[i] - pts[j]))
        compactness[sid] = float(np.mean(dists))

    return compactness


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT C — CATEGORY CENTROID DISTANCES
# ─────────────────────────────────────────────────────────────────────────────

def compute_centroid_distances(embedding: np.ndarray, labels: list) -> dict:
    """
    For each category compute its centroid (mean of all its scene points).
    Then compute the distance from every centroid to every other centroid.

    Simple idea:
      apartment centroid  = average position of all apartment dots in 2D
      coffee_shop centroid = average position of all coffee_shop dots in 2D
      distance(apartment, coffee_shop) = how far apart those two averages are

    Returns:
        centroids: dict  category → [x, y]  centroid coordinates
        distances: dict  "catA → catB" → float distance
        matrix:    dict  suitable for heatmap plotting
                         matrix[row_cat][col_cat] = distance
    """
    label_arr  = np.array(labels)
    categories = sorted(set(labels))

    # ── Compute one centroid per category ────────────────────────────────────
    centroids = {}
    for cat in categories:
        pts = embedding[label_arr == cat]          # all dots of this category
        centroids[cat] = pts.mean(axis=0)          # mean x, mean y → centroid

    # ── Pairwise distances between centroids ─────────────────────────────────
    distances = {}
    matrix    = {c: {} for c in categories}

    for i, ci in enumerate(categories):
        for j, cj in enumerate(categories):
            dist = float(np.linalg.norm(centroids[ci] - centroids[cj]))
            matrix[ci][cj] = dist
            if i < j:
                distances[f"{ci} → {cj}"] = dist

    # ── Print readable summary ───────────────────────────────────────────────
    print(f"\n  Category centroid distances:")
    print(f"  {'Pair':<45} Distance")
    print(f"  {'─'*55}")
    for pair, dist in sorted(distances.items(), key=lambda x: x[1]):
        print(f"  {pair:<45} {dist:.3f}")

    return centroids, distances, matrix


def plot_centroid_heatmap(ax, centroid_matrix: dict, categories: list,
                          centroids_xy: dict, embedding: np.ndarray,
                          labels: list, title: str):
    """
    Two-panel visualisation of centroid distances:

    Left side of ax  → distance heatmap (matrix)
    Annotations show the actual distance number in each cell.

    Colour scale: dark = far apart, light = close together.
    Diagonal is always 0 (a category is 0 away from itself).

    Also draws star markers at each centroid position
    on the t-SNE scatter (passed as embedding + labels).
    """
    n    = len(categories)
    mat  = np.array([[centroid_matrix[ci][cj] for cj in categories]
                     for ci in categories])

    # Normalise for colour (0 = same spot, 1 = furthest apart)
    mat_norm = mat / (mat.max() + 1e-8)

    im = ax.imshow(mat_norm, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')

    # Axis labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([c.replace('_', '\n') for c in categories],
                       fontsize=7, rotation=0)
    ax.set_yticklabels(categories, fontsize=7)

    # Annotate each cell with the actual distance value
    for i in range(n):
        for j in range(n):
            dist = centroid_matrix[categories[i]][categories[j]]
            text_color = 'white' if mat_norm[i, j] > 0.6 else '#333333'
            ax.text(j, i, f"{dist:.1f}",
                    ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color=text_color)

    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel("Category", fontsize=8)
    ax.set_ylabel("Category", fontsize=8)

    # Colour bar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="4%", pad=0.08)
    cb      = plt.colorbar(im, cax=cax)
    cb.set_label("Relative distance\n(0=same, 1=furthest)", fontsize=6)
    cb.ax.tick_params(labelsize=6)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING — EXPERIMENT A (category clustering)
# ─────────────────────────────────────────────────────────────────────────────

def plot_category_panel(ax, embedding, labels, scene_ids, title, metrics,
                        centroids_xy=None):
    ax.set_facecolor('#F8F8F8')
    ax.grid(True, linewidth=0.4, alpha=0.5, color='white')

    label_arr = np.array(labels)
    for cat in sorted(set(labels)):
        mask  = label_arr == cat
        pts   = embedding[mask]
        color = CATEGORY_COLORS.get(cat, DEFAULT_COLOR)
        sids  = [s for s, l in zip(scene_ids, labels) if l == cat]

        ax.scatter(pts[:, 0], pts[:, 1],
                   c=color, label=f"{cat} (n={mask.sum()})",
                   s=120, alpha=0.85, edgecolors='white', linewidths=0.8, zorder=3)

        for pt, sid in zip(pts, sids):
            ax.annotate(sid[-7:], xy=pt, fontsize=5.5, color='#333333',
                        ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points')

        # Draw centroid as a large star marker
        if centroids_xy is not None and cat in centroids_xy:
            cx, cy = centroids_xy[cat]
            ax.scatter(cx, cy, c=color, marker='*', s=320, zorder=6,
                       edgecolors='black', linewidths=0.8, alpha=1.0)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel("t-SNE dim 1", fontsize=8)
    ax.set_ylabel("t-SNE dim 2", fontsize=8)
    ax.tick_params(labelsize=7)

    if metrics:
        lines = []
        if 'silhouette' in metrics:
            lines.append(f"Silhouette: {metrics['silhouette']:.3f}")
        if 'separation_ratio' in metrics:
            lines.append(f"Separation: {metrics['separation_ratio']:.2f}×")
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontsize=7.5, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.85, edgecolor='#CCCCCC'))


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING — EXPERIMENT B (intra-scene consistency)
# ─────────────────────────────────────────────────────────────────────────────

def plot_consistency_panel(ax, embedding, labels, colors, markers,
                           scene_ids, anchor_ids, title, compactness):
    ax.set_facecolor('#F8F8F8')
    ax.grid(True, linewidth=0.4, alpha=0.5, color='white')

    colors_arr  = np.array(colors)
    markers_arr = np.array(markers)

    # ── Background (other) scenes first so anchors render on top ────────────
    other_mask = markers_arr == 'other'
    if other_mask.any():
        ax.scatter(embedding[other_mask, 0], embedding[other_mask, 1],
                   c=OTHER_COLOR, s=60, alpha=0.5,
                   edgecolors='white', linewidths=0.4, zorder=2,
                   label=f"Other scenes (n={other_mask.sum()})")

    # ── Anchor scenes — each colour = one scene ──────────────────────────────
    for anchor_idx, sid in enumerate(anchor_ids):
        color = ANCHOR_COLORS[anchor_idx % len(ANCHOR_COLORS)]
        mask  = np.array([l == sid and m == 'anchor'
                          for l, m in zip(labels, markers)])
        if not mask.any():
            continue
        pts = embedding[mask]
        n   = mask.sum()
        cpt = compactness.get(sid, float('nan'))

        ax.scatter(pts[:, 0], pts[:, 1],
                   c=color, s=140, alpha=0.9,
                   edgecolors='white', linewidths=1.0, zorder=4,
                   label=f"{sid[-11:]} (n={n}, Δ={cpt:.1f})")

        # Draw convex hull outline for visual clarity when ≥3 points
        if n >= 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax.plot(pts[simplex, 0], pts[simplex, 1],
                            color=color, alpha=0.35, linewidth=1.2, zorder=3)
            except Exception:
                pass

        # Label centroid
        centroid = pts.mean(axis=0)
        ax.annotate(sid[-7:], xy=centroid, fontsize=6, fontweight='bold',
                    color=color, ha='center', va='center',
                    xytext=(0, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.7, edgecolor=color, linewidth=0.8))

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel("t-SNE dim 1", fontsize=8)
    ax.set_ylabel("t-SNE dim 2", fontsize=8)
    ax.tick_params(labelsize=7)

    # Mean compactness annotation
    if compactness:
        mean_c = np.mean(list(compactness.values()))
        ax.text(0.02, 0.98,
                f"Mean intra-scene Δ: {mean_c:.2f}\n(lower = tighter clusters)",
                transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.85, edgecolor='#CCCCCC'))

    ax.legend(fontsize=6.5, loc='lower right', framealpha=0.85,
              edgecolor='#CCCCCC', ncol=1)


# ─────────────────────────────────────────────────────────────────────────────
# FULL FIGURE ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(exp_a_results, exp_b_results, output_dir: str):
    """
    Layout:
      Row 1 (Experiment A): scatter coloured by category  +  centroid stars
      Row 2 (Experiment B): intra-scene consistency
      Row 3 (Centroid heatmap): pairwise centroid distance matrix
    """
    has_baseline = (exp_a_results.get('baseline') is not None or
                    exp_b_results.get('baseline') is not None)
    ncols = 2 if has_baseline else 1

    n_a   = 1 if exp_a_results.get('semantic') is not None else 0
    n_b   = 1 if exp_b_results.get('semantic') is not None else 0
    # Row 3 only exists when we have Exp A results (centroids come from Exp A)
    n_c   = 1 if n_a else 0
    nrows = n_a + n_b + n_c

    if nrows == 0:
        print("[WARN] No results to plot.")
        return None

    # Row heights: Exp A and B are tall; heatmap row is shorter
    height_ratios = []
    if n_a: height_ratios.append(6.5)
    if n_b: height_ratios.append(6.5)
    if n_c: height_ratios.append(5.0)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, sum(height_ratios)),
        gridspec_kw={'height_ratios': height_ratios},
        squeeze=False,
    )
    fig.patch.set_facecolor('white')

    row = 0

    # ── Row 1: Experiment A — category scatter ────────────────────────────────
    if n_a:
        for col, key in enumerate(['baseline', 'semantic']):
            if not has_baseline and key == 'baseline':
                continue
            r = exp_a_results.get(key)
            if r is None:
                axes[row][col].set_visible(False)
                continue
            label_name = ("Baseline VAE\n(no semantic loss)" if key == 'baseline'
                          else f"Semantic VAE\n({r['name']})")
            label_name += "\n— Exp A: Category Clustering —"
            plot_category_panel(
                axes[row][col], r['embedding'], r['labels'],
                r['scene_ids'], label_name, r['metrics'],
                centroids_xy = r.get('centroids_xy'),   # ★ centroid stars
            )

        # Shared legend
        all_cats = set()
        for key in ['baseline', 'semantic']:
            r = exp_a_results.get(key)
            if r:
                all_cats.update(r['labels'])
        patches = [mpatches.Patch(color=CATEGORY_COLORS.get(c, DEFAULT_COLOR),
                                  label=c) for c in sorted(all_cats)]
        axes[row][0].legend(handles=patches, fontsize=7, loc='upper right',
                            framealpha=0.85, edgecolor='#CCCCCC',
                            title='Room category  ★=centroid', title_fontsize=7)
        row += 1

    # ── Row 2: Experiment B — intra-scene consistency ─────────────────────────
    if n_b:
        for col, key in enumerate(['baseline', 'semantic']):
            if not has_baseline and key == 'baseline':
                continue
            r = exp_b_results.get(key)
            if r is None:
                axes[row][col].set_visible(False)
                continue
            label_name = ("Baseline VAE\n(no semantic loss)" if key == 'baseline'
                          else f"Semantic VAE\n({r['name']})")
            label_name += "\n— Exp B: Intra-scene Consistency —"
            plot_consistency_panel(
                axes[row][col],
                r['embedding'], r['labels'], r['colors'],
                r['markers'], r['scene_ids'], r['anchor_ids'],
                label_name, r['compactness'])
        row += 1

    # ── Row 3: Centroid distance heatmaps ─────────────────────────────────────
    if n_c:
        for col, key in enumerate(['baseline', 'semantic']):
            if not has_baseline and key == 'baseline':
                continue
            r = exp_a_results.get(key)
            if r is None or 'centroid_matrix' not in r:
                axes[row][col].set_visible(False)
                continue
            label_name = ("Baseline VAE" if key == 'baseline'
                          else f"Semantic VAE")
            label_name += "\n— Centroid Distance Matrix —"
            cats = sorted(set(r['labels']))
            plot_centroid_heatmap(
                axes[row][col],
                r['centroid_matrix'],
                cats,
                r.get('centroids_xy', {}),
                r['embedding'],
                r['labels'],
                label_name,
            )
        row += 1

    fig.suptitle("Can3Tok Latent Space Analysis",
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout(h_pad=3.5, w_pad=2.0)

    os.makedirs(output_dir, exist_ok=True)
    png = os.path.join(output_dir, "tsne_semantic_comparison.png")
    pdf = os.path.join(output_dir, "tsne_semantic_comparison.pdf")
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {png}")
    print(f"  Saved: {pdf}")
    return png


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description='Can3Tok t-SNE — Experiment A (category) + B (intra-scene)'
    )
    ap.add_argument('--scene_config',            required=True)
    ap.add_argument('--dataset_root',            required=True)
    ap.add_argument('--split',
                    default='train_grid1.0cm_chunk8x8_stride6x6')
    ap.add_argument('--checkpoint_semantic',     required=True)
    ap.add_argument('--checkpoint_baseline',     default=None)
    ap.add_argument('--output_dir',              default='tsne_results')

    # Experiment A
    ap.add_argument('--min_scenes_per_category', type=int, default=2)
    ap.add_argument('--max_scenes_per_category', type=int, default=15)

    # Experiment B
    ap.add_argument('--n_anchor_scenes',   type=int, default=5,
                    help='How many scenes to use as anchors in Exp B')
    ap.add_argument('--n_subsamplings',    type=int, default=5,
                    help='Random subsamplings per anchor scene (Exp B)')
    ap.add_argument('--n_other_scenes',    type=int, default=8,
                    help='Background scenes encoded once in Exp B')

    # Aggregation
    ap.add_argument('--aggregate_chunks', action='store_true', default=False,
                    help='Encode ALL chunks per scene and average their latents. '
                         'Gives a holistic full-scene representation. '
                         'If False (default), use single central chunk only.')

    # Shared
    ap.add_argument('--tsne_perplexity',   type=float, default=None)
    ap.add_argument('--tsne_n_iter',       type=int,   default=2000)
    ap.add_argument('--tsne_seed',         type=int,   default=42)
    ap.add_argument('--resol',             type=int,   default=200)

    # Which experiments to run
    ap.add_argument('--skip_exp_a', action='store_true',
                    help='Skip category clustering experiment')
    ap.add_argument('--skip_exp_b', action='store_true',
                    help='Skip intra-scene consistency experiment')
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*65}")
    print(f"Can3Tok t-SNE — Experiment A (category) + B (intra-scene)")
    print(f"{'='*65}")
    print(f"  Device:  {device}")
    print(f"  Output:  {args.output_dir}")

    split_dir = Path(args.dataset_root) / args.split
    if not split_dir.exists():
        print(f"[ERROR] Not found: {split_dir}")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load scene config ────────────────────────────────────────────────────
    scenes_by_cat, valid_cats, all_ids = load_scene_config(
        args.scene_config,
        min_per_cat = args.min_scenes_per_category,
        max_per_cat = args.max_scenes_per_category,
    )

    # Select anchor scenes for Exp B — prefer those with most subsamplings needed
    # Use apartment (largest category) as first anchors, then diversify
    anchor_ids = []
    for cat in sorted(valid_cats, key=lambda c: -len(scenes_by_cat[c])):
        for sid in scenes_by_cat[cat]:
            if sid not in anchor_ids:
                anchor_ids.append(sid)
            if len(anchor_ids) >= args.n_anchor_scenes:
                break
        if len(anchor_ids) >= args.n_anchor_scenes:
            break

    other_ids = [s for s in all_ids
                 if s not in anchor_ids][:args.n_other_scenes]

    print(f"\nExp B anchors ({len(anchor_ids)}): {anchor_ids}")
    print(f"Exp B others  ({len(other_ids)}):  {other_ids}")

    # ── Run both experiments for each model ──────────────────────────────────
    exp_a_results = {'baseline': None, 'semantic': None}
    exp_b_results = {'baseline': None, 'semantic': None}

    checkpoints = {}
    if args.checkpoint_baseline:
        checkpoints['baseline'] = args.checkpoint_baseline
    checkpoints['semantic'] = args.checkpoint_semantic

    all_metrics = {}

    for model_key, ckpt_path in checkpoints.items():
        print(f"\n{'='*65}")
        print(f"MODEL: {model_key.upper()}  —  {Path(ckpt_path).parent.name}")
        print(f"{'='*65}")

        model, training_flags = load_model(ckpt_path, device)
        ckpt_name = Path(ckpt_path).parent.name

        # ── Experiment A ────────────────────────────────────────────────────
        if not args.skip_exp_a and valid_cats:
            print(f"\n{'─'*50}")
            print(f"Experiment A: Category Clustering")
            print(f"{'─'*50}")

            lat, lab, sids, skip = encode_for_category_experiment(
                model, scenes_by_cat, valid_cats, split_dir,
                training_flags, device, args.resol,
                aggregate_chunks = args.aggregate_chunks)

            if skip:
                print(f"  Skipped {len(skip)}: "
                      + ", ".join(f"{s}:{r}" for s, r in skip[:3]))

            if lat is not None and len(lat) >= 3:
                emb_a   = run_tsne(lat, args.tsne_perplexity,
                                   args.tsne_n_iter, args.tsne_seed)
                metrics = compute_metrics(emb_a, lab)
                print(f"\n  Metrics: {metrics}")

                # ── Centroid distances ───────────────────────────────────────
                centroids_xy, cent_distances, cent_matrix = \
                    compute_centroid_distances(emb_a, lab)

                np.save(f"{args.output_dir}/latents_expA_{model_key}.npy", lat)
                np.save(f"{args.output_dir}/embedding_expA_{model_key}.npy", emb_a)
                exp_a_results[model_key] = {
                    'embedding':       emb_a,
                    'labels':          lab,
                    'scene_ids':       sids,
                    'metrics':         metrics,
                    'name':            ckpt_name,
                    'centroids_xy':    centroids_xy,    # ★ centroid coords
                    'centroid_matrix': cent_matrix,     # ★ pairwise distances
                }
                all_metrics[f"expA_{model_key}"] = metrics
                all_metrics[f"expA_{model_key}_centroid_distances"] = cent_distances
            else:
                print("  [WARN] Not enough scenes encoded for Exp A.")

        # ── Experiment B ────────────────────────────────────────────────────
        if not args.skip_exp_b and anchor_ids:
            print(f"\n{'─'*50}")
            print(f"Experiment B: Intra-scene Consistency")
            print(f"{'─'*50}")

            lat, lab, col, mrk, sids, skip = encode_for_consistency_experiment(
                model, anchor_ids, other_ids, split_dir,
                training_flags, device,
                n_subsamplings = args.n_subsamplings,
                resol          = args.resol)

            if skip:
                print(f"  Skipped {len(skip)}: "
                      + ", ".join(f"{s}:{r}" for s, r in skip[:3]))

            if lat is not None and len(lat) >= 3:
                emb_b = run_tsne(lat, args.tsne_perplexity,
                                 args.tsne_n_iter, args.tsne_seed)
                compact = intra_scene_compactness(emb_b, lab, mrk)
                print(f"\n  Per-scene compactness (lower = tighter):")
                for sid, val in compact.items():
                    print(f"    {sid}: {val:.3f}")
                mean_c = np.mean(list(compact.values())) if compact else float('nan')
                print(f"    Mean: {mean_c:.3f}")

                np.save(f"{args.output_dir}/latents_expB_{model_key}.npy", lat)
                np.save(f"{args.output_dir}/embedding_expB_{model_key}.npy", emb_b)
                exp_b_results[model_key] = {
                    'embedding':  emb_b,
                    'labels':     lab,
                    'colors':     col,
                    'markers':    mrk,
                    'scene_ids':  sids,
                    'anchor_ids': anchor_ids,
                    'compactness': compact,
                    'name': ckpt_name,
                }
                all_metrics[f"expB_{model_key}"] = {
                    'per_scene_compactness': compact,
                    'mean_compactness': float(mean_c),
                }
            else:
                print("  [WARN] Not enough points encoded for Exp B.")

        del model
        torch.cuda.empty_cache()

    # ── Save metrics ─────────────────────────────────────────────────────────
    import json as _json
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        _json.dump(all_metrics, f, indent=2)

    # ── Plot ─────────────────────────────────────────────────────────────────
    png = make_figure(exp_a_results, exp_b_results, args.output_dir)

    print(f"\n{'='*65}")
    print(f"DONE")
    if png:
        print(f"  Plot:    {png}")
    print(f"  Metrics: {args.output_dir}/metrics.json")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()