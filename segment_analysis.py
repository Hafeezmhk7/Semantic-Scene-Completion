"""
segment_analysis.py
====================
Analyses the ScanNet72 segment label distribution for each of your
labelled scenes, then checks whether scenes that share similar label
distributions also appear close together in the t-SNE plot.

WHY THIS MATTERS
─────────────────
The semantic loss pushes Gaussians with the same ScanNet72 label
(wall, floor, sofa, …) toward similar latent features across scenes.

So if apartment A has:  40% wall, 20% floor, 15% sofa, 10% table …
   apartment B has:     38% wall, 22% floor, 14% sofa, 11% table …
   coffee shop C has:   35% wall, 18% floor,  5% sofa, 20% counter …

Then A and B share similar label distributions → semantic loss pulls
their Gaussians to similar features → their scene latents end up close
in t-SNE. C has a very different distribution → ends up further away.

This script tells you EXACTLY that story with real numbers and plots.

WHAT YOU GET
─────────────
  1. segment_distributions.png
       Heatmap: each row = one scene, each column = one segment label
       Cell = % of that label in that scene
       Lets you instantly see which scenes share the same "surface mix"

  2. label_similarity_matrix.png
       Heatmap: which pairs of scenes have the most similar label distributions
       (cosine similarity 0→1, where 1 = identical mix)

  3. tsne_vs_label_similarity.png  (if t-SNE embeddings provided)
       Scatter: x = label similarity, y = t-SNE distance
       If the semantic model works: high similarity → short t-SNE distance
       Shows this for baseline vs semantic side by side

  4. segment_stats.json
       All raw numbers — percentages, totals, similarities, t-SNE distances

USAGE
──────
  # Minimal (just segment distributions, no t-SNE comparison)
  python segment_analysis.py \\
    --scene_config  scene_config.json \\
    --dataset_root  /path/to/interior_gs \\
    --split         train_grid1.0cm_chunk8x8_stride6x6 \\
    --output_dir    segment_analysis/

  # Full (with t-SNE comparison)
  python segment_analysis.py \\
    --scene_config        scene_config.json \\
    --dataset_root        /path/to/interior_gs \\
    --split               train_grid1.0cm_chunk8x8_stride6x6 \\
    --embedding_baseline  tsne_results_job_XXX/embedding_expA_baseline.npy \\
    --embedding_semantic  tsne_results_job_XXX/embedding_expA_semantic.npy \\
    --output_dir          segment_analysis/
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Label indices — NO assumed names
# ─────────────────────────────────────────────────────────────────────────────
# We intentionally do NOT map label indices to names.
# The SceneSplat dataset paper does not publish a verified index→name mapping.
# Using an assumed ScanNet mapping produced nonsense (e.g. coffee shop = 14%
# "bathtub"). All analysis uses raw index strings: "label_0" … "label_71".
# Once the correct mapping is confirmed from the dataset authors, swap in names.

NUM_LABELS = 72

# Room category colours (matches latent_tsne_analysis.py)
CATEGORY_COLORS = {
    'apartment':         '#E63946',
    'coffee_shop':       '#F4A261',
    'convenience_store': '#2A9D8F',
    'club':              '#457B9D',
    'spa_pool':          '#8338EC',
    'cinema':            '#06D6A0',
    'concert_hall':      '#FFB703',
    'library':           '#3A86FF',
    'gym':               '#FF006E',
    'office':            '#8D99AE',
    'hotel':             '#EF8D2F',
    'restaurant':        '#7B2D8B',
}
DEFAULT_COLOR = '#888888'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD SCENE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def load_scene_config(config_path: str):
    """
    Load scene_config.json.
    Returns:
        scene_to_cat   dict  scene_id → category string
        cat_to_scenes  dict  category → [scene_id, ...]
        all_scene_ids  list  all scene ids with known categories
    """
    with open(config_path) as f:
        entries = json.load(f)

    scene_to_cat  = {}
    cat_to_scenes = defaultdict(list)

    for e in entries:
        sid = e.get('scene_id', '').strip()
        cat = e.get('category', '').strip()
        if not sid or sid == 'unknown':
            continue
        if cat and cat != 'unknown':
            scene_to_cat[sid]   = cat
            cat_to_scenes[cat].append(sid)

    all_scene_ids = list(scene_to_cat.keys())

    print(f"\nScene config: {config_path}")
    print(f"  Total labelled scenes: {len(all_scene_ids)}")
    print(f"  Categories found:")
    for cat, ids in sorted(cat_to_scenes.items()):
        print(f"    {cat:<28} {len(ids)} scenes")

    return scene_to_cat, dict(cat_to_scenes), all_scene_ids


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD SEGMENT DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

def find_all_chunks(scene_id: str, split_dir: Path) -> list:
    """All chunk dirs whose name starts with scene_id."""
    return sorted([d for d in split_dir.iterdir()
                   if d.is_dir() and d.name.startswith(scene_id)])


def load_segment_distribution(scene_id: str, split_dir: Path) -> tuple:
    """
    Load segment.npy across ALL chunks of a scene and compute
    the percentage of each ScanNet72 label.

    Each chunk has a segment.npy of shape [N_gaussians] where each
    value is a ScanNet72 label index (0-71) or -1 for unlabelled.

    We aggregate across all chunks so we get a whole-scene picture.

    Returns:
        dist    dict  label_idx(int) → percentage(float)  [only present labels]
        counts  dict  label_idx(int) → raw count(int)
        total   int   total labelled Gaussians
        n_chunks int  number of chunks found
    """
    chunk_dirs = find_all_chunks(scene_id, split_dir)
    if not chunk_dirs:
        return {}, {}, 0, 0

    raw_counts = np.zeros(NUM_LABELS, dtype=np.int64)
    total      = 0

    for chunk_dir in chunk_dirs:
        seg_path = chunk_dir / 'segment.npy'
        if not seg_path.exists():
            continue
        seg   = np.load(seg_path)
        valid = seg[(seg >= 0) & (seg < NUM_LABELS)]
        for lbl in valid:
            raw_counts[int(lbl)] += 1
        total += len(valid)

    if total == 0:
        return {}, {}, 0, len(chunk_dirs)

    dist   = {int(i): float(raw_counts[i] / total * 100)
              for i in range(NUM_LABELS) if raw_counts[i] > 0}
    counts = {int(i): int(raw_counts[i])
              for i in range(NUM_LABELS) if raw_counts[i] > 0}

    return dist, counts, total, len(chunk_dirs)


def load_all_distributions(scene_ids: list, split_dir: Path) -> dict:
    """
    Load distributions for all scenes.

    Returns:
        results  dict  scene_id → {
            'distribution': {label_idx: pct},
            'counts':       {label_idx: count},
            'total':        int,
            'n_chunks':     int,
        }
    """
    results = {}
    print(f"\nLoading segment distributions ({len(scene_ids)} scenes)...")

    for sid in tqdm(scene_ids, desc="  segments"):
        dist, counts, total, n_chunks = load_segment_distribution(sid, split_dir)
        results[sid] = {
            'distribution': dist,
            'counts':       counts,
            'total':        total,
            'n_chunks':     n_chunks,
        }
        if total == 0:
            print(f"  [WARN] {sid}: no segment labels found "
                  f"({n_chunks} chunks checked)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BUILD MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def find_top_labels(scene_ids: list, distributions: dict, top_n: int = 20):
    """
    Find the top_n most common labels across all scenes by total Gaussian count.
    Returns label indices and names.
    """
    global_counts = np.zeros(NUM_LABELS)
    for sid in scene_ids:
        d     = distributions[sid]['distribution']
        total = distributions[sid]['total']
        for lbl, pct in d.items():
            global_counts[int(lbl)] += pct * total / 100.0

    top_indices = list(np.argsort(global_counts)[-top_n:][::-1])
    top_names   = [f"label_{i}" for i in top_indices]
    return top_indices, top_names, global_counts


def build_pct_matrix(scene_ids: list, distributions: dict,
                     top_labels: list) -> np.ndarray:
    """
    Build [N_scenes × N_labels] matrix of label percentages.
    Row i = scene i,  column j = % of top_labels[j] in that scene.
    """
    mat = np.zeros((len(scene_ids), len(top_labels)))
    for i, sid in enumerate(scene_ids):
        d = distributions[sid]['distribution']
        for j, lbl in enumerate(top_labels):
            mat[i, j] = d.get(lbl, 0.0)
    return mat


def compute_cosine_similarity(mat: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity between rows of mat.
    1.0 = identical label mix,  0.0 = completely different.
    """
    norms  = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    normed = mat / norms
    sim    = normed @ normed.T
    return np.clip(sim, 0, 1)


def compute_tsne_dist_matrix(embedding: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances in t-SNE 2D space."""
    N   = len(embedding)
    mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            mat[i, j] = np.linalg.norm(embedding[i] - embedding[j])
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_distribution_heatmap(scene_ids, scene_to_cat, pct_matrix,
                               top_names, output_path):
    """
    Main heatmap: rows = scenes, columns = segment labels, cell = %.

    Scenes are grouped by category. Each row label shows:
      scene_id  [category]

    This is the key plot — it lets you visually see which scenes share
    the same surface composition.
    """
    N_scenes, N_labels = pct_matrix.shape

    # Group scenes by category for cleaner display
    cat_order = sorted(set(scene_to_cat.get(s, 'unknown') for s in scene_ids))
    ordered_scenes = []
    for cat in cat_order:
        ordered_scenes.extend([s for s in scene_ids
                                if scene_to_cat.get(s, 'unknown') == cat])

    # Reorder matrix rows
    idx_map    = {s: i for i, s in enumerate(scene_ids)}
    row_order  = [idx_map[s] for s in ordered_scenes]
    pct_ordered = pct_matrix[row_order, :]

    fig_h = max(7, N_scenes * 0.6)
    fig_w = max(14, N_labels * 0.75)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    im = ax.imshow(pct_ordered, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=max(pct_ordered.max(), 1))

    # X axis — label names
    ax.set_xticks(range(N_labels))
    ax.set_xticklabels(top_names, rotation=45, ha='right', fontsize=8.5)

    # Y axis — scene id + category
    y_labels = []
    cat_colors_bar = []
    for sid in ordered_scenes:
        cat   = scene_to_cat.get(sid, 'unknown')
        short = sid[-13:] if len(sid) > 13 else sid
        y_labels.append(f"{short}   [{cat}]")
        cat_colors_bar.append(CATEGORY_COLORS.get(cat, DEFAULT_COLOR))

    ax.set_yticks(range(N_scenes))
    ax.set_yticklabels(y_labels, fontsize=8)

    # Colour each y-tick label by room category
    for tick, color in zip(ax.get_yticklabels(), cat_colors_bar):
        tick.set_color(color)
        tick.set_fontweight('bold')

    # Annotate cells with the percentage value (skip values < 1%)
    for i in range(N_scenes):
        for j in range(N_labels):
            val = pct_ordered[i, j]
            if val >= 1.0:
                text_color = 'white' if val > pct_ordered.max() * 0.65 else '#333333'
                ax.text(j, i, f"{val:.0f}",
                        ha='center', va='center',
                        fontsize=6.5, color=text_color, fontweight='bold')

    # Category divider lines
    prev_cat  = None
    divider_y = -0.5
    for row_i, sid in enumerate(ordered_scenes):
        cat = scene_to_cat.get(sid, 'unknown')
        if cat != prev_cat and row_i > 0:
            ax.axhline(row_i - 0.5, color='#333333', linewidth=1.5, alpha=0.7)
        prev_cat = cat

    # Colour bar
    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label("% of Gaussians with this label", fontsize=9)

    # Category legend
    cats_present = sorted(set(scene_to_cat.get(s, 'unknown') for s in scene_ids))
    patches = [mpatches.Patch(color=CATEGORY_COLORS.get(c, DEFAULT_COLOR),
                               label=c) for c in cats_present]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.18, 1),
              fontsize=8, title='Room category', title_fontsize=8,
              framealpha=0.9)

    ax.set_title("Segment Label Distribution per Scene\n"
                 "(% of Gaussians labelled as each surface type)",
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("ScanNet72 Segment Label", fontsize=10)
    ax.set_ylabel("Scene  [Room Category]",  fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_similarity_matrix(scene_ids, scene_to_cat, sim_matrix, output_path):
    """
    Heatmap of pairwise label similarity between all scenes.

    Bright = similar label distribution (likely similar t-SNE position).
    Dark   = different label distribution (likely further apart in t-SNE).

    Scenes are grouped by category so you can immediately see whether
    within-category pairs are more similar than cross-category pairs.
    """
    N = len(scene_ids)

    # Group by category
    cat_order      = sorted(set(scene_to_cat.get(s, '?') for s in scene_ids))
    ordered_scenes = []
    for cat in cat_order:
        ordered_scenes.extend([s for s in scene_ids
                                if scene_to_cat.get(s, '?') == cat])
    idx_map  = {s: i for i, s in enumerate(scene_ids)}
    order    = [idx_map[s] for s in ordered_scenes]
    sim_ord  = sim_matrix[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(max(8, N * 0.7), max(7, N * 0.65)))
    fig.patch.set_facecolor('white')

    im = ax.imshow(sim_ord, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    short_labels = [s[-11:] + f"\n[{scene_to_cat.get(s,'?')[:4]}]"
                    for s in ordered_scenes]
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)

    # Annotate each cell
    for i in range(N):
        for j in range(N):
            val        = sim_ord[i, j]
            text_color = 'white' if val < 0.35 else '#222222'
            ax.text(j, i, f"{val:.2f}",
                    ha='center', va='center',
                    fontsize=6.5, color=text_color)

    # Category divider lines
    prev_cat = None
    for row_i, sid in enumerate(ordered_scenes):
        cat = scene_to_cat.get(sid, '?')
        if cat != prev_cat and row_i > 0:
            ax.axhline(row_i - 0.5, color='black', linewidth=1.5)
            ax.axvline(row_i - 0.5, color='black', linewidth=1.5)
        prev_cat = cat

    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("Cosine similarity of label distributions\n"
                 "(1.0 = identical surface mix)", fontsize=9)

    ax.set_title("Pairwise Label Distribution Similarity Between Scenes\n"
                 "(Do scenes that look similar in t-SNE share the same surface types?)",
                 fontsize=11, fontweight='bold', pad=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_tsne_vs_label_sim(scene_ids, scene_to_cat, sim_matrix,
                            embeddings_dict, output_path):
    """
    Scatter plot: x = label similarity, y = t-SNE distance.

    Each dot = one pair of scenes.
    Coloured by whether both scenes are the same category (within)
    or different categories (cross).

    KEY INSIGHT:
      If semantic loss works → high label similarity → low t-SNE distance
      → dots slope downward from left to right
      → baseline shows no pattern (flat / random scatter)
      → semantic model shows negative correlation
    """
    n_models = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_models,
                             figsize=(7 * n_models, 6),
                             squeeze=False)
    fig.patch.set_facecolor('white')

    N = len(scene_ids)

    # Collect all upper-triangle pairs
    pairs_sim      = []
    pairs_same_cat = []
    pair_labels    = []

    for i in range(N):
        for j in range(i + 1, N):
            pairs_sim.append(sim_matrix[i, j])
            same = (scene_to_cat.get(scene_ids[i], '?') ==
                    scene_to_cat.get(scene_ids[j], '?'))
            pairs_same_cat.append(same)
            pair_labels.append(
                f"{scene_ids[i][-7:]} ↔ {scene_ids[j][-7:]}"
            )

    pairs_sim      = np.array(pairs_sim)
    pairs_same_cat = np.array(pairs_same_cat)

    for col, (model_name, embedding) in enumerate(embeddings_dict.items()):
        ax = axes[0][col]
        ax.set_facecolor('#F8F8F8')
        ax.grid(True, linewidth=0.5, alpha=0.5, color='white')

        # t-SNE distances for the same pairs
        tsne_dist_mat = compute_tsne_dist_matrix(embedding)
        pairs_tsne = np.array([tsne_dist_mat[i, j]
                                for i in range(N) for j in range(i + 1, N)])

        # Plot within-category pairs
        w_mask = pairs_same_cat
        ax.scatter(pairs_sim[w_mask],  pairs_tsne[w_mask],
                   c='#E63946', s=80, alpha=0.85, label='Same category',
                   edgecolors='white', linewidths=0.6, zorder=3)

        # Plot cross-category pairs
        c_mask = ~pairs_same_cat
        ax.scatter(pairs_sim[c_mask], pairs_tsne[c_mask],
                   c='#457B9D', s=60, alpha=0.65, label='Different category',
                   edgecolors='white', linewidths=0.6, zorder=2)

        # Annotate each dot with scene pair names
        for k, (sim_v, tsne_v, lbl) in enumerate(
                zip(pairs_sim, pairs_tsne, pair_labels)):
            ax.annotate(lbl, (sim_v, tsne_v),
                        fontsize=5, color='#444444',
                        xytext=(3, 3), textcoords='offset points')

        # Trend line (linear regression over all pairs)
        if len(pairs_sim) >= 3:
            m, b   = np.polyfit(pairs_sim, pairs_tsne, 1)
            x_line = np.linspace(pairs_sim.min(), pairs_sim.max(), 100)
            ax.plot(x_line, m * x_line + b,
                    color='#FF9800', linewidth=2, linestyle='--',
                    label=f"Trend (slope={m:.1f})", zorder=4)

            # Pearson correlation
            corr = np.corrcoef(pairs_sim, pairs_tsne)[0, 1]
            ax.text(0.97, 0.97,
                    f"Pearson r = {corr:.3f}\n"
                    f"(negative = semantic model works)",
                    transform=ax.transAxes, fontsize=8,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                              alpha=0.9, edgecolor='#CCCCCC'))

        ax.set_xlabel("Label distribution similarity\n"
                      "(1.0 = identical surface mix)", fontsize=9)
        ax.set_ylabel("t-SNE distance between scene latents\n"
                      "(lower = model thinks they are similar)", fontsize=9)
        ax.set_title(f"{model_name}\n"
                     f"Label Similarity  vs  t-SNE Distance",
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, framealpha=0.9)

    fig.suptitle("Does shared surface composition explain t-SNE proximity?\n"
                 "If semantic loss works: similar labels → similar latents "
                 "(negative Pearson r)",
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def print_top_labels_per_scene(scene_ids, scene_to_cat,
                                distributions, top_n_print=10):
    """Print the top-N labels for each scene to the console."""
    print(f"\n{'='*65}")
    print(f"TOP {top_n_print} SEGMENT LABELS PER SCENE")
    print(f"{'='*65}")

    for sid in scene_ids:
        cat   = scene_to_cat.get(sid, 'unknown')
        d     = distributions[sid]['distribution']
        total = distributions[sid]['total']
        nc    = distributions[sid]['n_chunks']

        if not d:
            print(f"\n  {sid} [{cat}] — NO SEGMENT DATA")
            continue

        # Sort by percentage descending
        top = sorted(d.items(), key=lambda x: x[1], reverse=True)[:top_n_print]

        print(f"\n  {sid}  [{cat}]  "
              f"({total:,} labelled Gaussians across {nc} chunks)")
        print(f"  {'Label':<22} {'%':>6}   {'Count':>8}")
        print(f"  {'─'*42}")
        for lbl_idx, pct in top:
            name  = f"label_{lbl_idx}"
            count = distributions[sid]['counts'].get(lbl_idx, 0)
            print(f"  {name:<22} {pct:>6.1f}%  {count:>8,}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — SAVE JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_stats_json(scene_ids, scene_to_cat, distributions,
                    sim_matrix, output_path):
    """
    Save all statistics to a JSON file.
    Includes per-scene label percentages and pairwise similarities.
    """
    out = {
        'per_scene': {},
        'pairwise_similarity': {},
    }

    for sid in scene_ids:
        d     = distributions[sid]
        dist  = d['distribution']
        named = {f"label_{int(k)}": round(v, 2)
                 for k, v in dist.items()}
        named_sorted = dict(sorted(named.items(),
                                   key=lambda x: x[1], reverse=True))
        out['per_scene'][sid] = {
            'category':     scene_to_cat.get(sid, 'unknown'),
            'total_gaussians': d['total'],
            'n_chunks':     d['n_chunks'],
            'label_pct':    named_sorted,
        }

    N = len(scene_ids)
    for i in range(N):
        for j in range(i + 1, N):
            key = f"{scene_ids[i]} ↔ {scene_ids[j]}"
            out['pairwise_similarity'][key] = round(float(sim_matrix[i, j]), 4)

    # Sort pairwise by similarity descending
    out['pairwise_similarity'] = dict(
        sorted(out['pairwise_similarity'].items(),
               key=lambda x: x[1], reverse=True)
    )

    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_config',         required=True,
                    help='scene_config.json with scene_id and category fields')
    ap.add_argument('--dataset_root',         required=True,
                    help='Root of interior_gs dataset')
    ap.add_argument('--split',
                    default='train_grid1.0cm_chunk8x8_stride6x6',
                    help='Which split folder to look for segment.npy files')
    ap.add_argument('--embedding_baseline',   default=None,
                    help='Path to embedding_expA_baseline.npy from t-SNE run')
    ap.add_argument('--embedding_semantic',   default=None,
                    help='Path to embedding_expA_semantic.npy from t-SNE run')
    ap.add_argument('--top_n_labels',         type=int, default=20,
                    help='Number of top labels to show in heatmap')
    ap.add_argument('--top_n_print',          type=int, default=10,
                    help='Number of top labels to print per scene')
    ap.add_argument('--output_dir',           default='segment_analysis')
    return ap.parse_args()


def main():
    args      = parse_args()
    split_dir = Path(args.dataset_root) / args.split
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"Segment Label Analysis")
    print(f"{'='*65}")
    print(f"  Split:  {split_dir}")
    print(f"  Output: {args.output_dir}")

    if not split_dir.exists():
        print(f"[ERROR] Split dir not found: {split_dir}")
        return

    # ── 1. Load scene config ──────────────────────────────────────────────────
    scene_to_cat, cat_to_scenes, all_ids = load_scene_config(args.scene_config)

    # ── 2. Load segment distributions ────────────────────────────────────────
    distributions = load_all_distributions(all_ids, split_dir)

    # Remove scenes with no data
    valid_ids = [s for s in all_ids if distributions[s]['total'] > 0]
    print(f"\n  Valid scenes (have segment data): {len(valid_ids)} / {len(all_ids)}")

    # ── 3. Print top labels per scene ────────────────────────────────────────
    print_top_labels_per_scene(valid_ids, scene_to_cat,
                                distributions, args.top_n_print)

    # ── 4. Build matrices ─────────────────────────────────────────────────────
    top_labels, top_names, global_counts = find_top_labels(
        valid_ids, distributions, args.top_n_labels)

    print(f"\n  Top {args.top_n_labels} labels across all scenes:")
    for i, (lbl, name) in enumerate(zip(top_labels, top_names)):
        print(f"    {i+1:2d}. {name:<22}  "
              f"total Gaussians: {int(global_counts[lbl]):,}")

    pct_matrix = build_pct_matrix(valid_ids, distributions, top_labels)
    sim_matrix = compute_cosine_similarity(pct_matrix)

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    print(f"\nGenerating plots...")

    plot_distribution_heatmap(
        valid_ids, scene_to_cat, pct_matrix, top_names,
        os.path.join(args.output_dir, 'segment_distributions.png')
    )

    plot_similarity_matrix(
        valid_ids, scene_to_cat, sim_matrix,
        os.path.join(args.output_dir, 'label_similarity_matrix.png')
    )

    # t-SNE correlation plots (only if embeddings provided)
    embeddings_dict = {}
    for name, path in [('Baseline VAE', args.embedding_baseline),
                       ('Semantic VAE', args.embedding_semantic)]:
        if path and os.path.exists(path):
            emb = np.load(path)
            # embedding rows must match valid_ids order
            if len(emb) == len(valid_ids):
                embeddings_dict[name] = emb
                print(f"  Loaded embedding: {path}  shape={emb.shape}")
            else:
                print(f"  [WARN] {path}: rows={len(emb)} "
                      f"but valid_ids={len(valid_ids)} — skipping")

    if embeddings_dict:
        plot_tsne_vs_label_sim(
            valid_ids, scene_to_cat, sim_matrix, embeddings_dict,
            os.path.join(args.output_dir, 'tsne_vs_label_similarity.png')
        )

    # ── 6. Save JSON ──────────────────────────────────────────────────────────
    save_stats_json(
        valid_ids, scene_to_cat, distributions, sim_matrix,
        os.path.join(args.output_dir, 'segment_stats.json')
    )

    print(f"\n{'='*65}")
    print(f"DONE  —  {args.output_dir}/")
    print(f"  segment_distributions.png   ← which labels each scene has")
    print(f"  label_similarity_matrix.png ← which scenes share similar labels")
    if embeddings_dict:
        print(f"  tsne_vs_label_similarity.png ← does label sim predict t-SNE dist?")
    print(f"  segment_stats.json          ← all raw numbers")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()