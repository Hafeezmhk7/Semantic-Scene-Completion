"""
label_similarity_tsne.py
=========================
Data-driven experiment: find scenes that genuinely share segment label
compositions, then verify that the semantic model places them closer
together in latent space than the baseline does.

WHY THIS IS A BETTER EXPERIMENT THAN THE PREVIOUS ONE
───────────────────────────────────────────────────────
The previous t-SNE used 23 manually labelled scenes across 13 categories
(~1-2 scenes per category). With so few scenes per category, t-SNE cannot
produce clean clusters regardless of model quality.

This experiment avoids manual labels entirely:
  1. Scan ~100 scenes automatically
  2. Compute their segment label distributions
  3. Find groups of scenes that share similar label compositions
     → these are the scenes the semantic loss SHOULD pull together
  4. Encode with baseline and semantic models
  5. t-SNE coloured by label-similarity group
  6. Measure: are same-group scenes closer in semantic than baseline?

If semantic loss works → YES, definitively and cleanly.

OUTPUTS
────────
  groups_summary.txt          — which scenes went into which group and why
  tsne_label_groups.png       — t-SNE coloured by label similarity group
  within_group_distances.png  — bar chart: baseline vs semantic within-group dist
  results.json                — all numbers

USAGE
──────
  python label_similarity_tsne.py \\
    --dataset_root      /path/to/interior_gs \\
    --split             train_grid1.0cm_chunk8x8_stride6x6 \\
    --checkpoint_baseline  checkpoints/.../final.pth \\
    --checkpoint_semantic  checkpoints/.../final.pth \\
    --n_scan            100   \\
    --n_groups          5     \\
    --min_group_size    4     \\
    --similarity_thresh 0.70  \\
    --output_dir        label_tsne_results/
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
from gs_dataset_scenesplat import gs_dataset


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NUM_LABELS = 72

# Visually distinct colours for up to 10 groups
GROUP_COLORS = [
    '#E63946',  # red
    '#2196F3',  # blue
    '#4CAF50',  # green
    '#FF9800',  # orange
    '#9C27B0',  # purple
    '#00BCD4',  # cyan
    '#FF5722',  # deep orange
    '#8BC34A',  # light green
    '#F06292',  # pink
    '#795548',  # brown
]
UNGROUPED_COLOR = '#CCCCCC'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — SCAN SCENES AND LOAD SEGMENT DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

def scan_scene_dirs(split_dir: Path, n_scan: int) -> list:
    """
    Get up to n_scan unique SCENE IDs from the split directory.

    Chunks are named like: 0207_840167_r0_c0  or  0207_840167_0
    Scene ID is always the first two underscore parts: 0207_840167

    We strictly deduplicate so each physical scene appears ONCE.
    This ensures groups contain cross-scene similarity, NOT just
    "chunks of the same room have the same labels" (trivially true).
    """
    all_dirs  = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    scene_ids = []
    seen      = set()

    for d in all_dirs:
        parts = d.name.split('_')
        # Scene ID = first two parts: XXXX_YYYYYY
        sid = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else d.name

        if sid not in seen:
            seen.add(sid)
            scene_ids.append(sid)

        if len(scene_ids) >= n_scan:
            break

    print(f"\nFound {len(scene_ids)} unique scenes in {split_dir.name}")
    print(f"  (one entry per physical scene, chunks aggregated)")
    return scene_ids


def load_segment_distribution(scene_id: str, split_dir: Path) -> tuple:
    """
    Aggregate segment.npy across ALL chunks of a scene.
    Returns (distribution_vector [72], total_gaussians, n_chunks_found).
    distribution_vector[i] = fraction (0-1) of Gaussians with label i.
    """
    chunks = sorted([d for d in split_dir.iterdir()
                     if d.is_dir() and d.name.startswith(scene_id)])
    if not chunks:
        return None, 0, 0

    counts = np.zeros(NUM_LABELS, dtype=np.int64)
    total  = 0

    for chunk in chunks:
        seg_path = chunk / 'segment.npy'
        if not seg_path.exists():
            continue
        seg   = np.load(seg_path)
        valid = seg[(seg >= 0) & (seg < NUM_LABELS)]
        for lbl in valid:
            counts[int(lbl)] += 1
        total += len(valid)

    if total == 0:
        return None, 0, len(chunks)

    dist = counts.astype(np.float32) / total   # fraction, sums to 1
    return dist, total, len(chunks)


def load_all_distributions(scene_ids: list, split_dir: Path) -> tuple:
    """
    Load distributions for all scene_ids.
    Returns:
        valid_ids   list  scene_ids that had segment data
        dist_matrix np.ndarray [N, 72]  one row per valid scene
        totals      dict  scene_id → total Gaussian count
    """
    valid_ids   = []
    dist_rows   = []
    totals      = {}

    print(f"\nLoading segment distributions for {len(scene_ids)} scenes...")
    for sid in tqdm(scene_ids, desc="  reading segment.npy"):
        dist, total, n_chunks = load_segment_distribution(sid, split_dir)
        if dist is None or total < 1000:
            # Skip scenes with too few labelled Gaussians (unreliable)
            continue
        valid_ids.append(sid)
        dist_rows.append(dist)
        totals[sid] = total

    dist_matrix = np.stack(dist_rows)   # [N, 72]
    print(f"  Valid scenes (≥1000 labelled Gaussians): {len(valid_ids)}")
    return valid_ids, dist_matrix, totals


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FIND LABEL-SIMILARITY GROUPS
# ─────────────────────────────────────────────────────────────────────────────

def compute_cosine_similarity(dist_matrix: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows. Returns [N, N] in [0, 1]."""
    norms = np.linalg.norm(dist_matrix, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    normed = dist_matrix / norms
    return np.clip(normed @ normed.T, 0, 1)


def find_dominant_label_groups(scene_ids: list,
                                dist_matrix: np.ndarray,
                                n_groups: int,
                                min_group_size: int,
                                dominance_thresh: float = 0.25) -> list:
    """
    Group scenes by their single most dominant label.

    WHY THIS IS BETTER THAN COSINE SIMILARITY
    ───────────────────────────────────────────
    Cosine similarity on the full 72-d label vector is dominated by the
    most globally common labels (label_16, label_18 appear in almost every
    scene). This means groups end up as "scenes that all have label_16 plus
    something else" — not truly distinct groups.

    Instead, we group by WHICH label dominates each scene:
      - For each scene, find the label with the highest fraction
      - If that fraction >= dominance_thresh, the scene belongs to that group
      - All scenes dominated by the same label form one group

    This produces truly distinct groups:
      Group A: scenes where label_19 > 25%  (one surface type dominates)
      Group B: scenes where label_45 > 25%  (a completely different surface)
      Group C: scenes where label_4  > 25%
      ...

    These groups are maximally informative for the semantic loss test because:
      - The semantic loss directly operates on per-label features
      - If label_X is dominant, the semantic loss should learn a strong,
        consistent representation for label_X across all those scenes
      - If the semantic model works, same-group scenes should cluster together

    Args:
        dominance_thresh: minimum fraction for a label to be considered
                          dominant. 0.25 = label must be >25% of all Gaussians.
    """
    N = len(scene_ids)

    # Find dominant label for each scene
    scene_dominant = {}  # scene_id → (dominant_label_idx, fraction)
    for i, sid in enumerate(scene_ids):
        top_label     = int(np.argmax(dist_matrix[i]))
        top_fraction  = float(dist_matrix[i, top_label])
        if top_fraction >= dominance_thresh:
            scene_dominant[sid] = (top_label, top_fraction)

    # Group by dominant label
    label_to_scenes = {}
    for sid, (lbl, frac) in scene_dominant.items():
        if lbl not in label_to_scenes:
            label_to_scenes[lbl] = []
        label_to_scenes[lbl].append((sid, frac))

    # Sort groups by size descending, take top n_groups
    sorted_groups = sorted(label_to_scenes.items(),
                           key=lambda x: len(x[1]), reverse=True)

    groups    = []
    used_sids = set()

    for lbl, members in sorted_groups:
        if len(groups) >= n_groups:
            break
        if len(members) < min_group_size:
            continue
        # Sort members by dominance fraction descending (most extreme first)
        members_sorted = [sid for sid, _ in
                          sorted(members, key=lambda x: x[1], reverse=True)]
        groups.append(members_sorted)
        used_sids.update(members_sorted)
        mean_frac = np.mean([f for _, f in members])
        print(f"  Group {len(groups)}: {len(members)} scenes dominated by "
              f"label_{lbl}  (mean fraction={mean_frac*100:.0f}%)")

    # Note any scenes that had no dominant label
    n_no_dominant = sum(1 for sid in scene_ids if sid not in scene_dominant)
    if n_no_dominant:
        print(f"  {n_no_dominant} scenes had no dominant label (all labels <{dominance_thresh*100:.0f}%)")

    print(f"\n  Found {len(groups)} groups with ≥{min_group_size} members")
    return groups


# Keep the old cosine-similarity grouper available under a different name
def find_similarity_groups(scene_ids: list,
                            dist_matrix: np.ndarray,
                            sim_matrix: np.ndarray,
                            n_groups: int,
                            min_group_size: int,
                            similarity_thresh: float) -> list:
    """Legacy cosine-similarity grouper. Use find_dominant_label_groups instead."""
    N        = len(scene_ids)
    pool     = set(range(N))
    groups   = []

    while len(groups) < n_groups and len(pool) >= min_group_size:
        best_sim  = -1
        best_seed = (None, None)
        pool_list = sorted(pool)
        for i in pool_list:
            for j in pool_list:
                if i >= j:
                    continue
                if sim_matrix[i, j] > best_sim:
                    best_sim  = sim_matrix[i, j]
                    best_seed = (i, j)
        if best_sim < similarity_thresh:
            break
        i, j = best_seed
        group = {i, j}
        changed = True
        while changed:
            changed = False
            for k in pool - group:
                avg_sim = np.mean([sim_matrix[k, m] for m in group])
                if avg_sim >= similarity_thresh:
                    group.add(k)
                    changed = True
        if len(group) >= min_group_size:
            groups.append([scene_ids[idx] for idx in sorted(group)])
            pool -= group
        else:
            pool -= {i, j}

    print(f"\n  Found {len(groups)} groups with ≥{min_group_size} members")
    return groups


def print_group_summary(groups: list,
                        scene_ids: list,
                        dist_matrix: np.ndarray,
                        sim_matrix: np.ndarray,
                        totals: dict):
    """Print a human-readable summary of each group."""
    idx_map = {s: i for i, s in enumerate(scene_ids)}

    print(f"\n{'='*65}")
    print(f"GROUP SUMMARY")
    print(f"{'='*65}")

    for g_idx, group in enumerate(groups):
        indices   = [idx_map[s] for s in group]
        sub_sim   = sim_matrix[np.ix_(indices, indices)]
        # Mean of upper triangle (exclude diagonal)
        n         = len(indices)
        tri_vals  = [sub_sim[i, j] for i in range(n) for j in range(i+1, n)]
        mean_sim  = np.mean(tri_vals) if tri_vals else 0.0

        print(f"\nGroup {g_idx+1}  ({len(group)} scenes, "
              f"mean pairwise similarity={mean_sim:.3f})")
        print(f"  {'Scene':<20} {'Total Gaussians':>16}  Top-3 labels")
        print(f"  {'─'*60}")

        for sid in group:
            dist  = dist_matrix[idx_map[sid]]
            top3  = np.argsort(dist)[-3:][::-1]
            top3s = "  ".join([f"label_{t}({dist[t]*100:.0f}%)"
                               for t in top3 if dist[t] > 0.01])
            print(f"  {sid:<20} {totals.get(sid, 0):>16,}  {top3s}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — ENCODE SCENES
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    print(f"\nLoading model: {Path(checkpoint_path).parent.name}")
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
          f"epoch={ckpt.get('epoch','?')}  "
          f"val_L2={ckpt.get('val_l2_error', ckpt.get('final_val_l2','?'))}")
    return model, flags


@torch.no_grad()
def encode_scene(model, scene_id: str, split_dir: Path,
                 training_flags: dict, device: torch.device,
                 resol: int = 200) -> np.ndarray:
    """
    Encode ALL chunks of a scene, average their mu vectors.
    Returns mean_mu [16384] or None on failure.
    """
    chunks = sorted([d for d in split_dir.iterdir()
                     if d.is_dir() and d.name.startswith(scene_id)])
    if not chunks:
        return None

    chunk_mus = []

    for chunk_dir in chunks:
        try:
            ds = gs_dataset(
                root             = str(chunk_dir.parent),
                resol            = resol,
                random_permute   = False,
                train            = False,
                sampling_method  = 'opacity',
                max_scenes       = None,
                normalize        = training_flags['use_canonical_norm'],
                normalize_colors = True,
                target_radius    = 10.0,
                scale_norm_mode  = training_flags['scale_norm_mode'],
            )

            names = [os.path.basename(d) for d in ds.scene_dirs]
            cname = chunk_dir.name
            if cname in names:
                idx = names.index(cname)
            else:
                match = [n for n in names if n.startswith(cname)]
                if not match:
                    continue
                idx = names.index(match[0])

            feats = ds[idx]['features']
            x     = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
            _, mu, _, _, _, _ = model(x, x, x, x[:, :, :3])
            chunk_mus.append(mu.squeeze(0).reshape(-1).cpu().numpy())

        except Exception:
            continue

    if not chunk_mus:
        return None

    return np.mean(chunk_mus, axis=0)   # [16384]


@torch.no_grad()
def encode_all_scenes(model, scene_ids: list, split_dir: Path,
                      training_flags: dict, device: torch.device,
                      resol: int = 200) -> tuple:
    """
    Encode all scene_ids. Returns (encoded_ids, latent_matrix [N, 16384]).
    """
    encoded_ids = []
    latents     = []

    print(f"\n  Encoding {len(scene_ids)} scenes (aggregate chunks)...")
    for sid in tqdm(scene_ids, desc="  encoding"):
        mu = encode_scene(model, sid, split_dir, training_flags, device, resol)
        if mu is not None:
            encoded_ids.append(sid)
            latents.append(mu)
        else:
            print(f"    [SKIP] {sid}")

    if not latents:
        return [], None

    return encoded_ids, np.stack(latents)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — t-SNE
# ─────────────────────────────────────────────────────────────────────────────

def run_tsne(latents: np.ndarray, seed: int = 42,
             n_iter: int = 2000) -> np.ndarray:
    N          = latents.shape[0]
    perplexity = float(max(2, min(15, N // 3)))
    print(f"  t-SNE: {N} points, perplexity={perplexity:.0f}")
    return TSNE(n_components=2, perplexity=perplexity,
                max_iter=n_iter, init='pca',
                random_state=seed, learning_rate='auto').fit_transform(latents)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_within_group_distances(embedding: np.ndarray,
                                   scene_ids: list,
                                   groups: list) -> dict:
    """
    For each group compute mean pairwise distance between its members
    in the t-SNE embedding.
    Lower = tighter = semantic model organised the space better.
    """
    idx_map = {s: i for i, s in enumerate(scene_ids)}
    results = {}

    for g_idx, group in enumerate(groups):
        indices = [idx_map[s] for s in group if s in idx_map]
        if len(indices) < 2:
            continue
        pts   = embedding[indices]
        dists = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dists.append(np.linalg.norm(pts[i] - pts[j]))
        results[f"group_{g_idx+1}"] = {
            'scenes':    group,
            'mean_dist': float(np.mean(dists)),
            'max_dist':  float(np.max(dists)),
            'n_scenes':  len(indices),
        }

    return results


def compute_cross_group_distances(embedding: np.ndarray,
                                  scene_ids: list,
                                  groups: list) -> float:
    """
    Mean distance between scenes from DIFFERENT groups.
    We want this to be large (groups well separated).
    """
    idx_map   = {s: i for i, s in enumerate(scene_ids)}
    group_ids = []
    for g_idx, group in enumerate(groups):
        for s in group:
            if s in idx_map:
                group_ids.append((idx_map[s], g_idx))

    cross_dists = []
    for i in range(len(group_ids)):
        for j in range(i + 1, len(group_ids)):
            idx_i, g_i = group_ids[i]
            idx_j, g_j = group_ids[j]
            if g_i != g_j:
                cross_dists.append(
                    np.linalg.norm(embedding[idx_i] - embedding[idx_j])
                )

    return float(np.mean(cross_dists)) if cross_dists else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def assign_point_colors(scene_ids: list, groups: list) -> tuple:
    """
    Assign a colour and group label to every scene.
    Ungrouped scenes get UNGROUPED_COLOR and label 'ungrouped'.
    """
    color_map = {}
    label_map = {}

    for g_idx, group in enumerate(groups):
        color = GROUP_COLORS[g_idx % len(GROUP_COLORS)]
        for sid in group:
            color_map[sid] = color
            label_map[sid] = f"group_{g_idx+1}"

    for sid in scene_ids:
        if sid not in color_map:
            color_map[sid] = UNGROUPED_COLOR
            label_map[sid] = 'ungrouped'

    return color_map, label_map


def plot_tsne_panel(ax, embedding, scene_ids, groups,
                   color_map, label_map, title,
                   within_dists, cross_dist,
                   dist_matrix, scene_ids_all,
                   xlim=None, ylim=None):
    """
    Scatter plot of t-SNE embedding.
    Grouped scenes: large dots in group colour + convex hull.
    Ungrouped: small grey dots.
    xlim/ylim: shared axis limits so both panels use the same scale.
    """
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, linewidth=0.4, alpha=0.5, color='white')

    idx_map = {s: i for i, s in enumerate(scene_ids)}

    # Draw ungrouped first (background)
    ung = [s for s in scene_ids if label_map[s] == 'ungrouped']
    if ung:
        pts = embedding[[idx_map[s] for s in ung]]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=UNGROUPED_COLOR, s=40, alpha=0.4,
                   edgecolors='white', linewidths=0.3, zorder=1,
                   label=f"Ungrouped (n={len(ung)})")

    # Draw each group
    for g_idx, group in enumerate(groups):
        color   = GROUP_COLORS[g_idx % len(GROUP_COLORS)]
        members = [s for s in group if s in idx_map]
        if not members:
            continue

        indices = [idx_map[s] for s in members]
        pts     = embedding[indices]
        wd      = within_dists.get(f"group_{g_idx+1}", {}).get('mean_dist', 0)

        ax.scatter(pts[:, 0], pts[:, 1],
                   c=color, s=150, alpha=0.9,
                   edgecolors='white', linewidths=1.0, zorder=3,
                   label=f"Group {g_idx+1} (n={len(members)}, Δ={wd:.2f})")

        # Convex hull
        if len(pts) >= 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax.plot(pts[simplex, 0], pts[simplex, 1],
                            color=color, alpha=0.3, linewidth=1.5, zorder=2)
            except Exception:
                pass

        # Centroid star
        centroid = pts.mean(axis=0)
        ax.scatter(*centroid, c=color, marker='*', s=350, zorder=5,
                   edgecolors='black', linewidths=0.8)

        # Label centroid with group number
        ax.annotate(f"G{g_idx+1}", xy=centroid, fontsize=8,
                    fontweight='bold', color=color,
                    ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel("t-SNE dim 1", fontsize=8)
    ax.set_ylabel("t-SNE dim 2", fontsize=8)
    ax.tick_params(labelsize=7)

    # Apply shared axis limits if provided — essential for fair visual comparison
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Metrics box
    mean_within = np.mean([v['mean_dist'] for v in within_dists.values()]) \
                  if within_dists else 0
    sep = cross_dist / mean_within if mean_within > 0 else 0
    ax.text(0.02, 0.98,
            f"Mean within-group Δ: {mean_within:.2f}\n"
            f"Mean cross-group Δ:  {cross_dist:.2f}\n"
            f"Separation ratio:    {sep:.2f}×",
            transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.9, edgecolor='#CCCCCC'))

    ax.legend(fontsize=7, loc='lower right', framealpha=0.85,
              edgecolor='#CCCCCC')


def plot_within_group_bars(ax_baseline, ax_semantic,
                            baseline_dists, semantic_dists, groups):
    """
    Side-by-side bar chart comparing within-group distances.
    One bar per group, baseline vs semantic.
    Lower bar = tighter cluster = model did better.
    """
    group_names = [f"Group {i+1}\n({len(g)} scenes)"
                   for i, g in enumerate(groups)]
    x           = np.arange(len(groups))
    width       = 0.35

    for ax, dists, label, color in [
        (ax_baseline, baseline_dists, 'Baseline VAE', '#5B8DB8'),
        (ax_semantic, semantic_dists, 'Semantic VAE', '#E06C75'),
    ]:
        vals = [dists.get(f"group_{i+1}", {}).get('mean_dist', 0)
                for i in range(len(groups))]

        bars = ax.bar(x, vals, width=0.6, color=color, alpha=0.85,
                      edgecolor='white', linewidth=0.8)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01 * max(vals),
                        f"{val:.2f}",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(group_names, fontsize=8)
        ax.set_ylabel("Mean within-group t-SNE distance\n(lower = tighter cluster)",
                      fontsize=8)
        ax.set_title(f"{label}\nWithin-group distances per label-similarity group",
                     fontsize=10, fontweight='bold')
        ax.set_facecolor('#F5F5F5')
        ax.grid(axis='y', linewidth=0.5, alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Draw improvement arrows between the two axes is handled by caller


def make_figure(baseline_results, semantic_results,
                groups, dist_matrix, all_scene_ids,
                output_path):
    """
    3-row figure:
      Row 1: t-SNE scatter  (baseline | semantic)
      Row 2: Within-group distance bars  (baseline | semantic)
      Row 3: Improvement bar  (baseline - semantic per group)
    """
    fig = plt.figure(figsize=(16, 18))
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(3, 2,
                          height_ratios=[2.2, 1.2, 1.0],
                          hspace=0.45, wspace=0.28)

    ax_tsne_base  = fig.add_subplot(gs[0, 0])
    ax_tsne_sem   = fig.add_subplot(gs[0, 1])
    ax_bar_base   = fig.add_subplot(gs[1, 0])
    ax_bar_sem    = fig.add_subplot(gs[1, 1])
    ax_improve    = fig.add_subplot(gs[2, :])   # full width

    color_map, label_map = assign_point_colors(
        baseline_results['scene_ids'], groups)

    # ── Row 1: t-SNE scatter ──────────────────────────────────────────────────
    # Compute SHARED axis limits across both embeddings so the two panels
    # are directly comparable. Without this, different t-SNE scales make
    # clustering look better or worse than it really is.
    pad = 2.0
    all_emb = np.concatenate([baseline_results['embedding'],
                               semantic_results['embedding']], axis=0)
    shared_xlim = (all_emb[:, 0].min() - pad, all_emb[:, 0].max() + pad)
    shared_ylim = (all_emb[:, 1].min() - pad, all_emb[:, 1].max() + pad)

    for ax, res, title in [
        (ax_tsne_base, baseline_results,
         "Baseline VAE (no semantic loss)\n"
         "Colour = label-similarity group"),
        (ax_tsne_sem,  semantic_results,
         "Semantic VAE (hidden, beta=0.3)\n"
         "Colour = label-similarity group"),
    ]:
        # Re-assign colours for this model's scene_ids (may differ if some failed)
        cm, lm = assign_point_colors(res['scene_ids'], groups)
        plot_tsne_panel(
            ax, res['embedding'], res['scene_ids'], groups,
            cm, lm, title,
            res['within_dists'], res['cross_dist'],
            dist_matrix, all_scene_ids,
            xlim=shared_xlim, ylim=shared_ylim,
        )

    # ── Row 2: within-group bar charts ────────────────────────────────────────
    plot_within_group_bars(ax_bar_base, ax_bar_sem,
                           baseline_results['within_dists'],
                           semantic_results['within_dists'],
                           groups)

    # ── Row 3: improvement chart ──────────────────────────────────────────────
    ax_improve.set_facecolor('#F5F5F5')
    ax_improve.grid(axis='y', linewidth=0.5, alpha=0.6)

    group_names = [f"Group {i+1}" for i in range(len(groups))]
    x           = np.arange(len(groups))

    baseline_vals = [baseline_results['within_dists']
                     .get(f"group_{i+1}", {}).get('mean_dist', 0)
                     for i in range(len(groups))]
    semantic_vals = [semantic_results['within_dists']
                     .get(f"group_{i+1}", {}).get('mean_dist', 0)
                     for i in range(len(groups))]

    improvements = [b - s for b, s in zip(baseline_vals, semantic_vals)]
    colors_bar   = ['#4CAF50' if imp > 0 else '#E63946'
                    for imp in improvements]

    bars = ax_improve.bar(x, improvements, color=colors_bar,
                          alpha=0.85, edgecolor='white', linewidth=0.8)

    for bar, imp in zip(bars, improvements):
        sign = "↓ tighter" if imp > 0 else "↑ looser"
        ax_improve.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.005 if imp >= 0 else -0.005),
                        f"{imp:+.2f}\n{sign}",
                        ha='center',
                        va='bottom' if imp >= 0 else 'top',
                        fontsize=8, fontweight='bold')

    ax_improve.axhline(0, color='black', linewidth=1.0)
    ax_improve.set_xticks(x)
    ax_improve.set_xticklabels(group_names, fontsize=9)
    ax_improve.set_ylabel("Improvement = baseline Δ − semantic Δ\n"
                          "(positive = semantic is tighter = better)",
                          fontsize=8)
    ax_improve.set_title(
        "Per-group improvement: does semantic model produce tighter clusters "
        "for scenes sharing label compositions?",
        fontsize=10, fontweight='bold')
    ax_improve.spines['top'].set_visible(False)
    ax_improve.spines['right'].set_visible(False)

    fig.suptitle(
        "Label-Similarity Guided t-SNE\n"
        "Groups = scenes selected by shared segment label composition  "
        "→  ground truth for what semantic loss should organise",
        fontsize=12, fontweight='bold', y=1.01)

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root',        required=True)
    ap.add_argument('--split',
                    default='train_grid1.0cm_chunk8x8_stride6x6')
    ap.add_argument('--checkpoint_baseline', required=True)
    ap.add_argument('--checkpoint_semantic', required=True)
    ap.add_argument('--n_scan',              type=int, default=100,
                    help='How many scenes to scan for label distributions')
    ap.add_argument('--n_groups',            type=int, default=5,
                    help='How many label-similarity groups to find')
    ap.add_argument('--min_group_size',      type=int, default=4,
                    help='Minimum scenes per group')
    ap.add_argument('--dominance_thresh',    type=float, default=0.25,
                    help='Min fraction for a label to be dominant in a scene (0.25 = label must be >25%% of all Gaussians)')
    ap.add_argument('--tsne_seed',           type=int, default=42)
    ap.add_argument('--tsne_n_iter',         type=int, default=2000)
    ap.add_argument('--resol',               type=int, default=200)
    ap.add_argument('--output_dir',          default='label_tsne_results')
    return ap.parse_args()


def main():
    args      = parse_args()
    device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    split_dir = Path(args.dataset_root) / args.split
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"Label-Similarity Guided t-SNE")
    print(f"{'='*65}")
    print(f"  Split:      {split_dir}")
    print(f"  Scan N:     {args.n_scan} scenes")
    print(f"  Groups:     {args.n_groups} × min {args.min_group_size} scenes")
    print(f"  Dominance:  label must be >{args.dominance_thresh*100:.0f}%% of Gaussians to count as dominant")
    print(f"  Output:     {args.output_dir}")

    # ── 1. Scan scenes ────────────────────────────────────────────────────────
    scene_ids = scan_scene_dirs(split_dir, args.n_scan)

    # ── 2. Load segment distributions ─────────────────────────────────────────
    valid_ids, dist_matrix, totals = load_all_distributions(scene_ids, split_dir)

    # ── 3. Find dominant-label groups ─────────────────────────────────────────
    # Compute cosine similarity matrix (still used for print_group_summary)
    sim_matrix = compute_cosine_similarity(dist_matrix)

    print(f"\nFinding dominant-label groups...")
    print(f"  Dominance threshold: label must be >{args.dominance_thresh*100:.0f}% of scene Gaussians")
    groups = find_dominant_label_groups(
        valid_ids, dist_matrix,
        args.n_groups, args.min_group_size, args.dominance_thresh)

    if not groups:
        print("[ERROR] No groups found. Try lowering --dominance_thresh (e.g. 0.15)")
        return

    print_group_summary(groups, valid_ids, dist_matrix, sim_matrix, totals)

    # Select scenes to encode: all group members + up to 15 ungrouped
    grouped_ids = [s for g in groups for s in g]
    ungrouped   = [s for s in valid_ids if s not in set(grouped_ids)][:15]
    encode_ids  = grouped_ids + ungrouped

    print(f"\nScenes to encode:")
    print(f"  Grouped:   {len(grouped_ids)} across {len(groups)} groups")
    print(f"  Ungrouped: {len(ungrouped)} background scenes")
    print(f"  Total:     {len(encode_ids)}")

    # ── 4. Encode with both models ────────────────────────────────────────────
    model_results = {}

    for key, ckpt_path in [('baseline', args.checkpoint_baseline),
                            ('semantic', args.checkpoint_semantic)]:
        print(f"\n{'─'*50}")
        print(f"Encoding with {key.upper()} model...")
        model, flags = load_model(ckpt_path, device)

        enc_ids, latents = encode_all_scenes(
            model, encode_ids, split_dir, flags, device, args.resol)

        del model
        torch.cuda.empty_cache()

        if latents is None:
            print(f"  [ERROR] No scenes encoded for {key}")
            continue

        # ── 5. t-SNE ──────────────────────────────────────────────────────────
        print(f"\n  Running t-SNE...")
        embedding = run_tsne(latents, args.tsne_seed, args.tsne_n_iter)

        # ── 6. Metrics ────────────────────────────────────────────────────────
        within = compute_within_group_distances(embedding, enc_ids, groups)
        cross  = compute_cross_group_distances(embedding, enc_ids, groups)

        mean_w = np.mean([v['mean_dist'] for v in within.values()]) \
                 if within else 0
        sep    = cross / mean_w if mean_w > 0 else 0

        print(f"\n  Results for {key}:")
        print(f"  {'Group':<12} {'Within Δ':>10}  Scenes")
        print(f"  {'─'*40}")
        for gname, gdata in within.items():
            print(f"  {gname:<12} {gdata['mean_dist']:>10.3f}  "
                  f"{', '.join(s[-7:] for s in gdata['scenes'][:3])}…")
        print(f"  {'─'*40}")
        print(f"  Mean within-group Δ:  {mean_w:.3f}")
        print(f"  Mean cross-group Δ:   {cross:.3f}")
        print(f"  Separation ratio:     {sep:.2f}×")

        # Save arrays
        np.save(f"{args.output_dir}/latents_{key}.npy",   latents)
        np.save(f"{args.output_dir}/embedding_{key}.npy", embedding)

        model_results[key] = {
            'scene_ids':    enc_ids,
            'embedding':    embedding,
            'within_dists': within,
            'cross_dist':   cross,
            'separation':   sep,
        }

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    if 'baseline' in model_results and 'semantic' in model_results:
        make_figure(
            model_results['baseline'],
            model_results['semantic'],
            groups, dist_matrix, valid_ids,
            os.path.join(args.output_dir, 'tsne_label_groups.png'),
        )

        # Summary comparison
        b_mean = np.mean([v['mean_dist']
                          for v in model_results['baseline']['within_dists'].values()])
        s_mean = np.mean([v['mean_dist']
                          for v in model_results['semantic']['within_dists'].values()])
        improvement_pct = (b_mean - s_mean) / b_mean * 100 if b_mean > 0 else 0

        print(f"\n{'='*65}")
        print(f"FINAL COMPARISON")
        print(f"{'='*65}")
        print(f"  Baseline mean within-group Δ:  {b_mean:.3f}")
        print(f"  Semantic mean within-group Δ:  {s_mean:.3f}")
        if improvement_pct > 0:
            print(f"  Semantic is {improvement_pct:.1f}% tighter ✓")
            print(f"  → Semantic loss IS pulling label-similar scenes together")
        else:
            print(f"  Semantic is {abs(improvement_pct):.1f}% LOOSER")
            print(f"  → Semantic loss is NOT helping at this beta value")
        print(f"{'='*65}")

    # ── 8. Save JSON ──────────────────────────────────────────────────────────
    out = {
        'groups': [
            {'group_id': i+1, 'scenes': g,
             'size': len(g)}
            for i, g in enumerate(groups)
        ],
        'n_ungrouped_background': len(ungrouped),
    }
    for key in ['baseline', 'semantic']:
        if key in model_results:
            r = model_results[key]
            out[key] = {
                'mean_within_group_dist': float(
                    np.mean([v['mean_dist'] for v in r['within_dists'].values()])),
                'mean_cross_group_dist':  float(r['cross_dist']),
                'separation_ratio':       float(r['separation']),
                'per_group': {k: {'mean_dist': v['mean_dist'],
                                  'n_scenes':  v['n_scenes']}
                              for k, v in r['within_dists'].items()},
            }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {args.output_dir}/results.json")


if __name__ == '__main__':
    main()