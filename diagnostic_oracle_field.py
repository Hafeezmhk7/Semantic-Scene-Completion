"""
ORACLE WITHIN-BLOCK FIELD DIAGNOSTIC
====================================
Question this answers (before we build any new decoder):
    Inside one Hilbert block (the ~20 Gaussians one token decodes), is the per-Gaussian
    COLOUR RESIDUAL (colour minus block mean) and the SURFACE NORMAL (thin covariance axis)
    a *recoverable spatial field* -- i.e. a function of the Gaussian's within-block position
    that GENERALISES to unseen points in the same block -- or is it effectively noise?

Why it is decisive:
    A position-keyed / field decoder (triplane, hash-grid, or our Fourier head) with a
    perfect per-block code can, at best, reproduce the part of the within-block signal that
    is a function of position. We measure that ceiling directly with per-block cross-validated
    ridge regression of the signal on a basis of the within-block offset, swept over the
    basis frequency. We report it on the SAME blocks the model decodes (the dataset's
    scaffold_token_ids) and, as an upper bound, on coarser contiguous Hilbert blocks.

Reading the result:
    1) VARIANCE SPLIT tells you how much is even at stake: if within-block variance is a small
       fraction of total, solving it barely moves the metric. If it is large, it is the game.
    2) ORACLE CV-R2 (held-out points) is the ceiling. High (>~0.5) => structured field =>
       a field decoder is worth building, and the frequency where R2 saturates tells you the
       resolution to target. Low (<~0.25) even at high frequency => the fine detail is white
       noise relative to position => no decoder recovers it => characterise the ceiling instead.
    3) The TRAIN-R2 (no CV) column is the in-sample upper bound; a big gap to CV-R2 is exactly
       the overfitting that broke the colour head, made explicit.

This script TRAINS NOTHING and needs no GPU. It loads scenes exactly as the training pipeline
does (same opacity selection, same Hilbert sort + frame radius, same normalisation), so the
numbers reflect the real data the decoder sees.
"""

import argparse, os, sys
import numpy as np


# --------------------------------------------------------------------------------------
# Geometry: surface normal = thin (smallest-scale) axis of Sigma = R diag(s^2) R^T.
# R built with the SAME (w,x,y,z) 3DGS convention as the model's _build_R_from_quat.
# --------------------------------------------------------------------------------------
def build_R_from_quat_np(q):
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], axis=-1).reshape(-1, 3, 3)
    return R


def normals_from_scale_quat(scale, quat):
    """Return (normal [N,3] unit, aniso_mid [N]) where aniso_mid = s_mid/s_min measures how
    well-defined the thin axis is (large => clear normal)."""
    R = build_R_from_quat_np(quat)
    k = np.argmin(scale, axis=1)                       # index of smallest scale = thin axis
    normal = R[np.arange(len(R)), :, k]                # that COLUMN of R is the eigenvector
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12)
    s_sorted = np.sort(scale, axis=1)                  # ascending: s_min, s_mid, s_max
    aniso_mid = s_sorted[:, 1] / (s_sorted[:, 0] + 1e-12)
    return normal, aniso_mid


# --------------------------------------------------------------------------------------
# Bases of the (normalised) within-block 3D offset. Constant is added separately.
# --------------------------------------------------------------------------------------
def poly_features(o, degree):
    x, y, z = o[:, 0], o[:, 1], o[:, 2]
    feats = []
    if degree >= 1:
        feats += [x, y, z]
    if degree >= 2:
        feats += [x*x, y*y, z*z, x*y, x*z, y*z]
    if degree >= 3:
        feats += [x**3, y**3, z**3, x*x*y, x*x*z, y*y*x, y*y*z, z*z*x, z*z*y, x*y*z]
    return np.stack(feats, axis=-1) if feats else np.zeros((len(o), 0), np.float64)


class RFF:
    """Random Fourier features of the offset (fixed B). Matches the head we'd build."""
    def __init__(self, n_freqs, sigma, seed=0):
        self.B = np.random.default_rng(seed).normal(0.0, sigma, size=(3, n_freqs))

    def __call__(self, o):
        proj = 2.0 * np.pi * (o @ self.B)
        return np.concatenate([np.sin(proj), np.cos(proj)], axis=-1)


def make_design(o_norm, cfg):
    if cfg['kind'] == 'poly':
        feats = poly_features(o_norm, cfg['degree'])
    else:
        feats = cfg['rff'](o_norm)
    ones = np.ones((len(o_norm), 1), np.float64)
    return np.concatenate([ones, feats], axis=1)


# --------------------------------------------------------------------------------------
# Per-block ridge fit. Returns pooled sums so we can aggregate a single global R2.
# --------------------------------------------------------------------------------------
def ridge_fit(X, Y, lam):
    p = X.shape[1]
    A = X.T @ X + lam * np.eye(p)
    return np.linalg.solve(A, X.T @ Y)            # [p, c]


def block_sums(design, Y, folds, lam, rng, want_angles=False, normal_target=False):
    """Cross-validated and in-sample sums for one block.
    Returns dict with cv_res, cv_tot, tr_res, tr_tot (pooled over channels), and optional
    held-out angular errors if normal_target."""
    n = design.shape[0]
    out = {'cv_res': 0.0, 'cv_tot': 0.0, 'tr_res': 0.0, 'tr_tot': 0.0, 'angles': []}

    # in-sample (upper bound)
    W = ridge_fit(design, Y, lam)
    pred = design @ W
    out['tr_res'] = float(np.sum((Y - pred) ** 2))
    out['tr_tot'] = float(np.sum((Y - Y.mean(0, keepdims=True)) ** 2))

    # cross-validated (the real ceiling)
    k = min(folds, n)
    if k >= 2:
        idx = rng.permutation(n)
        for fold in np.array_split(idx, k):
            test = fold
            train = np.setdiff1d(idx, test, assume_unique=False)
            if len(train) < 2 or len(test) < 1:
                continue
            W = ridge_fit(design[train], Y[train], lam)
            pr = design[test] @ W
            mu = Y[train].mean(0, keepdims=True)
            out['cv_res'] += float(np.sum((Y[test] - pr) ** 2))
            out['cv_tot'] += float(np.sum((Y[test] - mu) ** 2))
            if want_angles and normal_target:
                pn = pr / (np.linalg.norm(pr, axis=1, keepdims=True) + 1e-12)
                cos = np.abs(np.sum(pn * Y[test], axis=1)).clip(0, 1)   # sign-invariant
                out['angles'].append(np.degrees(np.arccos(cos)))
    return out


def sign_align(normals, idxs):
    """Flip each block's normals into the hemisphere of the block-mean direction so the
    field is continuous (resolves the antipodal ambiguity locally)."""
    out = normals.copy()
    for ix in idxs:
        if len(ix) == 0:
            continue
        m = out[ix].mean(0)
        m = m / (np.linalg.norm(m) + 1e-12)
        flip = (out[ix] @ m) < 0
        out[ix[flip]] *= -1.0
    return out


# --------------------------------------------------------------------------------------
# Variance decomposition (between-block vs within-block) for a [N,c] signal.
# --------------------------------------------------------------------------------------
def variance_split(Y, block_idxs):
    g_mean = Y.mean(0, keepdims=True)
    total = float(np.sum((Y - g_mean) ** 2))
    between = 0.0
    within = 0.0
    for ix in block_idxs:
        if len(ix) == 0:
            continue
        bm = Y[ix].mean(0, keepdims=True)
        between += len(ix) * float(np.sum((bm - g_mean) ** 2))
        within += float(np.sum((Y[ix] - bm) ** 2))
    return between, within, total


# --------------------------------------------------------------------------------------
# Block partitions
# --------------------------------------------------------------------------------------
def blocks_from_scaffold(token_ids):
    """Exact partition the decoder uses (one block per token id)."""
    order = np.argsort(token_ids, kind='stable')
    sorted_ids = token_ids[order]
    boundaries = np.where(np.diff(sorted_ids) != 0)[0] + 1
    groups = np.split(order, boundaries)
    return [g for g in groups if len(g) > 0]


def blocks_contiguous(n_points, group_size):
    """Coarser upper-bound partition: contiguous chunks of the (Hilbert-ordered) points."""
    idx = np.arange(n_points)
    return [idx[i:i + group_size] for i in range(0, n_points, group_size)]


# --------------------------------------------------------------------------------------
# Core analysis over a set of blocks
# --------------------------------------------------------------------------------------
def analyze(label, scenes, block_lists, cfgs, args, rng):
    print(f"\n{'='*78}\nPARTITION: {label}\n{'='*78}")

    # ---- variance split (headline) ----
    col_b = col_w = col_t = 0.0
    nrm_b = nrm_w = nrm_t = 0.0
    n_blocks = 0
    n_pts = 0
    for (color, normal, _off, _rms), idxs in zip(scenes, block_lists):
        b, w, t = variance_split(color, idxs)
        col_b += b; col_w += w; col_t += t
        naligned = sign_align(normal, idxs)
        b, w, t = variance_split(naligned, idxs)
        nrm_b += b; nrm_w += w; nrm_t += t
        n_blocks += sum(1 for ix in idxs if len(ix) >= args.min_pts)
        n_pts += sum(len(ix) for ix in idxs)

    print(f"  blocks used (>= {args.min_pts} pts): {n_blocks:,}   points: {n_pts:,}")
    print("\n  VARIANCE SPLIT  (fraction of total per-Gaussian variance)")
    print(f"    {'signal':10s} {'between-block':>14s} {'within-block':>14s}")
    print(f"    {'colour':10s} {col_b/max(col_t,1e-9):>13.1%} {col_w/max(col_t,1e-9):>13.1%}")
    print(f"    {'normal':10s} {nrm_b/max(nrm_t,1e-9):>13.1%} {nrm_w/max(nrm_t,1e-9):>13.1%}")
    print("    (between-block = already reachable by a per-token constant; within-block = the")
    print("     part a field decoder must supply. If within-block is small, little is at stake.)")

    # ---- oracle CV-R2 sweep ----
    print("\n  ORACLE  (per-block ridge, held-out points; TRAIN-R2 is the in-sample upper bound)")
    header = (f"    {'basis':16s} {'colour CV-R2':>13s} {'colour train-R2':>16s} "
              f"{'normal CV-R2':>13s} {'normal train-R2':>16s} {'normal med.ang':>15s}")
    print(header)

    for cfg in cfgs:
        c_cvr = c_cvt = c_trr = c_trt = 0.0
        n_cvr = n_cvt = n_trr = n_trt = 0.0
        ang = []
        for (color, normal, off, rms), idxs in zip(scenes, block_lists):
            naligned = sign_align(normal, idxs)
            for ix in idxs:
                if len(ix) < args.min_pts:
                    continue
                o = off[ix] / (rms[ix][0] + 1e-9)          # block-normalised offset
                design = make_design(o, cfg)
                # colour residual target (block-mean removed; constant in design handles it)
                s = block_sums(design, color[ix].astype(np.float64), args.folds, args.ridge, rng)
                c_cvr += s['cv_res']; c_cvt += s['cv_tot']; c_trr += s['tr_res']; c_trt += s['tr_tot']
                # normal target
                s = block_sums(design, naligned[ix].astype(np.float64), args.folds, args.ridge,
                               rng, want_angles=True, normal_target=True)
                n_cvr += s['cv_res']; n_cvt += s['cv_tot']; n_trr += s['tr_res']; n_trt += s['tr_tot']
                ang.extend(s['angles'])
        c_cv = 1 - c_cvr / max(c_cvt, 1e-9)
        c_tr = 1 - c_trr / max(c_trt, 1e-9)
        n_cv = 1 - n_cvr / max(n_cvt, 1e-9)
        n_tr = 1 - n_trr / max(n_trt, 1e-9)
        med = float(np.median(np.concatenate(ang))) if ang else float('nan')
        print(f"    {cfg['name']:16s} {c_cv:>13.3f} {c_tr:>16.3f} "
              f"{n_cv:>13.3f} {n_tr:>16.3f} {med:>14.1f}°")


# --------------------------------------------------------------------------------------
def load_scenes(args):
    sys.path.insert(0, args.repo_root)
    from gs_dataset_scenesplat import gs_dataset
    gs_dataset.TARGET_POINTS = args.num_gaussians     # match the run's Gaussian count

    ds = gs_dataset(
        root=args.chunks_dir, max_scenes=args.num_scenes, skip_scenes=args.skip_scenes,
        normalize=True, normalize_colors=True, use_chunk_norm_factor=True,
        target_radius=args.order_frame_radius, scale_norm_mode='linear',
        color_residual=True, position_scaffold=True, scaffold_mode='hilbert_block',
        morton_order=True, order_curve='hilbert', order_frame_radius=args.order_frame_radius,
        preload=True, disable_semantics=True)

    scenes = []           # list of (color[N,3], normal[N,3], offset[N,3], rms[N,1])
    scaffold_blocks = []  # list of [list of index arrays] per scene
    for i in range(len(ds)):
        d = ds[i]
        feats = d['features']
        pos = feats[:, 4:7]
        color = feats[:, 7:10].astype(np.float64)        # scene-mean-centred residual
        scale = feats[:, 11:14]
        quat = feats[:, 14:18]
        normal, _aniso = normals_from_scale_quat(scale, quat)
        offset = d['position_offsets'].astype(np.float64)  # pos - anchor (model's offset)
        token_ids = d['scaffold_token_ids']
        idxs = blocks_from_scaffold(token_ids)
        # per-block RMS radius (broadcast to each point) for offset normalisation
        rms = np.ones((len(pos), 1), np.float64)
        for ix in idxs:
            if len(ix):
                rms[ix] = np.sqrt(np.mean(np.sum(offset[ix] ** 2, axis=1)) + 1e-12)
        scenes.append((color, normal.astype(np.float64), offset, rms))
        scaffold_blocks.append(idxs)
    return scenes, scaffold_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo_root', default='.')
    ap.add_argument('--chunks_dir', default='/home/yli7/scratch/datasets/gaussian_world/'
                    'preprocessed/interior_gs/train_grid1.0cm_chunk8x8_stride6x6')
    ap.add_argument('--num_scenes', type=int, default=30)
    ap.add_argument('--skip_scenes', type=int, default=3800,
                    help="Skip the first N sorted chunks -> measure on HELD-OUT chunks "
                         "(the ones the model must generalise to). Set 0 to use train chunks.")
    ap.add_argument('--num_gaussians', type=int, default=10000)
    ap.add_argument('--order_frame_radius', type=float, default=10.0)
    ap.add_argument('--min_pts', type=int, default=8, help="Skip blocks smaller than this.")
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--ridge', type=float, default=1e-3)
    ap.add_argument('--coarse_group', type=int, default=80,
                    help="Coarser contiguous block size for the upper-bound partition "
                         "(more points/block -> can probe higher frequency). 0 to skip.")
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # frequency sweep: low -> high. RFF sigma is at unit (block-normalised) scale.
    cfgs = [
        {'name': 'linear(deg1)',  'kind': 'poly', 'degree': 1},
        {'name': 'quadratic(deg2)','kind': 'poly', 'degree': 2},
        {'name': 'fourier K6 s2', 'kind': 'rff', 'rff': RFF(6, 2.0, args.seed)},
        {'name': 'fourier K10 s4','kind': 'rff', 'rff': RFF(10, 4.0, args.seed + 1)},
    ]

    print("Loading scenes (faithful pipeline: opacity select + Hilbert sort + norm)...")
    scenes, scaffold_blocks = load_scenes(args)
    print(f"Loaded {len(scenes)} scenes. quat convention (w,x,y,z) matches the model.")
    print("Colour is scene-mean-centred (color_residual=True); ratios/R2 are unaffected.")

    # (1) the REAL partition the decoder uses
    analyze("scaffold tokens (EXACT decoder blocks, ~20 Gaussians each)",
            scenes, scaffold_blocks, cfgs, args, rng)

    # (2) coarser upper bound: contiguous Hilbert blocks with more points each
    if args.coarse_group and args.coarse_group > 0:
        coarse_blocks = [blocks_contiguous(len(s[0]), args.coarse_group) for s in scenes]
        analyze(f"contiguous Hilbert blocks of {args.coarse_group} (UPPER BOUND: more samples)",
                scenes, coarse_blocks, cfgs, args, rng)

    print("\nHOW TO READ:")
    print("  within-block variance small  -> little is at stake; the metric floor is mostly")
    print("                                  block-mean error, not within-block detail.")
    print("  CV-R2 high (>~0.5) & rising with frequency -> structured field -> BUILD the field")
    print("                                  decoder; saturation frequency = resolution to target.")
    print("  CV-R2 low (<~0.25) at all frequencies, train-R2 high -> position-independent noise")
    print("                                  (the colour-head overfit). Characterise the ceiling.")


if __name__ == '__main__':
    main()