#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_anisotropy.py
======================
Answer one question before spending effort on rotation: CAN orientation matter
for these Gaussians at all? A Gaussian's orientation only affects its covariance
when the Gaussian is anisotropic (its scale axes differ). If most Gaussians are
near-isotropic (s_max / s_min ~ 1), rotation is geometrically almost irrelevant
and a gauge-invariant covariance loss cannot move it -- so effort belongs on
colour / generalisation instead. If they are strongly anisotropic, orientation
carries real signal and the covariance objective (and yaw augmentation) is
justified.

This script loads scale.npy from a sample of scene directories under a dataset
root, computes the per-Gaussian sorted scale-axis ratios, and prints percentiles
and a text histogram. It is intentionally dependency-light (numpy only) and does
NOT touch the training code.

Usage
-----
  python diagnose_anisotropy.py --root /path/to/dataset_split [--n_scenes 50]
                                [--opacity_topk 10000] [--max_points 200000]

  # grid chunks:
  python diagnose_anisotropy.py \
    --root /home/yli7/scratch/datasets/gaussian_world/preprocessed/interior_gs/train_grid1.0cm_chunk8x8_stride6x6

Notes
-----
* Ratios are computed on the RAW scale.npy values (the 3DGS scales). Normalisation
  multiplies all three axes by the same scalar, so it does not change the ratios --
  the anisotropy you measure here is exactly what the model must represent.
* --opacity_topk mirrors the training pipeline: if opacity.npy is present, only the
  top-k most opaque Gaussians per scene are considered (the ones actually kept).
"""

import os
import argparse
import glob
import numpy as np


def _scene_dirs(root, n_scenes):
    dirs = sorted(d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d))
    if not dirs:
        # maybe root itself is a single scene
        if os.path.exists(os.path.join(root, 'scale.npy')):
            return [root]
        raise SystemExit(f"No scene directories (with scale.npy) found under {root}")
    return dirs[:n_scenes]


def _text_hist_log(values, lo_exp=0, hi_exp=7, width=50):
    """Histogram of log10(values) over decades [10^lo_exp, 10^hi_exp]. Percentages are
    out of the GRAND total (so they sum to 100, including the under/over bins)."""
    v = np.asarray(values, dtype=np.float64)
    total = max(len(v), 1)
    logv = np.log10(np.clip(v, 1e-12, None))
    edges = np.arange(lo_exp, hi_exp + 1)
    lines = []
    under = int((logv < lo_exp).sum())
    if under:
        frac = under / total
        lines.append(f"  [< 10^{lo_exp}]         {'#'*int(round(frac*width)):<{width}} "
                     f"{100*frac:5.1f}%  ({under})")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        cnt = int(((logv >= lo) & (logv < hi)).sum())
        frac = cnt / total
        bar = '#' * int(round(frac * width))
        lines.append(f"  [10^{lo} , 10^{hi})   {bar:<{width}} {100*frac:5.1f}%  ({cnt})")
    over = int((logv >= hi_exp).sum())
    if over:
        frac = over / total
        lines.append(f"  [>= 10^{hi_exp}]        {'#'*int(round(frac*width)):<{width}} "
                     f"{100*frac:5.1f}%  ({over})")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Dataset split dir containing scene subdirs')
    ap.add_argument('--n_scenes', type=int, default=50)
    ap.add_argument('--opacity_topk', type=int, default=10000,
                    help='Keep only the top-k most opaque Gaussians per scene (0 = all).')
    ap.add_argument('--max_points', type=int, default=400000,
                    help='Global cap on pooled Gaussians (subsample for speed).')
    args = ap.parse_args()

    dirs = _scene_dirs(args.root, args.n_scenes)
    print(f"Scanning {len(dirs)} scene(s) under:\n  {args.root}\n")

    r_maxmin, r_maxmid = [], []
    n_used = 0
    for d in dirs:
        sp = os.path.join(d, 'scale.npy')
        if not os.path.exists(sp):
            continue
        scale = np.load(sp).astype(np.float64)
        if scale.ndim != 2 or scale.shape[1] != 3:
            continue
        if args.opacity_topk and os.path.exists(os.path.join(d, 'opacity.npy')):
            op = np.load(os.path.join(d, 'opacity.npy')).reshape(-1)
            if len(op) == len(scale) and len(scale) > args.opacity_topk:
                keep = np.argsort(op)[-args.opacity_topk:]
                scale = scale[keep]
        s = np.sort(np.abs(scale), axis=1)            # ascending: [s_min, s_mid, s_max]
        s = np.clip(s, 1e-8, None)
        r_maxmin.append(s[:, 2] / s[:, 0])
        r_maxmid.append(s[:, 2] / s[:, 1])
        n_used += len(scale)

    if not r_maxmin:
        raise SystemExit("No usable scale.npy arrays found.")

    r_maxmin = np.concatenate(r_maxmin)
    r_maxmid = np.concatenate(r_maxmid)
    if len(r_maxmin) > args.max_points:
        sel = np.random.default_rng(0).choice(len(r_maxmin), args.max_points, replace=False)
        r_maxmin, r_maxmid = r_maxmin[sel], r_maxmid[sel]

    pct = [5, 25, 50, 75, 90, 95, 99]
    print(f"Gaussians pooled: {n_used:,} (showing {len(r_maxmin):,})\n")
    print("Anisotropy ratio  s_max / s_min   (1.0 = isotropic sphere), log10 bins:")
    qs = np.percentile(r_maxmin, pct)
    print("  percentiles: " + "  ".join(f"p{p}={q:.2f}" for p, q in zip(pct, qs)))
    print(f"  mean={r_maxmin.mean():.2f}  median={np.median(r_maxmin):.2f}")
    print(_text_hist_log(r_maxmin, lo_exp=0, hi_exp=7))
    print()
    print("Ratio  s_max / s_mid   (flatness of the largest axis), log10 bins:")
    qs2 = np.percentile(r_maxmid, pct)
    print("  percentiles: " + "  ".join(f"p{p}={q:.2f}" for p, q in zip(pct, qs2)))
    print(f"  mean={r_maxmid.mean():.2f}  median={np.median(r_maxmid):.2f}")
    print(_text_hist_log(r_maxmid, lo_exp=0, hi_exp=4))
    print()
    frac_iso = float((r_maxmin < 1.5).mean())
    frac_aniso = float((r_maxmin > 3.0).mean())
    print("Interpretation:")
    print(f"  near-isotropic (ratio < 1.5): {100*frac_iso:5.1f}% of Gaussians")
    print(f"  strongly anisotropic (>3.0) : {100*frac_aniso:5.1f}% of Gaussians")
    if frac_iso > 0.7:
        print("  -> Mostly isotropic: orientation is largely irrelevant. A gauge-invariant\n"
              "     covariance loss cannot move rotation; focus on colour + generalisation.")
    elif frac_aniso > 0.3:
        print("  -> Strongly anisotropic: orientation carries real signal. The covariance\n"
              "     objective and yaw augmentation are well justified; consider raising\n"
              "     --geom_shape_weight so the shape term gets meaningful gradient.")
    else:
        print("  -> Mixed: orientation matters for a substantial minority. The covariance\n"
              "     loss is reasonable; watch CovBures, not raw Rot, to judge it.")


if __name__ == '__main__':
    main()