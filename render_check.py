"""
render_check.py  --  Milestone 1 for the virtual-camera rendering loss.

Goal: prove, on the cluster, that we can take the GROUND-TRUTH Gaussians of a
chunk (the exact tensors the model is trained to reconstruct), convert them to
gsplat inputs with the SAME conventions the dataset uses, place virtual cameras
around the chunk, and render images that actually look like the scene.

Why this first: the rendering LOSS renders the predicted Gaussians and the GT
Gaussians from the same virtual cameras and minimises the image difference. The
only risky parts are (a) the param -> renderable conversion (colour residual +
mean, opacity activation, linear scale, quaternion order) and (b) the camera
placement. This script isolates exactly those two things with no training in the
loop, so if the renders are wrong we debug here, cheaply.

It does NOT need the original ScanNet camera poses or photos. The GT Gaussians
are the supervision; the cameras are synthetic.

Run (from the repo root, on a GPU node, in the can3tok env):
    pip install gsplat            # JIT-compiles on first import; needs nvcc + CUDA
    python render_check.py --num_scenes 4 --skip_scenes 3800 \
        --num_gaussians 40000 --img 512 --out ./render_check_out

Then open render_check_out/sceneNN_view*.png and compare to the same chunk in
SuperSplat. If they match, the conversion + cameras are correct and we wire the
loss. If opacity looks wrong, read the printed OPACITY line (auto-detected).
"""

import argparse
import math
import os
import sys

import numpy as np
import torch


# --------------------------------------------------------------------------- #
#  small vector helpers
# --------------------------------------------------------------------------- #
def _normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)


def look_at_viewmat(eye, target, up_hint):
    """
    OpenCV-convention world->camera matrix (camera looks down +z, +x right,
    +y down), which is what gsplat expects.

    Returns a [4,4] float32 numpy array.
    """
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    f = _normalize(target - eye)                     # forward = camera +z
    up_hint = _normalize(np.asarray(up_hint, np.float64))
    # If forward is (nearly) parallel to the up hint, pick a different hint so
    # the cross products stay well-conditioned.
    if abs(float(np.dot(f, up_hint))) > 0.95:
        up_hint = np.array([1.0, 0.0, 0.0]) if abs(f[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    r = _normalize(np.cross(f, up_hint))             # right = camera +x
    u = np.cross(f, r)                               # down  = camera +y (r x u = f)
    R_wc = np.stack([r, u, f], axis=0)               # rows are camera axes in world
    t_wc = -R_wc @ eye
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R_wc
    M[:3, 3] = t_wc
    return M.astype(np.float32)


def make_cameras(centroid, radius, fov_deg, width, height, device,
                 n_ring=6, ring_elevations_deg=(15.0, 45.0), top_view=True,
                 up_axis=2, dist_mult=2.6):
    """
    Place cameras on rings around the chunk centroid, all looking at it.

    up_axis selects which world axis is treated as 'up' for elevation. We do not
    know the canonical up axis for sure, so we orbit in azimuth and use two
    elevations plus a top view; at least some views will frame the chunk well
    regardless of the true up axis.

    Returns viewmats [C,4,4], Ks [C,3,3] (both torch.float32 on device).
    """
    centroid = np.asarray(centroid, np.float64)
    up = np.zeros(3)
    up[up_axis] = 1.0
    # two in-plane axes orthogonal to 'up'
    a0 = np.zeros(3); a0[(up_axis + 1) % 3] = 1.0
    a1 = np.zeros(3); a1[(up_axis + 2) % 3] = 1.0

    dist = float(radius) * float(dist_mult)
    eyes = []
    for elev in ring_elevations_deg:
        ce = math.cos(math.radians(elev))
        se = math.sin(math.radians(elev))
        for k in range(n_ring):
            az = 2.0 * math.pi * k / n_ring
            dir_world = ce * (math.cos(az) * a0 + math.sin(az) * a1) + se * up
            eyes.append(centroid + dist * _normalize(dir_world))
    if top_view:
        eyes.append(centroid + dist * up)

    viewmats = np.stack([look_at_viewmat(e, centroid, up) for e in eyes], axis=0)

    fx = 0.5 * width / math.tan(0.5 * math.radians(fov_deg))
    fy = 0.5 * height / math.tan(0.5 * math.radians(fov_deg))
    K = np.array([[fx, 0.0, width / 2.0],
                  [0.0, fy, height / 2.0],
                  [0.0, 0.0, 1.0]], np.float32)
    Ks = np.broadcast_to(K, (viewmats.shape[0], 3, 3)).copy()

    return (torch.from_numpy(viewmats).to(device),
            torch.from_numpy(Ks).to(device))


# --------------------------------------------------------------------------- #
#  data loading  (same pattern as diagnostic_oracle_field.py)
# --------------------------------------------------------------------------- #
def load_gt_chunks(args):
    """
    Returns a list of dicts, one per scene, each with numpy arrays:
        means [N,3], rgb [N,3] in [0,1], opacity_raw [N], scales [N,3], quats [N,4]
    using the exact gs_full_params column layout the model trains on.
    """
    sys.path.insert(0, args.repo_root)
    from gs_dataset_scenesplat import gs_dataset
    gs_dataset.TARGET_POINTS = args.num_gaussians

    ds = gs_dataset(
        root=args.chunks_dir, max_scenes=args.num_scenes, skip_scenes=args.skip_scenes,
        normalize=True, normalize_colors=True, use_chunk_norm_factor=True,
        target_radius=args.order_frame_radius, scale_norm_mode='linear',
        color_residual=True, position_scaffold=True, scaffold_mode='hilbert_block',
        morton_order=True, order_curve='hilbert', order_frame_radius=args.order_frame_radius,
        preload=True, disable_semantics=True)

    out = []
    for i in range(len(ds)):
        d = ds[i]
        feats = np.asarray(d['features'])
        mean_color = np.asarray(d['mean_color'], np.float32)        # [3]
        means = feats[:, 4:7].astype(np.float32)                    # canonical-frame xyz
        color_res = feats[:, 7:10].astype(np.float32)               # scene-mean-centred RGB
        opacity_raw = feats[:, 10].astype(np.float32)               # raw (logit OR [0,1])
        scales = feats[:, 11:14].astype(np.float32)                 # linear std in canonical frame
        quats = feats[:, 14:18].astype(np.float32)                  # wxyz (scalar-first)
        rgb = np.clip(color_res + mean_color[None, :], 0.0, 1.0)    # true RGB, same as recon PLY
        out.append(dict(means=means, rgb=rgb, opacity_raw=opacity_raw,
                        scales=scales, quats=quats, mean_color=mean_color))
    return out


def resolve_opacity(opacity_raw, mode):
    """
    Map raw stored opacity to [0,1] for rendering.

    mode='auto': if values stray outside [0,1] it is a logit -> sigmoid; else use
    as-is. Returns (opacities_float32, human_readable_mode_string).
    """
    lo, hi = float(opacity_raw.min()), float(opacity_raw.max())
    if mode == 'sigmoid':
        chosen = 'sigmoid (forced)'
        op = 1.0 / (1.0 + np.exp(-opacity_raw))
    elif mode == 'raw':
        chosen = 'raw (forced)'
        op = np.clip(opacity_raw, 0.0, 1.0)
    else:  # auto
        if lo < -1e-4 or hi > 1.0 + 1e-4:
            chosen = 'sigmoid (auto: values outside [0,1] -> logit)'
            op = 1.0 / (1.0 + np.exp(-opacity_raw))
        else:
            chosen = 'raw (auto: already in [0,1])'
            op = np.clip(opacity_raw, 0.0, 1.0)
    return op.astype(np.float32), f"raw range [{lo:.3f}, {hi:.3f}] -> {chosen}"


# --------------------------------------------------------------------------- #
#  rendering
# --------------------------------------------------------------------------- #
def render(params, viewmats, Ks, width, height, device, bg=0.0):
    try:
        from gsplat import rasterization
    except Exception as ex:  # noqa
        print("\n[FATAL] could not import gsplat:", repr(ex))
        print("Install it in the can3tok env on a GPU node:")
        print("    pip install gsplat")
        print("(first import JIT-compiles CUDA kernels; needs nvcc + a CUDA torch).")
        sys.exit(1)

    means = torch.from_numpy(params['means']).to(device)
    quats = torch.from_numpy(params['quats']).to(device)
    scales = torch.from_numpy(params['scales']).to(device)
    opac = torch.from_numpy(params['opacities']).to(device)
    rgb = torch.from_numpy(params['rgb']).to(device)
    C = viewmats.shape[0]
    backgrounds = torch.full((C, 3), float(bg), device=device)

    colors, alphas, _meta = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=rgb,
        viewmats=viewmats, Ks=Ks, width=width, height=height,
        sh_degree=None, render_mode="RGB", backgrounds=backgrounds,
        rasterize_mode="classic", packed=True,
    )
    # colors: [C, H, W, 3] in roughly [0,1]
    return colors.clamp(0.0, 1.0).detach().cpu().numpy(), alphas.detach().cpu().numpy()


def save_png(img_hw3, path):
    arr = (np.clip(img_hw3, 0.0, 1.0) * 255.0).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
        return
    except Exception:
        pass
    try:
        import matplotlib.image as mpimg
        mpimg.imsave(path, arr)
        return
    except Exception:
        np.save(path + ".npy", arr)
        print(f"  (no PIL/matplotlib; saved {path}.npy instead)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo_root', default='.')
    ap.add_argument('--chunks_dir', default='/home/yli7/scratch/datasets/gaussian_world/'
                    'preprocessed/interior_gs/train_grid1.0cm_chunk8x8_stride6x6')
    ap.add_argument('--num_scenes', type=int, default=4)
    ap.add_argument('--skip_scenes', type=int, default=3800,
                    help="Skip the first N sorted chunks (use held-out chunks).")
    ap.add_argument('--num_gaussians', type=int, default=40000)
    ap.add_argument('--order_frame_radius', type=float, default=10.0)
    ap.add_argument('--img', type=int, default=512, help="Render width=height.")
    ap.add_argument('--fov', type=float, default=50.0)
    ap.add_argument('--n_ring', type=int, default=6)
    ap.add_argument('--dist_mult', type=float, default=2.6,
                    help="Camera distance = dist_mult * chunk radius.")
    ap.add_argument('--up_axis', type=int, default=2, choices=[0, 1, 2],
                    help="World up axis for elevation/top view. Try 1 if 2 looks tilted.")
    ap.add_argument('--opacity', default='auto', choices=['auto', 'sigmoid', 'raw'])
    ap.add_argument('--bg', type=float, default=0.0, help="Background grey level.")
    ap.add_argument('--radius_pct', type=float, default=90.0,
                    help="Use this percentile of centroid distance as chunk radius "
                         "(robust to a few far outlier splats).")
    ap.add_argument('--out', default='./render_check_out')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FATAL] no CUDA device. Run on a GPU node.")
        sys.exit(1)
    device = torch.device('cuda')
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.num_scenes} GT chunks from {args.chunks_dir}")
    print(f"  num_gaussians={args.num_gaussians}  skip_scenes={args.skip_scenes}")
    scenes = load_gt_chunks(args)
    print(f"Loaded {len(scenes)} scenes.\n")

    for si, p in enumerate(scenes):
        means = p['means']
        centroid = means.mean(axis=0)
        d = np.linalg.norm(means - centroid[None, :], axis=1)
        radius = float(np.percentile(d, args.radius_pct))
        bbox = means.max(0) - means.min(0)

        opac, op_msg = resolve_opacity(p['opacity_raw'], args.opacity)
        p['opacities'] = opac

        # quaternion sanity: gsplat normalises internally, but check it is finite
        qn = np.linalg.norm(p['quats'], axis=1)

        print(f"--- scene {si:02d} ---")
        print(f"  N Gaussians : {means.shape[0]:,}")
        print(f"  centroid    : ({centroid[0]:+.3f}, {centroid[1]:+.3f}, {centroid[2]:+.3f})")
        print(f"  bbox extent : ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f})  "
              f"radius(p{args.radius_pct:.0f})={radius:.3f}")
        print(f"  scale std   : min={p['scales'].min():.4f}  med={np.median(p['scales']):.4f}  "
              f"max={p['scales'].max():.4f}")
        print(f"  RGB         : min={p['rgb'].min():.3f}  mean={p['rgb'].mean():.3f}  "
              f"max={p['rgb'].max():.3f}  (mean_color={p['mean_color']})")
        print(f"  OPACITY     : {op_msg}")
        print(f"  quat |q|    : min={qn.min():.3f} max={qn.max():.3f} (gsplat re-normalises)")

        viewmats, Ks = make_cameras(
            centroid, radius, args.fov, args.img, args.img, device,
            n_ring=args.n_ring, top_view=True, up_axis=args.up_axis,
            dist_mult=args.dist_mult)
        imgs, alphas = render(p, viewmats, Ks, args.img, args.img, device, bg=args.bg)

        cover = alphas.reshape(alphas.shape[0], -1).mean(axis=1)
        for vi in range(imgs.shape[0]):
            fn = os.path.join(args.out, f"scene{si:02d}_view{vi:02d}.png")
            save_png(imgs[vi], fn)
        print(f"  wrote {imgs.shape[0]} views -> {args.out}/scene{si:02d}_view*.png")
        print(f"  view coverage (mean alpha): "
              f"min={cover.min():.3f} max={cover.max():.3f}  "
              f"{'(LOW: cameras may be too close/far or up_axis wrong)' if cover.max() < 0.05 else ''}")
        print()

    print("Done. Compare these PNGs to the same chunk in SuperSplat.")
    print("If everything is black/empty: try --dist_mult 4, a different --up_axis,")
    print("or --opacity sigmoid. If colour is right but geometry looks scrambled,")
    print("the quaternion order may differ (we'll flip wxyz<->xyzw in the loss).")


if __name__ == '__main__':
    main()