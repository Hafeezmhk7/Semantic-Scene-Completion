"""
render_loss.py  --  Milestone 2: virtual-camera rendering loss for Can3Tok.

Renders the PREDICTED Gaussians and the GROUND-TRUTH Gaussians of each scene from
the same synthetic cameras and minimises the image difference (L1 + D-SSIM). No
real ScanNet poses or photos are needed: the GT Gaussians are the supervision and
the cameras are synthetic. Permutation-invariant by construction, and it optimises
image quality directly, which is the crispness signal that parameter-space losses
cannot give.

Conventions (verified against gs_ply_reconstructor.py + gs_dataset_scenesplat.py):
  model output / GT target are ALREADY activated:
    pos    [.,0:3]  canonical-frame xyz
    color  [.,3:6]  RGB residual (scene-mean-subtracted); true RGB = color + mean_color
    opac   [.,6]    in [0,1]            (model applied sigmoid; GT opacity.npy in [0,1])
    scale  [.,7:10] linear std, metres  (model applied exp; GT scale*scale_factor)
    quat   [.,10:14] unit, wxyz scalar-first (3DGS order; gsplat also expects wxyz)
  so the render conversion is: add mean colour, clip, pass through. No sigmoid/exp.

Gradients flow through gsplat into the predicted Gaussians; the GT render is detached.
Designed to COMPLEMENT the set loss (a fast stable parameter-space signal), not
replace it: total = set_loss + render_loss_weight * render_loss.

Cost note: each scene is one rasterisation call, so cost scales with
(scenes_per_step * views * img_size^2 * N). Keep render_max_scenes small (e.g. 8),
views few (4), and img_size modest (128) at 40k Gaussians, and ramp it in as a
fine-tuning phase once the set loss has converged.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

# Set once after the first render to avoid spamming the (loud) empty-render warning.
_RENDER_EMPTY_WARNED = False

# --------------------------------------------------------------------------- #
#  LPIPS perceptual term (optional)  --  lazily loaded, robust to missing deps
# --------------------------------------------------------------------------- #
# LPIPS (Zhang et al. 2018) is a learned perceptual metric: it compares deep
# features of two images, so it rewards getting the IMAGE right at multiple scales
# rather than per-pixel intensity. It is the standard render-space term for 3DGS
# autoencoders/generators (alongside L1 + D-SSIM). The backbone (AlexNet/VGG) is
# pretrained and FROZEN; gradient flows through the rendered image into the
# Gaussians. Weights are fetched from the internet on first use, which fails on
# offline compute nodes -- so this loader degrades gracefully (warn once, fall
# back to L1+D-SSIM) instead of crashing a training run.
_LPIPS_MODEL  = None
_LPIPS_FAILED = False

def _get_lpips(device, net='alex'):
    """Return a cached, frozen LPIPS module on `device`, or None if unavailable.
    To use on an offline cluster, pre-download the weights ONCE on a login node
    with internet: `python -c "import lpips; lpips.LPIPS(net='alex')"` (caches the
    AlexNet backbone in ~/.cache/torch/hub and the linear head in the lpips pkg)."""
    global _LPIPS_MODEL, _LPIPS_FAILED
    if _LPIPS_FAILED:
        return None
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    try:
        import lpips as _lpips_pkg
        m = _lpips_pkg.LPIPS(net=net, verbose=False).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _LPIPS_MODEL = m
        print(f"[render_loss] LPIPS({net}) loaded; perceptual term active.", flush=True)
        return m
    except Exception as e:
        _LPIPS_FAILED = True
        print(f"[render_loss WARNING] LPIPS unavailable ({type(e).__name__}: {e}). "
              f"Falling back to L1 + D-SSIM only (render_lpips_weight ignored). To enable: "
              f"`pip install lpips` and pre-download weights on a node with internet.",
              flush=True)
        return None


# --------------------------------------------------------------------------- #
#  cameras (numpy, no grad)  --  identical math to render_check.py (verified)
# --------------------------------------------------------------------------- #
def _normalize_np(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)


def _look_at_viewmat(eye, target, up_hint):
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    f = _normalize_np(target - eye)
    up_hint = _normalize_np(np.asarray(up_hint, np.float64))
    if abs(float(np.dot(f, up_hint))) > 0.95:
        up_hint = np.array([1.0, 0.0, 0.0]) if abs(f[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    r = _normalize_np(np.cross(f, up_hint))
    u = np.cross(f, r)
    R = np.stack([r, u, f], axis=0)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = -R @ eye
    return M.astype(np.float32)


def build_cameras(centroid, radius, fov_deg, img_size, device, dtype,
                  n_ring=4, ring_elevations_deg=(30.0,), top_view=False,
                  up_axis=2, dist_mult=2.6):
    """Returns viewmats [C,4,4], Ks [C,3,3] on `device` (no grad)."""
    centroid = np.asarray(centroid, np.float64)
    up = np.zeros(3); up[up_axis] = 1.0
    a0 = np.zeros(3); a0[(up_axis + 1) % 3] = 1.0
    a1 = np.zeros(3); a1[(up_axis + 2) % 3] = 1.0
    dist = max(float(radius), 1e-3) * float(dist_mult)

    eyes = []
    for elev in ring_elevations_deg:
        ce, se = math.cos(math.radians(elev)), math.sin(math.radians(elev))
        for k in range(n_ring):
            az = 2.0 * math.pi * k / n_ring
            d = ce * (math.cos(az) * a0 + math.sin(az) * a1) + se * up
            eyes.append(centroid + dist * _normalize_np(d))
    if top_view:
        eyes.append(centroid + dist * up)

    viewmats = np.stack([_look_at_viewmat(e, centroid, up) for e in eyes], axis=0)
    fx = 0.5 * img_size / math.tan(0.5 * math.radians(fov_deg))
    K = np.array([[fx, 0.0, img_size / 2.0],
                  [0.0, fx, img_size / 2.0],
                  [0.0, 0.0, 1.0]], np.float32)
    Ks = np.broadcast_to(K, (viewmats.shape[0], 3, 3)).copy()
    return (torch.from_numpy(viewmats).to(device=device, dtype=dtype),
            torch.from_numpy(Ks).to(device=device, dtype=dtype))


# --------------------------------------------------------------------------- #
#  param -> renderable  (differentiable for pred)
# --------------------------------------------------------------------------- #
def to_renderable(p14, mean_color, quat_order='wxyz',
                  opacity_clamp=(1e-3, 1.0), scale_min=1e-4):
    """
    p14: [N,14] model-output / GT params (already activated). mean_color: [3].
    Returns dict of float32 tensors for gsplat. Differentiable in p14.
    """
    p = p14.float()
    mc = mean_color.float()
    means = p[:, 0:3]
    rgb = (p[:, 3:6] + mc[None, :]).clamp(0.0, 1.0)
    opac = p[:, 6].clamp(opacity_clamp[0], opacity_clamp[1])
    scales = p[:, 7:10].clamp_min(scale_min)
    quats = p[:, 10:14]
    if quat_order == 'xyzw':          # data is wxyz; if gsplat wants xyzw, roll
        quats = quats[:, [1, 2, 3, 0]]
    quats = quats / (quats.norm(dim=1, keepdim=True) + 1e-8)
    return dict(means=means, quats=quats, scales=scales, opacities=opac, colors=rgb)


def _render(r, viewmats, Ks, img_size, bg, near_plane):
    from gsplat import rasterization
    # Do NOT pass `backgrounds` to rasterization: its expected shape differs across
    # gsplat versions (the [C,3] form trips an assertion in some builds). With no
    # background, gsplat composites over black, i.e. it returns the accumulated
    # colour and a separate alpha. We then composite any non-black background
    # ourselves, which is version-independent and fine for a pred-vs-GT loss
    # (both renders get identical treatment).
    colors, alphas, _meta = rasterization(
        means=r['means'], quats=r['quats'], scales=r['scales'],
        opacities=r['opacities'], colors=r['colors'],
        viewmats=viewmats, Ks=Ks, width=img_size, height=img_size,
        sh_degree=None, render_mode="RGB",
        near_plane=near_plane, rasterize_mode="classic", packed=True,
    )
    colors = colors.clamp(0.0, 1.0)            # [C, H, W, 3] over black
    if float(bg) != 0.0:
        colors = (colors + (1.0 - alphas) * float(bg)).clamp(0.0, 1.0)  # alphas: [C,H,W,1]
    return colors


# --------------------------------------------------------------------------- #
#  differentiable SSIM (torch only)
# --------------------------------------------------------------------------- #
def _gaussian_window(ws, sigma, channels, device, dtype):
    coords = torch.arange(ws, device=device, dtype=dtype) - (ws - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    w2d = g[:, None] * g[None, :]
    return w2d.expand(channels, 1, ws, ws).contiguous()


def ssim(img1, img2, ws=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    """img1, img2: [B, C, H, W] in [0,1]. Returns scalar mean SSIM."""
    ch = img1.shape[1]
    win = _gaussian_window(ws, sigma, ch, img1.device, img1.dtype)
    pad = ws // 2
    mu1 = F.conv2d(img1, win, padding=pad, groups=ch)
    mu2 = F.conv2d(img2, win, padding=pad, groups=ch)
    mu1_sq, mu2_sq, mu1mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(img1 * img1, win, padding=pad, groups=ch) - mu1_sq
    s2 = F.conv2d(img2 * img2, win, padding=pad, groups=ch) - mu2_sq
    s12 = F.conv2d(img1 * img2, win, padding=pad, groups=ch) - mu1mu2
    ssim_map = ((2 * mu1mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean()


# --------------------------------------------------------------------------- #
#  the loss
# --------------------------------------------------------------------------- #
def compute_render_loss(prediction, target, mean_color,
                        num_ring=4, ring_elevations_deg=(30.0,), top_view=False,
                        img_size=128, fov_deg=50.0,
                        l1_weight=1.0, ssim_weight=0.2, lpips_weight=0.0, lpips_net='alex',
                        max_scenes=8, up_axis=2, dist_mult=2.6,
                        bg=0.0, near_plane=0.01,
                        quat_order='wxyz', radius_pct=90.0,
                        opacity_clamp=(1e-3, 1.0), scale_min=1e-4,
                        generator=None, return_components=False):
    """
    prediction, target: [B, N, 14] (already-activated params). mean_color: [B, 3].
    Renders a random subset of `max_scenes` scenes each call (cost control); the
    gradient flows through the predicted render only.
    """
    B, N, _ = prediction.shape
    device = prediction.device
    nS = B if (max_scenes is None or max_scenes <= 0) else min(int(max_scenes), B)
    perm = torch.randperm(B, device=device, generator=generator)[:nS]

    l1_acc = prediction.new_zeros(())
    dssim_acc = prediction.new_zeros(())
    lpips_acc = prediction.new_zeros(())
    cnt = 0
    for b in perm.tolist():
        rp = to_renderable(prediction[b], mean_color[b], quat_order, opacity_clamp, scale_min)
        with torch.no_grad():
            rt = to_renderable(target[b], mean_color[b], quat_order, opacity_clamp, scale_min)
            centroid = rt['means'].mean(0)
            dd = (rt['means'] - centroid[None, :]).norm(dim=1)
            radius = float(torch.quantile(dd, radius_pct / 100.0))
        viewmats, Ks = build_cameras(
            centroid.detach().cpu().numpy(), radius, fov_deg, img_size,
            device, rp['means'].dtype, n_ring=num_ring,
            ring_elevations_deg=ring_elevations_deg, top_view=top_view,
            up_axis=up_axis, dist_mult=dist_mult)

        img_p = _render(rp, viewmats, Ks, img_size, bg, near_plane)        # grad
        with torch.no_grad():
            img_t = _render(rt, viewmats, Ks, img_size, bg, near_plane)    # target
            # One-time sanity check: the GT render must not be blank. If it is, the
            # render loss is ~0 and has NO effect (this is what an empty-render bug
            # looks like). Warn loudly once so it shows up in the training log.
            global _RENDER_EMPTY_WARNED
            if not _RENDER_EMPTY_WARNED:
                gt_max = float(img_t.max())
                gt_cov = float((img_t.amax(dim=-1) > 1e-3).float().mean())
                # The usual reason 40k tiny, strongly-anisotropic Gaussians render to
                # ~nothing is that each one lands SUB-PIXEL at this resolution and
                # distance, so its rasterised footprint is negligible. Compute the
                # projected splat size so the cause is visible in the training log
                # (this is the render_check diagnostic, brought in-line).
                fx = 0.5 * img_size / math.tan(0.5 * math.radians(fov_deg))
                cam_dist = max(float(radius), 1e-3) * float(dist_mult)
                large = rt['scales'].amax(dim=1)               # largest axis per Gaussian
                med_large = float(large.median())
                proj_px = (2.0 * med_large / cam_dist) * fx    # ~2-sigma diam in pixels
                frac_sub = float(((2.0 * large / cam_dist) * fx < 1.0).float().mean())
                op = rt['opacities']
                print(f"[render_loss] GT render: max={gt_max:.4g} coverage={gt_cov:.4g} "
                      f"| scene radius={radius:.4g} cam_dist={cam_dist:.4g} img={img_size} "
                      f"| opacity[min/med/max]={float(op.min()):.3f}/"
                      f"{float(op.median()):.3f}/{float(op.max()):.3f} "
                      f"| scale_large[med]={med_large:.4g} -> proj diam ~{proj_px:.2f}px, "
                      f"{100.0*frac_sub:.0f}% of splats <1px", flush=True)
                if gt_max < 1e-3 or gt_cov < 0.01:
                    if proj_px < 1.0:
                        _need = int(math.ceil(img_size * (1.5 / max(proj_px, 1e-6))))
                        print(f"[render_loss WARNING] GT render is BLANK and splats are "
                              f"SUB-PIXEL (~{proj_px:.2f}px diameter). The render loss is "
                              f"~0 and has NO effect. Fix: raise --render_img to about "
                              f"{_need} and/or lower --render_dist_mult (move the camera "
                              f"closer). This is a resolution issue, not a convention bug.",
                              flush=True)
                    else:
                        print(f"[render_loss WARNING] GT render is BLANK but splats are "
                              f"NOT sub-pixel (~{proj_px:.2f}px). Suspect a convention/"
                              f"framing issue: check the opacity range above and "
                              f"--render_up_axis / --render_quat_order. Run render_check.py "
                              f"to inspect the PNGs.", flush=True)
                else:
                    print(f"[render_loss] GT render OK; render loss is live.", flush=True)
                _RENDER_EMPTY_WARNED = True

        l1_acc = l1_acc + (img_p - img_t).abs().mean()
        if ssim_weight > 0:
            dssim_acc = dssim_acc + (1.0 - ssim(
                img_p.permute(0, 3, 1, 2), img_t.permute(0, 3, 1, 2)))
        if lpips_weight > 0:
            _lp = _get_lpips(prediction.device, lpips_net)
            if _lp is not None:
                # LPIPS expects NCHW RGB in [-1, 1]; grad flows through the pred image.
                xp = img_p.permute(0, 3, 1, 2).clamp(0, 1) * 2.0 - 1.0
                xt = (img_t.permute(0, 3, 1, 2).clamp(0, 1) * 2.0 - 1.0)
                lpips_acc = lpips_acc + _lp(xp, xt).mean()
        cnt += 1

    cnt = max(cnt, 1)
    l1_acc = l1_acc / cnt
    dssim_acc = dssim_acc / cnt
    lpips_acc = lpips_acc / cnt
    loss = l1_weight * l1_acc + ssim_weight * dssim_acc + lpips_weight * lpips_acc
    if return_components:
        return loss, {'l1': float(l1_acc.detach()), 'dssim': float(dssim_acc.detach()),
                      'lpips': float(lpips_acc.detach()), 'scenes': cnt}
    return loss