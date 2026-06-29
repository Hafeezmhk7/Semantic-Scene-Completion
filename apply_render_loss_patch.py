#!/usr/bin/env python3
"""
apply_render_loss_patch.py
==========================

Adds a differentiable-render loss option to gs_can3tok_2.py for testing whether
supervising shape via the rendered image — rather than per-Gaussian L2 on (s,q)
or on Σ — fixes the small-isotropic-Gaussian failure mode.

DIAGNOSIS THIS TEST FALSIFIES
-----------------------------
The Σ-loss test refuted the simple gauge-ambiguity story. The Σ-on-Σ Frobenius
loss IS gauge-invariant, but it still collapsed to small isotropic blobs because
the per-slot Σ target distribution across scenes is wide and its conditional
mean is approximately a small isotropic matrix. Both element-wise L2 paths --
on (s,q) AND on Σ -- find that same isotropic minimum because they're both
asking the same wrong question: "what is the per-slot shape, conditioned on
scene identity?" Per-slot shape isn't a function of scene identity.

A render loss bypasses this entirely. The renderer pools many Gaussians per
pixel, so the supervision signal is "do these Gaussians, jointly, paint the
correct picture?" rather than "is each Gaussian individually shaped right?"
Per-pixel pooling allows individual Gaussians to be anisotropic in different
ways and still produce a correct image; the model is no longer pressured into
the isotropic mean.

WHAT THIS PATCH DOES
--------------------
1. Writes render_loss_helpers.py next to gs_can3tok_2.py (one-time companion
   module containing the rasteriser wrapper, camera sampler, SSIM, and the
   RenderLossModule callable).
2. Adds CLI flags --render_loss, --render_loss_weight, --render_num_cameras,
   --render_image_size, --render_fov_deg, --render_ssim_weight.
3. Extends compute_reconstruction_loss with a use_render_loss flag that drops
   scale/rotation from the element-wise L2 term (same as exclude_scale_rotation).
4. Instantiates a single RenderLossModule before the training loop when
   --render_loss is on, with a sanity render saved to disk for visual
   verification before any real training time is spent.
5. Computes the render loss in the training loop and in evaluate_model,
   adding render_loss_weight * render_loss to the total loss.
6. Adds Render=... to the training-log line so the term is visible.
7. Records every flag in checkpoint metadata for reproducibility.
8. Appends a sentinel comment for idempotency.

RASTERISER
----------
Auto-detected at first use. gsplat is preferred; the original Inria
diff-gaussian-rasterization is supported as a fallback. If neither is
installed, the run fails fast at startup with an actionable error
("pip install gsplat").

CAMERAS
-------
Synthesised per step: placed on a sphere at 1.5 * scene_radius around each
scene's centroid, looking inward, with a tiny target jitter. No real capture
poses are required. This is appropriate for the falsifiability test (you don't
need photographs to grade per-Gaussian shape and orientation) but is a weaker
signal than rendering against real photos would give.

LOSS
----
(1 - ssim_weight) * L1 + ssim_weight * (1 - SSIM) on RGB images in [0, 1],
the 3DGS-standard recipe. ssim_weight = 0.2 by default.

USAGE
-----
    python apply_render_loss_patch.py             # default training-script path
    python apply_render_loss_patch.py PATH        # custom path

Idempotent: re-running on a patched file is a no-op. A backup of the original
is written to PATH + '.bak.render' before the patched content is written.

The companion module (render_loss_helpers.py) is written next to the training
script; the patch refuses to overwrite it if it already exists with different
content (use --force-helpers to override).
"""
from __future__ import annotations
import os, sys, shutil

SENTINEL = "# === RENDER_LOSS_PATCH_APPLIED ==="
DEFAULT_PATH = "/home/yli11/scratch-project/Hafeez_thesis/Semantic-Scene-Completion/gs_can3tok_2.py"

# ============================================================================
# EMBEDDED HELPERS CONTENT
# ============================================================================
# The full text of render_loss_helpers.py, written verbatim next to
# gs_can3tok_2.py. Keep this in sync if the helper API changes.
HELPERS_CONTENT = '''"""render_loss_helpers.py
=========================

Render-loss companion module for Can3Tok diagnostic experiments.

USED BY: gs_can3tok_2.py when --render_loss is on. Lives next to it so the
import resolves with no PYTHONPATH change.
"""
from __future__ import annotations
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F


_BACKEND_CACHE = None

def detect_rasteriser() -> Optional[str]:
    """Return \'gsplat\', \'inria\', or None depending on which is importable."""
    global _BACKEND_CACHE
    if _BACKEND_CACHE is not None:
        return _BACKEND_CACHE
    try:
        import gsplat  # noqa: F401
        _BACKEND_CACHE = \'gsplat\'
        return \'gsplat\'
    except ImportError:
        pass
    try:
        import diff_gaussian_rasterization  # noqa: F401
        _BACKEND_CACHE = \'inria\'
        return \'inria\'
    except ImportError:
        pass
    _BACKEND_CACHE = \'none\'
    return None


def _look_at_view_matrix(eye, target, up):
    """World-to-camera matrix in OpenCV convention (camera looks down +Z)."""
    forward = target - eye
    forward = forward / forward.norm().clamp_min(1e-8)
    right = torch.linalg.cross(forward, up)
    right = right / right.norm().clamp_min(1e-8)
    cam_up = torch.linalg.cross(right, forward)
    R = torch.stack([right, -cam_up, forward], dim=0)
    t = -R @ eye
    M = torch.eye(4, device=eye.device, dtype=eye.dtype)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _intrinsic_matrix(fov_deg, image_size, device, dtype):
    f = image_size / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = torch.zeros((3, 3), device=device, dtype=dtype)
    K[0, 0] = f; K[1, 1] = f
    K[0, 2] = image_size / 2.0; K[1, 2] = image_size / 2.0
    K[2, 2] = 1.0
    return K


def sample_cameras(centroid, radius, num_cameras, image_size, fov_deg, device, dtype):
    """Sample num_cameras around a scene centroid, all looking inward."""
    cam_radius = 1.5 * float(radius)
    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    viewmats = []
    for _ in range(int(num_cameras)):
        theta = float(torch.rand((), device=\'cpu\').item()) * 2.0 * math.pi
        phi = (float(torch.rand((), device=\'cpu\').item()) - 0.5) * (2.0 / 3.0 * math.pi)
        x = cam_radius * math.cos(phi) * math.cos(theta)
        y = cam_radius * math.sin(phi)
        z = cam_radius * math.cos(phi) * math.sin(theta)
        eye = centroid + torch.tensor([x, y, z], device=device, dtype=dtype)
        target = centroid + (torch.rand(3, device=device, dtype=dtype) - 0.5) * (0.1 * radius)
        viewmats.append(_look_at_view_matrix(eye, target, up))
    viewmats = torch.stack(viewmats, dim=0)
    K = _intrinsic_matrix(fov_deg, image_size, device, dtype)
    Ks = K.unsqueeze(0).expand(int(num_cameras), -1, -1).contiguous()
    return viewmats, Ks


def _gaussian_kernel_1d(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, -1).expand(channels, 1, -1)


def ssim(img1, img2, window_size=11, sigma=1.5):
    """SSIM between two image batches [B, C, H, W] in [0, 1]."""
    C = img1.shape[1]
    kernel = _gaussian_kernel_1d(window_size, sigma, C, img1.device, img1.dtype)
    pad = window_size // 2
    def _filter(x):
        x = F.conv2d(x, kernel.unsqueeze(2), padding=(0, pad), groups=C)
        x = F.conv2d(x, kernel.unsqueeze(3), padding=(pad, 0), groups=C)
        return x
    mu1, mu2 = _filter(img1), _filter(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2
    sigma1_sq = _filter(img1*img1) - mu1_sq
    sigma2_sq = _filter(img2*img2) - mu2_sq
    sigma12 = _filter(img1*img2) - mu1_mu2
    C1, C2 = 0.01**2, 0.03**2
    num = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (num / den.clamp_min(1e-8)).mean()


class _GSplatRenderer:
    def __init__(self, image_size, fov_deg, bg=(0., 0., 0.)):
        import gsplat
        self._gsplat = gsplat
        self.image_size = int(image_size)
        self.fov_deg = float(fov_deg)
        self.bg_tensor = torch.tensor(bg)

    def render(self, means, quats, scales, opacities, colors, viewmats, Ks):
        out, _, _ = self._gsplat.rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmats, Ks=Ks,
            width=self.image_size, height=self.image_size,
            render_mode=\'RGB\', packed=False,
            backgrounds=self.bg_tensor.to(means.device, means.dtype).unsqueeze(0).expand(viewmats.shape[0], -1).contiguous(),
        )
        return out.permute(0, 3, 1, 2).clamp(0.0, 1.0)


class _InriaRenderer:
    def __init__(self, image_size, fov_deg, bg=(0., 0., 0.)):
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings, GaussianRasterizer)
        self._GRS = GaussianRasterizationSettings
        self._GR = GaussianRasterizer
        self.image_size = int(image_size)
        self.fov_deg = float(fov_deg)
        self.bg = torch.tensor(bg)

    @staticmethod
    def _make_proj_matrix(fov_deg, znear=0.01, zfar=100.0, device=\'cuda\', dtype=torch.float32):
        tan = math.tan(math.radians(fov_deg) / 2.0)
        right = tan * znear; top = tan * znear
        P = torch.zeros((4, 4), device=device, dtype=dtype)
        P[0, 0] = znear / right
        P[1, 1] = znear / top
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        P[3, 2] = 1.0
        return P

    def render(self, means, quats, scales, opacities, colors, viewmats, Ks):
        device, dtype = means.device, means.dtype
        tanfov = math.tan(math.radians(self.fov_deg) / 2.0)
        P = self._make_proj_matrix(self.fov_deg, device=device, dtype=dtype)
        bg = self.bg.to(device=device, dtype=dtype)
        renders = []
        for c in range(viewmats.shape[0]):
            V = viewmats[c]
            full_proj = (P @ V).T.contiguous()
            view = V.T.contiguous()
            campos = -V[:3, :3].T @ V[:3, 3]
            settings = self._GRS(
                image_height=self.image_size, image_width=self.image_size,
                tanfovx=tanfov, tanfovy=tanfov, bg=bg, scale_modifier=1.0,
                viewmatrix=view, projmatrix=full_proj,
                sh_degree=0, campos=campos, prefiltered=False, debug=False,
            )
            rasterizer = self._GR(raster_settings=settings)
            means2D = torch.zeros_like(means, requires_grad=True)
            try:
                means2D.retain_grad()
            except Exception:
                pass
            img, _ = rasterizer(
                means3D=means, means2D=means2D, shs=None,
                colors_precomp=colors, opacities=opacities.unsqueeze(-1),
                scales=scales, rotations=quats, cov3D_precomp=None,
            )
            renders.append(img.clamp(0.0, 1.0))
        return torch.stack(renders, dim=0)


class RenderLossModule:
    """Render predicted and GT Gaussians from synthesised cameras and compute
    L1 + ssim_weight * (1 - SSIM) on the resulting images."""

    def __init__(self, num_cameras=2, image_size=128, fov_deg=60.0,
                 ssim_weight=0.2, color_residual=False,
                 max_scale=1.0, frame_radius=10.0, verbose=False):
        self.num_cameras = int(num_cameras)
        self.image_size = int(image_size)
        self.fov_deg = float(fov_deg)
        self.ssim_weight = float(ssim_weight)
        self.color_residual = bool(color_residual)
        self.max_scale = float(max_scale)
        self.frame_radius = float(frame_radius)
        self.verbose = bool(verbose)
        backend = detect_rasteriser()
        if backend == \'gsplat\':
            self.renderer = _GSplatRenderer(image_size, fov_deg)
        elif backend == \'inria\':
            self.renderer = _InriaRenderer(image_size, fov_deg)
        else:
            raise RuntimeError(
                "No differentiable Gaussian rasteriser found. Install one of:\\n"
                "    pip install gsplat                       # preferred\\n"
                "    pip install diff-gaussian-rasterization  # Inria original")
        self.backend = backend
        if self.verbose:
            print(f"  [render_loss] backend = {self.backend}, image_size = {self.image_size}, "
                  f"cameras/step = {self.num_cameras}, fov = {self.fov_deg:.0f} deg, "
                  f"ssim_weight = {self.ssim_weight}")

    def _prepare(self, params14, mean_color_per_scene):
        """Convert [G, 14] to rasteriser-ready (means, quats, scales, opacities, colors)."""
        means = params14[:, 0:3].float()
        col = params14[:, 3:6].float()
        opa = params14[:, 6].float()
        scale = params14[:, 7:10].float()
        quat = params14[:, 10:14].float()
        if self.color_residual:
            col = col + mean_color_per_scene.float().unsqueeze(0)
        col = col.clamp(0.0, 1.0)
        if (opa.min() < -0.05) or (opa.max() > 1.05):
            opa = torch.sigmoid(opa)
        else:
            opa = opa.clamp(0.0, 1.0)
        if scale.min() < 0.0:
            scale = F.softplus(scale)
        scale = scale.clamp(min=1e-4, max=self.max_scale)
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return means, quat, scale, opa, col

    def __call__(self, pred_abs, target_abs, mean_color, valid_mask=None):
        """Compute the render loss for a batch."""
        B = pred_abs.shape[0]
        device = pred_abs.device
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        n_used = 0
        for b in range(B):
            t_means = target_abs[b, :, 0:3].float()
            if valid_mask is not None:
                vm_b = valid_mask[b]
                t_means_valid = t_means[vm_b > 0]
            else:
                t_means_valid = t_means
            if t_means_valid.shape[0] < 4:
                continue
            centroid = t_means_valid.mean(dim=0)
            radius = float(((t_means_valid - centroid).norm(dim=-1).max()).clamp_min(1.0).item())
            viewmats, Ks = sample_cameras(
                centroid, radius, self.num_cameras, self.image_size, self.fov_deg,
                device, t_means.dtype)
            p_means, p_quat, p_scale, p_opa, p_col = self._prepare(pred_abs[b], mean_color[b])
            t_means_g, t_quat, t_scale, t_opa, t_col = self._prepare(target_abs[b], mean_color[b])
            if valid_mask is not None:
                vm = valid_mask[b].float()
                p_opa = p_opa * vm
                t_opa = t_opa * vm
            try:
                pred_imgs = self.renderer.render(
                    p_means, p_quat, p_scale, p_opa, p_col, viewmats, Ks)
                with torch.no_grad():
                    targ_imgs = self.renderer.render(
                        t_means_g, t_quat, t_scale, t_opa, t_col, viewmats, Ks)
            except Exception as e:
                if self.verbose:
                    print(f"  [render_loss] scene {b} skipped ({type(e).__name__}: {e})")
                continue
            l1 = (pred_imgs - targ_imgs).abs().mean()
            if self.ssim_weight > 0.0:
                s = ssim(pred_imgs, targ_imgs)
                loss_b = (1.0 - self.ssim_weight) * l1 + self.ssim_weight * (1.0 - s)
            else:
                loss_b = l1
            loss_sum = loss_sum + loss_b
            n_used += 1
        if n_used == 0:
            return loss_sum
        return loss_sum / float(n_used)

    def save_sanity_renders(self, pred_abs, target_abs, mean_color, valid_mask,
                            output_dir, num_scenes=2):
        """Render first num_scenes of the batch, save pred|GT pairs as PNGs."""
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            print("  [render_loss] sanity PNGs skipped (Pillow / numpy missing)")
            return
        os.makedirs(output_dir, exist_ok=True)
        B = pred_abs.shape[0]
        n = min(int(num_scenes), B)
        device = pred_abs.device
        for b in range(n):
            t_means = target_abs[b, :, 0:3].float()
            if valid_mask is not None:
                vm_b = valid_mask[b]
                t_means_valid = t_means[vm_b > 0]
            else:
                t_means_valid = t_means
            if t_means_valid.shape[0] < 4:
                continue
            centroid = t_means_valid.mean(dim=0)
            radius = float(((t_means_valid - centroid).norm(dim=-1).max()).clamp_min(1.0).item())
            viewmats, Ks = sample_cameras(
                centroid, radius, self.num_cameras, self.image_size, self.fov_deg,
                device, t_means.dtype)
            p_means, p_quat, p_scale, p_opa, p_col = self._prepare(pred_abs[b], mean_color[b])
            t_means_g, t_quat, t_scale, t_opa, t_col = self._prepare(target_abs[b], mean_color[b])
            if valid_mask is not None:
                vm = valid_mask[b].float()
                p_opa = p_opa * vm
                t_opa = t_opa * vm
            with torch.no_grad():
                try:
                    pred_imgs = self.renderer.render(
                        p_means, p_quat, p_scale, p_opa, p_col, viewmats, Ks)
                    targ_imgs = self.renderer.render(
                        t_means_g, t_quat, t_scale, t_opa, t_col, viewmats, Ks)
                except Exception as e:
                    print(f"  [render_loss] sanity render failed on scene {b}: {e}")
                    continue
            for c in range(self.num_cameras):
                pred_np = (pred_imgs[c].permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(\'uint8\')
                targ_np = (targ_imgs[c].permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(\'uint8\')
                combined = np.concatenate((pred_np, targ_np), axis=1)
                Image.fromarray(combined).save(os.path.join(
                    output_dir, f"scene{b:02d}_cam{c}_pred_vs_gt.png"))
        print(f"  [render_loss] sanity PNGs written to {output_dir}")


_MODULE_CACHE: Optional[RenderLossModule] = None

def get_render_module(num_cameras=2, image_size=128, fov_deg=60.0,
                      ssim_weight=0.2, color_residual=False,
                      max_scale=1.0, frame_radius=10.0, verbose=False):
    """Return the per-process RenderLossModule, creating it on first call."""
    global _MODULE_CACHE
    if _MODULE_CACHE is None:
        _MODULE_CACHE = RenderLossModule(
            num_cameras=num_cameras, image_size=image_size, fov_deg=fov_deg,
            ssim_weight=ssim_weight, color_residual=color_residual,
            max_scale=max_scale, frame_radius=frame_radius, verbose=verbose)
    return _MODULE_CACHE
'''


# ============================================================================
# PATCH BLOCKS (str_replace operations on gs_can3tok_2.py)
# ============================================================================

# ----- Block 1: add the conditional helpers import -----
B1_OLD = """from gs_ply_reconstructor import save_reconstructed_gaussians
from accelerate import Accelerator, DistributedDataParallelKwargs
"""
B1_NEW = """from gs_ply_reconstructor import save_reconstructed_gaussians
from accelerate import Accelerator, DistributedDataParallelKwargs

# Render-loss companion (written next to this file by apply_render_loss_patch.py).
# Imported defensively so the script still runs when --render_loss is OFF and
# render_loss_helpers.py is absent.
try:
    from render_loss_helpers import get_render_module as _get_render_module
    _RENDER_HELPERS_AVAILABLE = True
except Exception as _e:
    _RENDER_HELPERS_AVAILABLE = False
    _RENDER_IMPORT_ERROR = _e
"""

# ----- Block 2: extend compute_reconstruction_loss signature -----
B2_OLD = """def compute_reconstruction_loss(prediction, target, batch_size, color_weight=1.0,
                                valid_mask=None, exclude_scale_rotation=False,
                                use_sigma_loss=False, sigma_weight=1.0):
"""
B2_NEW = """def compute_reconstruction_loss(prediction, target, batch_size, color_weight=1.0,
                                valid_mask=None, exclude_scale_rotation=False,
                                use_sigma_loss=False, sigma_weight=1.0,
                                use_render_loss=False):
"""

# ----- Block 3: make exclude_scale_rotation branch ALSO fire for render_loss -----
B3_OLD = """    if exclude_scale_rotation:
        # Diagnostic path: split the loss so the third term covers ONLY opacity
        # (6:7). Scale (7:10) and rotation (10:14) get no gradient. Always go
        # through this split even when color_weight==1.0 (i.e. skip the
        # whole-tensor norm shortcut below).
"""
B3_NEW = """    if exclude_scale_rotation or use_render_loss:
        # Diagnostic path: third term covers ONLY opacity (6:7). Scale (7:10) and
        # rotation (10:14) get no gradient from THIS loss term. Under
        # --render_loss the shape gradient is supplied by the rendered-image
        # comparison computed externally in the training loop; here we simply
        # remove the element-wise (s, q) term that was driving collapse.
"""

# ----- Block 4: add render-loss CLI flags (anchor on the sigma_weight flag) -----
B4_OLD = """parser.add_argument('--sigma_weight', type=float, default=1.0,
    help='Multiplier on the Sigma Frobenius term when --sigma_loss is on. 1.0 = raw '
         'Frobenius distance summed over all (B*N) Gaussians. Increase if the early '
         'Pos/Col loss dominates and Sigma stays unfit; decrease if Sigma explodes and '
         'Pos/Col stop converging. Default 1.0 matches the magnitude of the prior L2 '
         'scale+rotation term roughly for typical normalised splat sizes.')
parser.add_argument('--scale_penalty_weight', type=float, default=0.0)
"""
B4_NEW = """parser.add_argument('--sigma_weight', type=float, default=1.0,
    help='Multiplier on the Sigma Frobenius term when --sigma_loss is on. 1.0 = raw '
         'Frobenius distance summed over all (B*N) Gaussians. Increase if the early '
         'Pos/Col loss dominates and Sigma stays unfit; decrease if Sigma explodes and '
         'Pos/Col stop converging. Default 1.0 matches the magnitude of the prior L2 '
         'scale+rotation term roughly for typical normalised splat sizes.')
parser.add_argument('--render_loss', action='store_true', default=False,
    help='DIAGNOSTIC (image-supervised shape): replace element-wise L2 on (s, q) '
         'with a differentiable-render comparison. Predicted and ground-truth '
         'Gaussians are rasterised through synthesised cameras and the resulting '
         'images are compared with L1 + ssim_weight*(1-SSIM). Position (0:3), '
         'color (3:6), opacity (6:7) still use element-wise L2; scale (7:10) and '
         'rotation (10:14) get gradient only through the rendered image. Tests '
         'whether per-pixel pooling at the renderer escapes the conditional-mean '
         'isotropic minimum that both L2-on-(s,q) and L2-on-Sigma converged to. '
         'Requires gsplat or diff-gaussian-rasterization installed.')
parser.add_argument('--render_loss_weight', type=float, default=1.0,
    help='Multiplier on the render-loss term in the total loss. 1.0 is a sensible '
         'default for an L1+SSIM image loss in [0,1] alongside the element-wise '
         'L2 terms in their current scale. Increase if Pos/Col/Opa dominate and '
         'shapes never converge; decrease if the renders dominate and Pos/Col '
         'stop improving.')
parser.add_argument('--render_num_cameras', type=int, default=2,
    help='Number of synthesised cameras rendered per scene per training step. '
         'More cameras = more informative shape gradient but linearly more '
         'rasteriser cost. 2-4 is the published sweet spot.')
parser.add_argument('--render_image_size', type=int, default=128,
    help='Square image side length for render-loss rasterisation. 128 keeps '
         'training-step cost reasonable; 256+ may help shape detail but slows '
         'training meaningfully.')
parser.add_argument('--render_fov_deg', type=float, default=60.0,
    help='Vertical field of view (degrees) for synthesised cameras.')
parser.add_argument('--render_ssim_weight', type=float, default=0.2,
    help='Weight on the (1-SSIM) term inside the render loss. Standard 3DGS '
         'recipe uses 0.2 (so the loss is 0.8 * L1 + 0.2 * (1-SSIM)).')
parser.add_argument('--scale_penalty_weight', type=float, default=0.0)
"""

# ----- Block 5: instantiate render module before training loop -----
B5_OLD = """print(f\"\\n{'='*70}\\nSTARTING TRAINING  (epoch {start_epoch} -> {args.num_epochs-1})\\n{'='*70}\\n\")

_kl_anneal_active = (args.kl_anneal_steps > 0)
"""
B5_NEW = """print(f\"\\n{'='*70}\\nSTARTING TRAINING  (epoch {start_epoch} -> {args.num_epochs-1})\\n{'='*70}\\n\")

# Render-loss module: built once, reused every train/eval step. Built lazily
# (only when --render_loss is on) so a default run never imports gsplat.
_render_module = None
if args.render_loss:
    if not _RENDER_HELPERS_AVAILABLE:
        raise RuntimeError(
            \"--render_loss requires render_loss_helpers.py next to this script. \"
            \"Run apply_render_loss_patch.py to install it. \"
            f\"Import error was: {_RENDER_IMPORT_ERROR}\")
    _render_module = _get_render_module(
        num_cameras=args.render_num_cameras,
        image_size=args.render_image_size,
        fov_deg=args.render_fov_deg,
        ssim_weight=args.render_ssim_weight,
        color_residual=args.color_residual,
        frame_radius=10.0,
        verbose=accelerator.is_main_process,
    )
    if accelerator.is_main_process:
        print(f\"  [render_loss] module ready (backend={_render_module.backend}, \"
              f\"weight={args.render_loss_weight}, sanity renders at epoch 0 batch 0)\")

_kl_anneal_active = (args.kl_anneal_steps > 0)
"""

# ----- Block 6: extend the training-loop call site (and add the external render call) -----
B6_OLD = """        recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight,
                                                 valid_mask=vmask_gpu,
                                                 exclude_scale_rotation=args.no_scale_rotation_loss,
                                                 use_sigma_loss=args.sigma_loss,
                                                 sigma_weight=args.sigma_weight)
"""
B6_NEW = """        recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight,
                                                 valid_mask=vmask_gpu,
                                                 exclude_scale_rotation=args.no_scale_rotation_loss,
                                                 use_sigma_loss=args.sigma_loss,
                                                 sigma_weight=args.sigma_weight,
                                                 use_render_loss=args.render_loss)
        # --- RENDER LOSS (external, needs absolute params + mean_color) ----------
        render_loss = torch.zeros((), device=device)
        if args.render_loss and _render_module is not None:
            _pred_abs_render = UV_gs_recover.reshape(B, -1, 14)
            # The rasteriser path is fp32; bf16 autocast can mix dtypes and crash
            # gsplat. Disable autocast strictly around the render call.
            try:
                with torch.autocast('cuda', enabled=False):
                    render_loss = _render_module(_pred_abs_render.float(),
                                                 target_abs.float(),
                                                 mean_color_gt.float(),
                                                 vmask_gpu)
                # One-shot visual sanity check at the very start of training, so
                # any rasteriser-wiring bug shows up before hours of compute go by.
                if epoch == start_epoch and i_batch == 0 and accelerator.is_main_process:
                    _sanity_dir = os.path.join(save_path, 'render_sanity', f'epoch_{epoch:04d}')
                    _render_module.save_sanity_renders(
                        _pred_abs_render.float(), target_abs.float(),
                        mean_color_gt.float(), vmask_gpu,
                        output_dir=_sanity_dir, num_scenes=min(B, 2))
            except Exception as _render_err:
                if accelerator.is_main_process and i_batch == 0:
                    print(f\"  [render_loss] training-step render failed: \"
                          f\"{type(_render_err).__name__}: {_render_err}\")
                render_loss = torch.zeros((), device=device)
"""

# ----- Block 7: add render_loss to total_loss -----
B7_OLD = """        total_loss = (recon_loss
                      + _kl_current                * KL_loss
"""
B7_NEW = """        total_loss = (recon_loss
                      + args.render_loss_weight    * render_loss
                      + _kl_current                * KL_loss
"""

# ----- Block 8: add 'render' to the per-epoch accumulator dict -----
B8_OLD = """    e = {k: 0.0 for k in [
        'loss','recon','kl','sem','color_pred','scene_sem','anchor',
        'layout','cross_recon','ortho','seg_pred','scale_pen',
        'z_s_nce','z_s_npos',
        'zs_tok_nce','zs_tok_ncats',
        'zs_lay_nce','zs_lay_ncats',
        'zs_pool_nce','zs_pool_ncats',
        'pos','col','opa','scl','rot']}
"""
B8_NEW = """    e = {k: 0.0 for k in [
        'loss','recon','kl','sem','color_pred','scene_sem','anchor',
        'layout','cross_recon','ortho','seg_pred','scale_pen',
        'z_s_nce','z_s_npos',
        'zs_tok_nce','zs_tok_ncats',
        'zs_lay_nce','zs_lay_ncats',
        'zs_pool_nce','zs_pool_ncats',
        'render',
        'pos','col','opa','scl','rot']}
"""

# ----- Block 9: accumulate render in the per-epoch dict (after 'rot') -----
B9_OLD = """        e['pos'] += ind['position']; e['col'] += ind['color']
        e['opa'] += ind['opacity'];  e['scl'] += ind['scale']
        e['rot'] += ind['rotation']
"""
B9_NEW = """        e['pos'] += ind['position']; e['col'] += ind['color']
        e['opa'] += ind['opacity'];  e['scl'] += ind['scale']
        e['rot'] += ind['rotation']
        e['render'] += float(render_loss.detach().item()) if args.render_loss else 0.0
"""

# ----- Block 10: add Render=... to the epoch print line -----
B10_OLD = """        print(f\"\\nEpoch {epoch:04d} | \"
              f\"Loss={e['loss']/nb:.4f} | \"
              f\"Recon={e['recon']/nb:.4f} | \"
"""
B10_NEW = """        print(f\"\\nEpoch {epoch:04d} | \"
              f\"Loss={e['loss']/nb:.4f} | \"
              f\"Recon={e['recon']/nb:.4f} | \"
              f\"Render={e['render']/nb:.4f} | \"
"""

# ----- Block 11: extend the eval call site -----
B11_OLD = """            recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight,
                                                     valid_mask=vmask_gpu,
                                                     exclude_scale_rotation=args.no_scale_rotation_loss,
                                                     use_sigma_loss=args.sigma_loss,
                                                     sigma_weight=args.sigma_weight)
"""
B11_NEW = """            recon_loss = compute_reconstruction_loss(pred_3d, target, B, args.color_loss_weight,
                                                     valid_mask=vmask_gpu,
                                                     exclude_scale_rotation=args.no_scale_rotation_loss,
                                                     use_sigma_loss=args.sigma_loss,
                                                     sigma_weight=args.sigma_weight,
                                                     use_render_loss=args.render_loss)
"""

# ----- Block 12: startup-summary notification when --render_loss is on -----
B12_OLD = """    if args.sigma_loss:
        print(f\"  SIGMA LOSS      : ON  weight={args.sigma_weight}  \"
              f\"(gauge-INVARIANT Frobenius on Sigma=R(q)*diag(s^2)*R(q)^T; \"
              f\"replaces element-wise L2 on (s, q))\")
    print(f\"  cross_recon={args.cross_recon_weight} ortho={args.ortho_weight}\")
"""
B12_NEW = """    if args.sigma_loss:
        print(f\"  SIGMA LOSS      : ON  weight={args.sigma_weight}  \"
              f\"(gauge-INVARIANT Frobenius on Sigma=R(q)*diag(s^2)*R(q)^T; \"
              f\"replaces element-wise L2 on (s, q))\")
    if args.render_loss:
        print(f\"  RENDER LOSS     : ON  weight={args.render_loss_weight}  \"
              f\"({args.render_num_cameras}cam@{args.render_image_size}px, \"
              f\"fov={args.render_fov_deg:.0f}deg, ssim_w={args.render_ssim_weight}; \"
              f\"replaces element-wise L2 on (s, q))\")
    print(f\"  cross_recon={args.cross_recon_weight} ortho={args.ortho_weight}\")
"""

# ----- Block 13: checkpoint metadata -----
B13_OLD = """    'voxel_snap':                 args.voxel_snap,
    'sigma_loss':                 args.sigma_loss,
    'sigma_weight':               args.sigma_weight,
}
"""
B13_NEW = """    'voxel_snap':                 args.voxel_snap,
    'sigma_loss':                 args.sigma_loss,
    'sigma_weight':               args.sigma_weight,
    'render_loss':                args.render_loss,
    'render_loss_weight':         args.render_loss_weight,
    'render_num_cameras':         args.render_num_cameras,
    'render_image_size':          args.render_image_size,
    'render_fov_deg':             args.render_fov_deg,
    'render_ssim_weight':         args.render_ssim_weight,
}
"""

# ----- Block 14: append sentinel after the previous sentinels -----
B14_OLD = """# === SIGMA_LOSS_PATCH_APPLIED ==="""
B14_NEW = """# === SIGMA_LOSS_PATCH_APPLIED ===
""" + SENTINEL


BLOCKS = [
    ("1: helpers import",                          B1_OLD,  B1_NEW),
    ("2: compute_reconstruction_loss signature",   B2_OLD,  B2_NEW),
    ("3: exclude_scale_rotation branch trigger",   B3_OLD,  B3_NEW),
    ("4: render-loss CLI flags",                   B4_OLD,  B4_NEW),
    ("5: render module instantiation",             B5_OLD,  B5_NEW),
    ("6: training-loop call site + render call",   B6_OLD,  B6_NEW),
    ("7: total_loss addition",                     B7_OLD,  B7_NEW),
    ("8: epoch accumulator dict",                  B8_OLD,  B8_NEW),
    ("9: per-epoch render accumulation",           B9_OLD,  B9_NEW),
    ("10: epoch-print Render=... line",            B10_OLD, B10_NEW),
    ("11: eval call site",                         B11_OLD, B11_NEW),
    ("12: startup summary notification",           B12_OLD, B12_NEW),
    ("13: checkpoint metadata",                    B13_OLD, B13_NEW),
    ("14: append sentinel",                        B14_OLD, B14_NEW),
]


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    force_helpers = '--force-helpers' in sys.argv
    print(f"Target file: {path}")
    if not os.path.exists(path):
        print(f"ERROR: target file not found: {path}", file=sys.stderr)
        return 1

    helpers_dir = os.path.dirname(os.path.abspath(path))
    helpers_path = os.path.join(helpers_dir, 'render_loss_helpers.py')

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if SENTINEL in content:
        print(f"Sentinel '{SENTINEL}' already present -- patch is a no-op. Exiting.")
        return 0

    # Pre-flight: verify every anchor exists exactly once.
    print("\nPre-flight: verifying anchor strings in gs_can3tok_2.py...")
    errors = []
    for name, old, _new in BLOCKS:
        n = content.count(old)
        if n != 1:
            errors.append(f"  Block {name}: anchor found {n} times (expected 1)")
        else:
            print(f"  Block {name}: anchor OK")
    if errors:
        print("\nERROR: anchor mismatch. The file does not match the expected state.\n"
              "Common cause: a previous patch already modified this region, or the\n"
              "file is not the one this patch was written against.", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        return 2

    # Helpers file: write or refuse-to-overwrite.
    if os.path.exists(helpers_path):
        with open(helpers_path, "r", encoding="utf-8") as f:
            existing = f.read()
        if existing != HELPERS_CONTENT and not force_helpers:
            print(f"\nERROR: {helpers_path} already exists with different content.\n"
                  f"Pass --force-helpers to overwrite, or remove the file first.",
                  file=sys.stderr)
            return 3
        elif existing == HELPERS_CONTENT:
            print(f"\nHelpers file already up to date: {helpers_path}")
        else:
            shutil.copy2(helpers_path, helpers_path + ".bak.render")
            with open(helpers_path, "w", encoding="utf-8") as f:
                f.write(HELPERS_CONTENT)
            print(f"\nHelpers file overwritten (backup: {helpers_path}.bak.render)")
    else:
        with open(helpers_path, "w", encoding="utf-8") as f:
            f.write(HELPERS_CONTENT)
        print(f"\nHelpers file written: {helpers_path}")

    # Apply blocks
    print("\nApplying blocks to gs_can3tok_2.py...")
    patched = content
    for name, old, new in BLOCKS:
        before = len(patched)
        patched = patched.replace(old, new, 1)
        delta = len(patched) - before
        print(f"  Block {name}: applied (+{delta} chars)")

    if SENTINEL not in patched:
        print(f"ERROR: sentinel '{SENTINEL}' missing from patched content -- aborting"
              " without writing.", file=sys.stderr)
        return 4

    backup = path + ".bak.render"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"\nBackup written: {backup}")
    else:
        print(f"\nBackup already exists, not overwriting: {backup}")

    tmp = path + ".tmp.render"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(patched)
    os.replace(tmp, path)
    print(f"Patched file written: {path}")

    print("\nDone. Verify with:")
    print(f"  grep -n 'render_loss\\|render_loss_helpers' {path} | head -20")
    print(f"  ls -la {helpers_path}")
    print("\nRun the test with --render_loss added to your usual command (or use")
    print("the can3tok_overfit_render_loss.job script).")
    print("\nA visual sanity render is saved automatically at epoch 0 batch 0 to")
    print("  CHECKPOINT_DIR/render_sanity/epoch_0000/scene*_cam*_pred_vs_gt.png")
    print("Inspect those BEFORE letting the full training run continue -- if the GT")
    print("renders look obviously broken (all black, garbage, wrong perspective), the")
    print("rasteriser convention is wrong and we need to fix the wrapper before any")
    print("real compute is spent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())