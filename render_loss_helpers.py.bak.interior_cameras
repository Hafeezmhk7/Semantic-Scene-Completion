"""render_loss_helpers.py
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
    """Return 'gsplat', 'inria', or None depending on which is importable."""
    global _BACKEND_CACHE
    if _BACKEND_CACHE is not None:
        return _BACKEND_CACHE
    try:
        import gsplat  # noqa: F401
        _BACKEND_CACHE = 'gsplat'
        return 'gsplat'
    except ImportError:
        pass
    try:
        import diff_gaussian_rasterization  # noqa: F401
        _BACKEND_CACHE = 'inria'
        return 'inria'
    except ImportError:
        pass
    _BACKEND_CACHE = 'none'
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
        theta = float(torch.rand((), device='cpu').item()) * 2.0 * math.pi
        phi = (float(torch.rand((), device='cpu').item()) - 0.5) * (2.0 / 3.0 * math.pi)
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
            render_mode='RGB', packed=False,
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
    def _make_proj_matrix(fov_deg, znear=0.01, zfar=100.0, device='cuda', dtype=torch.float32):
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
        if backend == 'gsplat':
            self.renderer = _GSplatRenderer(image_size, fov_deg)
        elif backend == 'inria':
            self.renderer = _InriaRenderer(image_size, fov_deg)
        else:
            raise RuntimeError(
                "No differentiable Gaussian rasteriser found. Install one of:\n"
                "    pip install gsplat                       # preferred\n"
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
                pred_np = (pred_imgs[c].permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype('uint8')
                targ_np = (targ_imgs[c].permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype('uint8')
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
