# SiT transport/path.py  (Ma et al., ECCV 2024)
# https://github.com/willisma/SiT  —  MIT License
# Kept verbatim from the original repository.

import torch as th
import numpy as np


def expand_t_like_x(t, x):
    """Reshape time t to be broadcastable against x.
    t : [B,]          →  [B, 1, 1, ...]  matching ndim of x
    """
    dims = [1] * (len(x.size()) - 1)
    return t.view(t.size(0), *dims)


# ============================================================================
# Coupling plans
# ============================================================================

class ICPlan:
    """Linear (independent coupling) path — the standard rectified flow."""

    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma

    # ── Path coefficients ────────────────────────────────────────────────────

    def compute_alpha_t(self, t):
        """Coefficient of x1 (data) in x_t = alpha_t * x1 + sigma_t * x0."""
        return t, 1

    def compute_sigma_t(self, t):
        """Coefficient of x0 (noise) in x_t."""
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t):
        return 1 / t

    # ── SDE drift / diffusion ────────────────────────────────────────────────

    def compute_drift(self, x, t):
        t           = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift       = alpha_ratio * x
        diffusion   = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        t = expand_t_like_x(t, x)
        choices = {
            "constant":              norm,
            "SBDM":                  norm * self.compute_drift(x, t)[1],
            "sigma":                 norm * self.compute_sigma_t(t)[0],
            "linear":                norm * (1 - t),
            "decreasing":            0.25 * (norm * th.cos(np.pi * t) + 1) ** 2,
            "inccreasing-decreasing": norm * th.sin(np.pi * t) ** 2,
        }
        try:
            return choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form '{form}' not implemented.")

    # ── Conversion between model output types ────────────────────────────────

    def get_score_from_velocity(self, velocity, x, t):
        t              = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var   = sigma_t ** 2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - x) / var
        return score

    def get_noise_from_velocity(self, velocity, x, t):
        t              = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var   = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - x) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        t              = expand_t_like_x(t, x)
        drift, var     = self.compute_drift(x, t)
        return var * score - drift

    # ── Forward path ─────────────────────────────────────────────────────────

    def compute_mu_t(self, t, x0, x1):
        t              = expand_t_like_x(t, x1)
        alpha_t, _     = self.compute_alpha_t(t)
        sigma_t, _     = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        return self.compute_mu_t(t, x0, x1)

    def compute_ut(self, t, x0, x1, xt):
        t              = expand_t_like_x(t, x1)
        _, d_alpha_t   = self.compute_alpha_t(t)
        _, d_sigma_t   = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(self, t, x0, x1):
        """Return (t, x_t, velocity_target) for a training step."""
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut


# ============================================================================
# Variance-preserving path  (DDPM-style)
# ============================================================================

class VPCPlan(ICPlan):
    """VP (variance-preserving) path."""

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_mean_coeff = lambda t: (
            -0.25 * ((1 - t) ** 2) * (sigma_max - sigma_min)
            - 0.5  * (1 - t)       *  sigma_min
        )
        self.d_log_mean_coeff = lambda t: (
            0.5 * (1 - t) * (sigma_max - sigma_min) + 0.5 * sigma_min
        )

    def compute_alpha_t(self, t):
        alpha_t   = th.exp(self.log_mean_coeff(t))
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        p_sigma_t = 2 * self.log_mean_coeff(t)
        sigma_t   = th.sqrt(1 - th.exp(p_sigma_t))
        d_sigma_t = th.exp(p_sigma_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * sigma_t)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        return self.d_log_mean_coeff(t)

    def compute_drift(self, x, t):
        t      = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2


# ============================================================================
# Generalised VP (cosine schedule)
# ============================================================================

class GVPCPlan(ICPlan):
    """Generalised VP path with cosine schedule."""

    def __init__(self, sigma: float = 0.0):
        super().__init__(sigma)

    def compute_alpha_t(self, t):
        alpha_t   = th.sin(t * np.pi / 2)
        d_alpha_t = np.pi / 2 * th.cos(t * np.pi / 2)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        sigma_t   = th.cos(t * np.pi / 2)
        d_sigma_t = -np.pi / 2 * th.sin(t * np.pi / 2)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        return np.pi / (2 * th.tan(t * np.pi / 2))