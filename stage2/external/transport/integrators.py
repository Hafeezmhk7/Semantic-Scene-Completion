# SiT transport/integrators.py  (Ma et al., ECCV 2024)
# https://github.com/willisma/SiT  —  MIT License
# Kept verbatim except two URL-encoding artefacts fixed:
#   [th.no](http://...)_grad()  →  th.no_grad()
#   [self.t.to](http://...)     →  self.t.to

import torch as th
import numpy as np
import torch.nn as nn
from tqdm import tqdm

# torchdiffeq is only needed inside ode.sample() at inference time.
# Import lazily so training works without it installed.
try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None


class sde:
    """SDE solver (Euler-Maruyama and Heun)."""

    def __init__(self, drift, diffusion, *, t0, t1, num_steps, sampler_type):
        assert t0 < t1, "SDE sampler must run in forward time."
        self.num_timesteps = num_steps
        self.t             = th.linspace(t0, t1, num_steps)
        self.dt            = self.t[1] - self.t[0]
        self.drift         = drift
        self.diffusion     = diffusion
        self.sampler_type  = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur  = th.randn(x.size()).to(x)
        t_vec  = th.ones(x.size(0)).to(x) * t
        dw     = w_cur * th.sqrt(self.dt)
        drift  = self.drift(x, t_vec, model, **model_kwargs)
        diff   = self.diffusion(x, t_vec)
        mean_x = x + drift * self.dt
        x      = mean_x + th.sqrt(2 * diff) * dw
        return x, mean_x

    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur   = th.randn(x.size()).to(x)
        dw      = w_cur * th.sqrt(self.dt)
        t_cur   = th.ones(x.size(0)).to(x) * t
        diff    = self.diffusion(x, t_cur)
        xhat    = x + th.sqrt(2 * diff) * dw
        K1      = self.drift(xhat, t_cur, model, **model_kwargs)
        xp      = xhat + self.dt * K1
        K2      = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat

    def __forward_fn(self):
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun":  self.__Heun_step,
        }
        try:
            return sampler_dict[self.sampler_type]
        except KeyError:
            raise NotImplementedError(f"Sampler type '{self.sampler_type}' not implemented.")

    def sample(self, init, model, **model_kwargs):
        x, mean_x = init, init
        samples   = []
        sampler   = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)
        return samples


class ode:
    """ODE solver using torchdiffeq (Euler, Heun, dopri5, etc.)."""

    def __init__(self, drift, *, t0, t1, sampler_type, num_steps, atol, rtol):
        self.drift        = drift
        self.t            = th.linspace(t0, t1, num_steps)
        self.atol         = atol
        self.rtol         = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        if odeint is None:
            raise ImportError(
                "torchdiffeq is required for ode.sample(). "
                "Install it with: pip install torchdiffeq\n"
                "For inference you can use euler_sample() in sample_stage2.py instead, "
                "which has no extra dependencies."
            )
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            t_vec = (
                th.ones(x[0].size(0)).to(device) * t
                if isinstance(x, tuple)
                else th.ones(x.size(0)).to(device) * t
            )
            return self.drift(x, t_vec, model, **model_kwargs)

        t    = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        return odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)