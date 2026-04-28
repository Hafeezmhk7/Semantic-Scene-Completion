# SiT transport/transport.py  (Ma et al., ECCV 2024)
# https://github.com/willisma/SiT  —  MIT License
# Kept verbatim from the original repository.

import torch as th
import numpy as np
import enum

from . import path
from .utils import mean_flat
from .integrators import ode, sde


class ModelType(enum.Enum):
    NOISE    = enum.auto()   # model predicts epsilon
    SCORE    = enum.auto()   # model predicts ∇ log p(x)
    VELOCITY = enum.auto()   # model predicts v(x)  ← default for rectified flow


class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP    = enum.auto()
    VP     = enum.auto()


class WeightType(enum.Enum):
    NONE       = enum.auto()
    VELOCITY   = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    """
    Wraps a coupling path and provides training_losses() and get_drift() for
    use with any velocity/noise/score model backbone.
    """

    def __init__(self, *, model_type, path_type, loss_type, train_eps, sample_eps):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP:    path.GVPCPlan,
            PathType.VP:     path.VPCPlan,
        }
        self.loss_type    = loss_type
        self.model_type   = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps    = train_eps
        self.sample_eps   = sample_eps

    # ── Prior ────────────────────────────────────────────────────────────────

    def prior_logp(self, z: th.Tensor) -> th.Tensor:
        """Log-probability under the N(0, I) prior."""
        shape = th.tensor(z.size())
        N     = th.prod(shape[1:])
        return th.vmap(lambda x: -N / 2.0 * np.log(2 * np.pi) - th.sum(x ** 2) / 2.0)(z)

    # ── Epsilon boundaries ────────────────────────────────────────────────────

    def check_interval(
        self, train_eps, sample_eps, *,
        diffusion_form="SBDM", sde=False, reverse=False, eval=False, last_step_size=0.0,
    ):
        t0, t1 = 0, 1
        eps    = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [path.VPCPlan]:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        elif (
            type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]
            and (self.model_type != ModelType.VELOCITY or sde)
        ):
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        if reverse:
            t0, t1 = 1 - t0, 1 - t1
        return t0, t1

    # ── Training ─────────────────────────────────────────────────────────────

    def sample(self, x1: th.Tensor):
        """Sample noise x0 and time t for a training step."""
        x0       = th.randn_like(x1)
        t0, t1   = self.check_interval(self.train_eps, self.sample_eps)
        t        = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t        = t.to(x1)
        return t, x0, x1

    def training_losses(self, model, x1: th.Tensor, model_kwargs=None):
        """
        Compute flow matching training loss.

        model        : callable with signature model(x_t, t, **model_kwargs) → velocity
        x1           : [B, ...] clean data (target)
        model_kwargs : extra keyword arguments forwarded to model (e.g. z_s_clean)

        Returns dict with keys 'loss' [B,] and 'pred' [B, ...].
        """
        if model_kwargs is None:
            model_kwargs = {}

        t, x0, x1      = self.sample(x1)
        t, xt, ut       = self.path_sampler.plan(t, x0, x1)
        model_output    = model(xt, t, **model_kwargs)

        B, *_, C = xt.shape
        assert model_output.size() == xt.size(), (
            f"Model output {model_output.size()} must match x_t {xt.size()}"
        )

        terms = {"pred": model_output}
        if self.model_type == ModelType.VELOCITY:
            terms["loss"] = mean_flat(((model_output - ut) ** 2))
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _   = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type == WeightType.VELOCITY:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type == WeightType.LIKELIHOOD:
                weight = drift_var / (sigma_t ** 2)
            else:
                weight = 1
            if self.model_type == ModelType.NOISE:
                terms["loss"] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms["loss"] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

        return terms

    # ── Inference ────────────────────────────────────────────────────────────

    def get_drift(self):
        """Return the drift function for ODE/SDE sampling."""
        def velocity_ode(x, t, model, **kw):
            return model(x, t, **kw)

        def noise_ode(x, t, model, **kw):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _            = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            score                 = model(x, t, **kw) / -sigma_t
            return -drift_mean + drift_var * score

        def score_ode(x, t, model, **kw):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            return -drift_mean + drift_var * model(x, t, **kw)

        drift_fn = {
            ModelType.VELOCITY: velocity_ode,
            ModelType.NOISE:    noise_ode,
            ModelType.SCORE:    score_ode,
        }[self.model_type]

        def body_fn(x, t, model, **kw):
            out = drift_fn(x, t, model, **kw)
            assert out.shape == x.shape
            return out

        return body_fn

    def get_score(self):
        """Return score function derived from model output type."""
        if self.model_type == ModelType.NOISE:
            return lambda x, t, model, **kw: (
                model(x, t, **kw) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
            )
        elif self.model_type == ModelType.SCORE:
            return lambda x, t, model, **kw: model(x, t, **kw)
        else:  # VELOCITY
            return lambda x, t, model, **kw: (
                self.path_sampler.get_score_from_velocity(model(x, t, **kw), x, t)
            )


# ============================================================================
# Sampler
# ============================================================================

class Sampler:
    """
    High-level sampler wrapping Transport for ODE and SDE sampling.
    Used in sample_stage2.py.
    """

    def __init__(self, transport: Transport):
        self.transport = transport
        self.drift     = transport.get_drift()
        self.score     = transport.get_score()

    def __get_sde_diffusion_and_drift(self, *, diffusion_form="SBDM", diffusion_norm=1.0):
        def diff_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm
            )
        sde_drift     = lambda x, t, model, **kw: self.drift(x, t, model, **kw) + diff_fn(x, t) * self.score(x, t, model, **kw)
        sde_diffusion = diff_fn
        return sde_drift, sde_diffusion

    def __get_last_step(self, sde_drift, *, last_step, last_step_size):
        if last_step is None:
            return lambda x, t, model, **kw: x
        elif last_step == "Mean":
            return lambda x, t, model, **kw: x + sde_drift(x, t, model, **kw) * last_step_size
        elif last_step == "Euler":
            return lambda x, t, model, **kw: x + self.drift(x, t, model, **kw) * last_step_size
        elif last_step == "Tweedie":
            alpha  = self.transport.path_sampler.compute_alpha_t
            sigma  = self.transport.path_sampler.compute_sigma_t
            return lambda x, t, model, **kw: (
                x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **kw)
            )
        else:
            raise NotImplementedError(f"last_step='{last_step}' not implemented.")

    def sample_ode(self, *, sampling_method="dopri5", num_steps=50, atol=1e-6, rtol=1e-3, reverse=False):
        """Return a sampling function for the probability-flow ODE."""
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            sde=False, eval=True, reverse=reverse, last_step_size=0.0,
        )
        _ode = ode(
            drift=self.drift, t0=t0, t1=t1,
            sampler_type=sampling_method, num_steps=num_steps, atol=atol, rtol=rtol,
        )
        return _ode.sample

    def sample_sde(
        self, *, sampling_method="Euler", diffusion_form="SBDM", diffusion_norm=1.0,
        last_step="Mean", last_step_size=0.04, num_steps=250,
    ):
        """Return a sampling function for the SDE."""
        if last_step is None:
            last_step_size = 0.0
        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form, diffusion_norm=diffusion_norm
        )
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            diffusion_form=diffusion_form, sde=True, eval=True,
            reverse=False, last_step_size=last_step_size,
        )
        _sde     = sde(sde_drift, sde_diffusion, t0=t0, t1=t1, num_steps=num_steps, sampler_type=sampling_method)
        last_fn  = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, model, **model_kwargs):
            xs  = _sde.sample(init, model, **model_kwargs)
            ts  = th.ones(init.size(0), device=init.device) * t1
            x   = last_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)
            assert len(xs) == num_steps
            return xs

        return _sample