# SiT transport package  (Ma et al., ECCV 2024)
# https://github.com/willisma/SiT  —  MIT License
# Kept verbatim from the original repository.

from .transport import Transport, ModelType, WeightType, PathType, Sampler


def create_transport(
    path_type:  str   = "Linear",
    prediction: str   = "velocity",
    loss_weight: str  = None,
    train_eps:  float = None,
    sample_eps: float = None,
):
    """
    Factory for Transport objects.

    path_type   : "Linear" | "GVP" | "VP"
    prediction  : "velocity" | "noise" | "score"
    loss_weight : None | "velocity" | "likelihood"
    """
    # ── Model type ────────────────────────────────────────────────────────────
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    # ── Loss weighting ────────────────────────────────────────────────────────
    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    # ── Path type ─────────────────────────────────────────────────────────────
    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP":    PathType.GVP,
        "VP":     PathType.VP,
    }
    path_type = path_choice[path_type]

    # ── Epsilon boundaries (numerical stability) ──────────────────────────────
    if path_type in [PathType.VP]:
        train_eps  = 1e-5 if train_eps  is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        train_eps  = 1e-3 if train_eps  is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:
        # velocity + Linear/GVP: stable everywhere
        train_eps  = 0
        sample_eps = 0

    return Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )