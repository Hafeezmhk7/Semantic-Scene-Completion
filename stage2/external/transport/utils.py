# SiT transport/utils.py  (Ma et al., ECCV 2024)
# https://github.com/willisma/SiT  —  MIT License
# Kept verbatim from the original repository.

import torch as th


class EasyDict:
    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)


def mean_flat(x: th.Tensor) -> th.Tensor:
    """Mean over all non-batch dimensions → [B,]."""
    return th.mean(x, dim=list(range(1, len(x.size()))))


def log_state(state: dict) -> str:
    result = []
    for key, value in dict(sorted(state.items())).items():
        if "<object" in str(value) or "object at" in str(value):
            result.append(f"{key}: [{value.__class__.__name__}]")
        else:
            result.append(f"{key}: {value}")
    return "\n".join(result)