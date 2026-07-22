"""Angular loss and metric over unit gaze vectors.

Objective is arccos of the normalized dot product, in radians. The clamp is
mandatory: arccos' gradient diverges at cos = +-1, which is reachable when a
prediction is (or rounds to) exact, and would poison every weight with NaN.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

EPS = 1e-7


def _check(pred, target):
    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}; "
            "both must be (B, 3)"
        )
    if pred.ndim != 2 or pred.shape[-1] != 3:
        raise ValueError(f"expected (B, 3) tensors, got {tuple(pred.shape)}")


def _cos(pred, target):
    p = F.normalize(pred.float(), p=2, dim=-1, eps=1e-8)
    t = F.normalize(target.float(), p=2, dim=-1, eps=1e-8)
    return (p * t).sum(dim=-1).clamp(-1.0 + EPS, 1.0 - EPS)


def angular_loss(pred, target):
    """(B,3), (B,3) -> scalar float32. Mean arccos(clamped normalized dot), radians."""
    _check(pred, target)
    return torch.arccos(_cos(pred, target)).mean()


def angular_error_degrees(pred, target):
    """(B,3), (B,3) -> (B,) float32. Per-sample angle, degrees."""
    _check(pred, target)
    return torch.arccos(_cos(pred, target)) * (180.0 / math.pi)


def cosine_loss(pred, target):
    """(B,3), (B,3) -> scalar float32. Mean (1 - normalized dot). Range [0, 2].

    No EPS clamp: unlike angular_loss this never calls arccos, so its gradient
    (-t on the normalized dot) is finite everywhere. The clamp in _cos exists
    only to tame arccos' divergence and is deliberately not reused here.
    """
    _check(pred, target)
    p = F.normalize(pred.float(), p=2, dim=-1, eps=1e-8)
    t = F.normalize(target.float(), p=2, dim=-1, eps=1e-8)
    return (1.0 - (p * t).sum(dim=-1)).mean()


_LOSSES = {"angular": angular_loss, "cosine": cosine_loss}


def get_loss(name):
    """Resolve a config loss name to its callable. Single source of truth."""
    try:
        return _LOSSES[name]
    except KeyError:
        raise ValueError(f"unknown loss {name!r}; valid: {sorted(_LOSSES)}") from None
