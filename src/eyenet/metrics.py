"""Metrics for gaze estimation (distinct from loss functions)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from eyenet.losses import angular_error_degrees


def angular_variance(vectors: torch.Tensor) -> torch.Tensor:
    """(N,3) -> scalar float32. Mean angle from the mean vector to each vector, degrees.

    Captures directional spread in the space that matters (angles), not component variance.
    Useful as a reference baseline to detect mean collapse in predictions.
    """
    if vectors.ndim != 2 or vectors.shape[-1] != 3:
        raise ValueError(f"expected (N, 3) tensor, got {tuple(vectors.shape)}")
    if vectors.shape[0] < 2:
        return torch.tensor(0.0, device=vectors.device, dtype=vectors.dtype)

    # Compute mean direction
    mean_vector = vectors.mean(dim=0)  # (3,)
    mean_vector = F.normalize(mean_vector, p=2, dim=-1, eps=1e-8)  # (3,)

    # Expand mean to match batch size for angular_error_degrees
    mean_expanded = mean_vector.unsqueeze(0).expand(vectors.shape[0], -1)

    # Compute angular error between each vector and the mean
    errors = angular_error_degrees(vectors, mean_expanded)  # (N,)

    return errors.mean()
