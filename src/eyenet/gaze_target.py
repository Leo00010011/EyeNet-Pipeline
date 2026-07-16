"""Spherical (theta, phi) -> 3D unit gaze vector, MPIIGaze convention.

Pure function, no I/O.
"""

from __future__ import annotations

import numpy as np
import torch

from eyenet.losses import EPS


def spherical_to_unit(theta, phi) -> np.ndarray:
    """Convert roll-removed spherical gaze (EveDataset's g_tobii convention)
    to a 3D unit vector: g = [-cos(theta)*sin(phi), -sin(theta), -cos(theta)*cos(phi)].

    theta, phi: Python floats or (N,) arrays.
    Returns (3,) or (N, 3) float32 array.
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    g = np.stack([
        -np.cos(theta) * np.sin(phi),
        -np.sin(theta),
        -np.cos(theta) * np.cos(phi),
    ], axis=-1)
    return g.astype(np.float32)


def unit_to_spherical(g: torch.Tensor) -> torch.Tensor:
    """Inverse of spherical_to_unit, MPIIGaze convention.

    g: (B, 3) or (3,) unit-norm tensor. NOT normalized here -- callers pass
    either GazeResNet18 output (F.normalize'd) or a spherical_to_unit target.
    Returns (B, 2) or (2,): [theta, phi] in radians, input dtype and device.
    """
    if g.ndim > 2 or g.shape[-1] != 3:
        raise ValueError(f"expected (B, 3) or (3,), got {tuple(g.shape)}")
    # arcsin' diverges at +/-1 and a straight-down gaze reaches it in float32 --
    # same failure mode as the arccos clamp in losses.py, same EPS.
    theta = torch.asin(torch.clamp(-g[..., 1], -1.0 + EPS, 1.0 - EPS))
    phi = torch.atan2(-g[..., 0], -g[..., 2])
    return torch.stack([theta, phi], dim=-1)
