"""Spherical (theta, phi) -> 3D unit gaze vector, MPIIGaze convention.

Pure function, no I/O.
"""

from __future__ import annotations

import numpy as np


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
