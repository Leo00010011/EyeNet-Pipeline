"""ImageNet preprocessing of the 128x128 uint8 eye crop.

Pure function, no I/O.
"""

from __future__ import annotations

import numpy as np
import torch

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_eye_crop(image: np.ndarray) -> torch.Tensor:
    """Convert a (128,128,3) uint8 RGB eye crop into a (3,128,128) float32 tensor,
    scaled to [0,1] then normalized with ImageNet mean/std per channel.

    Raises ValueError if image is not (128,128,3) uint8.
    """
    if image.shape != (128, 128, 3) or image.dtype != np.uint8:
        raise ValueError(f"expected (128,128,3) uint8, got {image.shape} {image.dtype}")
    x = image.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())
