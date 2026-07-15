"""Eye-crop geometry helpers — pure functions, no H5/accessor I/O.

Mirrors EveDataset's face_crop_tools.py pattern for the eye-crop extraction
step this repo owns. EveDataset delivers 512×512 face crops; this module cuts
and resizes per-eye 128×128 patches from those crops.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Default eye-crop window size in face-crop pixel space.
# Needs empirical sizing against real eye-corner coordinate spans (same method
# EveDataset used to size its 512px face-crop window). 96px is the starting
# reference; tune if mean angular error suggests a too-tight or too-loose crop.
_EYE_WINDOW_PX = 96


def compute_eye_crop_window(
    eye_corners: np.ndarray,
    frame_shape: tuple[int, ...],
    window_px: int = _EYE_WINDOW_PX,
) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) window centred on the eye-corner midpoint, clipped to frame.

    eye_corners: (2, 2) float — [[x0, y0], [x1, y1]] in crop-space pixel coords (x=col, y=row).
    frame_shape: (H, W) or (H, W, C) of the face crop.
    Raises ValueError if the frame is smaller than window_px in either dimension.
    """
    h, w = frame_shape[:2]
    if h < window_px or w < window_px:
        raise ValueError(
            f"Frame {frame_shape[:2]} is smaller than eye crop window {window_px}px"
        )
    cx = float(eye_corners[:, 0].mean())
    cy = float(eye_corners[:, 1].mean())
    x0 = int(round(cx - window_px / 2))
    y0 = int(round(cy - window_px / 2))
    x0 = max(0, min(x0, w - window_px))
    y0 = max(0, min(y0, h - window_px))
    return x0, y0, x0 + window_px, y0 + window_px


def crop_eye(
    face_crop: np.ndarray,
    window: tuple[int, int, int, int],
    output_size: int = 128,
) -> np.ndarray:
    """Cut the eye region from face_crop and resize to (output_size, output_size, 3) uint8.

    face_crop: (H, W, 3) uint8 RGB.
    window: (x0, y0, x1, y1) as returned by compute_eye_crop_window.
    """
    x0, y0, x1, y1 = window
    patch = face_crop[y0:y1, x0:x1]
    img = Image.fromarray(patch).resize((output_size, output_size), Image.BILINEAR)
    return np.ascontiguousarray(img, dtype=np.uint8)


def flip_for_canonical_eye(
    image: np.ndarray,
    vector: np.ndarray,
    eye: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the shared-weight model's left/right flip convention.

    Left eye: image and vector returned unchanged.
    Right eye: image horizontally flipped; vector x-component negated so the
    target stays geometrically consistent with the mirrored image.

    image: (H, W, 3) uint8 eye crop.
    vector: (3,) float unit gaze vector in normalized camera space.
    eye: "left" or "right".
    Returns (canonical_image, canonical_vector).
    """
    if eye == "left":
        return image, vector
    if eye == "right":
        flipped_image = np.ascontiguousarray(image[:, ::-1])
        flipped_vector = vector * np.array([-1.0, 1.0, 1.0], dtype=vector.dtype)
        return flipped_image, flipped_vector
    raise ValueError(f"eye must be 'left' or 'right', got {eye!r}")
