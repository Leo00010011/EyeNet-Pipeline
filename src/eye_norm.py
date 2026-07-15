"""Eye-image data normalization — Zhang et al. 2018 perspective warp.

Pure functions, no H5/accessor I/O. Callers supply all arrays.
"""

from __future__ import annotations

import cv2
import numpy as np


def compose_warp(
    W: np.ndarray,
    x0: int,
    y0: int,
) -> np.ndarray:
    """Compose stored W with the inverse crop-to-frame translation.

    W maps original 1920×1080 frame pixels → normalized-patch pixels.
    T_inv maps face-crop pixels → original frame pixels via (x0, y0) offset.

    H_crop = W @ T_inv,  T_inv = [[1, 0, x0], [0, 1, y0], [0, 0, 1]]

    Returns (3, 3) float64 homography for cv2.warpPerspective.
    """
    T_inv = np.array([[1, 0, x0],
                      [0, 1, y0],
                      [0, 0, 1 ]], dtype=np.float64)
    return W.astype(np.float64) @ T_inv


def normalize_eye(
    crop: np.ndarray,
    H_crop: np.ndarray,
    out_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Warp a 512×512 RGB face crop into the normalized eye patch.

    `H_crop` maps face-crop pixels into EVE's own normalized-patch frame, whose
    principal point (where the eye center lands, by the Zhang construction) sits
    at ~(63, 61). EVE's native eye patch is ~128 px, so warping directly to
    `out_size=(128, 128)` reproduces that patch with the eye centered — no
    intermediate canvas, no center crop, no intrinsics rescale required.

    (Earlier revisions warped to a 256×256 canvas and center-cropped [64:192],
    assuming the eye sat at (128, 128); it actually sits at ~(63, 61), so that
    crop landed on the cheek. Warp straight to the output size instead.)

    crop: (512, 512, 3) uint8 RGB face crop from get_face_crop().
    H_crop: (3, 3) result of compose_warp().
    out_size: (H, W) final output size.

    Returns (out_H, out_W, 3) uint8 RGB.
    """
    out_H, out_W = out_size

    return cv2.warpPerspective(
        crop, H_crop, (out_W, out_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
