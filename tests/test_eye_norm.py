"""Unit tests for eye_norm — pure geometry, no H5/accessor I/O.

Covers compose_warp and normalize_eye. All inputs are synthetic arrays.
"""

import cv2
import numpy as np
import pytest
from pytest import approx

from eye_norm import compose_warp, normalize_eye


# ---------------------------------------------------------------------------
# Group 1 — compose_warp correctness
# ---------------------------------------------------------------------------

class TestComposeWarp:
    def test_identity_origin(self):
        """(x0=0, y0=0) with W=I → H_crop == identity cast to float64."""
        W = np.eye(3, dtype=np.float32)
        H = compose_warp(W, 0, 0)
        assert H == approx(np.eye(3), abs=1e-10)

    def test_translation_offsets(self):
        """T_inv correctly shifts pixel coords by (x0, y0)."""
        W = np.eye(3, dtype=np.float32)
        H = compose_warp(W, x0=100, y0=200)
        pt_crop = np.array([0.0, 0.0, 1.0])
        pt_out = H @ pt_crop
        assert pt_out[0] / pt_out[2] == approx(100.0, abs=1e-6)
        assert pt_out[1] / pt_out[2] == approx(200.0, abs=1e-6)

    def test_arbitrary_offset(self):
        """Non-identity W constructed to map (x0, y0) → (64, 64); H_crop @ [0,0,1] gives (64,64)."""
        x0, y0 = 204, 312
        # W translates so that W @ [x0, y0, 1] = [64, 64, 1]
        W = np.array([[1, 0, 64 - x0],
                      [0, 1, 64 - y0],
                      [0, 0, 1      ]], dtype=np.float32)
        H = compose_warp(W, x0, y0)
        pt = H @ np.array([0.0, 0.0, 1.0])
        assert pt[0] / pt[2] == approx(64.0, abs=1e-6)
        assert pt[1] / pt[2] == approx(64.0, abs=1e-6)

    def test_output_dtype_and_shape(self):
        """Return dtype is float64, shape is (3, 3)."""
        H = compose_warp(np.eye(3, dtype=np.float32), 10, 20)
        assert H.dtype == np.float64
        assert H.shape == (3, 3)

    def test_matrix_form_equals_W_at_T_inv(self):
        """Result equals W.astype(float64) @ T_inv exactly."""
        np.random.seed(42)
        W = np.random.rand(3, 3).astype(np.float32)
        x0, y0 = 137, 254
        H = compose_warp(W, x0, y0)
        T_inv = np.array([[1, 0, x0],
                          [0, 1, y0],
                          [0, 0, 1 ]], dtype=np.float64)
        expected = W.astype(np.float64) @ T_inv
        assert H == approx(expected, abs=1e-10)

    def test_associativity(self):
        """compose_warp(W2 @ W1, x0, y0) == W2 @ compose_warp(W1, x0, y0).

        Both sides use float64 matrices throughout so no float32 rounding is
        introduced between the two sides, and associativity holds to 1e-8.
        """
        np.random.seed(7)
        W1 = np.random.rand(3, 3)  # float64
        W2 = np.random.rand(3, 3)  # float64
        x0, y0 = 55, 123
        lhs = compose_warp(W2 @ W1, x0, y0)
        rhs = W2 @ compose_warp(W1, x0, y0)
        assert lhs == approx(rhs, abs=1e-8)


# ---------------------------------------------------------------------------
# Group 2 — normalize_eye shape, dtype, and content
# ---------------------------------------------------------------------------

class TestNormalizeEye:
    def _identity_H(self) -> np.ndarray:
        return compose_warp(np.eye(3, dtype=np.float32), 0, 0)

    def test_output_shape(self):
        """Returns (128, 128, 3) by default."""
        crop = np.zeros((512, 512, 3), dtype=np.uint8)
        out = normalize_eye(crop, self._identity_H())
        assert out.shape == (128, 128, 3)

    def test_output_dtype(self):
        """Return dtype is uint8."""
        crop = np.zeros((512, 512, 3), dtype=np.uint8)
        out = normalize_eye(crop, self._identity_H())
        assert out.dtype == np.uint8

    def test_custom_out_size(self):
        """out_size parameter is respected."""
        crop = np.zeros((512, 512, 3), dtype=np.uint8)
        out = normalize_eye(crop, self._identity_H(), out_size=(64, 64))
        assert out.shape == (64, 64, 3)

    def test_uniform_gray_no_equalization(self):
        """Uniform gray crop → output mean within 2.0 of 128 (confirms no equalizeHist)."""
        crop = np.full((512, 512, 3), 128, dtype=np.uint8)
        out = normalize_eye(crop, self._identity_H())
        assert out.mean() == approx(128.0, abs=2.0)

    def test_rgb_channels_independent(self):
        """Only red channel set → only red channel non-zero in output."""
        crop = np.zeros((512, 512, 3), dtype=np.uint8)
        crop[:, :, 0] = 200  # red only
        out = normalize_eye(crop, self._identity_H())
        assert out[:, :, 0].mean() > 100
        assert out[:, :, 1].mean() < 5
        assert out[:, :, 2].mean() < 5

    def test_does_not_mutate_crop(self):
        """Input crop array is unchanged after the call."""
        crop = np.full((512, 512, 3), 77, dtype=np.uint8)
        original = crop.copy()
        normalize_eye(crop, self._identity_H())
        assert np.array_equal(crop, original)

    def test_no_equalizeHist(self):
        """Output must differ from the equalizeHist-applied version on a non-uniform crop."""
        crop = np.zeros((512, 512, 3), dtype=np.uint8)
        # Non-uniform content: gradient
        crop[:, :, 0] = np.tile(np.arange(512, dtype=np.uint8), (512, 1))
        crop[:, :, 1] = 100
        crop[:, :, 2] = 50

        out = normalize_eye(crop, self._identity_H())

        # Apply equalizeHist per-channel to the output
        eq = np.stack(
            [cv2.equalizeHist(out[:, :, c]) for c in range(3)], axis=2
        )
        # The two arrays must differ (no equalization was applied in normalize_eye)
        assert not np.array_equal(out, eq), (
            "normalize_eye output matched equalizeHist output — "
            "histogram equalization must not be applied"
        )


# ---------------------------------------------------------------------------
# Group 3 — composition end-to-end (pure, no bundle)
# ---------------------------------------------------------------------------

class TestComposeAndWarp:
    def test_warp_lands_at_output_center(self):
        """Bright pixel at crop origin, W mapping it to (64,64), lands at output center.

        normalize_eye warps directly to out_size, so a W that sends the crop
        origin to (64, 64) must place that pixel at output (64, 64) — the center
        of the 128×128 patch. (This is the corrected contract: the eye lands at
        EVE's principal point ~(63,61), near the 128-patch center, with no
        intermediate 256 canvas.)
        """
        # W translates crop origin (0,0) → output center (64, 64)
        W = np.array([[1, 0, 64],
                      [0, 1, 64],
                      [0, 0, 1 ]], dtype=np.float32)
        crop = np.zeros((512, 512, 3), dtype=np.uint8)
        crop[0, 0] = (255, 255, 255)  # bright pixel at crop origin

        H = compose_warp(W, x0=0, y0=0)
        out = normalize_eye(crop, H)

        assert out[64, 64, 0] > 200, "bright pixel not at expected center position"

    def test_warp_does_not_mutate_crop(self):
        """Input crop is unchanged after normalize_eye."""
        crop = np.full((512, 512, 3), 99, dtype=np.uint8)
        original = crop.copy()
        H = compose_warp(np.eye(3, dtype=np.float32), 0, 0)
        normalize_eye(crop, H)
        assert np.array_equal(crop, original)
