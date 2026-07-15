"""Unit tests for eyenet.geometry — pure geometry, no H5/accessor I/O.

Covers the three functions: compute_eye_crop_window, crop_eye,
flip_for_canonical_eye. All spec-required flip invariants are included:
flip-of-flip = identity (image and vector), flipped vector stays unit-norm,
synthetic pure-x vector correctly negates.
"""

import numpy as np
import pytest

from eyenet.geometry import compute_eye_crop_window, crop_eye, flip_for_canonical_eye


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_face_crop(h: int = 512, w: int = 512) -> np.ndarray:
    """Solid grey 512×512 face-crop placeholder."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _unit_vec(x: float, y: float, z: float) -> np.ndarray:
    v = np.array([x, y, z], dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# compute_eye_crop_window
# ---------------------------------------------------------------------------

class TestComputeEyeCropWindow:
    def test_centered_window_no_clipping(self):
        # Eye corners centred in a 512×512 face crop, window fits without clipping
        eye_corners = np.array([[256.0, 256.0], [256.0, 256.0]], dtype=np.float32)
        x0, y0, x1, y1 = compute_eye_crop_window(eye_corners, (512, 512), window_px=96)

        assert x1 - x0 == 96
        assert y1 - y0 == 96
        # Centre of the window should be at 256
        assert x0 == 256 - 48
        assert y0 == 256 - 48

    def test_window_size_matches_window_px(self):
        eye_corners = np.array([[200.0, 300.0], [220.0, 300.0]], dtype=np.float32)
        x0, y0, x1, y1 = compute_eye_crop_window(eye_corners, (512, 512), window_px=64)

        assert x1 - x0 == 64
        assert y1 - y0 == 64

    def test_clips_when_center_near_left_edge(self):
        # Center at x=10, window_px=96 — would go negative; should clip to x0=0
        eye_corners = np.array([[10.0, 256.0], [10.0, 256.0]], dtype=np.float32)
        x0, y0, x1, y1 = compute_eye_crop_window(eye_corners, (512, 512), window_px=96)

        assert x0 == 0
        assert x1 == 96

    def test_clips_when_center_near_right_edge(self):
        # Center at x=502, window_px=96 — would exceed 512; shift left
        eye_corners = np.array([[502.0, 256.0], [502.0, 256.0]], dtype=np.float32)
        x0, y0, x1, y1 = compute_eye_crop_window(eye_corners, (512, 512), window_px=96)

        assert x1 == 512
        assert x1 - x0 == 96

    def test_clips_when_center_near_top_edge(self):
        eye_corners = np.array([[256.0, 5.0], [256.0, 5.0]], dtype=np.float32)
        x0, y0, x1, y1 = compute_eye_crop_window(eye_corners, (512, 512), window_px=96)

        assert y0 == 0
        assert y1 == 96

    def test_midpoint_of_two_distinct_corners(self):
        # Midpoint of (100, 200) and (200, 300) is (150, 250)
        eye_corners = np.array([[100.0, 200.0], [200.0, 300.0]], dtype=np.float32)
        x0, y0, x1, y1 = compute_eye_crop_window(eye_corners, (512, 512), window_px=96)

        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        assert abs(cx - 150.0) <= 1.0
        assert abs(cy - 250.0) <= 1.0

    def test_raises_when_frame_smaller_than_window(self):
        eye_corners = np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="smaller than eye crop window"):
            compute_eye_crop_window(eye_corners, (64, 64), window_px=96)

    def test_accepts_frame_shape_with_channel_dim(self):
        # frame_shape (H, W, C) — should use only first two dims
        eye_corners = np.array([[256.0, 256.0], [256.0, 256.0]], dtype=np.float32)
        result = compute_eye_crop_window(eye_corners, (512, 512, 3), window_px=96)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# crop_eye
# ---------------------------------------------------------------------------

class TestCropEye:
    def test_output_shape_is_128x128x3(self):
        face_crop = _make_face_crop()
        eye_corners = np.array([[256.0, 256.0], [256.0, 256.0]], dtype=np.float32)
        window = compute_eye_crop_window(eye_corners, face_crop.shape)
        result = crop_eye(face_crop, window)

        assert result.shape == (128, 128, 3)

    def test_output_dtype_is_uint8(self):
        face_crop = _make_face_crop()
        eye_corners = np.array([[256.0, 256.0], [256.0, 256.0]], dtype=np.float32)
        window = compute_eye_crop_window(eye_corners, face_crop.shape)
        result = crop_eye(face_crop, window)

        assert result.dtype == np.uint8

    def test_extracts_correct_region(self):
        # Fill the target window with a distinctive colour (200); rest is 0.
        # After cropping a uniform patch, resize must preserve the colour.
        face_crop = np.zeros((512, 512, 3), dtype=np.uint8)
        # Place the uniform patch at a known location
        eye_corners = np.array([[256.0, 256.0], [256.0, 256.0]], dtype=np.float32)
        window = compute_eye_crop_window(eye_corners, face_crop.shape, window_px=96)
        x0, y0, x1, y1 = window
        face_crop[y0:y1, x0:x1] = 200

        result = crop_eye(face_crop, window)

        assert result.mean() == pytest.approx(200.0, abs=1.0)

    def test_custom_output_size(self):
        face_crop = _make_face_crop()
        eye_corners = np.array([[256.0, 256.0], [256.0, 256.0]], dtype=np.float32)
        window = compute_eye_crop_window(eye_corners, face_crop.shape)
        result = crop_eye(face_crop, window, output_size=64)

        assert result.shape == (64, 64, 3)

    def test_output_is_contiguous(self):
        face_crop = _make_face_crop()
        eye_corners = np.array([[256.0, 256.0], [256.0, 256.0]], dtype=np.float32)
        window = compute_eye_crop_window(eye_corners, face_crop.shape)
        result = crop_eye(face_crop, window)

        assert result.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# flip_for_canonical_eye
# ---------------------------------------------------------------------------

class TestFlipForCanonicalEye:
    def test_left_eye_image_unchanged(self):
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        vector = _unit_vec(0.3, -0.4, 0.8)
        out_img, _ = flip_for_canonical_eye(image, vector, "left")

        np.testing.assert_array_equal(out_img, image)

    def test_left_eye_vector_unchanged(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        vector = _unit_vec(0.3, -0.4, 0.8)
        _, out_vec = flip_for_canonical_eye(image, vector, "left")

        np.testing.assert_array_almost_equal(out_vec, vector)

    def test_right_eye_image_horizontally_flipped(self):
        # Create an asymmetric image so we can detect the flip
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[:, :64] = 100   # left half = 100
        image[:, 64:] = 200   # right half = 200
        vector = _unit_vec(0.3, -0.4, 0.8)

        out_img, _ = flip_for_canonical_eye(image, vector, "right")

        # After horizontal flip: left half should be 200, right half 100
        assert out_img[:, :64].mean() == pytest.approx(200.0)
        assert out_img[:, 64:].mean() == pytest.approx(100.0)

    def test_right_eye_x_component_negated(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        vector = _unit_vec(0.5, -0.3, 0.8)
        _, out_vec = flip_for_canonical_eye(image, vector, "right")

        assert out_vec[0] == pytest.approx(-vector[0])
        assert out_vec[1] == pytest.approx(vector[1])
        assert out_vec[2] == pytest.approx(vector[2])

    def test_pure_x_vector_negates_correctly(self):
        # [1, 0, 0] → [-1, 0, 0]; spec calls this out explicitly
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        _, out_vec = flip_for_canonical_eye(image, vector, "right")

        np.testing.assert_array_almost_equal(out_vec, [-1.0, 0.0, 0.0])

    def test_flip_of_flip_is_identity_image(self):
        # Applying the right-eye flip twice must recover the original image
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        vector = _unit_vec(0.3, -0.4, 0.8)

        img1, vec1 = flip_for_canonical_eye(image, vector, "right")
        img2, vec2 = flip_for_canonical_eye(img1, vec1, "right")

        np.testing.assert_array_equal(img2, image)

    def test_flip_of_flip_is_identity_vector(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        vector = _unit_vec(0.3, -0.4, 0.8)

        _, vec1 = flip_for_canonical_eye(image, vector, "right")
        _, vec2 = flip_for_canonical_eye(image, vec1, "right")

        np.testing.assert_array_almost_equal(vec2, vector)

    def test_flipped_vector_stays_unit_norm(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        vector = _unit_vec(0.5, -0.3, 0.8)
        _, out_vec = flip_for_canonical_eye(image, vector, "right")

        assert np.linalg.norm(out_vec) == pytest.approx(1.0, abs=1e-6)

    def test_invalid_eye_raises(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        vector = _unit_vec(0.0, 0.0, 1.0)
        with pytest.raises(ValueError, match="'left' or 'right'"):
            flip_for_canonical_eye(image, vector, "center")
