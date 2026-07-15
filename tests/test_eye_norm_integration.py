"""Integration tests for eye_norm against real EveBundle data.

Uses the sample_bundle, face_crops_root, and gaze_covered_exp_key fixtures
from tests/conftest.py. Tests skip automatically when the bundle is absent.
"""

import numpy as np
import pytest

from eye_norm import compose_warp, normalize_eye


class TestNormalizeEyeIntegration:
    def test_left_eye_produces_valid_128x128_patch(
        self, sample_bundle, face_crops_root, gaze_covered_exp_key
    ):
        """Left-eye patch: shape (128,128,3), uint8, non-black."""
        exp_key = gaze_covered_exp_key
        warp = sample_bundle.get_warp_matrix(exp_key, "left")
        combined = warp["validity"] & sample_bundle.get_frame_validity(exp_key)
        assert combined.any(), "no valid frame in test exp_key"

        t = int(np.argmax(combined))
        W_t = warp["W"][t]
        x0, y0 = sample_bundle.get_crop_origin(exp_key)[t]

        crop = sample_bundle.get_face_crop(exp_key, t, face_crops_root)
        H_crop = compose_warp(W_t, int(x0), int(y0))
        patch = normalize_eye(crop, H_crop)

        assert patch.shape == (128, 128, 3)
        assert patch.dtype == np.uint8
        assert patch.mean() > 5.0, "patch is all-black — warp maps outside crop bounds"

    def test_right_eye_patch_shape(
        self, sample_bundle, face_crops_root, gaze_covered_exp_key
    ):
        """Right-eye patch: shape (128,128,3), uint8, non-black."""
        exp_key = gaze_covered_exp_key
        warp = sample_bundle.get_warp_matrix(exp_key, "right")
        combined = warp["validity"] & sample_bundle.get_frame_validity(exp_key)
        assert combined.any(), "no valid frame for right eye in test exp_key"

        t = int(np.argmax(combined))
        W_t = warp["W"][t]
        x0, y0 = sample_bundle.get_crop_origin(exp_key)[t]

        crop = sample_bundle.get_face_crop(exp_key, t, face_crops_root)
        patch = normalize_eye(crop, compose_warp(W_t, int(x0), int(y0)))

        assert patch.shape == (128, 128, 3)
        assert patch.mean() > 5.0

    def test_validity_gate_respected(
        self, sample_bundle, face_crops_root, gaze_covered_exp_key
    ):
        """normalize_eye on an invalid frame must not raise (caller gates validity)."""
        exp_key = gaze_covered_exp_key
        warp = sample_bundle.get_warp_matrix(exp_key, "left")
        frame_val = sample_bundle.get_frame_validity(exp_key)
        invalid_mask = ~warp["validity"] & frame_val

        if not invalid_mask.any():
            pytest.skip("no frame that has face crop validity but invalid W — skipping")

        t = int(np.argmax(invalid_mask))
        W_t = warp["W"][t]
        x0, y0 = sample_bundle.get_crop_origin(exp_key)[t]
        crop = sample_bundle.get_face_crop(exp_key, t, face_crops_root)

        # Should not raise; output may be degenerate (black) but no exception
        try:
            patch = normalize_eye(crop, compose_warp(W_t, int(x0), int(y0)))
            assert patch.shape == (128, 128, 3)
        except Exception as exc:
            pytest.fail(f"normalize_eye raised on invalid frame: {exc}")

    def test_crop_origin_consistency(self, sample_bundle, gaze_covered_exp_key):
        """get_crop_origin returns non-trivial (not all-zero) per-frame origins."""
        exp_key = gaze_covered_exp_key
        origins = sample_bundle.get_crop_origin(exp_key)  # (90, 2) int32

        assert origins.shape == (90, 2)
        assert origins.dtype == np.int32

        # At least some origins must be non-zero — all (0,0) signals a migration bug
        assert not np.all(origins == 0), (
            "all crop origins are (0, 0) — likely a migration bug in FC_CROP_ORIGIN"
        )

    def test_warp_matrix_shape(self, sample_bundle, gaze_covered_exp_key):
        """get_warp_matrix returns W:(90,3,3) float32, validity:(90,) bool."""
        exp_key = gaze_covered_exp_key
        result = sample_bundle.get_warp_matrix(exp_key, "left")

        assert result["W"].shape == (90, 3, 3)
        assert result["W"].dtype == np.float32
        assert result["validity"].shape == (90,)
        assert result["validity"].dtype == bool
