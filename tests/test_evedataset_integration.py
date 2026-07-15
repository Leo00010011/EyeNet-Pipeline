"""Confirms the evedataset package installs and its EveBundle accessor is
usable end-to-end against a real sample bundle, per Roadmap.md R0."""

import numpy as np


def test_evedataset_importable():
    import evedataset

    assert hasattr(evedataset, "EveBundle")


def test_bundle_loads_and_has_samples(sample_bundle):
    assert len(sample_bundle.samples_df) > 0
    assert {"exp_key", "subject", "split", "valid"} <= set(sample_bundle.samples_df.columns)


def test_get_stimulus_returns_real_image(sample_bundle):
    exp_key = sample_bundle.samples_df["exp_key"].iloc[0]

    stim = sample_bundle.get_stimulus(exp_key)

    assert stim.ndim == 3 and stim.shape[2] == 3
    assert stim.dtype == np.uint8


def test_get_scanpath_returns_real_array(sample_bundle):
    exp_key = sample_bundle.samples_df["exp_key"].iloc[0]

    scanpath = sample_bundle.get_scanpath(exp_key)

    assert scanpath.shape[0] == 4
    assert scanpath.dtype == np.float32


def test_get_frame_validity_returns_real_mask(sample_bundle, gaze_covered_exp_key):
    validity = sample_bundle.get_frame_validity(gaze_covered_exp_key)

    assert validity.shape == (90,)
    assert validity.dtype == bool
    assert validity.any(), "expected at least one valid frame in a gaze-covered exp_key"


def test_get_normalized_gaze_returns_real_arrays(sample_bundle, gaze_covered_exp_key):
    gaze = sample_bundle.get_normalized_gaze(gaze_covered_exp_key, "left")

    assert set(gaze.keys()) == {"g_tobii", "R", "h", "o", "validity"}
    assert gaze["g_tobii"].shape == (90, 2)
    assert gaze["R"].shape == (90, 3, 3)
    assert gaze["h"].shape == (90, 2)
    assert gaze["o"].shape == (90, 3)
    assert gaze["validity"].shape == (90,)
    assert gaze["validity"].dtype == bool
    assert gaze["validity"].any()


def test_get_normalized_gaze_rejects_invalid_patch(sample_bundle, gaze_covered_exp_key):
    import pytest

    with pytest.raises(ValueError):
        sample_bundle.get_normalized_gaze(gaze_covered_exp_key, "not_a_patch")


def test_get_face_crop_and_eye_coords_for_a_valid_frame(
    sample_bundle, gaze_covered_exp_key, face_crops_root
):
    frame_validity = sample_bundle.get_frame_validity(gaze_covered_exp_key)
    gaze_validity = sample_bundle.get_normalized_gaze(gaze_covered_exp_key, "left")["validity"]
    valid_frames = [
        f for f in range(90) if frame_validity[f] and gaze_validity[f]
    ]
    assert valid_frames, "expected at least one frame valid for both frame_validity and gaze"
    frame = valid_frames[0]

    face_crop = sample_bundle.get_face_crop(gaze_covered_exp_key, frame, face_crops_root)
    coords = sample_bundle.get_eye_coords_in_crop(gaze_covered_exp_key, frame)

    assert face_crop.shape == (512, 512, 3)
    assert face_crop.dtype == np.uint8
    assert set(coords.keys()) == {"left", "right"}
    assert coords["left"].shape == (2, 2)
    assert coords["right"].shape == (2, 2)
    # eye-corner coords must fall inside the 512x512 face crop
    for eye_coords in coords.values():
        assert np.all(eye_coords >= 0) and np.all(eye_coords <= 512)
