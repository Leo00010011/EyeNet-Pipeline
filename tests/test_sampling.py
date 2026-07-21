import numpy as np

from eyenet.sampling import CALIBRATION_PREFIX_FRAMES, build_sample_index


def test_index_rows_pass_validity_gate(sample_bundle, gaze_covered_exp_key):
    index = build_sample_index(sample_bundle, [gaze_covered_exp_key])
    frame_valid = sample_bundle.get_frame_validity(gaze_covered_exp_key)
    gaze_valid = {
        patch: sample_bundle.get_normalized_gaze(gaze_covered_exp_key, patch)["validity"]
        for patch in ("left", "right")
    }
    assert len(index) > 0
    for _, row in index.iterrows():
        assert row["exp_key"] == gaze_covered_exp_key
        assert frame_valid[row["frame"]] == True
        assert gaze_valid[row["patch"]][row["frame"]] == True
        assert row["frame"] >= CALIBRATION_PREFIX_FRAMES


def test_no_gaze_norm_contributes_zero_rows(sample_bundle):
    no_gaze_key = None
    for exp_key in sample_bundle.samples_df["exp_key"]:
        if not sample_bundle.has_gaze_norm(exp_key):
            no_gaze_key = exp_key
            break
    if no_gaze_key is None:
        import pytest
        pytest.skip("no exp_key without gaze-norm coverage in sample bundle")
    index = build_sample_index(sample_bundle, [no_gaze_key])
    assert len(index) == 0


def test_row_count_matches_hand_count(sample_bundle, gaze_covered_exp_key):
    frame_valid = sample_bundle.get_frame_validity(gaze_covered_exp_key)
    expected = 0
    for patch in ("left", "right"):
        patch_valid = sample_bundle.get_normalized_gaze(gaze_covered_exp_key, patch)["validity"]
        mask = frame_valid & patch_valid
        mask[:CALIBRATION_PREFIX_FRAMES] = False
        expected += int(np.count_nonzero(mask))
    index = build_sample_index(sample_bundle, [gaze_covered_exp_key])
    assert len(index) == expected


class _FakeBundle:
    """Minimal stub exposing the accessor surface build_sample_index depends on."""

    def __init__(self, n_frames=90):
        self.n_frames = n_frames

    def has_gaze_norm(self, exp_key):
        return True

    def has_face_crops(self, exp_key):
        return True

    def get_frame_validity(self, exp_key):
        return np.ones(self.n_frames, dtype=bool)

    def get_normalized_gaze(self, exp_key, patch):
        return {"validity": np.ones(self.n_frames, dtype=bool)}


def test_calibration_prefix_excluded():
    bundle = _FakeBundle()
    index = build_sample_index(bundle, ["fake_exp"])

    last_excluded = CALIBRATION_PREFIX_FRAMES - 1
    first_included = CALIBRATION_PREFIX_FRAMES
    assert last_excluded not in set(index[index["patch"] == "left"]["frame"])
    assert last_excluded not in set(index[index["patch"] == "right"]["frame"])
    assert first_included in set(index[index["patch"] == "left"]["frame"])
    assert first_included in set(index[index["patch"] == "right"]["frame"])
    assert 89 in set(index[index["patch"] == "left"]["frame"])
    assert 89 in set(index[index["patch"] == "right"]["frame"])

    assert len(index) == 2 * (90 - CALIBRATION_PREFIX_FRAMES)
    assert index["frame"].min() >= CALIBRATION_PREFIX_FRAMES
