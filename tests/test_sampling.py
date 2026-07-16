import numpy as np

from eyenet.sampling import build_sample_index


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
        expected += int(np.count_nonzero(frame_valid & patch_valid))
    index = build_sample_index(sample_bundle, [gaze_covered_exp_key])
    assert len(index) == expected
