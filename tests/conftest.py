from pathlib import Path

import pytest

EVEDATASET_REPO = Path(__file__).resolve().parents[2] / "eve_shared" / "EveDataset"
SAMPLE_BUNDLE_DIR = EVEDATASET_REPO / "bundle"
FACE_CROPS_ROOT = EVEDATASET_REPO / "eve_out"


@pytest.fixture(scope="session")
def sample_bundle():
    from evedataset import EveBundle

    if not (SAMPLE_BUNDLE_DIR / "bundle.h5").exists():
        pytest.skip(f"no sample bundle at {SAMPLE_BUNDLE_DIR}")
    return EveBundle.load(SAMPLE_BUNDLE_DIR)


@pytest.fixture(scope="session")
def face_crops_root():
    if not FACE_CROPS_ROOT.exists():
        pytest.skip(f"no face-crop tree at {FACE_CROPS_ROOT}")
    return FACE_CROPS_ROOT


@pytest.fixture(scope="session")
def gaze_covered_exp_key(sample_bundle):
    """First exp_key in the sample bundle with both face-crop (F6) and
    gaze-norm (F7) coverage, so gaze-vector accessors have real data to hit."""
    for exp_key in sample_bundle.samples_df["exp_key"]:
        if sample_bundle.has_gaze_norm(exp_key) and sample_bundle.has_face_crops(exp_key):
            return exp_key
    pytest.skip("no exp_key in sample bundle has both gaze-norm and face-crop coverage")
