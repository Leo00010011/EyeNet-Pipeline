"""Validity-gated sample index over an EveBundle.

Depends on EveBundle's public accessor surface only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_sample_index(bundle, exp_keys) -> pd.DataFrame:
    """Enumerate every (exp_key, frame, patch) passing the strictest validity gate:
    frame_validity[frame] AND per-patch gaze validity[frame], both True.

    An exp_key missing gaze-norm or face-crop coverage contributes zero rows
    (skipped, not an error).

    Returns a DataFrame with columns exp_key, frame, patch.
    """
    rows = []
    for exp_key in exp_keys:
        if not (bundle.has_gaze_norm(exp_key) and bundle.has_face_crops(exp_key)):
            continue
        frame_valid = bundle.get_frame_validity(exp_key)
        for patch in ("left", "right"):
            gaze = bundle.get_normalized_gaze(exp_key, patch)
            patch_valid = gaze["validity"]
            for frame in np.nonzero(frame_valid & patch_valid)[0]:
                rows.append({"exp_key": exp_key, "frame": int(frame), "patch": patch})
    return pd.DataFrame(rows, columns=["exp_key", "frame", "patch"])
