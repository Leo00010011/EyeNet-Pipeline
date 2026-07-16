# F-CALIB — Exclude calibration-prefix frames from the validity gate

## Goal

Tobii calibrates at recording start, so the first ~20 of each experiment's 90 frames are strongly biased toward screen center — median screen-intercept distance to center is **14.5 px** for frames 0–19 vs **171.7 px** for frames 20–89 (~12×; see `notebooks/inspect_calibration_bias.ipynb`, 300-exp_key sample, seed 42). EveDataset's validity flags do **not** catch this (frame-0–19 validity coverage is 91.2% vs 94.0% for the rest — a calibration-biased sample can still be flagged valid). This feature removes those biased frames from the training/eval population by extending `build_sample_index`'s validity gate with a flat frame-index cutoff, preventing a silent center-bias from contaminating R2 training before it starts.

## Scope

**In scope:**
- Add a module-level constant `CALIBRATION_PREFIX_FRAMES = 20` to `src/eyenet/sampling.py`.
- Extend `build_sample_index`'s per-(exp_key, patch) validity gate to also exclude `frame < CALIBRATION_PREFIX_FRAMES`, ANDed with the existing `frame_validity` and per-patch `validity` conditions.
- Update the existing `tests/test_sampling.py` unit tests to cover the exclusion (frame 19 excluded, frame 20 included, otherwise unchanged behavior).
- Re-run `notebooks/inspect_data_pipeline.ipynb`'s coverage-rate check to confirm the expected slight drop from the ≈84.6% baseline.

**Out of scope (explicitly):**
- Per-recording / per-exp_key calibration-cutoff detection (e.g. a distance-variance changepoint). Rejected in the Roadmap: more "correct" in principle but adds real complexity and a new failure mode for a ~3 pp validity gain already mostly captured by the flat cutoff. Revisit only if the fixed cutoff is later shown to mismatch some subjects.
- Any configurable/override parameter on `build_sample_index` — the cutoff is a single module constant (design decision confirmed).
- Changes to `EyeGazeDataset` / `EyeGazeDataModule` (`src/eyenet/dataset.py`) — they consume `build_sample_index`'s output as-is.
- Any change to the geometry (`eye_norm.py`), gaze-target, preprocessing, or splits modules.

## Functional Requirements

**FR1.** `src/eyenet/sampling.py` defines a module-level constant `CALIBRATION_PREFIX_FRAMES: int = 20`, documented with a one-line pointer to the evidence notebook `notebooks/inspect_calibration_bias.ipynb`.

**FR2.** `build_sample_index(bundle, exp_keys)` retains its signature `(bundle, exp_keys) -> pd.DataFrame` with columns `["exp_key", "frame", "patch"]` (string, int, string). No new parameters.

**FR3.** For each `(exp_key, patch)`, a frame is emitted **iff all three** hold:
1. `get_frame_validity(exp_key)[frame] == True`
2. `get_normalized_gaze(exp_key, patch)["validity"][frame] == True`
3. `frame >= CALIBRATION_PREFIX_FRAMES`

Frames `0 .. CALIBRATION_PREFIX_FRAMES - 1` (i.e. `0..19`) are never emitted, regardless of their validity flags.

**FR4.** The exclusion is applied uniformly across all exp_keys and both patches — it is a temporal cutoff (calibration happens at recording start for the whole session), not camera- or patch-specific.

**FR5.** An exp_key missing gaze-norm or face-crop coverage still contributes zero rows (unchanged — the coverage guard runs before the frame gate).

**FR6.** No emitted row has `frame < CALIBRATION_PREFIX_FRAMES`. The maximum possible row count per exp_key drops from `2 × 90` to `2 × (90 − CALIBRATION_PREFIX_FRAMES) = 2 × 70`.

**FR7 (error/edge conditions).** No new error paths. The 90-frame validity arrays are assumed length ≥ `CALIBRATION_PREFIX_FRAMES`; slicing the boolean mask at `[:CALIBRATION_PREFIX_FRAMES]` is safe for the fixed 90-frame EVE layout. Output row order and dtypes are unchanged aside from the removed early frames.

## Public API Summary

```python
# src/eyenet/sampling.py

# First CALIBRATION_PREFIX_FRAMES frames of each experiment are Tobii-calibration
# biased toward screen center; see notebooks/inspect_calibration_bias.ipynb.
CALIBRATION_PREFIX_FRAMES: int = 20

def build_sample_index(bundle, exp_keys) -> pd.DataFrame:
    """(exp_key, frame, patch) index passing: frame_validity AND per-patch
    gaze validity AND frame >= CALIBRATION_PREFIX_FRAMES."""
```

## Dependencies

| Reads from | Via | Purpose |
|---|---|---|
| `EveBundle.has_gaze_norm` / `has_face_crops` | public accessor | coverage gate (unchanged) |
| `EveBundle.get_frame_validity(exp_key)` | public accessor | `(90,)` bool frame validity |
| `EveBundle.get_normalized_gaze(exp_key, patch)["validity"]` | public accessor | `(90,)` bool per-patch validity |

| Writes to / consumed by | How |
|---|---|
| `EyeGazeDataset` / `EyeGazeDataModule` | consume the returned DataFrame unchanged |
| `notebooks/inspect_data_pipeline.ipynb` | coverage-rate check re-run against new output |
