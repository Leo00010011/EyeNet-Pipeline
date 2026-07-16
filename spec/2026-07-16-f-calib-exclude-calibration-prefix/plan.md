# F-CALIB — Implementation Plan

## Context and Design Decisions

**Why a flat constant, not a per-recording heuristic.** The bias is caused by Tobii's calibration at recording start — a fixed temporal artifact, not a per-subject one. The evidence notebook shows a visibly sharp break at the frame 19→20 boundary (median center-distance 14.5 px → 171.7 px). A per-exp_key changepoint detector would be marginally more precise but adds real complexity and a new silent-failure surface for a ~3 pp validity gain already mostly captured by the flat cutoff. The Roadmap explicitly resolves this in favor of `CALIBRATION_PREFIX_FRAMES = 20` and defers the heuristic unless the fixed cutoff is shown to mismatch subjects. This mirrors the constitution's preference for explicit, testable constants over silent heuristics.

**Why extend the existing gate rather than a new filter stage.** The validity gate in `build_sample_index` is the single chokepoint where `(exp_key, frame, patch)` rows are admitted. Adding the frame-index condition to the same boolean AND keeps one place responsible for admission, so `EyeGazeDataset`/`DataModule` need no change and there is no second, separately-maintained filter that could drift out of sync (the exact positional-coupling / silent-desync failure mode the constitution warns about).

**Why a module constant, not a parameter.** Confirmed decision: a single explicit `CALIBRATION_PREFIX_FRAMES` referenced directly inside `build_sample_index`. No override keyword — keeping the API surface identical means the cutoff is a single source of truth and tests exercise exactly the production value.

**Why apply the boolean mask via a slice.** The existing code computes `frame_valid & patch_valid` (a `(90,)` bool array) then `np.nonzero(...)`. Zeroing the first `CALIBRATION_PREFIX_FRAMES` entries of that mask before `np.nonzero` is the minimal, order-preserving change and cannot emit an excluded frame.

## Step 1 — Add the constant (`src/eyenet/sampling.py`)

Add, below the imports:

```python
# The first CALIBRATION_PREFIX_FRAMES frames of each 90-frame experiment are
# biased toward screen center by Tobii's start-of-recording calibration and are
# excluded from the validity gate. Evidence: notebooks/inspect_calibration_bias.ipynb
# (median screen-intercept distance to center 14.5px for frames 0-19 vs 171.7px
# for 20-89; the existing validity flags do not catch this).
CALIBRATION_PREFIX_FRAMES = 20
```

## Step 2 — Extend the validity gate (`src/eyenet/sampling.py`)

Inside `build_sample_index`, in the per-patch loop, mask out the calibration prefix before enumerating frames:

```python
for patch in ("left", "right"):
    gaze = bundle.get_normalized_gaze(exp_key, patch)
    patch_valid = gaze["validity"]
    mask = frame_valid & patch_valid
    mask[:CALIBRATION_PREFIX_FRAMES] = False   # F-CALIB: drop calibration-prefix frames
    for frame in np.nonzero(mask)[0]:
        rows.append({"exp_key": exp_key, "frame": int(frame), "patch": patch})
```

Note: assign `mask` to a fresh array (`frame_valid & patch_valid` already returns a new array, so in-place `mask[:...] = False` does not mutate the bundle's cached validity arrays). Keep the existing docstring in sync — update it to state the third gate condition (`AND frame >= CALIBRATION_PREFIX_FRAMES`).

## Step 3 — Update unit tests (`tests/test_sampling.py`)

- **Keep** `test_index_rows_pass_validity_gate` but strengthen it: assert every emitted `row["frame"] >= CALIBRATION_PREFIX_FRAMES` in addition to the existing validity assertions.
- **Update** `test_row_count_matches_hand_count` so the hand count also zeroes the first `CALIBRATION_PREFIX_FRAMES` of `frame_valid & patch_valid` before counting — otherwise it will now over-count.
- **Add** `test_calibration_prefix_excluded` using a small synthetic fake-bundle (a tiny stub object exposing `has_gaze_norm`, `has_face_crops`, `get_frame_validity`, `get_normalized_gaze`) with all 90 frames valid for both patches: assert frame 19 is absent, frame 20 is present, and total rows == `2 * (90 - CALIBRATION_PREFIX_FRAMES)`. This pins the boundary independent of which real frames happen to be valid.
- `test_no_gaze_norm_contributes_zero_rows` needs no change (coverage guard is upstream of the frame gate).

## Step 4 — Re-run the data-pipeline notebook

Re-execute `notebooks/inspect_data_pipeline.ipynb` via `jupyter nbconvert --execute` and confirm the coverage-rate check drops slightly from the ≈84.6% baseline (losing up to 20/90 frames per exp_key that were previously valid). Persist the executed outputs. This is a data-validity confirmation, not a code change.

## Implementation Order

1. Add `CALIBRATION_PREFIX_FRAMES` constant (Step 1).
2. Extend the gate + docstring in `build_sample_index` (Step 2).
3. Update/add unit tests, run `pytest tests/test_sampling.py` green (Step 3).
4. Run the full suite to confirm no regression elsewhere.
5. Re-execute `notebooks/inspect_data_pipeline.ipynb` and verify the coverage drop (Step 4).
