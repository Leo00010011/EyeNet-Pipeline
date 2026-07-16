# F-CALIB — Validation

## Code Correctness

### Group 1 — Constant and API
- [ ] `CALIBRATION_PREFIX_FRAMES` is importable from `eyenet.sampling` and equals `20` (`int`).
- [ ] `build_sample_index`'s signature is unchanged: `(bundle, exp_keys) -> pd.DataFrame` with columns exactly `["exp_key", "frame", "patch"]`, no new keyword args.

### Group 2 — Boundary exclusion (synthetic fake-bundle, all 90 frames valid)
- [ ] `test_calibration_prefix_excluded`: with a stub bundle where `get_frame_validity` and both patches' `validity` are all-`True` over 90 frames, `build_sample_index` emits **no** row with `frame == 19` and **at least one** row with `frame == 20` per patch.
- [ ] Total row count for that fake bundle equals `2 * (90 - CALIBRATION_PREFIX_FRAMES)` == `140` (exactly, both patches, no early frames).
- [ ] No emitted row has `frame < CALIBRATION_PREFIX_FRAMES` (assert `index["frame"].min() >= 20`).
- [ ] Frame 20 present, frame 89 present — the upper end of the range is untouched.

### Group 3 — Real-bundle gate integrity
- [ ] `test_index_rows_pass_validity_gate` (updated): every emitted row satisfies `frame_validity[frame] == True` AND per-patch `validity[frame] == True` AND `frame >= CALIBRATION_PREFIX_FRAMES`.
- [ ] `test_row_count_matches_hand_count` (updated): hand count zeroes `(frame_valid & patch_valid)[:CALIBRATION_PREFIX_FRAMES]` before counting; `len(index)` matches exactly.
- [ ] `test_no_gaze_norm_contributes_zero_rows`: unchanged — an exp_key without gaze-norm coverage still yields zero rows (coverage guard precedes the frame gate).

### Group 4 — No mutation / isolation
- [ ] `build_sample_index` does not mutate the bundle's cached validity arrays: calling `get_frame_validity(exp_key)` / `get_normalized_gaze(exp_key, patch)["validity"]` before and after `build_sample_index` returns arrays with identical `frame < 20` entries still `True` where they were (the in-place `mask[:...] = False` operates on the fresh `frame_valid & patch_valid` result, not the cached source).

### Group 5 — Downstream untouched
- [ ] Full suite (`pytest`) passes with no changes to `test_dataset.py` / `test_splits.py` — `EyeGazeDataset`/`DataModule` consume the smaller index transparently.

## Data Validity

- [ ] Re-run `notebooks/inspect_data_pipeline.ipynb`: the validity-gate coverage rate drops from the ≈84.6% baseline (on the 20-exp_key sample) by an amount consistent with losing up to 20/90 previously-valid frames per exp_key — i.e. a small drop, not a collapse. Record the new figure in the notebook.
- [ ] Spot-check one real exp_key: the set of emitted frames for each patch is exactly the previously-emitted set with all `frame < 20` removed (no frame ≥ 20 dropped, no frame < 20 retained).
- [ ] Coverage drop is bounded: `new_count >= old_count - 2 * 20 * n_exp_keys` and `new_count <= old_count` (cannot add rows, cannot lose more than the prefix).

## Data Architecture Integrity

- [ ] The output index remains addressed by `(exp_key, frame, patch)` triples only — no positional coupling introduced; row order is still deterministic given exp_key iteration order.
- [ ] No phantom keys: every emitted `exp_key` still passes `has_gaze_norm` AND `has_face_crops`; the frame gate only removes rows, never adds an unkeyed one.
- [ ] The exclusion is not bypassable via a different code path — `build_sample_index` is the sole admission point, and `EyeGazeDataset` derives its samples solely from this index (verified: no independent frame enumeration downstream).
