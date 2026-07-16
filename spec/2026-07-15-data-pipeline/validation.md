# Validation — Data Pipeline (R1)

## Code Correctness

### Group 1 — `spherical_to_unit` (FR1)
- [ ] `theta=0, phi=0` → `g ≈ (0, 0, -1)`, `‖g‖ = 1 ± 1e-5`.
- [ ] `theta=0, phi=π/2` → `g ≈ (-1, 0, 0)` (tolerance `1e-5`).
- [ ] `theta=π/2, phi=0` → `g ≈ (0, -1, 0)` (tolerance `1e-5`).
- [ ] Vectorized input: `theta, phi` as `(50,)` arrays of random values in `[-π/2, π/2]` → output `(50, 3)`, every row `‖g‖ = 1 ± 1e-5`.
- [ ] Output dtype is `float32` for both scalar and array input.

### Group 2 — `preprocess_eye_crop` (FR2)
- [ ] Input `np.zeros((128,128,3), uint8)` → output shape `(3,128,128)`, dtype `float32`; value at any pixel equals `(0 - mean_c)/std_c` for its channel (exact per-channel constants, tolerance `1e-6`).
- [ ] Input `np.full((128,128,3), 255, uint8)` → value at any pixel equals `(1 - mean_c)/std_c`.
- [ ] Wrong shape (`(64,64,3)`) raises `ValueError`.
- [ ] Wrong dtype (`float32` input) raises `ValueError`.

### Group 3 — `build_sample_index` (FR3)
- [ ] Against `sample_bundle` fixture: for one known `gaze_covered_exp_key`, every `(exp_key, frame, patch)` row in the returned index satisfies `get_frame_validity(exp_key)[frame] == True` and `get_normalized_gaze(exp_key, patch)["validity"][frame] == True` (recomputed independently in the test, not just re-reading the same code path).
- [ ] An `exp_key` with `has_gaze_norm() == False` contributes zero rows and raises no error.
- [ ] Row count for a known fixture `exp_key` matches a hand-counted expectation (count of `True & True` entries across both patches) computed directly from the bundle in the test, not copy-pasted from the implementation.

### Group 4 — Split assignment (FR4/FR5)
- [ ] `assign_splits` on a synthetic `samples_df` with subjects in each of `set ∈ {"train","val","test"}`: subjects with `set=="val"` map to `"test"`; subjects with `set=="test"` are absent from the returned dict; subjects with `set=="train"` are absent from the returned dict (left to `make_train_val_split`).
- [ ] `make_train_val_split(subjects, val_fraction=0.2, seed=0)` on 10 synthetic subjects → exactly 2 map to `"val"`, 8 to `"train"`; re-running with the same seed produces an identical assignment (dict equality); a different seed produces a different assignment (not required to differ on every subject, but the two dicts must not be identical for at least one tested seed pair).
- [ ] `val_fraction=0` or `val_fraction=1.5` raises `ValueError`.
- [ ] Empty `train_subjects` raises `ValueError`.
- [ ] `save_split` then `load_split` round-trip: loaded dict equals the original `split` dict exactly (key and value equality).
- [ ] `load_split` on a nonexistent path raises `FileNotFoundError`.
- [ ] `load_split` on a JSON file missing the `"assignment"` key, or containing a value outside `{"train","val"}`, raises `ValueError`.

### Group 5 — `EyeGazeDataset` (FR6)
- [ ] Constructing with `target_split="bogus"` raises `KeyError`.
- [ ] For a real fixture bundle: `len(dataset)` for `target_split="train"` + `"val"` + `"test"` sums to the total row count of the un-split sample index (no sample silently dropped or duplicated across splits).
- [ ] `__getitem__(0)`: image tensor shape `(3,128,128)` dtype `float32`; target tensor shape `(3,)` dtype `float32`, `‖target‖ = 1 ± 1e-4`.
- [ ] For a right-eye (`patch="right"`) sample, the returned image is the horizontal mirror of calling F-NORM's `normalize_eye` directly without flip (pixel-for-pixel, via `np.testing.assert_array_equal` on the flip), and the returned target's x-component is the negation of the unflipped `spherical_to_unit` output for that frame — this is the flip-integration check specific to R1 (F-FLIP's own correctness is out of scope here, but its correct *wiring* into the dataset is in scope).
- [ ] For a left-eye (`patch="left"`) sample, image and target are returned unchanged from the pre-flip pipeline (no accidental mirroring).

### Group 6 — `EyeGazeDataModule` (FR7) integration
- [ ] `setup()` with a fresh `{"seed": 0, "val_fraction": 0.2}` split source succeeds against the real fixture bundle; `train_dataloader()`, `val_dataloader()`, `test_dataloader()` each yield at least one batch with the expected tensor shapes `(B,3,128,128)` and `(B,3)`.
- [ ] `setup()` with `{"path": <manifest from a prior run>}` reproduces the exact same `train_ds`/`val_ds` membership (same set of `(exp_key,frame,patch)` triples) as the run that generated the manifest.
- [ ] No `exp_key` whose subject is in EVE's official `set=="test"` group appears in any of `train_ds`, `val_ds`, or `test_ds`.

## Data Validity

Notebook cells (`notebooks/inspect_data_pipeline.ipynb`), executed end-to-end via `jupyter nbconvert --execute`:

- [ ] **Split sizes**: print subject counts for our `train`/`val`/`test` and confirm `test` subject count equals EVE's `set=="val"` subject count exactly (1:1, since test = EVE's val with no further split); confirm `train + val` subject count equals EVE's `set=="train"` subject count.
- [ ] **Coverage rate**: for a sample of ~20 exp_keys, print the fraction of the 90×2 (frame×patch) grid that passes the validity gate — expect a non-trivial fraction retained (not 0%, not silently 100% for every exp_key, which would suggest the gate isn't discriminating).
- [ ] **Target unit-norm check**: for 100 random dataset items, `‖target‖` histogram concentrated at `1.0 ± 1e-4`.
- [ ] **Visual spot-check**: plot 6 sample eye crops (mix of left/right, train/val/test) with their gaze target drawn as an arrow overlay, confirming arrows point in visually plausible directions (matches the eye's apparent gaze in the image) — a qualitative cross-check against ground truth, same spirit as F-NORM's landing-point scatter.
- [ ] **No frame/exp_key leakage across our train/val**: confirm zero subject overlap between `train_ds` and `val_ds` subject sets (set intersection empty), and same for `train_ds`/`test_ds`, `val_ds`/`test_ds`.

## Data Architecture Integrity

- [ ] **Subject hard-partitioning holds**: for the merged split assignment used by `EyeGazeDataModule`, every `exp_key`'s subject maps to exactly one of `train`/`val`/`test` — no subject appears with two different labels across the assignment dict (contradiction check).
- [ ] **No positional coupling**: `EyeGazeDataset.__getitem__` results are re-derivable by `(exp_key, frame, patch)` alone — shuffling the underlying `sample_index` DataFrame's row order before constructing the dataset produces the same per-key outputs (same image/target for the same triple), just reordered.
- [ ] **Split manifest round-trip is not bypassable**: attempting to `load_split` a hand-edited manifest with a duplicate subject key under two different values, or a subject value outside `{"train","val"}`, raises `ValueError` rather than silently picking one (covered functionally in Group 4, restated here as an architecture-integrity property — malformed split files must fail loudly, not silently corrupt a run).
- [ ] **EVE `test` subjects never leak in**: re-verify (independent of Group 6's dataloader check) that no `exp_key` belonging to an EVE `set=="test"` subject appears anywhere in `build_sample_index`'s output when the module is wired end-to-end — this is the one hard boundary this feature must never cross, since those subjects have no usable ground truth.
