# Roadmap

Status: R0 evedataset integration confirmed and tested; scope and ordering beyond that will still shift as open questions below get resolved.

## R0 — Foundations

- [x] **`evedataset` package installability confirmed.** Installed globally (not in a venv — this repo is cloned straight to the training server, no packaging/distribution needed) via `requirements.txt`'s `-e ../eve_shared/EveDataset` (editable, relative path dependency — this *is* the pin strategy; revisit only if EveDataset ships a wheel later). Spec corrected: the accessor class is `EveBundle` (not `EveGazeVectorAccessor`, which never existed — see Mission.md/TechStack.md).
- [x] **Integration tests against real data** — `tests/test_evedataset_integration.py` (8 tests, all passing): package import, `EveBundle.load`, `samples_df`, `get_stimulus`, `get_scanpath`, `get_frame_validity`, `get_normalized_gaze` (incl. invalid-patch error path), `get_face_crop` + `get_eye_coords_in_crop` for a real valid `(exp_key, frame)`. Fixtures in `tests/conftest.py` point at the sibling `eve_shared/EveDataset/bundle` (`sample_bundle`) and `eve_out` (`face_crops_root`), and pick a real gaze-covered `exp_key` (`gaze_covered_exp_key`). The sample bundle now has full F6/F7 coverage (2487/3096 exp_keys have both `has_gaze_norm` and `has_face_crops` — it was rebuilt with `include_gaze_vector_data=True` mid-session after an initial empty-coverage bundle blocked these tests).
- [x] **Visual inspection notebook** — `notebooks/inspect_evedataset.ipynb`: loads the bundle, picks a gaze-covered exp_key, plots the stimulus and the face crop with left/right eye-corner points overlaid, a rough (bounding-box, non-final) eye-crop preview, and prints `g_tobii`/`R`/`h`/`o` for one valid frame. Runs clean end-to-end via `jupyter nbconvert --execute`.
- [x] **F-NORM.1 — `W` and crop origin exposed through `EveBundle`.** EveDataset exposes `{left,right}_W` `(N,3,3)` via `get_warp_matrix` and the crop origin `(N,2)` via `get_crop_origin`. (`left`/`right` = eye patch, not camera.) Do not read the H5 directly. **`camera_matrix` is not exposed and is not needed** for F-NORM (the warp lands the eye at EVE's principal point by construction — see TechStack §Patch size / intrinsics).

- [x] **F-NORM — Zhang et al. 2018 eye-image data normalization.** ✓ DONE

    **Scope:** Produce a 128×128 RGB eye patch by applying the stored perspective transform `W` to warp the eye into the canonical normalized frame — the same frame as the ground-truth target `g_tobii`. The flip convention for right-eye crops is **not** part of this feature; it is tracked separately as F-FLIP below.

    **Resolved (F-NORM.0):** The crop→camera affine is the translation `T_inv = [[1,0,x0],[0,1,y0],[0,0,1]]` where `(x0, y0)` is `bundle.get_crop_origin(exp_key)[t]`. `H_crop = W @ T_inv` is composed and applied to the face crop directly — no intermediate pre-cut eye crop needed.

    **Implemented (`src/eye_norm.py`):**
      - `compose_warp(W, x0, y0)` → `(3,3) float64` — compose stored `W` with crop→frame translation `T_inv`.
      - `normalize_eye(crop, H_crop, out_size=(128,128))` → `(128,128,3) uint8` — warp the face crop **straight to 128×128**. EVE's native eye patch is ~128 px with principal point ≈ (63, 61) (measured across 40 exps), so the eye lands centred with no intermediate canvas, no center-crop, no intrinsics rescale, and EVE's `focal_norm`/`roiSize` never consulted.
      - Warp in RGB; no `cv2.equalizeHist`; no re-derivation of `W`, `R`, or head pose.

    **Tests:** 15 unit tests (`tests/test_eye_norm.py`), 4 integration tests + 1 skipped (`tests/test_eye_norm_integration.py`). All pass.

    **Bundle migrated:** `bundle.h5` rebuilt with `left_W`/`right_W`/`crop_origin` (F9 data) after confirming `dataset_cache.h5` had already been migrated.

    **Visual check:** `notebooks/inspect_eye_norm_executed.ipynb` — augmented with four additional sections (all executed via nbconvert, outputs persisted):
      1. Left vs right patch side-by-side for the same frame — confirms both eyes centred, landing points in DV7 band.
      2. Multi-frame strip (8 consecutive valid frames, left patch) — eye-centre std ≈ 0.7–1.0 px across frames.
      3. Multi-experiment grid (12 experiments, evenly sampled) — consistent frontalized eye across subjects/sessions.
      4. DV7 landing-point scatter (40 experiments, both patches) — `left_W` mean ≈ (60, 62), `right_W` mean ≈ (66, 61), clusters separated; 2 marginal outliers at t=0 (both within ~2 px of band edge, not implementation regressions).

    FR8 pixel-vs-`{camera}_eyes.mp4` comparison is still open (see R1 note).

    **Design-history note:** the first implementation warped to a 256×256 canvas then center-cropped `[64:192]`, assuming the eye sat at (128,128); it actually sits at ~(63,61), so the crop landed on the cheek. Fixed by warping direct-to-128.

- [ ] **F-FLIP — Canonical-eye flip convention** *(next feature; implement after F-NORM).*

    **Scope:** Apply the left/right flip so the shared-weight ResNet18 always sees one canonical eye orientation. This is a distinct transform from F-NORM's geometric warp and is implemented and tested independently to avoid conflating two separate coordinate operations.

    **Requirements:**
      - `flip_for_canonical_eye(image, gaze_vector, eye)` — horizontally flip right-eye images; negate the x-component of the corresponding ground-truth unit vector to stay geometrically consistent with the flipped image.
      - **Determine `eye` from the `W`-patch name** (`"left"`/`"right"` passed to `get_warp_matrix`), which shares its frame with the `g_tobii` target — **never** from `get_eye_coords_in_crop`, whose left/right labels are the opposite convention (see TechStack §Left/Right Flip Convention). Mixing the two silently mirrors half the dataset.
      - At inference/export time: unflip right-eye predictions (negate x) before persisting, so all exported vectors are in the original (non-mirrored) normalized camera space.
      - **Unit tests (mandatory before any dataset or training code depends on this):** flip-then-unflip is the identity on both image and vector; flipped vectors remain unit-norm; a synthetic pure-x-direction vector correctly negates. This pairing is a prime candidate for a silent sign-bug.

    **Visual pre-check:** `notebooks/inspect_f_flip.ipynb` — exercises the flip logic inline (no `src/` module yet) for 10 randomly sampled valid `(exp_key, frame, patch)` cases (seed 42; 6 right-eye, 4 left-eye, frames spread mid-session). Layout per row: raw patch + gaze arrow | canonical patch + gaze arrow | gaze vector table (raw → flipped). Sanity checks: flip-then-flip = identity on image and vector, unit-norm preserved — all 10 PASS. The `spherical_to_unit` helper (MPIIGaze convention: `g = [-cos θ sin φ, -sin θ, -cos θ cos φ]`) is also prototyped here; it moves to a tested module in R1.

- [x] Define train/val/test split policy on top of EveDataset's existing per-subject split column and the strictest validity gate (`frame_validity` AND per-patch `validity`, both `True`). **Spec correction:** the column is exposed as `samples_df["split"]` (not `"set"` — `"set"` is `SampleTable`'s internal column name; `EveBundle.samples_df` renames it to `split`).

## R1 — Data Pipeline ✓ DONE

- [x] **Build a PyTorch `Dataset`/`DataModule` wrapping `EveBundle`.** For each valid `(exp_key, frame, patch)`, fetches the face crop, produces the Zhang-normalized + canonically-flipped 128×128 eye crop via the R0 (F-NORM + F-FLIP) geometry modules, converts EveDataset's spherical `(theta, phi)` ground truth to a 3D unit vector (MPIIGaze convention), and applies ImageNet preprocessing, yielding `(eye_crop_128, target_unit_vector, exp_key, frame, patch)`.
- [x] `spherical_to_unit(theta, phi)` (`src/eyenet/gaze_target.py`) — isolated, unit-tested spherical→unit-vector conversion.
- [x] `preprocess_eye_crop(image)` (`src/eyenet/preprocessing.py`) — ImageNet-normalized `(3,128,128)` float32 tensor from a `(128,128,3)` uint8 crop.
- [x] `build_sample_index(bundle, exp_keys)` (`src/eyenet/sampling.py`) — validity-gated `(exp_key, frame, patch)` DataFrame index.
- [x] `assign_splits`, `make_train_val_split`, `save_split`, `load_split` (`src/eyenet/splits.py`) — EVE `val`→our `test`, EVE `train`→our `train`/`val` (seeded, persisted JSON manifest).
- [x] `EyeGazeDataset` / `EyeGazeDataModule` (`src/eyenet/dataset.py`) — the R2 training script's integration point.
- Validated against real samples end-to-end (`notebooks/inspect_data_pipeline.ipynb`, executed via `nbconvert`): split sizes match EVE's `val`↔our `test` 1:1 and `train+val`⊆EVE's `train`; validity-gate coverage ≈84.6% on a 20-exp_key sample (non-trivial, discriminating); 100 random targets all `‖g‖ = 1.0 ± 1e-4`; visual arrow-overlay spot-check; zero subject overlap across train/val/test. Also confirmed: zero EVE-test-subject leakage into `build_sample_index`'s output, and dataset outputs are reorder-invariant (no positional coupling on `sample_index` row order).
- **Tests:** 30 new unit/integration tests (`tests/test_gaze_target.py`, `tests/test_preprocessing.py`, `tests/test_sampling.py`, `tests/test_splits.py`, `tests/test_dataset.py`), all passing (full suite: 78 passed, 1 pre-existing skip).

## F-CALIB — Exclude calibration-prefix frames from the validity gate *(next; implement before R2 training starts)*

**Evidence:** `notebooks/inspect_calibration_bias.ipynb` (300-exp_key sample, seed 42, using `EveBundle.get_screen_intercept`/`get_gaze_ray`/`get_frame_validity` — the new F-GAZE-RAY accessors, no re-derived geometry). Tobii calibrates at recording start; the first ~20 of 90 `center`-camera frames are strongly biased toward screen center — median screen-intercept distance to center is **14.5px** for frames 0–19 vs **171.7px** for frames 20–89 (~12x), with a visibly sharp break in the violin plot at that boundary. Validity-flag coverage is only mildly lower (91.2% vs 94.0%) — the existing validity gate does **not** catch this on its own, since a calibration-biased sample can still be flagged valid.

**Decision:** a single explicit constant, not a per-recording heuristic. Add `CALIBRATION_PREFIX_FRAMES = 20` to `src/eyenet/sampling.py` and exclude `frame < CALIBRATION_PREFIX_FRAMES` in `build_sample_index`'s validity gate, alongside the existing `frame_validity`/per-patch `validity` AND. Rejected alternative: detecting the calibration cutoff per-exp_key (e.g. a distance-variance changepoint) — more "correct" in principle but adds real complexity and a new failure mode for a ~3pp validity-rate gain that's already mostly captured by the flat cutoff; revisit only if the fixed cutoff is later shown to mismatch some subjects.

**Scope:**
- `sampling.py`: add the constant and the frame-index exclusion, documented with a one-line pointer to the evidence notebook.
- Update `build_sample_index`'s existing unit tests to cover the new exclusion (frame 19 excluded, frame 20 included, otherwise unchanged behavior).
- Re-run `notebooks/inspect_data_pipeline.ipynb`'s coverage-rate check — expect the ≈84.6% figure to drop slightly (losing up to 20/90 frames per exp_key that were previously counted valid).
- No change to `EyeGazeDataset`/`DataModule` — they consume `build_sample_index`'s output as-is.

## R2 — Model & Training Loop

- ResNet18 (pretrained) + regression head, Lightning module.
- Angular/cosine loss implementation, unit-tested against hand-computed examples (e.g. identical vectors → 0 error; orthogonal vectors → 90°).
- Baseline training run on a small subset to validate the loop end-to-end (loss decreases, no NaNs, checkpoint saves/loads).
- W&B integration: loss curves, angular error, sample crop visualizations.

## R3 — Full Training & Evaluation

- Full-split training run.
- Evaluation against published appearance-based gaze estimation baselines (angular error) to sanity-check the pipeline isn't silently biased (mirrors EveDataset's COCO FreeView cross-check philosophy).
- Ablations as needed (left/right shared vs. separate, augmentation choices) — scope TBD based on R2 results.

## R4 — Export Pipeline for Downstream Denoiser

- Build the keyed (`exp_key` + `frame` + `patch`) HDF5 export pipeline per TechStack.md's schema: runs the trained model over a full split, unflips right-eye predictions back to original camera space, persists `pred_gaze` + `validity`.
- Round-trip test: exported predictions addressable and retrievable by `(exp_key, frame, patch)`, never by position; loader raises on duplicate keys or malformed rows, matching EveDataset's anti-amnesia guard pattern.
- Document the exported dataset's location/format as the stable handoff point for the denoiser project.

## Deferred / Explicitly Out of Scope

- The denoiser model itself.
- Any face-crop/normalization-matrix generation — owned entirely by `EveDataset`.
- Multi-task heads (blink, landmarks, etc.) — out of scope per current single-model, single-task decision; revisit only if the denoiser project requires auxiliary signals.
