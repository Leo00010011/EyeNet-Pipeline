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

## F-CALIB — Exclude calibration-prefix frames from the validity gate ✓ DONE

**Evidence:** `notebooks/inspect_calibration_bias.ipynb` (300-exp_key sample, seed 42, using `EveBundle.get_screen_intercept`/`get_gaze_ray`/`get_frame_validity` — the new F-GAZE-RAY accessors, no re-derived geometry). Tobii calibrates at recording start; the first ~20 of 90 `center`-camera frames are strongly biased toward screen center — median screen-intercept distance to center is **14.5px** for frames 0–19 vs **171.7px** for frames 20–89 (~12x), with a visibly sharp break in the violin plot at that boundary. Validity-flag coverage is only mildly lower (91.2% vs 94.0%) — the existing validity gate does **not** catch this on its own, since a calibration-biased sample can still be flagged valid.

**Decision:** a single explicit constant, not a per-recording heuristic. Add `CALIBRATION_PREFIX_FRAMES = 40` to `src/eyenet/sampling.py` and exclude `frame < CALIBRATION_PREFIX_FRAMES` in `build_sample_index`'s validity gate, alongside the existing `frame_validity`/per-patch `validity` AND. The evidence notebook's break sits at frame 20; `40` was chosen as the shipped threshold for extra margin past that break. Rejected alternative: detecting the calibration cutoff per-exp_key (e.g. a distance-variance changepoint) — more "correct" in principle but adds real complexity and a new failure mode for a validity-rate gain that's already mostly captured by the flat cutoff; revisit only if the fixed cutoff is later shown to mismatch some subjects.

**Scope:**
- `sampling.py`: add the constant and the frame-index exclusion, documented with a one-line pointer to the evidence notebook.
- Update `build_sample_index`'s existing unit tests to cover the new exclusion, deriving the excluded/included boundary from `CALIBRATION_PREFIX_FRAMES` rather than a hardcoded frame number — the test must track the constant, not pin its current value.
- Re-run `notebooks/inspect_data_pipeline.ipynb`'s coverage-rate check — expect the ≈84.6% figure to drop, losing up to `CALIBRATION_PREFIX_FRAMES`/90 frames per exp_key that were previously counted valid.
- No change to `EyeGazeDataset`/`DataModule` — they consume `build_sample_index`'s output as-is.

**Implemented:** `CALIBRATION_PREFIX_FRAMES = 40` added to `src/eyenet/sampling.py`; `build_sample_index`'s per-`(exp_key, patch)` mask (`frame_valid & patch_valid`) now zeroes `[:CALIBRATION_PREFIX_FRAMES]` before `np.nonzero`, applied identically to both patches. No signature/column change. Tests: `tests/test_sampling.py` — 4 tests (existing 3 strengthened/updated, new `test_calibration_prefix_excluded` against a synthetic fake-bundle, deriving its excluded/included frame boundary and expected row count from `CALIBRATION_PREFIX_FRAMES` so the test stays correct if the constant changes again), all passing. `notebooks/inspect_data_pipeline.ipynb`'s coverage-rate figure has not been re-executed against the `40`-frame cutoff since this revision — treat the previously recorded ≈70.3% (measured at the earlier `20`-frame value) as stale until re-run.

## R2 — Model & Training Loop ✓ DONE

- [x] **Angular loss** (`src/eyenet/losses.py`) — `angular_loss` (mean arccos of the clamped normalized dot, radians, differentiable) and `angular_error_degrees` (per-sample `(B,)`, degrees). Both normalize their inputs internally (`eps=1e-8`), so they are correct in isolation and testable without a model. `EPS = 1e-7` clamps `cos` to `[-1+EPS, 1-EPS]` in a single shared code path (`_cos`) — **load-bearing, not cosmetic**: `d/dx arccos` diverges at `cos = ±1`, which float32 reaches *before* the model is actually perfect, and an unclamped `arccos` would emit NaN gradients that silently poison every weight while the run still "trains" to completion. Loss floor is `arccos(1-1e-7) ≈ 0.026°`, far below any claimable angular error.
- [x] **ResNet18 + regression head** (`src/eyenet/model.py`) — `GazeResNet18(pretrained=True)`: `resnet18` backbone, `fc` replaced with `Linear(512, 3)`, output `F.normalize(..., eps=1e-8)` to unit length. No input resize — `AdaptiveAvgPool2d` handles 128×128 natively and F-NORM's framing must not be discarded. `pretrained=False` exists so the test suite is offline-deterministic.
- [x] **Lightning module** (`src/eyenet/lightning_module.py`) — `GazeEstimationModule(pretrained, lr, weight_decay)`: logs `train/loss`, `val/loss`, `val/angular_error_deg`, `test/angular_error_deg`; Adam, no scheduler. Consumes `batch[0], batch[1]` only — the R1 batch's `(exp_key, frame, patch)` metadata passes through untouched as R4's export key. **Never touches `EveBundle`**, so its tests run on synthetic batches with no fixture.
- [x] **Training script + config** (`scripts/train.py`, `configs/baseline.yaml`) — YAML-driven; the `trainer:` block is passed through to `pl.Trainer` unmodified, which is how the baseline run is scoped to a small subset (`limit_train_batches`/`limit_val_batches`) **without touching R1's subject-level split policy**. Bad `bundle_dir`/`crops_root` raise `FileNotFoundError` before the Trainer is built. `main(config_path)` is importable so tests drive it without a subprocess.
- [x] **Baseline run validated end-to-end** (`runs/baseline`, 2 epochs × 50 batches) — train loss 1.03 → 0.08 rad, zero NaNs, checkpoints save/load, predictions unit-norm on real data, F-FLIP convention intact. Evidence: `notebooks/inspect_r2_training.ipynb` (executed, outputs persisted).

    ⚠️ **The run's ~5.4° val angular error is not a result — do not quote it as one**, and do not compare it to published 4–7° baselines. Context R3 should start from:
      1. **The angular error scalar is a weak signal on this data.** EVE's labels are strongly biased toward the mean (everyone looks at a screen; spread ≈4.5°), which makes the **mean gaze a genuinely strong prior** — a constant mean-gaze predictor scores ≈4.5°. The subset model's ≈5.4° being *above* that is **not evidence of anything** after 2 epochs on 50 batches; a high-variance target with a strong central bias is expected to take a while to beat its own prior. The point is only that the raw error number cannot, on its own, separate a working model from a degenerate one here — so R3 should report it **alongside** the mean-gaze prior's score for scale, not treat the prior as a gate.
      2. **Prefer per-sample correlation as the correctness signal.** `phi` (horizontal) tracks ground truth at r ≈ +0.68 — this is what actually shows the image→label path is wired end to end, and it is what R2's acceptance rests on. `theta` (vertical) sits at r ≈ 0 (a flat band at its own mean) after the subset run; re-check it on the full-split run.
      3. validation.md's ≈90° untrained reference describes *uniform-random* vectors; an untrained ResNet18 emits a near-constant direction and measured 135°. Not a usable bar either way.

    None of this is a defect in the R2 code — it reflects the data's distribution and the deliberately tiny subset run.
- **Tests:** 33 new (`tests/test_losses.py` 16, `tests/test_model.py` 6, `tests/test_lightning_module.py` 7 — incl. a 30-epoch overfit run proving the loss falls and a checkpoint round-trip; `tests/test_train_script.py` 6 — a real 2-batch run against the sample bundle; `tests/test_batch_keys.py` 4 — key-path integrity). **No file under `src/eyenet/` that existed before R2 was modified**; only `requirements.txt` (`pyyaml`).

    **Deviation from the original R2 plan — W&B moved to R3.** R2's deliverable is a *correctness* result, and gating that on network access and an account turns a local check into an integration dependency. R2 ships `CSVLogger` only, which produces the same loss curve as a file the test suite can assert on directly — something W&B cannot do.

    **Spec corrections found during implementation:**
      - validation.md asserted `atol=1e-2` on the 180° case while itself stating the EPS clamp costs ≈0.026° there — self-contradictory. Measured cost is 0.028°; tests use `atol=5e-2`.
      - `train/loss` logged with `on_step` *and* `on_epoch` means Lightning writes `train/loss_step`/`train/loss_epoch` — there is **no bare `train/loss` column** in `metrics.csv`.
      - `default_collate` returns `exp_key`/`patch` as **tuples**, not lists (validation.md said "list"). Container type is incidental; one key per row is what matters.
      - FR16's `crops_root: ../eve_shared/eve_out` is wrong; the real tree (and `tests/conftest.py`) is `../eve_shared/EveDataset/eve_out`.

- [ ] **Image augmentation** — deferred to R3 (ablation). R2 trains on `preprocess_eye_crop` output only. ⚠️ **Horizontal flip must never be used as an augmentation** — it would silently invert the F-FLIP canonical-eye convention and desync every image from its label.

## F-WANDB — Weights & Biases experiment tracking ✓ DONE

**Motivation:** R2 shipped `CSVLogger` and the R2 diagnostics live in a notebook (`notebooks/inspect_r2_training.ipynb`) that must be re-executed by hand against a finished run. That is fine for one 2-epoch subset run and useless for R3's full-split runs and ablations, where the questions are *comparative* ("did this change help?") and need to be answerable **during** training, across runs. This feature moves the notebook's most decision-relevant plots into a live W&B dashboard. Supersedes the "W&B integration" bullet previously parked in R3.

**Scope:** logging only. This feature does not change the model, the loss, the split policy, or the data pipeline. `CSVLogger` stays (the R2 tests assert on `metrics.csv`); W&B is added alongside it, and must be **disableable from config** so the test suite and offline runs never touch the network.

### Required metrics (the minimum)

1. **Angular error in degrees, train and validation, on one chart.** Degrees — not the radian training objective — because degrees is the unit the literature and every downstream conversation use. Train and val on the same axes so overfitting is visible without cross-referencing two panels. (`train/angular_error_deg` is a new logged quantity: R2's `training_step` computes the degree metric already but only logs the radian loss.)
2. **Variance of the model's output, train and validation.** Per-component variance of the predicted unit vectors over each epoch. This is the collapse detector: a model converging to a constant vector still shows a falling loss, and on EVE's mean-biased labels that degenerate solution scores *well*. Variance is the cheap live signal that separates "learning the mapping" from "learning the mean" — the notebook needed a per-sample correlation plot to see this after the fact.
3. **Error split by eye — `left` vs `right`.** A persistent gap between the two is the signature of an F-FLIP convention break (right-eye crops are mirrored to the canonical left-eye orientation; a sign error there desyncs half the dataset and is invisible in the pooled mean). The batch already carries `patch` — R2's `training_step` deliberately ignores it (FR11), so this feature is the first consumer of that metadata outside R4's export.
4. **Error split by axis — `theta` (vertical) vs `phi` (horizontal),** in degrees. R2's subset run learned `phi` (r ≈ +0.68) while `theta` stayed at r ≈ 0; the pooled angular error hid that completely. Tracking the two axes separately makes an axis-specific failure legible while the run is still going.

### Requirements

- **Config-driven, off by default in tests.** A `logging:` block in `configs/*.yaml` (e.g. `wandb: {enabled, project, entity, run_name, tags}`). `enabled: false` ⇒ no `wandb` import path is exercised and no network call is made. `scripts/train.py` composes `WandbLogger` alongside the existing `CSVLogger` and passes both to `pl.Trainer(logger=[...])`.
- **`unit_to_spherical(g)` → `(theta, phi)` in `src/eyenet/gaze_target.py`** — the inverse of the existing `spherical_to_unit`, MPIIGaze convention: `theta = arcsin(-g_y)`, `phi = arctan2(-g_x, -g_z)`. Needed for metric 4. It is currently prototyped inline in `notebooks/inspect_r2_training.ipynb`; it moves to the tested module. **Unit test: round-trip `spherical_to_unit(unit_to_spherical(g)) == g`** — this is a prime sign-bug candidate, same class as F-FLIP.
- **Per-eye and per-axis metrics must aggregate over the epoch, not per batch** — a batch may contain only one patch, and a per-batch mean would make the left/right comparison noise. Accumulate in the module and emit `on_epoch`.
- **`patch` reaches the metric code as a tuple of `str`** (`default_collate`'s behavior, confirmed in R2) — not a tensor. Mask with a list comprehension or convert once; do not assume tensor indexing works.
- **No change to `losses.py`, `model.py`, `dataset.py`, `sampling.py`, `splits.py`.** The angular-error computation is reused as-is via `angular_error_degrees`.

### Tests

- `unit_to_spherical` round-trip + hand-computed cases (mirrors `tests/test_gaze_target.py`'s existing style).
- Per-eye aggregation: a synthetic epoch of known left/right errors yields the hand-computed per-eye means.
- Per-axis decomposition: a prediction differing from its target in `theta` only yields ≈0 `phi` error, and vice versa.
- `wandb.enabled: false` ⇒ the existing `tests/test_train_script.py` run still passes with no network access, and `pl.Trainer` receives no `WandbLogger`.

### Explicitly out of scope

- Sample-crop image panels, gradients/parameter histograms, sweeps, artifact/model registry — add only if a real question needs them.
- Replacing `CSVLogger` — it stays; R2's tests assert on `metrics.csv`.

**Implemented:**
- `unit_to_spherical(g)` (`src/eyenet/gaze_target.py`) — torch, batched inverse of `spherical_to_unit`; `EPS`-clamped `arcsin` at the pole, imported from `losses.py` (single source of truth). `spherical_to_unit` untouched (byte-for-byte, additive-only diff).
- `GazeEstimationModule` (`src/eyenet/lightning_module.py`) — `_step` now also surfaces `pred`/`target`; epoch-scoped buffers (`_reset_buffers`/`_accumulate`/`_emit`, driven from `on_{train,validation}_epoch_{start,end}`) log, per epoch: `{train,val}/angular_error_deg` (train only — val already logged per-step from R2), `{train,val}/pred_var_{x,y,z}` (skipped under 2 samples), `{train,val}/angular_error_deg_{left,right}` (skipped when that patch has zero samples in the epoch), `{train,val}/{theta,phi}_error_deg` (`phi` diff wrapped through `atan2(sin, cos)` to avoid a ~360° branch-cut artifact). `__init__` signature unchanged; R2's checkpoint round-trip test still loads unmodified.
- `build_loggers(cfg, out)` (`scripts/train.py`) — `CSVLogger` always `logger[0]`; `WandbLogger` appended only when `logging.wandb.enabled` is `true`, `WANDB_API_KEY` is set, and construction succeeds. Any failure (missing block, missing key, constructor exception) warns and degrades to CSV-only; never raises into `trainer.fit`. `wandb`/`WandbLogger` imported only on the enabled path.
- `configs/baseline.yaml` gained an explicit `logging.wandb.enabled: false` block (documentation; behavior unchanged — an absent block already meant disabled). `requirements.txt` gained `wandb`.

**Round-trip test result:** the 169-case `(theta, phi)` grid round-trip and the 200-case random-vector round-trip both pass at `atol=1e-5` — no sign or axis-swap bug between `spherical_to_unit` and `unit_to_spherical`.

**Tests:** 8 new (`tests/test_gaze_target.py`), 12 new (`tests/test_lightning_module.py`, incl. the FR13 epoch-vs-per-batch pin, the `phi` wraparound pin, and the variance collapse/spread/per-component pins), 8 new (`tests/test_train_script.py`, incl. all 6 `build_loggers` branches plus a `metrics.csv` check for the 3 new metric columns). Full suite: 121 passed, 1 pre-existing skip, zero regressions.

**Spec corrections found during implementation:** none — the plan's code listings (buffer lifecycle, `_accumulate`/`_emit`, `build_loggers`) were implemented close to verbatim and matched the R2-established conventions (`patch` as tuple, `on_step`/`on_epoch` flags on `train/loss` unchanged).

⚠️ **Data Validity checklist (real W&B dashboard run against a live account) not executed in this session** — no `WANDB_API_KEY` available in this environment. All code-correctness checks (Groups 1–6 of `validation.md`) pass; the `metrics.csv` cross-check (`test_new_metrics_reach_csv`) confirms the new columns reach a logger, which is the offline proof that `WandbLogger` would receive the identical `self.log` calls. The live-dashboard checks (co-plotted chart, W&B/CSV agreement, offline-mode sync, unattended job dry-run) remain open and should be run once a `WANDB_API_KEY` is available, before R3's full-split run.

## R3 — Full Training & Evaluation

- Full-split training run (after F-WANDB — the ablation questions need cross-run comparison).
- Evaluation against published appearance-based gaze estimation baselines (angular error) to sanity-check the pipeline isn't silently biased (mirrors EveDataset's COCO FreeView cross-check philosophy). ⚠️ Report the **mean-gaze prior's score alongside** the model's for scale (see the R2 note) — on EVE's mean-biased labels a raw ≈5° figure is not comparable to a published ≈5° figure.
- Re-check `theta` (vertical) correlation on the full run — R2's subset left it at r ≈ 0.
- Ablations as needed (left/right shared vs. separate, augmentation choices) — scope TBD based on R2 results.

## R4 — Export Pipeline for Downstream Denoiser

- Build the keyed (`exp_key` + `frame` + `patch`) HDF5 export pipeline per TechStack.md's schema: runs the trained model over a full split, unflips right-eye predictions back to original camera space, persists `pred_gaze` + `validity`.
- Round-trip test: exported predictions addressable and retrievable by `(exp_key, frame, patch)`, never by position; loader raises on duplicate keys or malformed rows, matching EveDataset's anti-amnesia guard pattern.
- Document the exported dataset's location/format as the stable handoff point for the denoiser project.

## Deferred / Explicitly Out of Scope

- The denoiser model itself.
- Any face-crop/normalization-matrix generation — owned entirely by `EveDataset`.
- Multi-task heads (blink, landmarks, etc.) — out of scope per current single-model, single-task decision; revisit only if the denoiser project requires auxiliary signals.
