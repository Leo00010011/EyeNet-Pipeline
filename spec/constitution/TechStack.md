# Tech Stack

## Core Dependency

**`evedataset`** — installable Python package, produced by the sibling `EveDataset` project. Provides `EveBundle`, the sole read interface into face crops and normalized-space ground-truth gaze. Treated as an external dependency (version-pinned once packaging stabilizes on the EveDataset side), never vendored or reimplemented.

Relevant accessor surface consumed here (see EveDataset's `TechStack.md` for full spec; actual signatures per `src/evedataset/bundle.py`):
- `EveBundle.load(bundle_dir)` → `EveBundle`, opens `bundle_dir/bundle.h5`
- `get_face_crop(exp_key, frame, crops_root)` → `(512,512,3)` uint8 RGB
- `get_eye_coords_in_crop(exp_key, frame)` → `{"left": (2,2), "right": (2,2)}` float32, crop-space eye-corner coordinates
- `get_normalized_gaze(exp_key, patch)` → `{"g_tobii": (90,2), "R": (90,3,3), "h": (90,2), "o": (90,3), "validity": (90,)}`, `patch ∈ ("face","left","right")` — arrays cover all 90 frames, index by `frame` to get a single frame's values
- `get_frame_validity(exp_key)` → `(90,)` bool
- `has_gaze_norm(exp_key)` / `has_face_crops(exp_key)` → `bool`, gate for whether an exp_key has F7/F6 coverage at all

**Now exposed (F-NORM.1 resolved):**
- `get_warp_matrix(exp_key, patch)` → `{"W": (90,3,3) float32, "validity": (90,) bool}` — the per-frame perspective transform matrix for the `face`/`left`/`right` patches (`left`/`right` here means eye patch, not camera — EVE's cameras are `basler`/`webcam_l`/`webcam_c`/`webcam_r`; each camera HDF stores its own `left_W`/`right_W`).
- `get_crop_origin(exp_key)` → `(90,2)` int32 — the `(x0,y0)` top-left corner of each frame's 512×512 face-crop window in the original frame.

**Not exposed — and not needed:** `camera_matrix` is **not** on `EveBundle` (the bundle's `gaze_norm` group has no `camera_matrix` dataset). F-NORM does not need it: the warp is applied via the stored `W`, and the eye lands at EVE's normalized principal point by construction (see §Patch size / intrinsics). If a later feature genuinely needs the intrinsics, that is a request to the EveDataset session, not a workaround here.

## Model & Training

| Component | Choice |
|---|---|
| Backbone | `torchvision.models.resnet18`, ImageNet-pretrained weights |
| Framework | PyTorch + PyTorch Lightning |
| Input | 128×128 RGB eye crop |
| Image normalization | ImageNet convention: scale to `[0,1]`, then `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` |
| Regression head | `Linear(512,hidden_dim) → Dropout(dropout1) → Linear(hidden_dim,3) → Dropout(dropout2)` on ResNet18 backbone → 3-vector, L2-normalized to unit length. `hidden_dim` (default 256) is config-adjustable; the two dropouts are independent (F-OPTUNA), with `dropout` (default 0.5) retained as a shim filling both. |
| Loss | Config-selectable via `losses.get_loss`: `angular` (arccos, radians — default) or `cosine` (`1 - ⟨x,x̂⟩`). Both over unit vectors. |
| Hyperparameter search | Optuna (F-OPTUNA), `scripts/tune.py` + `configs/optuna.yaml`; objective is `val/angular_error_deg` |
| Target derivation | EveDataset spherical `(theta, phi)` → 3D unit vector via MPIIGaze convention: `g = [-cos(theta)sin(phi), -sin(theta), -cos(theta)cos(phi)]` |
| Primary metric | Mean angular error (degrees) |
| Experiment tracking | `CSVLogger` (always on) + `WandbLogger` (F-WANDB, config-gated via `logging.wandb.enabled`, degrades gracefully) |

## Key Libraries

| Library | Role |
|---|---|
| `torch`, `torchvision` | Model, pretrained weights, image transforms |
| `pytorch-lightning` | Training loop, checkpointing, logging integration |
| `evedataset` | Data access (crops + normalized gaze ground truth) |
| `pyyaml` | Training-run config parsing (`configs/baseline.yaml`) |
| `wandb` | Experiment tracking — added by F-WANDB; config-gated (`logging.wandb.enabled`), never exercised in tests |
| `optuna` | Hyperparameter search (F-OPTUNA) — study, TPE/random samplers, median/ASHA/Nop pruners. **`optuna-integration` is deliberately NOT a dependency** — see F-OPTUNA §Pruning callback below |
| `numpy` | Array/vector math (spherical↔unit-vector conversion, normalization matrix assembly) |
| `opencv-python` (`cv2`) | `warpPerspective` for the Zhang data-normalization eye warp |
| `h5py` or `pandas`/`parquet` | Persisting exported prediction datasets (format TBD — see Roadmap) |
| `pytest` | Tests |

## Eye-Crop Extraction & Data Normalization (owned by this repo)

`evedataset` delivers only 512×512 face crops (`get_face_crop`) plus crop-space eye-corner coordinates (`get_eye_coords_in_crop` → `{"left": (2,2), "right": (2,2)}`). This repo implements its own geometry module, mirroring EveDataset's `face_crop_tools.py` pattern. The eye crop is produced by the **Zhang et al. 2018 data-normalization perspective warp**, not a plain bounding-box cut.

### Reference procedure (Zhang et al. 2018, ETRA)

Source: *"Revisiting Data Normalization for Appearance-Based Gaze Estimation"*, Xucong Zhang, Yusuke Sugano, Andreas Bulling — reference `normalizeData()` (`data_normalization_code/normalize_data.py`). CC BY-NC-SA 4.0; cite on publication.

Per eye, the reference computes a **transformation matrix** `W = cam_norm · S · R · inv(cam)` and applies it with `cv2.warpPerspective(img, W, roiSize)`. The three moving parts:

1. **`R` — rotation that aligns the head coordinate system's x-axis and removes roll.** Built from the eye-to-camera direction and the head-pose x-axis:
   - `forward = et / ‖et‖` — unit vector from camera to the eye center `et` (new **z** / optical axis).
   - `down = normalize(cross(forward, hRx))`, where `hRx = hR[:,0]` is the head rotation matrix's x-axis (new **y**).
   - `right = normalize(cross(down, forward))` (new **x**).
   - `R = [right, down, forward]ᵀ`. Constructing `right` from the head's x-axis is what "aligns the x-axis of the head coordinate system" and removes roll.
2. **`S = diag(1, 1, z_scale)` — distance normalization (scaling).** `z_scale = distance_norm / ‖et‖` scales the eye to a fixed canonical eye-to-camera distance.
3. **`cam_norm` — virtual normalized-camera intrinsics.** `[[focal_norm,0,W/2],[0,focal_norm,H/2],[0,0,1]]`, principal point at the crop center.

The reference also normalizes head rotation (`hR_norm = R·hR`) and the gaze vector. **Gaze normalization uses the "modified" variant: rotation `R` only, scaling `S` is NOT applied to the gaze direction** (`gc_norm = normalize(R · (gc − et))`). The `(theta, phi)` decode of the normalized gaze — `theta = arcsin(−g_y)`, `phi = arctan2(−g_x, −g_z)` — matches the MPIIGaze convention already used for the target (Mission/TechStack §Target derivation).

### `W` is already computed — consume it, do not re-derive it

Per EVE's `DATASET.md` (§HDF file format), the data-normalization procedure was already run when the dataset was built, and its outputs are stored **per patch, per frame**:

| EVE HDF key | Shape | Meaning |
|---|---|---|
| `{face,left,right}_W` | `(N, 3, 3)` | **The perspective transform matrix** — line 74 of the reference, i.e. `cam_norm · S · R · inv(cam)`. `face`/`left`/`right` = patch type (face, left-eye, right-eye), not camera. This repo uses `left_W` and `right_W`. |
| `{face,left,right}_R` | `(N, 3, 3)` | The rotation correction applied to the raw gaze vector (line 72) |
| `{face,left,right}_o` | `(N, 3)` | 3D origin of gaze for the patch |
| `{face,left,right}_g_tobii` | `(N, 2)` | Roll-removed gaze direction, spherical — the training target |
| `camera_matrix` | `(3, 3)` | Original pinhole intrinsics (`cam`) |
| `head_rvec` / `head_tvec` | `(N,180,3,1)` | Head pose from `cv2.solvePnP` |

**So this repo never runs `solvePnP`, never builds `R`, and never assembles `W`.** It fetches the stored per-frame `W` for the patch and applies it. This is what guarantees the eye image lands in the exact frame the target `g_tobii` was normalized into — re-deriving any of it invites a silent frame desync.

> **API gap resolved (F-NORM.1):** `{left,right}_W` (via `get_warp_matrix`) and the crop origin (via `get_crop_origin`) are exposed through `EveBundle`. Consume them through the public API — never read the H5 directly. (`camera_matrix` is not exposed and is not needed — see above.)

### The frame-composition requirement (critical)

`W` contains `inv(cam)` — **it is defined to warp the *original, undistorted camera frame*** into the normalized patch. Our eye crops were **extracted from the face crop with no normalization**, so they live in a *different pixel frame*. Feeding a pre-cut, axis-aligned eye crop straight into `cv2.warpPerspective(crop, W, ...)` is **geometrically invalid and will silently produce garbage**.

To warp an already-extracted crop, `W` must be composed with the affine `A` that maps **crop pixels → original camera pixels** (the offset + scale of how that crop was cut):

```
W_crop = W · A          # A: crop-pixel → camera-pixel (inverse of the extraction transform)
normalized = cv2.warpPerspective(eye_crop, W_crop, roiSize)
```

`A` must therefore be **recorded at extraction time** (crop origin + scale, and, if the 512×512 face crop is itself a warp of the camera frame, that warp too — composed). If `A` is not recoverable for the existing crops, the fallback is to apply `W` to the frame it was built for (the original camera image) and skip the intermediate crop entirely. Resolving this is the first task of F-NORM.

### Patch size / intrinsics — resolved: warp directly to 128×128

The stored `W` maps into **EVE's** own normalized-patch geometry. By the Zhang construction the rotation `R` puts the eye centre on the optical axis, so it projects **exactly onto `cam_norm`'s principal point** `(cx, cy)` — independent of head pose, gaze, or focal length (the focal terms multiply the on-axis x=y=0 coords and vanish). Measured empirically across 40 experiments, that principal point is **≈ (63, 61)** and stable to ~1.4 px:

| Patch | Eye it centres | Landing point (output px) |
|---|---|---|
| `left_W`  | image-**right** eye | (60, 62) |
| `right_W` | image-**left** eye  | (66, 60) |

So **EVE's native eye patch is ~128 px** with its principal point near a 128-patch centre. `normalize_eye` therefore **warps `W` straight to a 128×128 output** — the eye lands ~3 px from centre. No intermediate canvas, no center-crop, no intrinsics rescale, and EVE's `roiSize`/`focal_norm` never need to be known.

Other adaptations from the reference: warp the **RGB** image (reference uses grayscale), and **drop `cv2.equalizeHist`** — ImageNet mean/std normalization is used instead.

### Module surface

| Function | Role |
|---|---|
| `compose_warp(W, x0, y0)` → `(3,3)` float64 | Compose the stored `W` with the crop→frame translation `T_inv = [[1,0,x0],[0,1,y0],[0,0,1]]`: `H_crop = W @ T_inv`. |
| `normalize_eye(crop, H_crop, out_size=(128,128))` → `(128,128,3)` uint8 RGB | `cv2.warpPerspective` the face crop straight into the 128×128 normalized frame (no intermediate canvas / center-crop). |
| `flip_for_canonical_eye(image, vector, eye)` | Apply the left/right flip convention — **F-FLIP feature only**, see §Left/Right Flip Convention |

These are pure functions, unit-tested independently of any H5/accessor I/O — same testing philosophy as EveDataset's crop geometry helpers. F-NORM and F-FLIP are separate features: implement and test `compose_warp`/`normalize_eye` first, then `flip_for_canonical_eye` independently. F-NORM tests must verify the normalized image and the target vector end up in the **same** frame — the strongest available check is to reproduce EVE's own normalized eye patch (`{camera}_eyes.mp4`, which is the *post*-normalization eye video) for a real `(exp_key, frame, patch)` and compare pixel-wise.

## Left/Right Flip Convention

A single shared-weight ResNet18 sees only one canonical eye orientation:

> **Which eye is which (read before implementing F-FLIP).** Determine a sample's eye from the **`W`-patch name** (`get_warp_matrix(exp_key, "left"|"right")`), which is the same patch group as its `g_tobii`/`o` target — so image and label always share a frame. Do **not** derive the eye from `get_eye_coords_in_crop`, whose `left`/`right` labels are the **opposite** convention: `left_W` centres the eye that appears on the **image-right** side, `right_W` the image-left side. Mixing the two conventions is a silent mirror-bug. The flip decision keys off the patch name only.

- **Right-eye (`right_W`) crops are horizontally flipped** before being fed to the network; left-eye (`left_W`) crops pass through unflipped.
- **The corresponding target unit vector's x-component is negated** for flipped (originally-right) samples, to stay geometrically consistent with the flipped image.
- At inference/export time, right-eye predictions are flipped back (negate x) before being persisted, so exported vectors are always in the original (non-mirrored) normalized camera space.
- Test coverage: flip-then-unflip is the identity on both image and vector; flipped vectors remain unit-norm; a synthetic pure-x-direction vector correctly negates.

## Validity Policy

A `(exp_key, frame, patch)` sample is included in any split only if **both** hold:
- `get_frame_validity(exp_key)[frame] == True`
- `get_normalized_gaze(exp_key, patch)["validity"][frame] == True`

No partial-validity relaxation or interpolation across invalid frames — this repo consumes EveDataset's validity flags as-is, strictest reading, no leniency.

## R1 Data Pipeline (resolved)

**Spec correction:** `EveBundle.samples_df`'s per-subject split label is exposed as column **`split`**, not `set` — `set` is `SampleTable`'s internal column name (`VALID_SETS = ("train","val","test")`); the public `samples_df` property renames it to `split`. All split logic reads `samples_df["split"]`.

**Split policy:** EVE's `split=="test"` subjects are excluded entirely (no usable ground truth); EVE's `split=="val"` subjects become this project's `test` split untouched; EVE's `split=="train"` subjects are randomly partitioned **by subject** into this project's `train`/`val`, seeded and persisted as a JSON manifest (`{seed, val_fraction, assignment}`).

New modules:

| Module | Location | Functions |
|---|---|---|
| Gaze target conversion | `src/eyenet/gaze_target.py` | `spherical_to_unit(theta, phi)` → `(3,)`/`(N,3)` float32 unit vector |
| Image preprocessing | `src/eyenet/preprocessing.py` | `preprocess_eye_crop(image)` → `(3,128,128)` float32 tensor, ImageNet-normalized |
| Sample index | `src/eyenet/sampling.py` | `build_sample_index(bundle, exp_keys)` → validity-gated `(exp_key, frame, patch)` DataFrame |
| Split assignment | `src/eyenet/splits.py` | `assign_splits(samples_df)`, `make_train_val_split(train_subjects, val_fraction, seed)`, `save_split(path, split, seed, val_fraction)`, `load_split(path)` |
| Dataset/DataModule | `src/eyenet/dataset.py` | `EyeGazeDataset(bundle, crops_root, sample_index, split_assignment, target_split)` (`torch.utils.data.Dataset`); `EyeGazeDataModule(bundle, crops_root, split_source, batch_size, num_workers)` (`pytorch_lightning.LightningDataModule`) |

`requirements.txt` now also pins `torch`, `torchvision`, `pytorch-lightning`, `pandas`, `opencv-python` (previously implicit/undeclared — `cv2` was already used by `src/eye_norm.py`).

## Export Format (R4)

HDF5, `exp_key`-addressed, mirroring EveDataset's cache conventions:

| Dataset | Shape | dtype | Description |
|---|---|---|---|
| `exp_keys` | `(N,)` | vlen utf-8 | Foreign key into EveDataset's SampleTable/WebCamTable |
| `frame` | `(N,)` | int32 | Frame index within the experiment |
| `patch` | `(N,)` | vlen utf-8 | `"left"` or `"right"` |
| `pred_gaze` | `(N, 3)` | float32 | Predicted unit gaze vector, unflipped back to original camera space |
| `validity` | `(N,)` | bool | Validity flag the prediction was computed under (should be all-True given the validity policy, but stored for traceability) |

Every row is self-describing — no reliance on row order matching any other file. Loading code must verify no duplicate `(exp_key, frame, patch)` triples and raise on mismatch, per the anti-amnesia guard pattern EveDataset uses for its own caches.

## Open Questions

- Pin strategy for `evedataset` (path dependency during co-development vs. versioned wheel once EveDataset's packaging (F4) stabilizes).

## Resolved Design Decisions (F-NORM)

- **F-NORM.0 resolved:** The crop→camera affine is the pure translation `T_inv = [[1,0,x0],[0,1,y0],[0,0,1]]` where `(x0, y0) = bundle.get_crop_origin(exp_key)[t]`. Applied as `H_crop = W @ T_inv`. No rotation or scale needed — the 512×512 face crop is an axis-aligned window from the original frame.
- **Warp directly to 128×128 (F-NORM.2 resolved):** EVE's native eye patch is ~128 px with principal point ≈ (63, 61) (measured, §Patch size / intrinsics). `normalize_eye` warps straight to the 128×128 output — the eye lands ~3 px from centre by construction. No intermediate canvas, no center-crop, no intrinsics rescale; EVE's `focal_norm`/`roiSize` are never consulted. (Superseded the earlier 256×256-canvas-then-center-crop design, which mis-assumed the eye sat at (128,128) and clipped onto the cheek.)

## New Modules (F-NORM)

| Module | Location | Functions |
|---|---|---|
| Eye-image normalization | `src/eye_norm.py` | `compose_warp(W, x0, y0)` → `(3,3) float64`; `normalize_eye(crop, H_crop, out_size=(128,128))` → `(128,128,3) uint8` |

## New Modules (R2 — Model & Training Loop)

| Module | Location | Surface |
|---|---|---|
| Angular loss & metric | `src/eyenet/losses.py` | `EPS = 1e-7`; `angular_loss(pred, target)` → scalar, radians; `angular_error_degrees(pred, target)` → `(B,)`, degrees. Both take `(B,3)` and raise `ValueError` otherwise. |
| Model | `src/eyenet/model.py` | `GazeResNet18(pretrained=True, hidden_dim=256, dropout=0.5)` — `forward(x)`: `(B,3,128,128)` → `(B,3)` unit vector |
| Lightning module | `src/eyenet/lightning_module.py` | `GazeEstimationModule(pretrained=True, lr=1e-4, weight_decay=0.0, hidden_dim=256, dropout=0.5)` — `training_step`/`validation_step`/`test_step`/`configure_optimizers` |
| Training entrypoint | `scripts/train.py` | `main(config_path)`; CLI `py scripts/train.py --config configs/baseline.yaml` |
| Baseline run config | `configs/baseline.yaml` | `data` / `model` / `trainer` / `output` blocks; `trainer:` is passed to `pl.Trainer` verbatim |

### The `EPS` clamp in `losses.py` (do not remove)

`arccos`' derivative `-1/sqrt(1-x²)` diverges at `cos = ±1`. A perfect prediction is not hypothetical — it is what training steers toward, and float32 rounding reaches exactly `1.0` first. Unclamped, the first such batch emits NaN gradients that propagate through every weight on the optimizer step, and the run silently finishes with a dead model. `EPS = 1e-7` caps `|grad| ≈ 2236`. Both `angular_loss` and `angular_error_degrees` clamp through one shared `_cos` helper, so there is no path to an unclamped `arccos`. `tests/test_losses.py::test_no_nan_gradient_at_cos_one` pins this and must not be weakened.

### Run artifacts

| Path | Content |
|---|---|
| `<output.dir>/checkpoints/last.ckpt`, `epoch=..-val_angular_error_deg=...ckpt` | Weights + hparams (`ModelCheckpoint`, `monitor="val/angular_error_deg"`, `mode="min"`, `save_top_k=1`) |
| `<output.dir>/csv/version_*/metrics.csv` | Loss/metric history. Columns: `epoch`, `step`, **`train/loss_step`**, **`train/loss_epoch`** (logged `on_step`+`on_epoch` ⇒ Lightning splits it; there is no bare `train/loss`), `val/loss`, `val/angular_error_deg`. |

Checkpoints are weights, not datasets — R2 writes no `exp_key`-addressed artifact, so Mission.md §3's positional-coupling rule binds at R4. R1's `(exp_key, frame, patch)` batch metadata passes through the training step untouched (`tests/test_batch_keys.py`); note `default_collate` delivers `exp_key`/`patch` as **tuples** of `str` and `frame` as an int tensor.

## New Modules (F-WANDB)

| Module | Location | Surface |
|---|---|---|
| Gaze-target inverse | `src/eyenet/gaze_target.py` | `unit_to_spherical(g: torch.Tensor) -> torch.Tensor` — `(B,3)`/`(3,)` unit vector → `(B,2)`/`(2,)` `[theta, phi]` radians. Torch, batched; the numpy `spherical_to_unit` is unmodified. Shares `EPS=1e-7` with `losses.py` for the `arcsin` pole clamp. |
| Logger composition | `scripts/train.py` | `build_loggers(cfg: dict, out: Path) -> list` — `CSVLogger` always `logger[0]`; appends `WandbLogger` only when `logging.wandb.enabled` is `true` **and** `WANDB_API_KEY` is set **and** construction succeeds. Never raises — degrades to CSV-only with a `warnings.warn` on any failure. `wandb`/`WandbLogger` imported locally, only on the enabled path. |

**Config schema addition (`logging:` block, `configs/*.yaml`):**
```yaml
logging:
  wandb:
    enabled: false      # default; absent block is equivalent
    project: eyenet
    entity: null         # null => default entity for WANDB_API_KEY
    run_name: null       # null => W&B generates one
    tags: []
```

**`WANDB_API_KEY` contract:** authentication is by environment variable only — `scripts/train.py` never calls `wandb.login()` and never prompts; the key is read by `wandb` itself. Set it in the job's submit script (not in `configs/*.yaml`, which is version-controlled) or in `~/.netrc`. A missing key with `enabled: true` warns and continues with `CSVLogger` only — it never fails a queued run. `WANDB_MODE=offline` is a supported *environment* escape hatch for no-outbound-network compute nodes; it requires no code or config change.

**New logged metric names (`GazeEstimationModule`, epoch-level unless noted):** `train/angular_error_deg` (new; `val/angular_error_deg` unchanged from R2), `{train,val}/pred_var_{x,y,z}` (skipped if the epoch has <2 samples), `{train,val}/angular_error_deg_{left,right}` (skipped if that patch has 0 samples that epoch), `{train,val}/{theta,phi}_error_deg` (`phi` diff wrapped via `atan2(sin, cos)` before `abs()` to avoid a ~360° branch-cut artifact).

## New Modules (F-OPTUNA — hyperparameter search)

| Module | Location | Surface |
|---|---|---|
| Loss additions | `src/eyenet/losses.py` | `cosine_loss(pred, target)` → scalar, `1 - ⟨x,x̂⟩` meaned, range `[0,2]`; `get_loss(name)` → callable, `{"angular", "cosine"}`, `ValueError` otherwise. **Single source of truth for the `loss` config value** — no other module maps loss names. |
| Logger composition | `src/eyenet/logging_utils.py` | `build_loggers(cfg, out)` — **moved here from `scripts/train.py`** (which re-exports the name) so `train.py` and `hpo.py` share it without a `scripts`-as-package shim. Behavior identical to F-WANDB's. Plus `finish_wandb_run()` — the symmetric close; see §One W&B run per trial. |
| Search | `src/eyenet/hpo.py` | `suggest_params(trial, search_space)`, `build_sampler(cfg)`, `build_pruner(cfg)`, `build_loggers_for_trial(cfg, out, n)`, `trial_loggers(cfg, out, n)` (context manager), `build_objective(cfg, bundle, datamodule)`, `PruningCallback(trial, monitor)` |
| Study driver | `scripts/tune.py` | `main(config_path) -> optuna.Study`; CLI `py scripts/tune.py --config configs/optuna.yaml` |
| Search config | `configs/optuna.yaml` | `data`/`model`/`trainer`/`output`/`logging` (same schema as `baseline.yaml`) plus the `optuna:` block |

**Model signature changes (additive, backward-compatible):**
- `GazeResNet18(pretrained, hidden_dim, dropout, dropout1=None, dropout2=None)` — the two head dropouts are independent. `dropout` is a **shim**: alone it fills both (exact R2 behavior); explicit `dropout1`/`dropout2` win per-layer.
- `GazeEstimationModule(..., dropout1=None, dropout2=None, loss="angular")` — same shim, plus config-selected loss resolved through `get_loss` **at construction** (a bad name fails fast, not mid-run). R2-era checkpoints, which stored only `dropout`, still load.

### Pruning callback — why `optuna-integration` is not a dependency

`optuna_integration.PyTorchLightningPruningCallback` subclasses `lightning.pytorch.Callback` — the standalone **`lightning`** distribution, a different import root from this repo's **`pytorch_lightning`**. Installing it would place two Callback/Trainer hierarchies in one environment, and `pl.Trainer` rejects the foreign base class (`ImportError: Tried to import 'lightning'`). `hpo.PruningCallback` reimplements the ~10-line single-process logic against the Lightning this repo uses: report `objective_metric` to the trial each validation epoch, `raise optuna.TrialPruned()` when `should_prune()`, and **skip Lightning's pre-epoch-0 sanity-check validation pass** (reporting there would double-report step 0 and skew the pruner). DDP is out of scope, so the integration's distributed branch is not reproduced.

### One W&B run per trial — `finish_wandb_run` is load-bearing

`WandbLogger` is lazy: `wandb.init()` fires on first `.experiment` access, and that property
first checks the **process-global `wandb.run`** — if it is non-None it *reuses* that run (warning
only) instead of starting a new one. `WandbLogger.finalize()` uploads checkpoint artifacts and
**never calls `wandb.finish()`**. So in a single-process study, a fresh per-trial `WandbLogger`
is not enough: trial 0 opens the run and every later trial silently appends to it — interleaved
curves, `trainer/global_step` resetting each trial, and one `hparams` config for N configurations.

`logging_utils.finish_wandb_run()` closes the global run, and `hpo.trial_loggers` is the context
manager that pairs it with `build_loggers_for_trial` so open and close cannot drift apart — the
`finally` fires on a normal return, on `TrialPruned` (from `PruningCallback` or the missing-metric
path), and on an OOM `RuntimeError`. It is keyed off `"wandb" in sys.modules` so the disabled path
still never imports wandb (FR21), and it warns rather than raises (FR25).

Per-trial hyperparameters need no extra plumbing — `GazeEstimationModule.save_hyperparameters()`
puts each trial's sampled params in its own run's config once runs are separate.
`build_loggers_for_trial` also appends the `study_name` to the run's `tags` (deduplicated), so
one study's trials filter together in the W&B UI.

### F-OPTUNA config schema (`optuna:` block)

```yaml
optuna:
  study_name: eyenet-hpo
  storage: null            # null => in-memory, fresh each run;
                           # sqlite:///runs/hpo/study.db => persistent, resumes (load_if_exists)
  direction: minimize
  objective_metric: val/angular_error_deg   # degrees: comparable across losses
  n_trials: 30
  timeout: null
  sampler: {name: tpe, seed: 42}            # tpe | random
  pruner:  {name: median, n_warmup_steps: 1, n_startup_trials: 5}   # median | asha | none
  search_space:            # entirely config-driven; drop a key to drop the dimension
    dropout1:     {type: float, low: 0.0, high: 0.7}
    dropout2:     {type: float, low: 0.0, high: 0.7}
    weight_decay: {type: float, low: 1.0e-6, high: 1.0e-2, log: true}
    hidden_dim:   {type: categorical, choices: [128, 256]}
    loss:         {type: categorical, choices: [angular, cosine]}
```

`search_space` types are `float`/`int`/`categorical`; a searched key overrides the `model:` block's fixed value for that trial, and removing it falls back to `model:` with no code change. Empty/absent `search_space` is a `ValueError` — a search with nothing to search is a config error, not a silent single-config run.

### Run artifacts (F-OPTUNA)

| Path | Content |
|---|---|
| `<output.dir>/best_params.yaml` | `study_name`, `best_value`, `best_trial_number`, `best_params` — **the handoff**: paste `best_params` into a `configs/*.yaml` `model:` block for R3's full run. Written whenever ≥1 trial completed; skipped with a warning if none did. |
| `<output.dir>/csv/version_*/metrics.csv` | Per-trial metric history (CSVLogger, unchanged) |
| `<output.dir>/wandb/` | Per-trial W&B runs (named `{base}-t{n}`), only when enabled + key present |

F-OPTUNA writes **no** `exp_key`-keyed artifact — its output is hyperparameters, not predictions — so Mission.md §3's positional-coupling rule still binds at R4, not here.

### Run artifacts (F-WANDB addition)

`<output.dir>/wandb/` appears alongside `csv/` and `checkpoints/` only when `logging.wandb.enabled` is `true` and a `WandbLogger` was successfully constructed (or when `WANDB_MODE=offline` buffers a run there). `csv/version_*/metrics.csv` is unmoved — `CSVLogger` stays `logger[0]`, so `trainer.log_dir` and `ModelCheckpoint`'s default directory are unaffected by whether W&B is enabled.
