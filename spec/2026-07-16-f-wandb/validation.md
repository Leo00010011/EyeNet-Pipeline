# F-WANDB — Validation

## Code Correctness

### Group 1 — `unit_to_spherical`: round-trip and convention

The Roadmap flags this as a prime sign-bug candidate, same class as F-FLIP. These are the tests that hold the numpy/torch pair honest.

- [ ] **Round-trip through both backends (the load-bearing test).** For a grid of `theta ∈ linspace(-0.6, 0.6, 13)` × `phi ∈ linspace(-1.2, 1.2, 13)` (169 cases spanning EVE's realistic gaze range): `spherical_to_unit(theta, phi)` → `torch.from_numpy` → `unit_to_spherical` recovers `(theta, phi)` with `atol=1e-5`. A sign error in **either** function fails this. Failure mode if broken: the per-axis metrics report a plausible-looking but wrong decomposition, and R3's `theta`-correlation re-check is measured against a mirrored axis.
- [ ] **Round-trip vector → spherical → vector.** 200 seeded random unit vectors with `g_z < 0` (the forward-facing half-space the MPIIGaze convention covers), through `unit_to_spherical` then `spherical_to_unit`, recover `g` with `atol=1e-5`.
- [ ] **Hand-computed: straight ahead.** `g = [0, 0, -1]` → `theta ≈ 0`, `phi ≈ 0`, `atol=1e-6`.
- [ ] **Hand-computed: pure horizontal.** `g = [-1, 0, 0]` → `theta ≈ 0`, `phi ≈ pi/2`, `atol=1e-6`. Pins that `phi` is the **horizontal** axis — swapping `theta`/`phi` passes the straight-ahead case and fails here.
- [ ] **Hand-computed: pure vertical.** `g = [0, -1, 0]` → `theta ≈ pi/2`, `atol=1e-3` (the FR2 `EPS=1e-7` clamp costs `arcsin` accuracy at the pole; `phi` is unconstrained there and is **not** asserted).
- [ ] **No NaN at the pole.** `unit_to_spherical(torch.tensor([[0., -1., 0.]]))` is finite in every component of `theta`. Pins the FR2 clamp; without it float32 `-g_y = 1.0` exactly and `arcsin` is at its divergence point. This test must not be weakened — same standing as `tests/test_losses.py::test_no_nan_gradient_at_cos_one`.
- [ ] **Shapes.** `(B,3)` → `(B,2)`; `(3,)` → `(2,)`.
- [ ] **dtype and device preserved.** float32 in → float32 out; input `.device` == output `.device`.
- [ ] **Errors.** `(4,2)` raises `ValueError`; `(2,3,3)` raises `ValueError`; the message contains the offending shape.
- [ ] **`spherical_to_unit` is byte-for-byte unmodified** (FR5) — `git diff` on `gaze_target.py` shows additions only within that function's boundaries untouched. The existing `tests/test_gaze_target.py` cases pass with no edit.

### Group 2 — Per-eye aggregation (metric 3)

- [ ] **Hand-computed per-eye means.** A synthetic 5-tuple epoch with known per-sample errors and a **lopsided** patch split (batch 1 = 3×`left`, batch 2 = 1×`left` + 3×`right`). Assert `val/angular_error_deg_left` equals the mean over all **4** left samples (`atol=1e-4`).
- [ ] **Epoch-level, not per-batch (pins FR13).** With the same lopsided epoch, assert the logged `_left` is **not** equal to `mean(batch1_left_mean, batch2_left_mean)` — the two differ by construction (3 samples vs 1). This is the test that catches an implementation that averages batch means; it would pass every other test in this group.
- [ ] **Single-patch epoch (FR15).** An all-`left` epoch logs `val/angular_error_deg_left` and **`val/angular_error_deg_right` is absent** from `trainer.logged_metrics`. No `nan` is logged and no exception is raised.
- [ ] **`patch` handled as a tuple of `str` (FR14).** The synthetic batches pass `patch` as a `tuple`, matching `default_collate`'s confirmed R2 behavior. An implementation assuming tensor indexing raises here.
- [ ] **2-tuple batches unaffected (FR16).** The 7 existing `tests/test_lightning_module.py` tests — which drive `(image, target)` batches only — pass **with no modification**, and no per-eye key appears in `logged_metrics`.
- [ ] **Buffers reset between epochs.** A 2-epoch run where epoch 2's errors differ from epoch 1's yields epoch-2 metrics computed from epoch-2 samples only (no leakage). Assert the two epochs' logged values differ as hand-computed.

### Group 3 — Per-axis decomposition (metric 4)

- [ ] **`theta`-only offset.** Prediction and target sharing `phi` but differing in `theta` by a known 3°: `val/theta_error_deg ≈ 3` (`atol=1e-3`) and `val/phi_error_deg ≈ 0` (`atol=1e-3`).
- [ ] **`phi`-only offset.** The mirror case: `val/phi_error_deg ≈ 3`, `val/theta_error_deg ≈ 0`. Together with the previous case this pins that the two axes are not swapped — a swap passes a pooled-error test and both round-trip tests.
- [ ] **`phi` wraparound (pins FR19).** Two unit vectors straddling the `atan2` branch cut at `±pi`, ~1° apart in true angle: `val/phi_error_deg ≈ 1` with `atol=0.1`, **not ≈359**. Failure mode if unwrapped: a handful of frames per epoch inject ~360° into the mean and `phi_error_deg` becomes unreadable noise.
- [ ] **Degrees, not radians.** A known 10° offset reports ≈10, not ≈0.175. Cheap, catches a missing `rad2deg`.

### Group 4 — Output variance (metric 2)

- [ ] **Collapse ⇒ ~zero variance.** An epoch of a constant predicted vector yields `val/pred_var_x/y/z` all `< 1e-6`.
- [ ] **Spread ⇒ non-zero variance.** An epoch of varied predictions yields clearly non-zero variance in every component. The assertion is the **contrast** with the collapse case, not an absolute threshold — no threshold is defensible before R3 measures a real distribution.
- [ ] **Per-component, not pooled (FR9).** A synthetic epoch varying only in `x` (constant `y`, `z`) yields `pred_var_x > 0` while `pred_var_y ≈ pred_var_z ≈ 0`. This is the shape of R2's actual observed failure (`phi` learning, `theta` flat) and a pooled scalar would hide it.
- [ ] **Single-sample epoch logs no variance (FR11).** `limit_val_batches=1` with `batch_size=1`: the `pred_var_*` keys are **absent** from `logged_metrics`, no `nan` is logged, no exception raised.
- [ ] **Variance is on the raw (canonical-frame) output (FR10).** The value is computed from the tensor the loss sees — no unflip is applied. Assert by driving a known right-patch batch and confirming the variance matches the un-unflipped hand computation.

### Group 5 — Logger composition and headless operation

No test in this group constructs a real `WandbLogger` or touches the network. TechStack §Key Libraries records `wandb` as "never exercised in tests" and this group upholds that.

- [ ] **Disabled ⇒ CSV only (FR21).** `build_loggers({"logging": {"wandb": {"enabled": False}}}, tmp_path)` returns exactly `[CSVLogger]`.
- [ ] **No `wandb` import on the disabled path (FR21).** After the disabled call, `"wandb" not in sys.modules` — skipped if another test already imported it, so the assertion is guarded rather than order-dependent. This is what keeps the suite runnable on a machine with no `wandb` account and no network.
- [ ] **Missing `logging` block ⇒ disabled (FR22).** `build_loggers({}, tmp_path)` returns `[CSVLogger]` with no warning and no error. `configs/baseline.yaml` was valid before this feature and stays valid.
- [ ] **Enabled + no `WANDB_API_KEY` ⇒ warn, degrade (FR25).** With `monkeypatch.delenv("WANDB_API_KEY", raising=False)`: `pytest.warns(UserWarning, match="WANDB_API_KEY")`, result is `[CSVLogger]`, and **no exception propagates**. This is the queued-job contract — the run must survive a missing credential.
- [ ] **Enabled + constructor raises ⇒ warn, degrade (FR26).** `WANDB_API_KEY` set to a fake value and `WandbLogger` monkeypatched to raise `RuntimeError`: result is `[CSVLogger]`, a `UserWarning` carries the original message, no exception propagates.
- [ ] **`CSVLogger` is always `logger[0]` (FR23).** `isinstance(loggers[0], CSVLogger)` in every branch above. Lightning derives `trainer.log_dir` from `logger[0]`; a reorder silently relocates every artifact documented in TechStack §Run artifacts.
- [ ] **End-to-end regression (the gate).** The existing `tests/test_train_script.py` 2-batch real-bundle run passes unmodified against `configs/baseline.yaml`, offline, with `WANDB_API_KEY` unset, and `metrics.csv` still carries `train/loss_step` / `train/loss_epoch` / `val/loss` / `val/angular_error_deg`.
- [ ] **`train/loss` column names unchanged (FR8).** `metrics.csv` has `train/loss_step` and `train/loss_epoch` and **no bare `train/loss`** — confirming the new epoch-end logging did not perturb R2's `on_step`/`on_epoch` flags.
- [ ] **New metrics reach `metrics.csv`.** The same run's `metrics.csv` contains `train/angular_error_deg`, `val/pred_var_x`, and `val/theta_error_deg` columns. `CSVLogger` receiving the new metrics is the offline proof that `WandbLogger` would too — the metrics are logger-agnostic.

### Group 6 — Non-interference

- [ ] **Full suite green.** All 79 pre-existing tests (+1 known skip) pass, plus the new ones. Zero regressions.
- [ ] **Untouched files (Roadmap F-WANDB §Requirements).** `git diff --stat` shows **no changes** to `src/eyenet/losses.py`, `model.py`, `dataset.py`, `sampling.py`, `splits.py`, or `src/eye_norm.py`. Modified: `gaze_target.py` (additive), `lightning_module.py`, `scripts/train.py`, `configs/baseline.yaml`, `requirements.txt`, tests, constitution docs.
- [ ] **`GazeEstimationModule.__init__` signature unchanged.** R2's checkpoint round-trip test loads an existing `runs/baseline` checkpoint into the new module without error — `save_hyperparameters()` recorded `(pretrained, lr, weight_decay)` and the new buffers are not hyperparameters.

## Data Validity

Checks against a real run, not synthetic tensors. Run `scripts/train.py` on the R2 baseline config (2 epochs × 50 batches) with `logging.wandb.enabled: true` and `WANDB_API_KEY` exported.

- [ ] **The dashboard exists and is live.** A run appears under the configured project, and `train/angular_error_deg` and `val/angular_error_deg` are plottable **on one chart** (metric 1's actual requirement — both are scalars logged at epoch granularity, so W&B will co-plot them against `epoch`).
- [ ] **W&B and CSV agree.** For each epoch, `val/angular_error_deg` in the W&B run matches the same column in `<output.dir>/csv/version_*/metrics.csv` to `atol=1e-4`. Both loggers receive the identical `self.log` call; a mismatch means one of them is reading a different reduction. This is the cross-check that makes the dashboard trustworthy.
- [ ] **Degrees are plausible.** `val/angular_error_deg` lands in single-digit degrees (R2 measured ≈5.4° on this exact config). ⚠️ Per the Roadmap R2 note, **this number is not a result and must not be quoted as one or compared to published 4–7° baselines** — the check here is only that the unit is degrees and the magnitude is not absurd (e.g. not 0.09, which would mean radians leaked through).
- [ ] **Variance is non-zero on a real run.** `val/pred_var_x/y/z` are all clearly `> 0` after 2 epochs. Given R2's `theta` sat at r≈0, a **small `pred_var_y`** here is an expected and informative observation, not a test failure — record the measured values in the Roadmap as R3's starting reference.
- [ ] **Per-eye errors are both present and comparable.** Both `val/angular_error_deg_left` and `val/angular_error_deg_right` appear (a real epoch contains both patches). A **large persistent gap would indicate an F-FLIP break** — but note this run inherits R2's already-validated F-FLIP path, so a gap here means *this feature's masking* is wrong, not F-FLIP. Record the measured gap.
- [ ] **Per-axis errors are consistent with the pooled error.** `theta_error_deg` and `phi_error_deg` are each on the order of the pooled `angular_error_deg` (not 10× larger, which would indicate a `phi` wraparound leak, and not ~0, which would indicate the axes are being compared against themselves).
- [ ] **Per-axis matches the notebook's inline prototype.** `notebooks/inspect_r2_training.ipynb` prototypes `unit_to_spherical` inline. Decode the same checkpoint's predictions with both the notebook's inline version and the new module function; assert they agree to `atol=1e-5`. This validates the module against the code that produced R2's reported r≈+0.68 / r≈0 figures.
- [ ] **Offline escape hatch works (the no-internet contingency).** With `WANDB_MODE=offline` exported, the run completes, writes `<output.dir>/wandb/offline-run-*`, and `wandb sync <that path>` from a networked machine produces a dashboard with the same metrics. Confirms the §Running unattended step-4 fallback before R3 needs it in anger.
- [ ] **Unattended job dry-run (the actual deliverable of the queue requirement).** Submit the baseline config as a real queued job with `export WANDB_API_KEY=...` in the submit script and **no terminal attached**. Confirm: no login prompt, no hang waiting on stdin, the dashboard populates while the job runs, and the job exits 0. This is the check that the whole headless path works — everything else in this group is testable interactively and would not catch a prompt.
- [ ] **No credential leaks into version control.** `git grep` for the key's value across the repo returns nothing; `configs/*.yaml` contains no key; `requirements.txt` diff is the single `wandb` line. The key lives in the environment or `~/.netrc` (mode `600`) only.

## Data Architecture Integrity

Mission §3's positional-coupling rule binds at **R4**, when an `exp_key`-addressed artifact is first written. F-WANDB writes no dataset — it logs scalars. The relevant invariant here is narrower but real: this feature is the **first consumer of `patch` outside R4's export**, so it is the first place the key metadata can be silently mis-associated.

- [ ] **`patch` is read from the batch, never inferred.** The per-eye mask derives from `batch[4]` only. No positional assumption (e.g. "even indices are left"), no derivation from the image, no re-derivation from `get_eye_coords_in_crop` — whose left/right labels are the **opposite** convention (TechStack §Left/Right Flip Convention), and mixing them is a silent mirror bug.
- [ ] **`patch` alignment with the sample is preserved.** A synthetic batch where per-sample errors and patch labels are deliberately correlated (all `left` samples have error 1°, all `right` have 10°, interleaved within one batch) yields exactly `_left ≈ 1` and `_right ≈ 10`. An off-by-one or a mis-zipped mask produces a blend and fails. This is the positional-coupling test in miniature: the mask must track the sample, not its index.
- [ ] **Metadata passes through untouched.** `tests/test_batch_keys.py`'s 4 key-path integrity tests pass unmodified. This feature **reads** `batch[4]`; it must not mutate, reorder, consume, or otherwise disturb the `(exp_key, frame, patch)` tuple that R4's export depends on.
- [ ] **The eye determination stays with the `W`-patch name.** The `patch` string in the batch originates from `build_sample_index`, which keys off the `get_warp_matrix` patch name — the same patch group as the `g_tobii` target. This feature introduces **no new eye-determination path**; confirm by grep that `get_eye_coords_in_crop` appears nowhere in `lightning_module.py`.
- [ ] **Reorder-invariance.** Shuffling a synthetic epoch's sample order (and its patch labels with it) yields identical per-eye and per-axis metrics to `atol=1e-5`. Epoch-level means are order-independent by construction; this pins that the implementation did not sneak in an order-dependent reduction.
- [ ] **No new artifact, no new keying surface.** `git status` after a run shows the only new run-directory content is `wandb/` (W&B's own buffer) alongside the existing `csv/` and `checkpoints/`. No `exp_key`-addressed file is written, so no anti-amnesia guard is required by this feature.
