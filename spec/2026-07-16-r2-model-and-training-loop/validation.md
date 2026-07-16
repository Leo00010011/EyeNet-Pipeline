# R2 — Model & Training Loop — Validation

## Code Correctness

### Group 1 — Angular loss, hand-computed (`tests/test_losses.py`)

Every case builds its expected value by hand, not from a second implementation of the same formula.

- [ ] **Identical vectors → 0.** `angular_loss(t([[0.,0.,-1.]]), t([[0.,0.,-1.]]))` returns a scalar `< 1e-3` rad. Not exactly 0: the `EPS=1e-7` clamp floors it at `arccos(1-1e-7) ≈ 4.5e-4` rad. Asserting `== 0` would fail.
- [ ] **Orthogonal → π/2.** `angular_loss([[1.,0.,0.]], [[0.,1.,0.]]) ≈ pi/2`, `atol=1e-5`.
- [ ] **Orthogonal → 90°.** `angular_error_degrees([[1.,0.,0.]], [[0.,1.,0.]]) ≈ [90.0]`, `atol=1e-3`, shape `(1,)`.
- [ ] **60° case.** `angular_error_degrees([[1.,0.,0.]], [[0.5, sqrt(3)/2, 0.]]) ≈ [60.0]`, `atol=1e-3`.
- [ ] **Opposed → 180°.** `angular_error_degrees([[1.,0.,0.]], [[-1.,0.,0.]]) ≈ [180.0]`, `atol=1e-2` (the clamp costs ≈0.026° at the endpoint).
- [ ] **Non-unit input is normalized internally (FR2).** `angular_error_degrees([[2.,0.,0.]], [[0.,5.,0.]]) ≈ [90.0]`, `atol=1e-3` — magnitude must not affect the angle.
- [ ] **Batch shape and values.** A `(4,3)` batch mixing the 0°/60°/90°/180° cases → `angular_error_degrees` returns shape `(4,)`, dtype float32, matching `[≈0, ≈60, ≈90, ≈180]` elementwise; `angular_loss` equals `deg2rad(those).mean()` within `atol=1e-4`.

### Group 2 — Numerical safety (the critical group)

- [ ] **No NaN gradient at cos = 1.** `pred = torch.tensor([[1.,0.,0.]], requires_grad=True)`; `angular_loss(pred, torch.tensor([[1.,0.,0.]])).backward()`; assert `torch.isfinite(pred.grad).all()`. **Without the FR3 clamp this produces `NaN` and fails.** This test is the whole justification for `EPS`; it must not be weakened or removed.
- [ ] **No NaN gradient at cos = −1.** Same with `target = [[-1.,0.,0.]]`; `pred.grad` finite.
- [ ] **Gradient magnitude is bounded.** At `cos = 1`, `pred.grad.abs().max() < 1e4` — confirms the clamp caps the divergence rather than merely deferring it.
- [ ] **Loss is finite over a random batch.** 128 random (non-normalized, including near-zero-norm) rows → `angular_loss` finite, `angular_error_degrees` all finite and within `[0, 180]`.

### Group 3 — Loss error paths

- [ ] `angular_loss(torch.zeros(3), torch.zeros(3))` raises `ValueError` whose message names `(B, 3)` — unbatched input is rejected, not silently broadcast.
- [ ] `angular_loss(torch.zeros(2,4), torch.zeros(2,4))` raises `ValueError` (last dim ≠ 3).
- [ ] `angular_loss(torch.zeros(2,3), torch.zeros(3,3))` raises `ValueError` naming both shapes.

### Group 4 — Model (`tests/test_model.py`, all `pretrained=False`)

- [ ] **Shape.** `GazeResNet18(pretrained=False)(torch.randn(2,3,128,128))` → shape `(2,3)`, dtype float32.
- [ ] **Unit-norm output (FR6).** Every row of that output has `‖v‖ = 1.0`, `atol=1e-5`.
- [ ] **128×128 needs no resize (FR8).** The forward pass on a 128×128 input completes without error — no `Resize` anywhere in the module; assert `not any(isinstance(m, torch.nn.Upsample) for m in model.modules())` and that the head is `Linear(512, 3)`.
- [ ] **Gradients flow to the head.** `out.sum().backward()` → `model.backbone.fc.weight.grad` is not `None` and is finite.
- [ ] **Zero input is safe.** `model(torch.zeros(1,3,128,128))` → all-finite output (the `eps=1e-8` normalize guard; a manual `v/v.norm()` would risk `NaN`).

### Group 5 — Lightning module (`tests/test_lightning_module.py`, synthetic batches)

- [ ] **`training_step` returns a differentiable finite scalar.** Batch `(randn(4,3,128,128), unit(4,3))` → scalar, `requires_grad=True`, finite.
- [ ] **5-tuple R1 batch is accepted (FR11).** `(image, target, ["k1"]*4, tensor([0,1,2,3]), ["left"]*4)` → `training_step` succeeds and returns the same value as the 2-tuple form; the metadata is ignored, not consumed.
- [ ] **Loss decreases (Roadmap R2 acceptance).** `Trainer(overfit_batches=1, max_epochs=30, accelerator="cpu", logger=False, enable_checkpointing=False)` on a fixed 4-sample synthetic set → final `train/loss` < 0.5 × first-epoch `train/loss`. Fails if the optimizer is misconfigured or the graph is detached.
- [ ] **No NaNs through a real optimizer step.** After those 30 epochs, every parameter satisfies `torch.isfinite(p).all()`.
- [ ] **Checkpoint round-trip (Roadmap R2 acceptance).** `trainer.save_checkpoint(p)` → `GazeEstimationModule.load_from_checkpoint(p)` restores `hparams.lr` exactly and, in `eval()` mode on a fixed input, reproduces the original output within `atol=1e-6`.
- [ ] **Optimizer config (FR13).** `configure_optimizers()` returns `torch.optim.Adam` with `param_groups[0]["lr"] == 3e-4` when constructed with `lr=3e-4`, and no scheduler is returned.
- [ ] **No bundle dependency (FR14).** The whole test module imports and passes with no `sample_bundle` fixture — assert by construction (the file must not import `evedataset` or any conftest bundle fixture).

### Group 6 — Training script (`tests/test_train_script.py`)

- [ ] **End-to-end 2-batch run.** A `tmp_path` config (`max_epochs: 1, limit_train_batches: 2, limit_val_batches: 1, num_workers: 0, pretrained: false`) against the real `sample_bundle` / `face_crops_root` fixtures → `main(cfg)` returns without error.
- [ ] **Checkpoint written.** `<out>/checkpoints/last.ckpt` exists and is loadable via `GazeEstimationModule.load_from_checkpoint`.
- [ ] **CSV metrics written (FR18).** `<out>/csv/version_0/metrics.csv` exists, has a `train/loss` column, and every value in it is finite. This is the artifact that replaces W&B for R2's acceptance.
- [ ] **Bad path fails fast (FR19).** A config with `bundle_dir: /nonexistent` raises `FileNotFoundError` naming that path, and raises before any Trainer/model construction (assert via `monkeypatch` that `pl.Trainer` is never called).
- [ ] **`limit_*` pass-through (FR17).** With `limit_train_batches: 2`, the run's `metrics.csv` contains at most 2 `train/loss` step rows for epoch 0 — proving the subset scoping actually took effect rather than silently training the full split.
- [ ] **No regression in the existing suite.** Full `pytest` run: the current 79 passed / 1 skipped baseline still holds, plus the new R2 tests. No existing test is modified — R2 changes no file under `src/eyenet/` that exists today.

## Data Validity

Checks on the real baseline run (`notebooks/inspect_r2_training.ipynb`, executed via `nbconvert`, outputs persisted). These are sanity checks, not accuracy claims — competitive-baseline evaluation is R3.

- [ ] **Untrained baseline ≈ 90°.** A `pretrained=False`, untrained `GazeResNet18` over one real val batch reports mean angular error in **[75°, 105°]**. Rationale: a random unit 3-vector against a fixed one has an expected angle of 90°. A number far *below* this from an untrained net means the metric is broken; a number near 0° means labels are leaking into the prediction path.
- [ ] **Trained subset model beats the untrained baseline.** After the baseline run, mean val angular error is **< 60°** — well short of a publishable number (EVE/MPIIGaze appearance-based baselines report ≈4–7°), but conclusive that the loop learns from the signal rather than shuffling noise. A value stuck at ≈90° after training indicates image/label desync — the exact failure mode F-NORM and F-FLIP exist to prevent — and must be investigated, not tuned away.
- [ ] **Loss curve is monotone-ish and NaN-free.** `train/loss` from `metrics.csv` has zero `NaN`s and its last-quartile mean is below its first-quartile mean.
- [ ] **Predictions are unit vectors on real data.** Over one full val batch, `‖pred‖ = 1.0 ± 1e-5` for every row — FR6 holding on real inputs, not just synthetic.
- [ ] **Predictions are not collapsed.** The per-component std of predictions across a val batch is `> 1e-3` for at least two of the three components. A model that emits one constant vector can still show a falling loss (it converges to the dataset mean gaze); this check catches that degenerate solution.
- [ ] **Visual arrow overlay.** 8 val eye crops with ground-truth (green) and predicted (red) gaze arrows. Expect predicted arrows loosely tracking ground truth after the baseline run — not aligned, but not anti-correlated. Systematic mirroring (predictions consistently x-flipped relative to truth) would indicate an F-FLIP convention break and is the specific thing to look for.

## Data Architecture Integrity

R2 persists no `exp_key`-addressed dataset — checkpoints are weights, so Mission.md §3's positional-coupling rule has no artifact to bind to here. What R2 must guarantee is that R1's key path survives the training loop intact, so R4 inherits it working.

- [ ] **Key metadata survives collation.** Pull one batch from the real `val_dataloader`: `exp_key` is a length-`B` list of `str`, `frame` a length-`B` int tensor, `patch` a length-`B` list of `"left"`/`"right"` — the `(exp_key, frame, patch)` triple is reconstructable per row, position-independent.
- [ ] **Key metadata is unaltered by the training step.** `training_step` uses `batch[0], batch[1]` only; assert the batch's metadata objects are unchanged after the call (identity/equality check). Nothing in R2 may reorder, re-key, or drop them.
- [ ] **Row `i` of the batch corresponds to row `i` of the metadata.** For a batch drawn with `shuffle=False`, each `(exp_key[i], frame[i], patch[i])` matches the dataset's own `_index` row for that position, and the image at `i` re-derives byte-identically from that triple via the R1 per-item path. This is the invariant R4's export depends on: prediction `i` belongs to key `i`.
- [ ] **No positional coupling introduced by `limit_*`.** `limit_train_batches` truncates *how many* batches are drawn; it must not alter which key belongs to which row. Verify a limited run and an unlimited run yield identical `(exp_key, frame, patch)` triples for the batches they share (fixed seed, `shuffle=False`).
- [ ] **Split assignment is untouched by R2.** Re-run R1's existing zero-subject-overlap check (`tests/test_splits.py`) after R2 lands — train/val/test subject sets remain disjoint. R2 adds no code path that can reassign a subject.
