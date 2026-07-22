# F-OPTUNA — Validation

All Code-Correctness checks run **offline** with **no W&B network** and, except
Group 5, **no real bundle** (synthetic tensors / `FixedTrial`), mirroring the R2
/ F-WANDB testing philosophy. Group 5 uses the sample bundle exactly as
`tests/test_train_script.py` does.

## Code Correctness

### Group 1 — Cosine loss (`tests/test_losses.py`, additions)
- [ ] `cosine_loss(p, p) == 0` (within `atol=1e-6`) for a batch of random
  **unit** vectors — perfect agreement is the minimum.
- [ ] `cosine_loss(p, -p)` `== 2.0` (`atol=1e-6`) — antipodal is the maximum.
- [ ] Hand-computed case: `pred=[[1,0,0]]`, `target=[[0,1,0]]` (orthogonal) ⇒
  `cosine_loss == 1.0` (`atol=1e-6`).
- [ ] Scale-invariance: `cosine_loss(2*p, target) == cosine_loss(p, target)`
  (`atol=1e-6`) — the internal `F.normalize` makes magnitude irrelevant.
- [ ] `_check` is enforced: `cosine_loss` on a `(B,2)` or mismatched-shape input
  raises `ValueError` (same guard as `angular_loss`).
- [ ] **Gradient is finite at perfect agreement:** `pred = target.clone()
  .requires_grad_()`; `cosine_loss(pred, target).backward()`; assert
  `torch.isfinite(pred.grad).all()`. (Confirms the no-clamp claim in FR1 —
  contrast with `angular_loss`, which *needs* the clamp; the two behaviors are
  intentionally different and both must hold.)

### Group 2 — Loss resolver (`tests/test_losses.py`)
- [ ] `get_loss("angular") is angular_loss` and `get_loss("cosine") is cosine_loss`.
- [ ] `get_loss("mse")` raises `ValueError` whose message lists `['angular',
  'cosine']`.

### Group 3 — Independent dropouts + shim (`tests/test_model.py`)
- [ ] `GazeResNet18(dropout1=0.1, dropout2=0.3)` — the head's two `nn.Dropout`
  modules have `.p == 0.1` and `.p == 0.3` respectively (assert by walking
  `backbone.fc`).
- [ ] Shim: `GazeResNet18(dropout=0.4)` (no `dropout1`/`dropout2`) ⇒ both
  Dropout `.p == 0.4` — **preserves R2 behavior**.
- [ ] Precedence: `GazeResNet18(dropout=0.4, dropout1=0.1)` ⇒ first `.p == 0.1`,
  second `.p == 0.4` (explicit wins, shim fills the rest).
- [ ] Forward still returns unit-norm `(B,3)` for `(B,3,128,128)` input under the
  new signature (`‖out‖ = 1 ± 1e-5`), `pretrained=False`.

### Group 4 — Module: loss selection + checkpoint compat (`tests/test_lightning_module.py`)
- [ ] `GazeEstimationModule(loss="cosine")._loss_fn is cosine_loss`;
  `loss="angular"` (default) ⇒ `angular_loss`.
- [ ] `GazeEstimationModule(loss="bogus")` raises `ValueError` **at construction**
  (fail fast, not mid-training).
- [ ] A short overfit run (reuse the existing 30-epoch overfit test pattern) with
  `loss="cosine"` drives `train/loss` down monotonically-ish and ends below its
  first-epoch value — the cosine path actually trains.
- [ ] **R2 checkpoint round-trip still loads:** a checkpoint saved by a module
  constructed **without** `dropout1`/`dropout2`/`loss` (R2-era hparams) loads
  via `load_from_checkpoint` and produces identical forward output — the new
  kwargs default cleanly (`loss="angular"`, dropouts via shim).
- [ ] `angular_error_degrees` is still what `val/angular_error_deg` logs even
  when `loss="cosine"` (assert the logged metric equals a hand-recomputed
  `angular_error_degrees(pred, target).mean()`, independent of the loss).

### Group 5 — `suggest_params` & sampler/pruner builders (`tests/test_hpo.py`, new)
- [ ] `suggest_params(optuna.trial.FixedTrial({...}), search_space)` returns the
  fixed values with correct dispatch: a `float` dim comes back a float, a
  `categorical` `hidden_dim` comes back one of `{128,256}`, `loss` one of
  `{angular,cosine}`.
- [ ] A `search_space` with `type: "bogus"` raises `ValueError` naming the dim.
- [ ] Empty `search_space` (`{}`) raises `ValueError` (FR19).
- [ ] `build_sampler` returns `TPESampler` / `RandomSampler` for the two names;
  unknown ⇒ `ValueError`. Seed is honored (two `TPESampler(seed=42)` produce the
  same first suggestion on a trivial study).
- [ ] `build_pruner` returns `MedianPruner` / `SuccessiveHalvingPruner` /
  `NopPruner` for `median`/`asha`/`none`; unknown ⇒ `ValueError`.

### Group 6 — End-to-end study (`tests/test_tune_script.py`, new; real sample bundle)
- [ ] `main("<tmp optuna.yaml>")` with `n_trials: 2`, `storage: null`,
  `wandb.enabled: false`, `trainer` limited to `max_epochs: 1,
  limit_train_batches: 2, limit_val_batches: 1` completes and returns an
  `optuna.Study` with `len(study.trials) == 2`, **no network access**.
- [ ] `study.best_value` is a finite float and equals a completed trial's
  `val/angular_error_deg` order of magnitude (0–180°).
- [ ] `<output.dir>/best_params.yaml` exists and contains `best_params` whose
  keys are exactly the `search_space` keys; `best_value` matches `study.best_value`.
- [ ] Bad `data.bundle_dir` ⇒ `FileNotFoundError` **before** any trial runs
  (guard fires at `main` entry, same as `train.py`).
- [ ] With `pruner.name: none`, a 2-trial study still completes and records 0
  `PRUNED` trials (Nop pruner path exercised).
- [ ] `catch` path: a trial that raises `RuntimeError` (inject via a monkeypatched
  objective for one trial) is recorded `FAIL`, the study still finishes the
  remaining trial, and `best_params.yaml` is still written (FR20/FR17).

### Group 7 — Logger reuse / W&B gating (`tests/test_tune_script.py`)
- [ ] With `wandb.enabled: false` the Trainer for each trial receives exactly one
  logger and it is a `CSVLogger` (no `WandbLogger`, no network) — inherits the
  F-WANDB contract through the shared `build_loggers`.
- [ ] Per-trial run-name derivation: the composed W&B run name for trial `k` ends
  in `-t{k}` (unit-test `build_loggers_for_trial`'s cfg mutation without
  constructing a real `WandbLogger`).

## Data Validity

Run once against the sample bundle (notebook `notebooks/inspect_optuna_search.ipynb`,
executed via `nbconvert`, outputs persisted). These are sanity checks on the
search *behavior*, not accuracy claims — the subset budget is deliberately tiny.
- [ ] A ≥10-trial study on the subset produces a **spread** of `val/angular_error_deg`
  across trials (std > 0) — i.e. the searched dimensions actually move the
  objective; a flat objective would mean the search space or wiring is inert.
- [ ] The pruner prunes at least one trial when ≥10 trials are run with the
  median pruner (`n_startup_trials` respected) — confirms pruning is live, not a
  no-op that silently trains every trial to completion.
- [ ] `best_params` lands **inside** the declared bounds for every float dim and
  within `choices` for every categorical — no out-of-range leak from a bad
  `suggest_*` dispatch.
- [ ] Cross-check both loss branches are reachable: over the study, at least one
  trial trained with `loss=angular` and one with `loss=cosine` (or, for a fixed
  seed, the sampled sequence is recorded) — the categorical loss dimension is
  not silently collapsed to one value.
- [ ] The winning `best_params.yaml`, pasted into a `configs/*.yaml` `model:`
  block, is accepted by `scripts/train.py`'s `main` without error (the handoff
  format is actually consumable by the full-run path).

## Data Architecture Integrity

- [ ] **Split policy untouched:** the study builds **one** `EyeGazeDataModule`
  and reuses it for every trial; assert (in Group 6) that the datamodule's
  `split_source` is identical across trials and that no F-OPTUNA code imports
  `build_sample_index`/`assign_splits` or passes anything but `limit_*_batches`
  to vary trial cost. Trials differ only in hyperparameters, never in data.
- [ ] **No positional coupling introduced:** F-OPTUNA writes no `exp_key`-keyed
  prediction artifact (that is R4); its only output is `best_params.yaml`
  (hyperparameters) + CSV metrics. Confirm no prediction/label array is persisted
  by row position anywhere in the search path.
- [ ] **Objective is loss-invariant:** assert the objective metric string is
  `val/angular_error_deg` (degrees), not a raw loss column, so trials using
  different losses are compared on the same physical quantity (guards against a
  regression that points the study at `val/loss`).
- [ ] **Additive-diff check:** `angular_loss`, `angular_error_degrees`, `EPS`,
  `_cos`, and `model.forward`'s normalization are byte-for-byte unchanged; the
  R2 and F-WANDB test suites pass with zero modifications (regression gate).
