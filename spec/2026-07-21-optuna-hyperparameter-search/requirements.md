# F-OPTUNA — Optuna hyperparameter search

## Goal

Add an Optuna-driven hyperparameter search that tunes the model head and
optimizer over a small, config-defined search space, minimizing a
**loss-invariant** validation objective (mean angular error in degrees). The
search reuses the existing PyTorch Lightning training loop, YAML-config style,
and W&B logging unchanged — it only wraps them in an Optuna study that samples
hyperparameters, runs a short (subset) training per trial, prunes unpromising
trials early, and records the best configuration. This gives R3 a principled,
reproducible way to pick head/optimizer/loss settings before the expensive
full-split run, instead of hand-tuning.

## Scope

**In scope**
- A new config-selectable **cosine-distance loss** `1 - ⟨x, x̂⟩` (normalized
  gt/pred unit vectors) alongside the existing arccos `angular_loss`.
- Making the two MLP-head dropouts **independent** (`dropout1`, `dropout2`) so
  each can be searched separately.
- A config-driven Optuna study over five dimensions: `dropout1`, `dropout2`,
  `weight_decay`, `hidden_dim ∈ {128, 256}`, `loss ∈ {angular, cosine}`.
- Objective: **minimize `val/angular_error_deg`** (comparable across all trials
  regardless of which loss they trained under).
- Per-trial training on a **subset** (`limit_*_batches`, exactly as the R2
  baseline scopes its run) with an **Optuna pruning callback** for early
  stopping of weak trials.
- W&B, Lightning, and YAML config kept as the tracking/training/config stack;
  W&B stays config-gated and degrades gracefully (unchanged F-WANDB contract).
- A study driver script (`scripts/tune.py`) with an importable `main(config_path)`.

**Explicitly out of scope**
- Any change to R1's subject-level split policy (`build_sample_index`,
  `splits.py`, `dataset.py`). Per-trial cheapness comes from `limit_*_batches`,
  never from re-partitioning subjects.
- The full-split re-run of the winning config — that is an ordinary R3 training
  run of `scripts/train.py` with the tuned values pasted into a config; F-OPTUNA
  produces the values, it does not run the final model.
- Searching learning rate, batch size, backbone, augmentation, or architecture
  beyond the four listed knobs. (LR is deliberately fixed; add later only if a
  real question needs it.)
- Distributed / multi-GPU parallel trials, Optuna dashboards, and hyperband
  variants beyond the one median/ASHA pruner exposed here.
- Changing `losses.py`'s `EPS` clamp, `model.py`'s normalization, or the
  F-WANDB metric set.

## Functional Requirements

### Loss

- **FR1** — Add `cosine_loss(pred, target)` to `src/eyenet/losses.py`. Signature
  `(B,3), (B,3) -> scalar float32`. Computes `1 - (p·t)` where `p`, `t` are the
  L2-normalized (`F.normalize`, `eps=1e-8`) rows of `pred`/`target`, meaned over
  the batch. Reuses the existing `_check` shape guard (raises `ValueError` on
  non-`(B,3)` or mismatched shapes). Range `[0, 2]`, min `0` at perfect
  agreement. No `arccos`, so **no `EPS` clamp is required** — its gradient
  (`-t`) is finite everywhere; the clamp existing only for `angular_loss` stays
  untouched.
- **FR2** — Add `get_loss(name: str)` to `src/eyenet/losses.py`: a resolver
  mapping `"angular" -> angular_loss`, `"cosine" -> cosine_loss`. Unknown name
  raises `ValueError` listing the valid keys. This is the single source of truth
  for the `loss` config value; no other module hardcodes a loss-name string map.
- **FR3** — `angular_error_degrees` remains the reported metric for **both**
  losses. The objective and all F-WANDB angular-error logging are computed via
  `angular_error_degrees` regardless of the training loss, so trials are
  comparable.

### Model — independent dropouts

- **FR4** — `GazeResNet18.__init__` gains `dropout1: float = 0.5` and
  `dropout2: float = 0.5`, applied to the first and second `nn.Dropout` in the
  head respectively. The head becomes
  `Linear(512,hidden_dim) → Dropout(dropout1) → Linear(hidden_dim,3) → Dropout(dropout2)`.
- **FR5** — Backward compatibility: the existing `dropout: float = 0.5` keyword
  is retained as a shim. If a caller passes `dropout` but not `dropout1`/
  `dropout2`, both dropouts take the `dropout` value (preserving R2 behavior and
  the existing checkpoint round-trip). If `dropout1`/`dropout2` are given, they
  win. Passing both `dropout` and either explicit value where they conflict is
  allowed (explicit wins); document the precedence in a docstring line.
- **FR6** — `GazeEstimationModule.__init__` gains `dropout1`/`dropout2` with the
  same shim semantics and forwards them to `GazeResNet18`. `save_hyperparameters`
  continues to capture the full signature; the R2 checkpoint round-trip test must
  still load an R2-era checkpoint (which only stored `dropout`).
- **FR7** — `GazeEstimationModule.__init__` gains `loss: str = "angular"`. In
  `_step`, the training loss is `get_loss(self.hparams.loss)(pred, target)`
  instead of the hardcoded `angular_loss`. `angular_error_degrees` in `_step` is
  unchanged (FR3). No other line of `lightning_module.py` changes semantics.

### Search space & study config

- **FR8** — A new config file `configs/optuna.yaml` carries an `optuna:` block
  plus reused `data`/`model`/`trainer`/`output`/`logging` blocks. The
  `data`/`trainer`/`output`/`logging` blocks have the same schema as
  `configs/baseline.yaml`. The `model` block provides the **fixed** (non-searched)
  base values (`pretrained`, `lr`); searched keys present in `model` are
  overridden per-trial.
- **FR9** — The `optuna:` block schema:
  ```yaml
  optuna:
    study_name: eyenet-hpo
    storage: null            # null => in-memory; else e.g. sqlite:///runs/hpo/study.db
    direction: minimize
    objective_metric: val/angular_error_deg
    n_trials: 30
    timeout: null            # seconds; null => unbounded, gated by n_trials
    sampler: {name: tpe, seed: 42}
    pruner: {name: median, n_warmup_steps: 1, n_startup_trials: 5}
    search_space:
      dropout1:     {type: float, low: 0.0, high: 0.7}
      dropout2:     {type: float, low: 0.0, high: 0.7}
      weight_decay: {type: float, low: 1.0e-6, high: 1.0e-2, log: true}
      hidden_dim:   {type: categorical, choices: [128, 256]}
      loss:         {type: categorical, choices: [angular, cosine]}
  ```
- **FR10** — `sampler.name ∈ {tpe, random}`; unknown raises `ValueError`. `seed`
  is passed to the sampler for reproducibility. `pruner.name ∈ {median, asha,
  none}` (`asha` → `optuna.pruners.SuccessiveHalvingPruner`; `none` →
  `optuna.pruners.NopPruner`); unknown raises `ValueError`.
- **FR11** — `search_space` entries are typed: `type ∈ {float, int, categorical}`.
  `float`/`int` require `low`/`high` and accept optional `log: bool` and (int)
  `step`; `categorical` requires `choices`. The suggest call is dispatched from
  `type` (`suggest_float`/`suggest_int`/`suggest_categorical`) — the set of
  searched dimensions is **driven entirely by the config**, so removing a key
  from `search_space` drops it from the search (falling back to the `model`
  block's fixed value) with no code change.

### Objective & study driver

- **FR12** — `src/eyenet/hpo.py` provides `suggest_params(trial, search_space)
  -> dict` — reads the `search_space` config, issues the typed `trial.suggest_*`
  calls (FR11), returns the sampled `{name: value}` mapping. Pure w.r.t. the
  Lightning stack (no model import needed); unit-testable with a fake/`FixedTrial`.
- **FR13** — `src/eyenet/hpo.py` provides `build_objective(cfg, bundle,
  datamodule) -> Callable[[optuna.Trial], float]`. The returned objective, per
  trial:
  1. `params = suggest_params(trial, cfg["optuna"]["search_space"])`.
  2. Compose the effective model kwargs = `cfg["model"]` overlaid with the
     searched subset of `params` (`dropout1`, `dropout2`, `hidden_dim`, `loss`);
     `weight_decay` overlays into the module too.
  3. Construct `GazeEstimationModule(**model_kwargs)`.
  4. Build a `pl.Trainer` from `cfg["trainer"]` (which carries the subset
     `limit_*_batches`), adding the pruning callback (FR14) and per-trial loggers
     (FR16). No `ModelCheckpoint` is required for search trials (checkpoints of
     throwaway subset trials are not artifacts); may be omitted.
  5. `trainer.fit(module, datamodule=datamodule)`.
  6. Return `float(trainer.callback_metrics[cfg["optuna"]["objective_metric"]])`.
     If the metric is absent (e.g. trial pruned before any validation epoch),
     re-raise the `optuna.TrialPruned` (step 14 path) rather than returning a
     sentinel.
- **FR14** — Pruning uses `optuna.integration.PyTorchLightningPruningCallback(
  trial, monitor=cfg["optuna"]["objective_metric"])` added to the Trainer's
  callbacks (only when `pruner.name != "none"`). A pruned trial surfaces as
  `optuna.TrialPruned`, which Optuna records as `PRUNED`, not `FAIL`.
- **FR15** — `scripts/tune.py` exposes `main(config_path: str) -> optuna.Study`
  and a CLI (`py scripts/tune.py --config configs/optuna.yaml`). It: validates
  `data.bundle_dir`/`data.crops_root` exist (same guard as `train.py`, same
  `FileNotFoundError`); loads the bundle and builds **one** `EyeGazeDataModule`
  reused across trials (the split is fixed for the whole study — trials differ
  only in hyperparameters, not data); builds the sampler/pruner (FR10); creates
  the study (`storage`/`study_name`/`direction`); runs `study.optimize(objective,
  n_trials=..., timeout=...)`; then writes the best result (FR17) and returns the
  study.
- **FR16** — Per-trial W&B logging reuses the F-WANDB `build_loggers` contract
  (config-gated, degrades to CSV-only, never raises into `fit`). Each trial gets
  a distinct run name derived from the base `logging.wandb.run_name` (or study
  name) plus the trial number, so trials are separable in the dashboard. When
  `logging.wandb.enabled` is `false` (default, and always in tests), no W&B path
  is exercised. CSVLogger stays `logger[0]`.
- **FR17** — On completion `main` writes `<output.dir>/best_params.yaml`
  containing `study_name`, `best_value` (the objective), `best_trial_number`, and
  `best_params` (the full sampled dict), and logs a one-line summary
  (`best value=… params=…`). This file is the handoff: its `best_params` are
  meant to be pasted into a `configs/*.yaml` `model:` block for the full-split
  R3 run. `best_params.yaml` is written even if some trials failed, as long as
  at least one completed.
- **FR18** — Reproducibility: given a fixed `sampler.seed` and a persistent
  `storage` (SQLite), re-invoking `main` **resumes** the same study (Optuna's
  `load_if_exists=True`) rather than starting over; with `storage: null`
  (in-memory) each invocation is a fresh study. Document both modes.

### Error handling & edges

- **FR19** — Empty `search_space` (`{}` or absent) is a `ValueError` from
  `suggest_params` / study setup — a search with nothing to search is a config
  error, not a silent single-config run.
- **FR20** — A single trial raising a non-pruning exception (e.g. OOM) must not
  abort the whole study: `study.optimize(..., catch=(RuntimeError,))` is used so
  such trials are recorded `FAIL` and the study continues. `TrialPruned` is
  never caught by `catch` (Optuna handles it as `PRUNED`).
- **FR21** — `objective_metric` defaults to `val/angular_error_deg` if the
  `optuna` block omits it; `direction` defaults to `minimize`. A `direction:
  maximize` with the angular-error metric is not auto-corrected (the user owns
  the pairing) but the default pairing is correct.
- **FR22** — No new coupling to `EveBundle` internals; the study reads data only
  through the R1 `EyeGazeDataModule`, exactly as `train.py` does.

## Public API Summary

```python
# src/eyenet/losses.py  (additions; angular_loss / angular_error_degrees / EPS unchanged)
def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...   # 1 - <x,x̂>, mean
def get_loss(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: ...

# src/eyenet/model.py
class GazeResNet18(nn.Module):
    def __init__(self, pretrained: bool = True, hidden_dim: int = 256,
                 dropout: float = 0.5, dropout1: float | None = None,
                 dropout2: float | None = None) -> None: ...

# src/eyenet/lightning_module.py
class GazeEstimationModule(pl.LightningModule):
    def __init__(self, pretrained: bool = True, lr: float = 1e-4,
                 weight_decay: float = 0.0, hidden_dim: int = 256,
                 dropout: float = 0.5, dropout1: float | None = None,
                 dropout2: float | None = None, loss: str = "angular") -> None: ...

# src/eyenet/hpo.py  (new)
def suggest_params(trial, search_space: dict) -> dict: ...
def build_objective(cfg: dict, bundle, datamodule) -> Callable[["optuna.Trial"], float]: ...
def build_sampler(cfg: dict): ...
def build_pruner(cfg: dict): ...

# scripts/tune.py  (new)
def main(config_path: str) -> "optuna.Study": ...
```

## Dependencies

| Reads from | For |
|---|---|
| `configs/optuna.yaml` | search space, study, trainer-subset, data, logging config |
| `EyeGazeDataModule` (R1) | the fixed train/val subset every trial trains on |
| `GazeEstimationModule` (R2) | the per-trial model, now loss/dropout-parameterized |
| `losses.get_loss` | resolve the searched `loss` categorical to a callable |
| `scripts/train.build_loggers` (F-WANDB) | per-trial CSV+W&B loggers, degrade-graceful |
| `optuna`, `optuna-integration` (new deps) | study, samplers, pruners, Lightning pruning callback |

| Writes to | What |
|---|---|
| `<output.dir>/best_params.yaml` | best objective value + params (handoff to R3 full run) |
| `<output.dir>/csv/version_*/metrics.csv` | per-trial metric history (CSVLogger, unchanged) |
| `optuna storage` (SQLite, optional) | persisted study for resume/inspection |
| `<output.dir>/wandb/` | per-trial W&B runs, only when enabled + key present |
