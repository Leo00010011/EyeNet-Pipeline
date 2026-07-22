# F-OPTUNA — Implementation Plan

## Context and Design Decisions

**Why now / where it sits.** R2 (model + loop) and F-WANDB (tracking) are done;
R3 is "full training & evaluation + ablations." Hand-tuning the head/optimizer
before an expensive full-split run is exactly the kind of comparative question
R3 needs answered cheaply. F-OPTUNA is the search that produces the settings the
R3 full run then uses — it does **not** replace R3's full run.

**Objective is loss-invariant (constitution: model validity is measured in
degrees).** The search includes the loss function as a dimension, so the raw
training loss is not comparable across trials (arccos radians vs. `1-cos`). The
project's primary metric — mean angular error in degrees (`angular_error_degrees`,
TechStack §Primary metric) — is computed the same way for every trial, so it is
the only sound objective. This is why FR3 keeps `angular_error_degrees` as the
reported/optimized quantity regardless of training loss.

**Per-trial cheapness comes from `limit_*_batches`, never from re-splitting.**
The constitution is emphatic that the subject-level split policy (R1) is
load-bearing and must not be perturbed for convenience (see the repeated "no
change to R1's split policy" notes in Roadmap R2 / F-CALIB). The R2 baseline
already demonstrated the sanctioned way to make a run cheap: pass
`limit_train_batches`/`limit_val_batches` through the `trainer:` block, which
touches Lightning only, not the DataModule. F-OPTUNA reuses that exact lever and
builds **one** DataModule shared across all trials, so the data is identical
trial-to-trial and only hyperparameters vary.

**Additive-only changes to shipped modules.** `losses.py`, `model.py`,
`lightning_module.py` are modified only additively: a new `cosine_loss` +
resolver; new `dropout1`/`dropout2`/`loss` kwargs with shims that preserve every
existing default and the R2 checkpoint round-trip. This mirrors how F-WANDB was
implemented ("no file under `src/eyenet/` that existed before was modified"
semantics — here we make the minimum, backward-compatible additions and keep the
R2/F-WANDB tests green).

**Two independent dropouts (user decision).** The head has two `nn.Dropout`
layers that R2 tied to one `dropout` param. The search wants them independent,
so `GazeResNet18` and the module gain `dropout1`/`dropout2` with a `dropout`
shim (FR5/FR6) so old configs and old checkpoints still work.

**W&B stays exactly as F-WANDB defined it.** Reuse `build_loggers` unchanged so
the config-gated, degrade-graceful, offline-in-tests contract is inherited
verbatim; the only addition is a per-trial run name so trials are separable
(FR16).

**New dependency surface.** `optuna` (study/sampler/pruner) and
`optuna-integration` (the `PyTorchLightningPruningCallback`; recent Optuna moved
Lightning integration out of core). Both are added to `requirements.txt`. Tests
must run offline and without a real bundle where possible — `suggest_params`,
`get_loss`, `cosine_loss`, and the dropout shim are all unit-testable with no
Optuna study and no network; the end-to-end study test runs a tiny 2-trial study
on the sample bundle (mirroring `tests/test_train_script.py`'s real-but-tiny run).

---

## Step 1 — `src/eyenet/losses.py`: add `cosine_loss` + `get_loss`

Additive only. After `angular_error_degrees`:

```python
def cosine_loss(pred, target):
    """(B,3),(B,3) -> scalar float32. Mean (1 - normalized dot). Range [0,2].

    No EPS clamp: unlike angular_loss this never calls arccos, so its gradient
    (-t on the normalized dot) is finite everywhere. The EPS clamp exists only
    to tame arccos' divergence and is left untouched.
    """
    _check(pred, target)
    p = F.normalize(pred.float(), p=2, dim=-1, eps=1e-8)
    t = F.normalize(target.float(), p=2, dim=-1, eps=1e-8)
    return (1.0 - (p * t).sum(dim=-1)).mean()

_LOSSES = {"angular": angular_loss, "cosine": cosine_loss}

def get_loss(name):
    try:
        return _LOSSES[name]
    except KeyError:
        raise ValueError(f"unknown loss {name!r}; valid: {sorted(_LOSSES)}")
```

Note: `_cos` (with the clamp) is reused only by `angular_loss`/
`angular_error_degrees`; `cosine_loss` computes its own unclamped dot on purpose.

## Step 2 — `src/eyenet/model.py`: independent dropouts with a `dropout` shim

```python
def __init__(self, pretrained=True, hidden_dim=256, dropout=0.5,
             dropout1=None, dropout2=None):
    super().__init__()
    d1 = dropout if dropout1 is None else dropout1   # explicit wins, else shim
    d2 = dropout if dropout2 is None else dropout2
    ...
    self.backbone.fc = nn.Sequential(
        nn.Linear(512, hidden_dim),
        nn.Dropout(d1),
        nn.Linear(hidden_dim, 3),
        nn.Dropout(d2),
    )
```

`forward` unchanged. Docstring gains one line on precedence (explicit
`dropout1`/`dropout2` override the `dropout` shim).

## Step 3 — `src/eyenet/lightning_module.py`: plumb `dropout1`/`dropout2`/`loss`

Additive to `__init__` (keep every existing default and order; append new
kwargs so `save_hyperparameters()` still round-trips an R2 checkpoint that
lacked them — missing kwargs fall back to defaults on load):

```python
def __init__(self, pretrained=True, lr=1e-4, weight_decay=0.0,
             hidden_dim=256, dropout=0.5, dropout1=None, dropout2=None,
             loss="angular"):
    super().__init__()
    self.save_hyperparameters()
    self.model = GazeResNet18(pretrained=pretrained, hidden_dim=hidden_dim,
                              dropout=dropout, dropout1=dropout1, dropout2=dropout2)
    self._loss_fn = get_loss(loss)   # FR7; raises on bad name at construction
    self._buf = {}
```

`_step` changes one line — `angular_loss(pred, target)` → `self._loss_fn(pred,
target)`. `angular_error_degrees(pred, target)` stays (FR3). Import `get_loss`
from `eyenet.losses`. Nothing else in the module changes (all F-WANDB buffers/
emit logic untouched).

## Step 4 — `src/eyenet/hpo.py` (new): suggest + sampler/pruner + objective

```python
import optuna

def suggest_params(trial, search_space):
    if not search_space:
        raise ValueError("optuna.search_space is empty; nothing to search")  # FR19
    out = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "float":
            out[name] = trial.suggest_float(name, spec["low"], spec["high"],
                                            log=spec.get("log", False))
        elif t == "int":
            out[name] = trial.suggest_int(name, spec["low"], spec["high"],
                                          step=spec.get("step", 1), log=spec.get("log", False))
        elif t == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"{name}: unknown search type {t!r}")   # FR11
    return out

def build_sampler(cfg):
    s = (cfg["optuna"].get("sampler") or {})
    name, seed = s.get("name", "tpe"), s.get("seed")
    if name == "tpe":    return optuna.samplers.TPESampler(seed=seed)
    if name == "random": return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"unknown sampler {name!r}")   # FR10

def build_pruner(cfg):
    p = (cfg["optuna"].get("pruner") or {})
    name = p.get("name", "median")
    if name == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=p.get("n_warmup_steps", 1),
                                           n_startup_trials=p.get("n_startup_trials", 5))
    if name == "asha":   return optuna.pruners.SuccessiveHalvingPruner()
    if name == "none":   return optuna.pruners.NopPruner()
    raise ValueError(f"unknown pruner {name!r}")   # FR10
```

`build_objective` — closure over the shared `cfg`/`datamodule`:

```python
def build_objective(cfg, bundle, datamodule):
    metric = cfg["optuna"].get("objective_metric", "val/angular_error_deg")  # FR21
    pruner_on = (cfg["optuna"].get("pruner") or {}).get("name", "median") != "none"

    def objective(trial):
        params = suggest_params(trial, cfg["optuna"]["search_space"])
        # base model kwargs overlaid with searched subset
        model_kwargs = dict(cfg["model"])
        for k in ("dropout1", "dropout2", "hidden_dim", "loss", "weight_decay"):
            if k in params:
                model_kwargs[k] = params[k]
        module = GazeEstimationModule(**model_kwargs)

        callbacks = []
        if pruner_on:
            from optuna.integration import PyTorchLightningPruningCallback   # FR14
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor=metric))

        out = Path(cfg["output"]["dir"])
        trainer = pl.Trainer(
            logger=build_loggers_for_trial(cfg, out, trial.number),   # FR16
            callbacks=callbacks,
            **cfg["trainer"],
        )
        trainer.fit(module, datamodule=datamodule)
        if metric not in trainer.callback_metrics:
            raise optuna.TrialPruned()   # FR13 step 6
        return float(trainer.callback_metrics[metric])
    return objective
```

`build_loggers_for_trial` wraps F-WANDB's `build_loggers`: shallow-copy `cfg`,
set `logging.wandb.run_name = f"{base}-t{trial_number}"`, delegate. Import
`build_loggers` from `scripts.train` (add `scripts/__init__.py` if needed, or
factor `build_loggers` into an importable location — see Step 6 note).

## Step 5 — `configs/optuna.yaml` (new)

Reuse `baseline.yaml`'s `data`/`output`/`logging` verbatim; keep `trainer:` with
the subset limits (search must be cheap); `model:` holds the fixed base
(`pretrained`, `lr`) plus defaults for any dimension *not* searched; add the
`optuna:` block from FR9. Set `output.dir: runs/hpo`, `logging.wandb.enabled:
false` by default. Example `trainer:` `max_epochs: 5`, `limit_train_batches: 100`,
`limit_val_batches: 20` — enough epochs for the pruner to discriminate.

## Step 6 — `scripts/tune.py` (new)

```python
def main(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    for key in ("bundle_dir", "crops_root"):          # same guard as train.py
        if not Path(cfg["data"][key]).exists():
            raise FileNotFoundError(f"data.{key} does not exist: {cfg['data'][key]}")
    bundle = EveBundle.load(cfg["data"]["bundle_dir"])
    datamodule = EyeGazeDataModule(bundle, cfg["data"]["crops_root"],
        split_source=cfg["data"]["split_source"],
        batch_size=cfg["data"]["batch_size"], num_workers=cfg["data"]["num_workers"])

    o = cfg["optuna"]
    study = optuna.create_study(
        study_name=o.get("study_name", "eyenet-hpo"),
        storage=o.get("storage"), load_if_exists=True,      # FR18
        direction=o.get("direction", "minimize"),           # FR21
        sampler=build_sampler(cfg), pruner=build_pruner(cfg))
    objective = build_objective(cfg, bundle, datamodule)
    study.optimize(objective, n_trials=o.get("n_trials"),
                   timeout=o.get("timeout"), catch=(RuntimeError,))   # FR20

    _write_best(cfg, study)     # FR17 -> <output.dir>/best_params.yaml
    print(f"[hpo] best value={study.best_value:.4f} params={study.best_params}")
    return study
```

`_write_best` dumps `study_name`, `best_value`, `best_trial_number`,
`best_params` to `<output.dir>/best_params.yaml`. Guard: if no trial completed,
warn and skip the file rather than raising on `study.best_value`.

**Importability note (Step 4 dependency):** `build_loggers` currently lives in
`scripts/train.py`. `scripts/` is imported via the `sys.path.insert` shim, not a
package. Simplest: `hpo.py` imports it with the same path shim
(`from train import build_loggers` after inserting `scripts/` on `sys.path`), OR
move `build_loggers` into `src/eyenet/` (e.g. a small `logging_utils.py`) and
have both `train.py` and `hpo.py` import it. **Chosen:** move `build_loggers`
into `src/eyenet/logging_utils.py` and re-import it in `train.py` (one-line
change, keeps F-WANDB tests green by re-exporting the name from `train.py`).
This avoids `scripts`-as-package fragility and is the only structural change to a
shipped file beyond the additive kwargs.

## Step 7 — `requirements.txt`

Add `optuna` and `optuna-integration`. Nothing else.

## Implementation Order

1. `losses.py` — `cosine_loss` + `get_loss` (Step 1).
2. `model.py` — `dropout1`/`dropout2` + shim (Step 2).
3. `lightning_module.py` — plumb `dropout1`/`dropout2`/`loss`, use `get_loss` (Step 3).
4. `src/eyenet/logging_utils.py` — move `build_loggers` out of `train.py`; re-export in `train.py` (Step 6 note).
5. `src/eyenet/hpo.py` — `suggest_params`, `build_sampler`, `build_pruner`, `build_objective` (Step 4).
6. `configs/optuna.yaml` (Step 5).
7. `scripts/tune.py` — `main` + `_write_best` (Step 6).
8. `requirements.txt` — `optuna`, `optuna-integration` (Step 7).
9. Tests (see validation.md), then the tiny real 2-trial study check.
