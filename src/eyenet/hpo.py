"""Optuna search: config-driven suggest, sampler/pruner builders, objective.

The objective is `val/angular_error_deg` (degrees), never a raw training loss:
the search treats the loss function itself as a dimension, so radians-of-arccos
and 1-cos are not comparable across trials. Degrees is the project's primary
metric and is computed identically whatever a trial trained under.

Per-trial cheapness comes from `trainer.limit_*_batches` only. One DataModule is
built by the caller and shared across every trial, so R1's subject-level split
policy is untouched and trials differ in hyperparameters alone.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import optuna
import pytorch_lightning as pl

from eyenet.lightning_module import GazeEstimationModule
from eyenet.logging_utils import build_loggers

# Searched keys that overlay onto the `model:` block for GazeEstimationModule.
MODEL_PARAM_KEYS = ("dropout1", "dropout2", "dropout", "hidden_dim", "loss", "weight_decay", "lr")


class PruningCallback(pl.Callback):
    """Report the objective to Optuna each validation epoch; raise TrialPruned.

    Spec correction: the plan called for
    `optuna_integration.PyTorchLightningPruningCallback`, but that class
    subclasses `lightning.pytorch.Callback` -- the standalone `lightning`
    package, a different import root from this repo's `pytorch_lightning`.
    Installing both puts two Callback/Trainer hierarchies in one environment and
    `pl.Trainer` rejects the foreign base class outright. The single-process
    logic is ~10 lines, so it is reimplemented here against the Lightning this
    repo actually uses. DDP is explicitly out of scope (requirements.md), so the
    integration's distributed branch is deliberately not reproduced.
    """

    def __init__(self, trial, monitor: str) -> None:
        super().__init__()
        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module) -> None:
        # Lightning runs a sanity-check validation pass before epoch 0; reporting
        # from it would double-report step 0 and skew the pruner.
        if trainer.sanity_checking:
            return
        score = trainer.callback_metrics.get(self.monitor)
        if score is None:
            warnings.warn(
                f"pruning metric {self.monitor!r} is not in the evaluation logs; "
                "this trial will not be pruned"
            )
            return
        epoch = pl_module.current_epoch
        self._trial.report(float(score), step=epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")


def suggest_params(trial, search_space: dict) -> dict:
    """Issue the typed trial.suggest_* call for each configured dimension."""
    if not search_space:
        raise ValueError("optuna.search_space is empty; nothing to search")  # FR19
    out = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "float":
            out[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif t == "int":
            out[name] = trial.suggest_int(
                name, spec["low"], spec["high"],
                step=spec.get("step", 1), log=spec.get("log", False),
            )
        elif t == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"{name}: unknown search type {t!r}")  # FR11
    return out


def build_sampler(cfg: dict):
    s = (cfg["optuna"].get("sampler") or {})
    name, seed = s.get("name", "tpe"), s.get("seed")
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"unknown sampler {name!r}; valid: ['random', 'tpe']")  # FR10


def build_pruner(cfg: dict):
    p = (cfg["optuna"].get("pruner") or {})
    name = p.get("name", "median")
    if name == "median":
        return optuna.pruners.MedianPruner(
            n_warmup_steps=p.get("n_warmup_steps", 1),
            n_startup_trials=p.get("n_startup_trials", 5),
        )
    if name == "asha":
        return optuna.pruners.SuccessiveHalvingPruner()
    if name == "none":
        return optuna.pruners.NopPruner()
    raise ValueError(f"unknown pruner {name!r}; valid: ['asha', 'median', 'none']")  # FR10


def build_loggers_for_trial(cfg: dict, out: Path, trial_number: int) -> list:
    """FR16: delegate to F-WANDB's build_loggers with a per-trial run name.

    Copies the nested logging block so the caller's cfg is never mutated across
    trials -- otherwise run names would compound (-t0-t1-t2...).
    """
    logging_cfg = dict(cfg.get("logging") or {})
    wandb_cfg = dict(logging_cfg.get("wandb") or {})
    base = wandb_cfg.get("run_name") or cfg["optuna"].get("study_name", "eyenet-hpo")
    wandb_cfg["run_name"] = f"{base}-t{trial_number}"
    logging_cfg["wandb"] = wandb_cfg
    return build_loggers({**cfg, "logging": logging_cfg}, out)


def build_objective(cfg: dict, bundle, datamodule):
    """Return the Optuna objective closing over the shared cfg/datamodule."""
    metric = cfg["optuna"].get("objective_metric", "val/angular_error_deg")  # FR21
    pruner_on = (cfg["optuna"].get("pruner") or {}).get("name", "median") != "none"

    def objective(trial):
        params = suggest_params(trial, cfg["optuna"]["search_space"])
        model_kwargs = dict(cfg.get("model") or {})
        for k in MODEL_PARAM_KEYS:
            if k in params:
                model_kwargs[k] = params[k]
        module = GazeEstimationModule(**model_kwargs)

        callbacks = [PruningCallback(trial, monitor=metric)] if pruner_on else []  # FR14

        out = Path(cfg["output"]["dir"])
        trainer = pl.Trainer(
            logger=build_loggers_for_trial(cfg, out, trial.number),  # FR16
            callbacks=callbacks,
            **cfg["trainer"],
        )
        trainer.fit(module, datamodule=datamodule)

        # FR13 step 6: no metric => the trial never validated. PRUNED, not a
        # sentinel value that would pollute the study's best_value.
        if metric not in trainer.callback_metrics:
            raise optuna.TrialPruned()
        return float(trainer.callback_metrics[metric])

    return objective
