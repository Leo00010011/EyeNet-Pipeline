"""CLI entrypoint: run the Optuna hyperparameter search from a YAML config.

Usage: py scripts/tune.py --config configs/optuna.yaml

Builds ONE EyeGazeDataModule and reuses it for every trial -- the split is fixed
for the whole study, so trials differ only in hyperparameters, never in data.

Resume semantics (FR18): with `optuna.storage` set to a SQLite URL and a fixed
`sampler.seed`, re-invoking main() resumes the same study (load_if_exists);
with `storage: null` each invocation is a fresh in-memory study.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import optuna
import yaml
from evedataset import EveBundle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyenet.dataset import EyeGazeDataModule  # noqa: E402
from eyenet.hpo import build_objective, build_pruner, build_sampler  # noqa: E402


def _write_best(cfg: dict, study: optuna.Study) -> Path | None:
    """FR17: dump the handoff file. Skipped (with a warning) if no trial
    completed -- study.best_value would raise, and a study where everything
    failed has no result to hand off."""
    out = Path(cfg["output"]["dir"])
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        warnings.warn("no trial completed; not writing best_params.yaml")
        return None
    out.mkdir(parents=True, exist_ok=True)
    path = out / "best_params.yaml"
    path.write_text(yaml.safe_dump({
        "study_name": study.study_name,
        "best_value": float(study.best_value),
        "best_trial_number": study.best_trial.number,
        "best_params": dict(study.best_params),
    }, sort_keys=False))
    return path


def main(config_path: str) -> optuna.Study:
    cfg = yaml.safe_load(Path(config_path).read_text())

    # Fail on bad paths before the slow bundle load and the first trial.
    for key in ("bundle_dir", "crops_root"):
        path = Path(cfg["data"][key])
        if not path.exists():
            raise FileNotFoundError(f"data.{key} does not exist: {path}")

    bundle = EveBundle.load(cfg["data"]["bundle_dir"])
    datamodule = EyeGazeDataModule(
        bundle,
        cfg["data"]["crops_root"],
        split_source=cfg["data"]["split_source"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    o = cfg["optuna"]
    study = optuna.create_study(
        study_name=o.get("study_name", "eyenet-hpo"),
        storage=o.get("storage"),
        load_if_exists=True,  # FR18
        direction=o.get("direction", "minimize"),  # FR21
        sampler=build_sampler(cfg),
        pruner=build_pruner(cfg),
    )
    # FR20: one OOM must not abort the study. TrialPruned is never caught here --
    # Optuna handles it itself and records PRUNED, not FAIL.
    study.optimize(
        build_objective(cfg, bundle, datamodule),
        n_trials=o.get("n_trials"),
        timeout=o.get("timeout"),
        catch=(RuntimeError,),
    )

    if _write_best(cfg, study) is not None:
        print(f"[hpo] best value={study.best_value:.4f} params={study.best_params}")
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
