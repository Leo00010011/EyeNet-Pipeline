"""End-to-end tests for scripts/tune.py against the real sample bundle.

Mirrors tests/test_train_script.py: tiny real runs (2 trials x 1 epoch x 2
batches), scoped by Lightning's limit_* flags, with W&B disabled so nothing
touches the network.
"""

import importlib.util
from pathlib import Path

import optuna
import pytest
import yaml
from pytorch_lightning.loggers import CSVLogger

from eyenet.hpo import build_loggers_for_trial

from conftest import FACE_CROPS_ROOT, SAMPLE_BUNDLE_DIR

TUNE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tune.py"

SEARCH_SPACE = {
    "dropout1": {"type": "float", "low": 0.0, "high": 0.7},
    "dropout2": {"type": "float", "low": 0.0, "high": 0.7},
    "weight_decay": {"type": "float", "low": 1.0e-6, "high": 1.0e-2, "log": True},
    "hidden_dim": {"type": "categorical", "choices": [128, 256]},
    "loss": {"type": "categorical", "choices": ["angular", "cosine"]},
}


@pytest.fixture(scope="module")
def tune_module():
    spec = importlib.util.spec_from_file_location("tune_script", TUNE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_config(tmp_path, **overrides):
    cfg = {
        "data": {
            "bundle_dir": str(SAMPLE_BUNDLE_DIR),
            "crops_root": str(FACE_CROPS_ROOT),
            "batch_size": 4,
            "num_workers": 0,
            "split_source": {"seed": 42, "val_fraction": 0.2},
        },
        "model": {"pretrained": False, "lr": 1.0e-4},
        "trainer": {
            "max_epochs": 1,
            "limit_train_batches": 2,
            "limit_val_batches": 1,
            "accelerator": "cpu",
            "log_every_n_steps": 1,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
        "output": {"dir": str(tmp_path / "hpo")},
        "logging": {"wandb": {"enabled": False}},
        "optuna": {
            "study_name": "test-hpo",
            "storage": None,
            "direction": "minimize",
            "objective_metric": "val/angular_error_deg",
            "n_trials": 2,
            "timeout": None,
            "sampler": {"name": "random", "seed": 42},
            "pruner": {"name": "median", "n_warmup_steps": 1, "n_startup_trials": 5},
            "search_space": SEARCH_SPACE,
        },
    }
    for section, values in overrides.items():
        cfg[section].update(values)
    path = tmp_path / "optuna.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


@pytest.fixture(scope="module")
def study_run(tmp_path_factory, tune_module, sample_bundle, face_crops_root):
    tmp_path = tmp_path_factory.mktemp("hpo")
    cfg_path, cfg = write_config(tmp_path)
    study = tune_module.main(str(cfg_path))
    return study, Path(cfg["output"]["dir"])


def test_two_trial_study_completes(study_run):
    study, _ = study_run
    assert isinstance(study, optuna.Study)
    assert len(study.trials) == 2


def test_best_value_is_a_plausible_angular_error(study_run):
    """Degrees, so it must sit in [0, 180] -- a raw-loss regression would not."""
    study, _ = study_run
    import math

    assert math.isfinite(study.best_value)
    assert 0.0 <= study.best_value <= 180.0


def test_best_params_yaml_is_the_handoff_file(study_run):
    study, out = study_run
    path = out / "best_params.yaml"
    assert path.exists()
    data = yaml.safe_load(path.read_text())
    assert set(data["best_params"]) == set(SEARCH_SPACE)
    assert data["best_value"] == pytest.approx(study.best_value)
    assert data["best_trial_number"] == study.best_trial.number


def test_best_params_stay_inside_declared_bounds(study_run):
    study, _ = study_run
    p = study.best_params
    for dim in ("dropout1", "dropout2", "weight_decay"):
        spec = SEARCH_SPACE[dim]
        assert spec["low"] <= p[dim] <= spec["high"]
    assert p["hidden_dim"] in SEARCH_SPACE["hidden_dim"]["choices"]
    assert p["loss"] in SEARCH_SPACE["loss"]["choices"]


def test_objective_metric_is_degrees_not_a_raw_loss(study_run):
    """Guards the loss-invariance property: the search compares trials trained
    under different losses, so it must optimize a physical quantity."""
    _, out = study_run
    cfg = yaml.safe_load((Path(TUNE_SCRIPT).parents[1] / "configs" / "optuna.yaml").read_text())
    assert cfg["optuna"]["objective_metric"] == "val/angular_error_deg"


def test_bad_bundle_dir_fails_fast_before_any_trial(tmp_path, tune_module, monkeypatch):
    bogus = str(tmp_path / "nonexistent")
    cfg_path, _ = write_config(tmp_path, data={"bundle_dir": bogus})

    def boom(*args, **kwargs):
        raise AssertionError("no study may be created on a bad path")

    monkeypatch.setattr(tune_module.optuna, "create_study", boom)
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        tune_module.main(str(cfg_path))


def test_bad_crops_root_fails_fast(tmp_path, tune_module, monkeypatch):
    bogus = str(tmp_path / "no_crops_here")
    cfg_path, _ = write_config(tmp_path, data={"crops_root": bogus})

    def boom(*args, **kwargs):
        raise AssertionError("no study may be created on a bad path")

    monkeypatch.setattr(tune_module.optuna, "create_study", boom)
    with pytest.raises(FileNotFoundError, match="no_crops_here"):
        tune_module.main(str(cfg_path))


def test_nop_pruner_path_prunes_nothing(tmp_path, tune_module, sample_bundle, face_crops_root):
    cfg_path, cfg = write_config(tmp_path)
    cfg["optuna"]["pruner"] = {"name": "none"}
    Path(cfg_path).write_text(yaml.safe_dump(cfg))

    study = tune_module.main(str(cfg_path))
    assert len(study.trials) == 2
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    assert len(pruned) == 0


def test_runtime_error_in_one_trial_is_caught_and_study_finishes(
    tmp_path, tune_module, monkeypatch, sample_bundle, face_crops_root
):
    """FR20: one OOM must cost one trial, not the whole study."""
    cfg_path, cfg = write_config(tmp_path)
    real_build = tune_module.build_objective

    def flaky_build(config, bundle, datamodule):
        inner = real_build(config, bundle, datamodule)

        def objective(trial):
            if trial.number == 0:
                raise RuntimeError("simulated OOM")
            return inner(trial)

        return objective

    monkeypatch.setattr(tune_module, "build_objective", flaky_build)
    study = tune_module.main(str(cfg_path))

    states = [t.state for t in study.trials]
    assert optuna.trial.TrialState.FAIL in states
    assert optuna.trial.TrialState.COMPLETE in states
    assert (Path(cfg["output"]["dir"]) / "best_params.yaml").exists()  # FR17


# --- Group 7: logger reuse / W&B gating ---


def test_each_trial_gets_exactly_one_csv_logger_when_wandb_disabled(tmp_path):
    cfg = {"logging": {"wandb": {"enabled": False}}, "optuna": {"study_name": "s"}}
    loggers = build_loggers_for_trial(cfg, tmp_path, 3)
    assert len(loggers) == 1
    assert isinstance(loggers[0], CSVLogger)


def test_per_trial_run_name_ends_in_trial_number(tmp_path, monkeypatch):
    import eyenet.hpo as hpo

    seen = {}
    monkeypatch.setattr(hpo, "build_loggers", lambda cfg, out: seen.update(cfg) or [])

    base_cfg = {"logging": {"wandb": {"enabled": True, "run_name": "sweep"}},
                "optuna": {"study_name": "s"}}
    hpo.build_loggers_for_trial(base_cfg, tmp_path, 7)
    assert seen["logging"]["wandb"]["run_name"] == "sweep-t7"

    # The caller's cfg must be untouched, else names compound across trials.
    assert base_cfg["logging"]["wandb"]["run_name"] == "sweep"


def test_run_name_falls_back_to_study_name(tmp_path, monkeypatch):
    import eyenet.hpo as hpo

    seen = {}
    monkeypatch.setattr(hpo, "build_loggers", lambda cfg, out: seen.update(cfg) or [])
    hpo.build_loggers_for_trial(
        {"logging": {"wandb": {"enabled": True}}, "optuna": {"study_name": "eyenet-hpo"}},
        tmp_path, 0,
    )
    assert seen["logging"]["wandb"]["run_name"] == "eyenet-hpo-t0"


def test_study_reuses_one_datamodule_across_trials(tmp_path, tune_module, monkeypatch,
                                                   sample_bundle, face_crops_root):
    """Data architecture integrity: trials differ in hyperparameters, never in
    data. One DataModule instance must serve the whole study."""
    cfg_path, _ = write_config(tmp_path)
    seen = []
    real_build = tune_module.build_objective

    def recording_build(config, bundle, datamodule):
        inner = real_build(config, bundle, datamodule)

        def objective(trial):
            seen.append(id(datamodule))
            return inner(trial)

        return objective

    monkeypatch.setattr(tune_module, "build_objective", recording_build)
    tune_module.main(str(cfg_path))
    assert len(seen) == 2 and len(set(seen)) == 1
