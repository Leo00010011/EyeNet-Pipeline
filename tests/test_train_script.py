"""End-to-end tests for scripts/train.py against the real sample bundle.

This is the one R2 test that exercises the real data path: a 2-batch run,
scoped by Lightning's limit_* flags, through the R1 DataModule.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pytorch_lightning.loggers import CSVLogger

from eyenet.lightning_module import GazeEstimationModule

from conftest import FACE_CROPS_ROOT, SAMPLE_BUNDLE_DIR

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train.py"


@pytest.fixture(scope="module")
def train_module():
    spec = importlib.util.spec_from_file_location("train_script", TRAIN_SCRIPT)
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
        "model": {"pretrained": False, "lr": 1.0e-4, "weight_decay": 0.0},
        "trainer": {
            "max_epochs": 1,
            "limit_train_batches": 2,
            "limit_val_batches": 1,
            "accelerator": "cpu",
            "log_every_n_steps": 1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
        "output": {"dir": str(tmp_path / "run")},
    }
    for section, values in overrides.items():
        cfg[section].update(values)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


@pytest.fixture(scope="module")
def baseline_run(tmp_path_factory, train_module, sample_bundle, face_crops_root):
    tmp_path = tmp_path_factory.mktemp("baseline")
    cfg_path, cfg = write_config(tmp_path)
    train_module.main(str(cfg_path))
    return Path(cfg["output"]["dir"])


def test_end_to_end_two_batch_run_completes(baseline_run):
    assert baseline_run.exists()


def test_checkpoint_written_and_loadable(baseline_run):
    ckpt = baseline_run / "checkpoints" / "last.ckpt"
    assert ckpt.exists()
    assert GazeEstimationModule.load_from_checkpoint(ckpt) is not None


def test_csv_metrics_written_with_finite_train_loss(baseline_run):
    """FR18's artifact -- the CSV loss curve that replaces W&B for R2 acceptance.

    train/loss is logged on_step and on_epoch (FR11), so Lightning splits it
    into train/loss_step and train/loss_epoch; there is no bare train/loss.
    """
    metrics = baseline_run / "csv" / "version_0" / "metrics.csv"
    assert metrics.exists()
    df = pd.read_csv(metrics)
    assert {"train/loss_step", "train/loss_epoch"} <= set(df.columns)
    for column in ("train/loss_step", "train/loss_epoch"):
        values = df[column].dropna()
        assert len(values) > 0
        assert np.isfinite(values.to_numpy()).all()


def test_limit_train_batches_pass_through(baseline_run):
    """limit_train_batches: 2 must actually cap the epoch, not silently train the full split."""
    df = pd.read_csv(baseline_run / "csv" / "version_0" / "metrics.csv")
    epoch0 = df[(df["epoch"] == 0) & df["train/loss_step"].notna()]
    assert len(epoch0) <= 2


def test_bad_bundle_dir_fails_fast_before_trainer(tmp_path, train_module, monkeypatch):
    bogus = str(tmp_path / "nonexistent")
    cfg_path, _ = write_config(tmp_path, data={"bundle_dir": bogus})

    def boom(*args, **kwargs):
        raise AssertionError("pl.Trainer must not be constructed on a bad path")

    monkeypatch.setattr(train_module.pl, "Trainer", boom)
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        train_module.main(str(cfg_path))


def test_bad_crops_root_fails_fast(tmp_path, train_module, monkeypatch):
    bogus = str(tmp_path / "no_crops_here")
    cfg_path, _ = write_config(tmp_path, data={"crops_root": bogus})

    def boom(*args, **kwargs):
        raise AssertionError("pl.Trainer must not be constructed on a bad path")

    monkeypatch.setattr(train_module.pl, "Trainer", boom)
    with pytest.raises(FileNotFoundError, match="no_crops_here"):
        train_module.main(str(cfg_path))


def test_new_metrics_reach_csv(baseline_run):
    """CSVLogger receiving the new metrics is the offline proof WandbLogger would too."""
    df = pd.read_csv(baseline_run / "csv" / "version_0" / "metrics.csv")
    for col in ("train/angular_error_deg", "val/pred_var_x", "val/theta_error_deg"):
        assert col in df.columns
        assert df[col].dropna().shape[0] > 0


# --- build_loggers (F-WANDB) ---


def test_build_loggers_disabled_returns_csv_only(train_module, tmp_path):
    loggers = train_module.build_loggers(
        {"logging": {"wandb": {"enabled": False}}}, tmp_path
    )
    assert len(loggers) == 1
    assert isinstance(loggers[0], CSVLogger)


def test_build_loggers_no_wandb_import_on_disabled_path(train_module, tmp_path):
    already_imported = "wandb" in sys.modules
    train_module.build_loggers({"logging": {"wandb": {"enabled": False}}}, tmp_path)
    if not already_imported:
        assert "wandb" not in sys.modules


def test_build_loggers_missing_logging_block_is_disabled(train_module, tmp_path):
    loggers = train_module.build_loggers({}, tmp_path)
    assert len(loggers) == 1
    assert isinstance(loggers[0], CSVLogger)


def test_build_loggers_enabled_no_api_key_warns_and_degrades(train_module, tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = {"logging": {"wandb": {"enabled": True}}}
    with pytest.warns(UserWarning, match="WANDB_API_KEY"):
        loggers = train_module.build_loggers(cfg, tmp_path)
    assert len(loggers) == 1
    assert isinstance(loggers[0], CSVLogger)


def test_build_loggers_enabled_constructor_raises_warns_and_degrades(
    train_module, tmp_path, monkeypatch
):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    import pytorch_lightning.loggers as pl_loggers

    def boom(*args, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr(pl_loggers, "WandbLogger", boom)
    cfg = {"logging": {"wandb": {"enabled": True}}}
    with pytest.warns(UserWarning):
        loggers = train_module.build_loggers(cfg, tmp_path)
    assert len(loggers) == 1
    assert isinstance(loggers[0], CSVLogger)


def test_build_loggers_csv_always_first(train_module, tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    for cfg in (
        {},
        {"logging": {"wandb": {"enabled": False}}},
        {"logging": {"wandb": {"enabled": True}}},
    ):
        loggers = train_module.build_loggers(cfg, tmp_path)
        assert isinstance(loggers[0], CSVLogger)
