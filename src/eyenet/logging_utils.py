"""Logger composition shared by scripts/train.py and the F-OPTUNA study.

Lives under src/eyenet/ (not scripts/) so both the training entrypoint and
eyenet.hpo can import it without a sys.path shim. scripts/train.py re-exports
`build_loggers`, so the F-WANDB tests that import it from there stay green.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

from pytorch_lightning.loggers import CSVLogger


def build_loggers(cfg: dict, out: Path) -> list:
    # FR23: CSVLogger FIRST. Lightning derives trainer.log_dir from logger[0];
    # reordering silently moves every run artifact.
    loggers = [CSVLogger(save_dir=str(out), name="csv")]

    wandb_cfg = (cfg.get("logging") or {}).get("wandb") or {}  # FR22
    if not wandb_cfg.get("enabled", False):
        return loggers  # FR21: no wandb import

    # FR25: instrumentation must never cost a queued run. Warn, degrade to CSV.
    if not os.environ.get("WANDB_API_KEY"):
        warnings.warn(
            "logging.wandb.enabled is true but WANDB_API_KEY is unset; "
            "continuing with CSVLogger only. Export WANDB_API_KEY in your job "
            "script (see spec requirements.md, 'Running unattended in a job queue')."
        )
        return loggers

    try:
        from pytorch_lightning.loggers import WandbLogger  # FR21: local import

        loggers.append(WandbLogger(
            project=wandb_cfg.get("project", "eyenet"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name"),
            tags=wandb_cfg.get("tags") or [],
            save_dir=str(out),
        ))
    except Exception as e:  # FR26
        warnings.warn(f"W&B logging disabled ({type(e).__name__}: {e}); "
                      "continuing with CSVLogger only.")
    return loggers


def finish_wandb_run() -> None:
    """Close the process-global wandb run, if one is open -- the counterpart to
    `build_loggers`' open.

    Load-bearing for the Optuna study, not cosmetic. `WandbLogger.finalize()`
    does NOT call `wandb.finish()`, and `WandbLogger.experiment` *reuses* a
    non-None `wandb.run` instead of starting a new one (it only warns). So
    without an explicit close between trials, every trial after the first logs
    into trial 0's run: interleaved curves, a resetting global_step, and one
    hparams config for N different configurations.

    Checking the global `wandb.run` is exactly what Lightning itself checks, so
    there is no second notion of "a run is open" that could drift.

    Keyed off `sys.modules` so the disabled path never imports wandb (FR21), and
    never raises -- instrumentation must not cost a queued run (FR25).
    """
    if "wandb" not in sys.modules:
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception as e:
        warnings.warn(f"could not close the W&B run ({type(e).__name__}: {e})")
