"""Logger composition shared by scripts/train.py and the F-OPTUNA study.

Lives under src/eyenet/ (not scripts/) so both the training entrypoint and
eyenet.hpo can import it without a sys.path shim. scripts/train.py re-exports
`build_loggers`, so the F-WANDB tests that import it from there stay green.
"""

from __future__ import annotations

import os
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
