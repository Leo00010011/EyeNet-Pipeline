"""CLI entrypoint: train the gaze model from a YAML config.

Usage: py scripts/train.py --config configs/baseline.yaml

The `trainer:` block is passed through to pl.Trainer unmodified -- that is how
the R2 baseline run is scoped to a small subset (limit_train_batches /
limit_val_batches), with no change to R1's subject-level split policy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import yaml
from evedataset import EveBundle
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyenet.dataset import EyeGazeDataModule  # noqa: E402
from eyenet.lightning_module import GazeEstimationModule  # noqa: E402
from eyenet.logging_utils import build_loggers  # noqa: E402,F401  (re-export)


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    # Fail on bad paths before the slow bundle load and the weight download.
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
    module = GazeEstimationModule(**cfg["model"])

    out = Path(cfg["output"]["dir"])
    trainer = pl.Trainer(
        logger=build_loggers(cfg, out),
        callbacks=[
            ModelCheckpoint(
                dirpath=str(out / "checkpoints"),
                monitor="val/angular_error_deg",
                mode="min",
                save_top_k=1,
                save_last=True,
            )
        ],
        **cfg["trainer"],
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
