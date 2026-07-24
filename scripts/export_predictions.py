"""Run a trained checkpoint over the val + test splits and dump per-sample
predictions to a CSV, keyed by (exp_key, frame, patch).

This is a lightweight, human-inspectable precursor to R4's HDF5 export -- same
keying rule (every row self-describing by exp_key/frame/patch, never by
position), same F-FLIP unflip-to-original-camera-space rule for the persisted
vectors. It reuses R1's EyeGazeDataModule and R2's GazeEstimationModule as-is;
no model, loss, split, or data-pipeline code is touched.

Usage:
    py scripts/export_predictions.py --config configs/cluster_run.yaml --checkpoint runs/.../checkpoints/last.ckpt

Or just edit CHECKPOINT_PATH / CONFIG_PATH below and run with no args.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml
from evedataset import EveBundle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyenet.dataset import EyeGazeDataModule  # noqa: E402
from eyenet.lightning_module import GazeEstimationModule  # noqa: E402
from eyenet.losses import angular_error_degrees  # noqa: E402

# --------------------------------------------------------------------------- #
# EDIT THESE (or pass --checkpoint / --config on the CLI, which override them). #
# --------------------------------------------------------------------------- #
CHECKPOINT_PATH = "runs/cluster_run_20260723_190831/checkpoints/'epoch=1-step=10354.ckpt'"
CONFIG_PATH = "configs/cluster_run.yaml"
OUTPUT_CSV = "predictions.csv"          # written next to the checkpoint's run dir
SPLITS = ("val", "test")                # which splits to export
# --------------------------------------------------------------------------- #


def _unflip_x(vec: torch.Tensor, patch: str) -> torch.Tensor:
    """Undo the F-FLIP canonical-eye mirror so persisted vectors live in the
    original (non-mirrored) normalized camera space. Right-eye crops were fed to
    the net mirrored (x negated); left-eye passed through. See
    src/eyenet/geometry.flip_for_canonical_eye and TechStack Left/Right Flip."""
    if patch == "right":
        return vec * torch.tensor([-1.0, 1.0, 1.0], dtype=vec.dtype)
    return vec


@torch.no_grad()
def export(config_path: str, checkpoint_path: str, output_csv: str) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())

    for key in ("bundle_dir", "crops_root"):
        path = Path(cfg["data"][key])
        if not path.exists():
            raise FileNotFoundError(f"data.{key} does not exist: {path}")
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {ckpt}")

    bundle = EveBundle.load(cfg["data"]["bundle_dir"])
    datamodule = EyeGazeDataModule(
        bundle,
        cfg["data"]["crops_root"],
        split_source=cfg["data"]["split_source"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    datamodule.setup()

    module = GazeEstimationModule.load_from_checkpoint(str(ckpt))
    module.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device)

    loaders = {
        "val": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }

    out_path = Path(output_csv)
    if not out_path.is_absolute():
        out_path = Path(cfg["output"]["dir"]) / output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "exp_key", "frame", "patch",
            "pred_x", "pred_y", "pred_z",           # original camera space (unflipped)
            "target_x", "target_y", "target_z",     # original camera space (unflipped)
            "angular_error_deg",
        ])
        for split in SPLITS:
            for batch in loaders[split]:
                image, target, exp_key, frame, patch = batch
                image = image.to(device)
                target = target.to(device)
                pred = module(image)                # canonical (flipped) space

                # angular error is invariant to the flip (pred & target share it)
                err = angular_error_degrees(pred, target).cpu()  # (B,)

                pred = pred.cpu()
                target = target.cpu()
                for i in range(image.shape[0]):
                    p = patch[i]
                    pu = _unflip_x(pred[i], p)
                    tu = _unflip_x(target[i], p)
                    writer.writerow([
                        split, exp_key[i], int(frame[i]), p,
                        float(pu[0]), float(pu[1]), float(pu[2]),
                        float(tu[0]), float(tu[1]), float(tu[2]),
                        float(err[i]),
                    ])
                    n_written += 1

    print(f"Wrote {n_written} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()
    export(args.config, args.checkpoint, args.output)
