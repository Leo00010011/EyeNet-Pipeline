"""LightningModule binding GazeResNet18 + angular loss + Adam.

Receives tensors only -- never an EveBundle or accessor -- so its tests run
against synthetic batches with no bundle fixture. The bundle-dependent path is
already covered at the R1 seam (tests/test_dataset.py).
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from eyenet.gaze_target import unit_to_spherical
from eyenet.losses import angular_error_degrees, get_loss
from eyenet.metrics import angular_variance
from eyenet.model import GazeResNet18


class GazeEstimationModule(pl.LightningModule):
    def __init__(
        self,
        pretrained: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        dropout1: float | None = None,
        dropout2: float | None = None,
        loss: str = "angular",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = GazeResNet18(
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            dropout=dropout,
            dropout1=dropout1,
            dropout2=dropout2,
        )
        self._loss_fn = get_loss(loss)  # FR7: raises on a bad name at construction
        self._buf = {}

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        # R1 batch: (image, target, exp_key, frame, patch). The last three are
        # R4 export keys, unused here except patch (F-WANDB per-eye metric).
        image, target = batch[0], batch[1]
        pred = self(image)
        per_sample_deg = angular_error_degrees(pred, target)  # (B,)
        # FR3: the reported metric stays angular_error_degrees whatever the
        # training loss, so trials under different losses stay comparable.
        return self._loss_fn(pred, target), per_sample_deg, pred, target

    def _reset_buffers(self, stage: str) -> None:
        self._buf[stage] = {"pred": [], "target": [], "deg": [], "patch": [], "theta_err": [], "phi_err": []}

    def on_train_epoch_start(self) -> None:
        self._reset_buffers("train")

    def on_validation_epoch_start(self) -> None:
        self._reset_buffers("val")

    def _accumulate(self, stage, pred, target, per_sample_deg, batch):
        if stage not in self._buf:
            self._reset_buffers(stage)
        b = self._buf[stage]
        b["pred"].append(pred.detach().cpu())
        b["target"].append(target.detach().cpu())
        b["deg"].append(per_sample_deg.detach().cpu())

        # FR14: patch is a TUPLE OF str from default_collate, not a tensor.
        # FR16: synthetic 2-tuple test batches have no patch -- skip, don't fail.
        if len(batch) > 4:
            b["patch"].extend(batch[4])

        sp = unit_to_spherical(pred.detach())
        st = unit_to_spherical(target.detach())
        d_theta = sp[:, 0] - st[:, 0]
        # FR19: phi comes from atan2 and wraps at +/-pi. Two near-identical gazes
        # straddling the branch cut would read as ~360 deg of error unwrapped.
        d_phi = torch.atan2(torch.sin(sp[:, 1] - st[:, 1]), torch.cos(sp[:, 1] - st[:, 1])) #TODO Check if this is right!
        b["theta_err"].append(torch.rad2deg(d_theta.abs()).cpu())
        b["phi_err"].append(torch.rad2deg(d_phi.abs()).cpu())

    def _emit(self, stage: str) -> None:
        b = self._buf.get(stage)
        if not b or not b["deg"]:
            return
        pred = torch.cat(b["pred"])  # (N, 3)
        target = torch.cat(b["target"])  # (N, 3)
        deg = torch.cat(b["deg"])  # (N,)
        prefix = stage

        if stage == "train":  # FR6; val/angular_error_deg already logged per-step
            self.log(f"{prefix}/angular_error_deg", deg.mean())

        # FR9/FR11: var(correction=1) on <2 samples is nan -- skip rather than log nan.
        if pred.shape[0] >= 2:
            var = pred.var(dim=0)  # (3,) per-component, NOT pooled
            for i, axis in enumerate("xyz"):
                self.log(f"{prefix}/pred_var_{axis}", var[i])

            # Angular variance: mean angle from mean vector to each vector
            pred_ang_var = angular_variance(pred)
            target_ang_var = angular_variance(target)
            self.log(f"{prefix}/pred_angular_variance_deg", pred_ang_var)
            self.log(f"{prefix}/target_angular_variance_deg", target_ang_var)

        # FR12/FR15: per-eye, epoch-level. Absent patch => no rows => not logged.
        if b["patch"]:
            patches = b["patch"]
            for eye in ("left", "right"):
                mask = torch.tensor([p == eye for p in patches], dtype=torch.bool)
                if mask.any():
                    self.log(f"{prefix}/angular_error_deg_{eye}", deg[mask].mean())

        # FR18
        self.log(f"{prefix}/theta_error_deg", torch.cat(b["theta_err"]).mean())
        self.log(f"{prefix}/phi_error_deg", torch.cat(b["phi_err"]).mean())

    def on_train_epoch_end(self) -> None:
        self._emit("train")

    def on_validation_epoch_end(self) -> None:
        self._emit("val")

    def training_step(self, batch, batch_idx):
        loss, per_sample_deg, pred, target = self._step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self._accumulate("train", pred, target, per_sample_deg, batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, per_sample_deg, pred, target = self._step(batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/angular_error_deg", per_sample_deg.mean(), on_epoch=True, prog_bar=True)
        self._accumulate("val", pred, target, per_sample_deg, batch)

    def test_step(self, batch, batch_idx):
        _, per_sample_deg, _, _ = self._step(batch)
        self.log("test/angular_error_deg", per_sample_deg.mean(), on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
