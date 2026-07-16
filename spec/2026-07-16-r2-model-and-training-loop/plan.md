# R2 — Model & Training Loop — Implementation Plan

## Context and Design Decisions

### Why arccos, and why the clamp is load-bearing

The objective is `mean(arccos(clamp(cos(pred, target))))` in radians — the arccos of the normalized dot product, as used in the appearance-based gaze literature. This optimizes the reported metric directly: `val/angular_error_deg` is the same quantity in different units, so there is no proxy-objective gap to reason about when reading a loss curve.

The cost is a real numerical hazard, and the clamp is the entire mitigation. `d/dx arccos(x) = -1/sqrt(1-x²)` → `-inf` as `x → ±1`. A perfect prediction (`cos = 1`) is not a hypothetical during training; it is the thing we are steering toward, and float32 rounding reaches exactly `1.0` well before the model is actually perfect. Without clamping, the first such batch emits `NaN` gradients, `NaN` propagates through every weight on the optimizer step, and the run silently produces a dead model that still "trains" to completion. `EPS = 1e-7` caps `|grad| ≈ 1/sqrt(2e-7) ≈ 2236`, which is finite and survivable. **validation.md Group 1 pins this with an explicit `cos = 1.0` gradient test** — this is the single highest-value test in R2.

Note the clamp also means the loss floor is `arccos(1-1e-7) ≈ 4.5e-4 rad ≈ 0.026°`, not exactly zero. That is far below any angular error we could meaningfully claim, so it does not affect interpretation.

### Where normalization happens, and why in two places

`GazeResNet18.forward` L2-normalizes its output (FR6) — the model's contract is "emits a unit vector," which matters for R4's export, where predictions are persisted directly. `angular_loss` *also* normalizes internally (FR2). This is deliberate redundancy, not an oversight: it makes the loss correct as a standalone pure function, so its unit tests can use hand-written non-unit vectors and hand-computed answers without constructing a model. The double-normalize is a no-op on already-unit input and costs one negligible kernel.

`F.normalize(..., eps=1e-8)` is specified over a manual `v / v.norm()` because the manual form yields `NaN` on a zero vector, and a freshly-initialized head can plausibly emit near-zero rows.

### Why `pretrained=False` must be reachable

Mission.md demands tests against real/realistic samples rather than mocks that hide convention bugs — but that applies to the *data and geometry* path, which R0/R1 already covers. The model tests are about tensor shapes, norms, and gradient flow, where ImageNet weights add nothing and a network download adds a failure mode. So `GazeResNet18(pretrained=False)` exists purely so the test suite is offline-deterministic. The baseline *run* uses `pretrained=True`, per TechStack.md.

### Why the Lightning module never touches `EveBundle`

`GazeEstimationModule` receives tensors and nothing else (FR14). This keeps the training logic testable against synthetic batches — a `(4, 3, 128, 128)` random tensor and a `(4, 3)` unit target need no bundle fixture, no crops root, and no disk. The bundle-dependent path is already tested at the R1 seam (`tests/test_dataset.py`); re-testing it through the Lightning module would just make model tests slow and fixture-coupled for no added coverage.

### Why `limit_*` flags for the subset, not an exp_key cap

Roadmap.md's R2 asks for a "baseline training run on a small subset to validate the loop end-to-end." The honest way to shrink that is Lightning's `limit_train_batches`/`limit_val_batches`, passed straight through from config (FR17). The alternative — capping exp_keys before `build_sample_index` — would touch R1 code and, worse, silently skew the subject-level split that R1 was careful to establish (Roadmap R1: "zero subject overlap across train/val/test"). Capping batches leaves the split policy exactly as specified and only draws fewer samples from it. The R1 data layer is imported and used as-is; **no file under `src/eyenet/` that exists today is modified by this feature.**

### Why W&B is deferred to R3 (deviation from Roadmap)

Roadmap.md lists W&B under R2. This spec moves it to R3, decided during spec authoring. R2's deliverable is a *correctness* result — loss decreases, no NaNs, checkpoints round-trip — and gating that on network access and an account turns a local check into an integration dependency. `CSVLogger` produces the same loss curve as a file the test suite can assert on directly (validation.md Group 5), which W&B cannot. Step 8 updates Roadmap.md so the constitution and the code do not disagree.

### What R2 deliberately does not establish

No `exp_key`-addressed persistence exists yet — checkpoints are weights, not datasets, so Mission.md §3's positional-coupling rule has nothing to bind to here. But the batch tuple already carries `(exp_key, frame, patch)` through from R1, and `training_step` ignores it (FR11). That metadata is R4's export key, and validation.md §Data Architecture Integrity checks that it survives collation intact — so R4 inherits a working key path rather than discovering a broken one.

---

## Step 1 — `src/eyenet/losses.py`

New file. Pure tensor functions, no I/O, no bundle import.

```python
"""Angular loss and metric over unit gaze vectors.

Objective is arccos of the normalized dot product, in radians. The clamp is
mandatory: arccos' gradient diverges at cos = ±1, which is reachable when a
prediction is (or rounds to) exact, and would poison every weight with NaN.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

EPS = 1e-7


def _check(pred, target):
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 2 or pred.shape[-1] != 3:
        raise ValueError(f"expected (B, 3) tensors, got {tuple(pred.shape)}")


def _cos(pred, target):
    p = F.normalize(pred.float(), p=2, dim=-1, eps=1e-8)
    t = F.normalize(target.float(), p=2, dim=-1, eps=1e-8)
    return (p * t).sum(dim=-1).clamp(-1.0 + EPS, 1.0 - EPS)


def angular_loss(pred, target):        # -> scalar, radians
    _check(pred, target)
    return torch.arccos(_cos(pred, target)).mean()


def angular_error_degrees(pred, target):   # -> (B,), degrees
    _check(pred, target)
    return torch.arccos(_cos(pred, target)) * (180.0 / math.pi)
```

Note `_cos` clamps **before** returning, so both callers share one clamp — there is no path to an unclamped arccos.

## Step 2 — `tests/test_losses.py`

New file. This is the step that earns R2 its correctness claim; write it before Step 3.

Hand-computed cases (per Roadmap R2: "identical vectors → 0 error; orthogonal vectors → 90°"):
- identical unit vectors → loss `≈ 0` (`< 1e-3` rad, accounting for the EPS floor); degrees `≈ 0`.
- orthogonal (`[1,0,0]` vs `[0,1,0]`) → loss `≈ π/2` (`atol=1e-5`); degrees `≈ 90.0` (`atol=1e-3`).
- opposed (`[1,0,0]` vs `[-1,0,0]`) → degrees `≈ 180.0` (`atol=1e-2`; the EPS clamp costs ~0.026° here).
- 60° case: `[1,0,0]` vs `[0.5, sqrt(3)/2, 0]` → degrees `≈ 60.0` (`atol=1e-3`).
- non-unit input: `[2,0,0]` vs `[0,5,0]` → still `90.0°`, proving FR2's internal normalization.
- batch: a `(4,3)` mix of the above → `angular_error_degrees` returns shape `(4,)` matching each row's expected value; `angular_loss` equals the mean of those in radians.

**The NaN-gradient test (critical):** `pred = target = [[1.,0.,0.]]` with `pred.requires_grad_()`; `angular_loss(pred, target).backward()`; assert `torch.isfinite(pred.grad).all()`. Without the FR3 clamp this yields `NaN` and the test fails — which is precisely the regression it guards.

Error paths: `(3,)` input raises `ValueError` naming `(B, 3)`; `(B,4)` raises; mismatched `(2,3)` vs `(3,3)` raises.

## Step 3 — `src/eyenet/model.py`

New file. Depends on nothing in this spec.

```python
"""ResNet18 + regression head -> unit gaze vector."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class GazeResNet18(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(512, 3)

    def forward(self, x):                      # (B,3,128,128) -> (B,3) unit
        return F.normalize(self.backbone(x), p=2, dim=1, eps=1e-8)
```

No input resize (FR8) — `AdaptiveAvgPool2d` handles 128×128 natively.

## Step 4 — `tests/test_model.py`

New file. All cases use `pretrained=False` (offline, per Design Decisions).

- `forward` on `(2, 3, 128, 128)` → output shape `(2, 3)`, dtype float32.
- every output row has `‖v‖ = 1.0 ± 1e-5`.
- gradients flow: `out.sum().backward()`; the head's `backbone.fc.weight.grad` is non-`None` and finite.
- a zero-image input still produces finite output (no `NaN` from the eps-guarded normalize).
- `GazeResNet18(pretrained=False).backbone.fc.out_features == 3`.

## Step 5 — `src/eyenet/lightning_module.py`

New file. Imports Step 1 and Step 3.

```python
"""LightningModule binding GazeResNet18 + angular loss + Adam."""
from __future__ import annotations

import pytorch_lightning as pl
import torch

from eyenet.losses import angular_error_degrees, angular_loss
from eyenet.model import GazeResNet18


class GazeEstimationModule(pl.LightningModule):
    def __init__(self, pretrained: bool = True, lr: float = 1e-4, weight_decay: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = GazeResNet18(pretrained=pretrained)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        # R1 batch: (image, target, exp_key, frame, patch); the last three are
        # R4 export keys, unused here.
        image, target = batch[0], batch[1]
        pred = self(image)
        return angular_loss(pred, target), angular_error_degrees(pred, target).mean()

    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, deg = self._step(batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/angular_error_deg", deg, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, deg = self._step(batch)
        self.log("test/angular_error_deg", deg, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
```

`batch[0], batch[1]` rather than a 5-way unpack keeps the module tolerant of synthetic 2-tuple batches in tests while still consuming R1's real 5-tuple correctly.

## Step 6 — `tests/test_lightning_module.py`

New file. Synthetic batches only — no bundle fixture (FR14).

- `training_step` on a synthetic `(image=(4,3,128,128) randn, target=(4,3) unit)` returns a finite scalar with `requires_grad=True`.
- overfit check: `pl.Trainer(overfit_batches=1, max_epochs=30, logger=False, enable_checkpointing=False, accelerator="cpu")` on a fixed 4-sample synthetic set → final `train/loss` is meaningfully below the first epoch's. This is the "loss decreases, no NaNs" claim in miniature, and it runs in seconds.
- `configure_optimizers` returns an `Adam` whose `param_groups[0]["lr"]` matches the constructor arg.
- `save_hyperparameters` round-trip: `GazeEstimationModule.load_from_checkpoint(path)` after `trainer.save_checkpoint(path)` restores `lr` and produces identical outputs on a fixed input (`atol=1e-6`) — the checkpoint save/load half of Roadmap R2's acceptance.

## Step 7 — `scripts/train.py` and `configs/baseline.yaml`

New files. `configs/baseline.yaml` is the FR16 document verbatim.

```python
def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    # FR19: fail on bad paths before the slow bundle load / weight download.
    for key in ("bundle_dir", "crops_root"):
        p = Path(cfg["data"][key])
        if not p.exists():
            raise FileNotFoundError(f"data.{key} does not exist: {p}")

    bundle = EveBundle.load(cfg["data"]["bundle_dir"])
    dm = EyeGazeDataModule(bundle, cfg["data"]["crops_root"],
                           split_source=cfg["data"]["split_source"],
                           batch_size=cfg["data"]["batch_size"],
                           num_workers=cfg["data"]["num_workers"])
    module = GazeEstimationModule(**cfg["model"])

    out = Path(cfg["output"]["dir"])
    trainer = pl.Trainer(
        logger=CSVLogger(save_dir=str(out), name="csv"),
        callbacks=[ModelCheckpoint(dirpath=str(out / "checkpoints"),
                                   monitor="val/angular_error_deg", mode="min",
                                   save_top_k=1, save_last=True)],
        **cfg["trainer"],            # FR17: pass-through, incl. limit_* subset flags
    )
    trainer.fit(module, datamodule=dm)
```

`split_source` is forwarded as the dict R1's DataModule already expects (`{"seed", "val_fraction"}` or `{"path"}`) — no translation layer.

Add `pyyaml` to `requirements.txt`.

## Step 8 — `tests/test_train_script.py` and Roadmap correction

Test (FR20, `main` importable):
- a `tmp_path` config with `limit_train_batches: 2, limit_val_batches: 1, max_epochs: 1, num_workers: 0`, `pretrained: false`, pointed at the real `sample_bundle`/`face_crops_root` fixtures from `tests/conftest.py`; `main(cfg_path)` completes; `<out>/checkpoints/last.ckpt` exists; `<out>/csv/version_0/metrics.csv` exists and contains a `train/loss` column with finite values. This is the one R2 test that exercises the real bundle end-to-end.
- a config with a bogus `bundle_dir` raises `FileNotFoundError` naming the path, and does so **fast** (no Trainer constructed).

Then update `spec/constitution/Roadmap.md` R2: mark the model/loss/baseline-run bullets done with a summary line, **move the W&B bullet to R3**, and note the augmentation deferral (with the never-horizontal-flip warning). Update `TechStack.md` with a New Modules (R2) table for `losses.py`/`model.py`/`lightning_module.py`, and note `pyyaml`.

## Step 9 — `notebooks/inspect_r2_training.ipynb`

New notebook, executed via `jupyter nbconvert --execute` with outputs persisted (the pattern every prior feature used):
1. Plot `train/loss` and `val/angular_error_deg` from the baseline run's `metrics.csv` — the visual form of "loss decreases."
2. Report the untrained-model baseline angular error over one val batch (expect ≈ 90°, the mean angle between a random unit vector and a fixed one), then the trained subset model's — establishes the loop learns *something*, without claiming a competitive number (that is R3).
3. Grid of 8 val eye crops with ground-truth vs predicted gaze arrows overlaid, per validation.md §Data Validity.

## Implementation Order

1. `src/eyenet/losses.py` (Step 1)
2. `tests/test_losses.py` — incl. the NaN-gradient test (Step 2)
3. `src/eyenet/model.py` (Step 3)
4. `tests/test_model.py` (Step 4)
5. `src/eyenet/lightning_module.py` (Step 5)
6. `tests/test_lightning_module.py` — incl. overfit + checkpoint round-trip (Step 6)
7. `scripts/train.py`, `configs/baseline.yaml`, `pyyaml` pin (Step 7)
8. `tests/test_train_script.py`; Roadmap.md / TechStack.md corrections (Step 8)
9. `notebooks/inspect_r2_training.ipynb` (Step 9)

Steps 1–6 have no dependency on `EveBundle` and can be completed and verified with the bundle absent. Only Steps 7–9 touch real data.
