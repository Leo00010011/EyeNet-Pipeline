# R2 — Model & Training Loop — Requirements

## Goal

Turn the R1 data pipeline into a trainable model. This feature adds the three pieces that stand between `EyeGazeDataModule` and a trained checkpoint: an angular loss over unit gaze vectors, a ResNet18-backed regression head that emits an L2-normalized 3-vector, and a PyTorch Lightning module that binds them into a training/validation/test loop with mean-angular-error reporting. Success for R2 is not a good model — it is a *provably correct and end-to-end runnable* loop: loss decreases on a small subset, no NaNs, checkpoints save and load, and every non-trivial transform (the loss, the head's normalization, the angular metric) is unit-tested against hand-computed values per Mission.md §Data Quality Standard #1.

## Scope

**In scope**
- `src/eyenet/losses.py` — `angular_loss(pred, target)` (radians, differentiable) and `angular_error_degrees(pred, target)` (metric, degrees).
- `src/eyenet/model.py` — `GazeResNet18(pretrained=True)`, a `nn.Module`: ResNet18 backbone + FC head → 3-vector, L2-normalized to unit length in `forward`.
- `src/eyenet/lightning_module.py` — `GazeEstimationModule(pl.LightningModule)`: wires model + loss + optimizer, logs `train/loss`, `val/loss`, `val/angular_error_deg`, `test/angular_error_deg`.
- `scripts/train.py` — CLI entrypoint composing `EyeGazeDataModule` + `GazeEstimationModule` + `pl.Trainer` from a YAML config.
- `configs/baseline.yaml` — the small-subset baseline-run config (subset scoped via Lightning `limit_*` flags).
- Unit + integration tests for all of the above.

**Explicitly out of scope**
- **W&B integration — deferred to R3.** R2 ships `CSVLogger` only. *This is a deliberate deviation from Roadmap.md's R2 bullet list, decided during spec authoring:* the baseline run's purpose is to prove the loop is correct, and requiring network/credentials to do that couples a correctness check to an account. Roadmap.md must be updated to move the W&B bullet from R2 to R3 (see plan.md Step 8).
- **Image augmentation** — R3 ablation per Roadmap.md. R2 trains on `preprocess_eye_crop` output only. Note for whoever picks up R3: **horizontal flip must never be used as an augmentation** — it would silently invert the F-FLIP canonical-eye convention and desync every image from its label.
- **Full-split training and baseline-comparison evaluation** — R3.
- **Prediction export to HDF5** — R4. R2 produces checkpoints, not datasets. No `exp_key`-addressed persistence is written here, so the export-keying invariants are not yet exercised (see validation.md §Data Architecture Integrity for what *is* checked).
- **Hyperparameter tuning / LR search** — the baseline config's values are chosen to make the loop run, not to maximize accuracy.
- Any change to R1 modules (`dataset.py`, `sampling.py`, `splits.py`, `preprocessing.py`, `gaze_target.py`) or R0 geometry (`eye_norm.py`, `geometry.py`).

## Functional Requirements

### Loss and metric (`src/eyenet/losses.py`)

**FR1.** `angular_loss(pred, target)` accepts `pred` and `target` of shape `(B, 3)` float32 tensors and returns a **scalar** float32 tensor: the mean angle in **radians** between corresponding rows. It computes `cos = (pred_hat * target_hat).sum(-1)` where `_hat` denotes L2-normalization, then `arccos(clamp(cos, -1+EPS, 1-EPS)).mean()`. This is the arccos-of-normalized-dot-product formulation used in the appearance-based gaze literature, and it is the quantity backpropagated.

**FR2.** `angular_loss` L2-normalizes **both** arguments internally with `eps=1e-8` in the denominator, rather than assuming unit input. The model's `forward` already normalizes its output (FR6) and targets are unit by construction (`spherical_to_unit`), so this is defensive — but it makes the function correct in isolation and keeps its unit tests independent of the model.

**FR3.** `EPS = 1e-7` is a module-level constant. The clamp is **mandatory, not cosmetic**: `d/dx arccos(x) = -1/sqrt(1-x²)` diverges at `x = ±1`, so an unclamped `arccos` produces `NaN`/`inf` gradients exactly when the prediction is perfect or perfectly opposed — the former is reachable in training. Clamping to `1-1e-7` caps the gradient magnitude at a finite value.

**FR4.** `angular_error_degrees(pred, target)` accepts `(B, 3)` tensors and returns a `(B,)` float32 tensor of **per-sample** angles in **degrees** (not a mean — the caller aggregates, so Lightning's logger can handle batch-size weighting correctly). Same normalize-and-clamp path as `angular_loss`.

**FR5.** Both functions raise `ValueError` if either argument's last dimension is not 3, or if `pred.shape != target.shape`. A `(3,)` unbatched input is **not** accepted — the message must name the expected `(B, 3)` shape.

### Model (`src/eyenet/model.py`)

**FR6.** `GazeResNet18(pretrained: bool = True)` is an `nn.Module` whose `forward(x)` maps `(B, 3, 128, 128)` float32 → `(B, 3)` float32, **L2-normalized to unit length** (`‖out[i]‖ = 1.0 ± 1e-5` for every row). Normalization uses `F.normalize(v, p=2, dim=1, eps=1e-8)` so an all-zero head output degrades to a zero vector rather than `NaN`.

**FR7.** The backbone is `torchvision.models.resnet18` with `weights=ResNet18_Weights.IMAGENET1K_V1` when `pretrained=True`, else `weights=None`. `pretrained=False` exists so tests run offline with no weight download. The final `fc` is replaced with `nn.Linear(512, 3)`.

**FR8.** The 128×128 input requires no architectural change: ResNet18's `AdaptiveAvgPool2d((1,1))` makes it resolution-agnostic and 128×128 yields a 4×4 feature map before pooling. The model must **not** resize its input to 224×224 — the eye patch's 128×128 geometry is fixed by F-NORM and resizing would discard the canonical framing for no gain.

**FR9.** `GazeResNet18` exposes no eye/patch awareness. Per the F-FLIP convention, every crop reaching the model is already in canonical (left-eye) orientation with its target x-negated to match, so the network sees one orientation only.

### Lightning module (`src/eyenet/lightning_module.py`)

**FR10.** `GazeEstimationModule(pretrained=True, lr=1e-4, weight_decay=0.0)` subclasses `pl.LightningModule`, calls `save_hyperparameters()`, and holds a `GazeResNet18` as `self.model`. `forward` delegates to `self.model`.

**FR11.** `training_step(batch, batch_idx)` unpacks the R1 batch as `(image, target, exp_key, frame, patch)` — `exp_key`/`frame`/`patch` are collated metadata and are **not** used by the training step; they exist for R4's keyed export. It returns `angular_loss(self(image), target)` and logs it as `train/loss` (`on_step=True, on_epoch=True, prog_bar=True`).

**FR12.** `validation_step` logs `val/loss` (radians, the same quantity as the training objective) and `val/angular_error_deg` (`angular_error_degrees(...).mean()`), both `on_epoch=True`. `test_step` logs `test/angular_error_deg` identically. Logging degrees as a *separate* metric from the radian loss is what makes runs comparable against published baselines, which report degrees.

**FR13.** `configure_optimizers` returns `torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)`. No LR scheduler in R2 — a scheduler would make "did the loss go down?" a harder question to answer on a 2-epoch subset run.

**FR14.** The module must not depend on `EveBundle` or any accessor. It receives tensors only, so its tests can run against synthetic batches with no bundle fixture.

### Training script (`scripts/train.py`)

**FR15.** `python scripts/train.py --config configs/baseline.yaml` runs a training loop end-to-end. `--config` is required; the file is parsed with `yaml.safe_load`.

**FR16.** The config schema is:

```yaml
data:
  bundle_dir: ../eve_shared/EveDataset/bundle
  crops_root: ../eve_shared/eve_out
  batch_size: 32
  num_workers: 4
  split_source:            # exactly one of the two forms below
    seed: 42
    val_fraction: 0.2
    # path: splits/baseline_split.json
model:
  pretrained: true
  lr: 1.0e-4
  weight_decay: 0.0
trainer:                   # passed through to pl.Trainer(**trainer)
  max_epochs: 2
  limit_train_batches: 50
  limit_val_batches: 10
  accelerator: auto
  log_every_n_steps: 5
output:
  dir: runs/baseline
```

**FR17.** The `trainer:` block is **passed through to `pl.Trainer(**cfg["trainer"])` unmodified**. This is how the R2 baseline run is scoped to a small subset — `limit_train_batches`/`limit_val_batches` — with **no new data-layer code and no change to the split policy**. The subject-level split from R1 stays exactly as specified; only the number of batches drawn per epoch is capped.

**FR18.** The script configures `pl.loggers.CSVLogger(save_dir=cfg["output"]["dir"], name="csv")` and a `ModelCheckpoint(dirpath=<output.dir>/checkpoints, monitor="val/angular_error_deg", mode="min", save_top_k=1, save_last=True)`. No W&B (see Scope).

**FR19.** The script raises `FileNotFoundError` with the offending path if `bundle_dir` or `crops_root` does not exist, **before** constructing the Trainer — failing after a slow bundle load and a model download is a bad way to learn about a typo.

**FR20.** `main(config_path)` is importable and callable, so a test can drive a 2-batch run without a subprocess. The `if __name__ == "__main__"` block only parses argv and calls it.

## Public API Summary

```python
# src/eyenet/losses.py
EPS: float = 1e-7

def angular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B,3), (B,3) -> scalar float32. Mean arccos(clamped normalized dot), radians."""

def angular_error_degrees(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B,3), (B,3) -> (B,) float32. Per-sample angle, degrees."""

# src/eyenet/model.py
class GazeResNet18(nn.Module):
    def __init__(self, pretrained: bool = True) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,128,128) -> (B,3) unit
        ...

# src/eyenet/lightning_module.py
class GazeEstimationModule(pl.LightningModule):
    def __init__(self, pretrained: bool = True, lr: float = 1e-4,
                 weight_decay: float = 0.0) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def training_step(self, batch, batch_idx) -> torch.Tensor: ...
    def validation_step(self, batch, batch_idx) -> None: ...
    def test_step(self, batch, batch_idx) -> None: ...
    def configure_optimizers(self) -> torch.optim.Optimizer: ...

# scripts/train.py
def main(config_path: str) -> None: ...
```

## Dependencies

| Direction | What | Source / Sink |
|---|---|---|
| Reads | `(image, target, exp_key, frame, patch)` batches | `EyeGazeDataModule` (R1, `src/eyenet/dataset.py`) — unchanged |
| Reads | ImageNet-pretrained ResNet18 weights | `torchvision.models` (network on first use; `pretrained=False` in tests) |
| Reads | Run configuration | `configs/baseline.yaml` |
| Reads (indirect) | Face crops, `W`, crop origin, `g_tobii`, validity | `EveBundle` — only via the R1 DataModule, never directly |
| Writes | Checkpoints | `<output.dir>/checkpoints/{epoch=..-val_angular_error_deg=..}.ckpt`, `last.ckpt` |
| Writes | Loss/metric history | `<output.dir>/csv/version_*/metrics.csv` |
| New dependency | `pyyaml` | Added to `requirements.txt` for config parsing |

**Unchanged, depended upon:** `src/eyenet/dataset.py`, `sampling.py`, `splits.py`, `preprocessing.py`, `gaze_target.py`, `geometry.py`, `src/eye_norm.py`. R2 adds no requirement to any of them.
