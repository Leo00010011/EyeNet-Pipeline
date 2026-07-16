# F-WANDB — Weights & Biases experiment tracking

## Goal

R2 shipped `CSVLogger`, and its diagnostics live in `notebooks/inspect_r2_training.ipynb`, which must be re-executed by hand against a finished run. That is adequate for one 2-epoch subset run and useless for R3, whose questions are *comparative* ("did this change help?") and need answering **during** training, across runs. This feature adds a live W&B dashboard alongside the existing `CSVLogger`, carrying the four diagnostics that are actually decision-relevant on this data: angular error in degrees, output variance (the collapse detector), error split by eye (the F-FLIP break detector), and error split by axis (`theta` vs `phi`). It also makes the training script runnable **unattended in a job queue** — authentication by environment variable, no interactive login, and a W&B outage can never kill a queued run.

This is a logging feature. It changes no model, loss, split policy, or data-pipeline behavior.

## Scope

**In scope:**

- A `logging:` block in `configs/*.yaml` gating W&B, defaulting to disabled where tests can reach it.
- `WandbLogger` composed alongside `CSVLogger` and passed to `pl.Trainer(logger=[...])`.
- `unit_to_spherical(g)` in `src/eyenet/gaze_target.py` — torch, batched, the inverse of the existing numpy `spherical_to_unit`.
- Four new logged metric families in `GazeEstimationModule`: `train/angular_error_deg`, per-component prediction variance, per-eye error, per-axis error.
- Epoch-level aggregation buffers in the Lightning module for the per-eye and per-axis metrics.
- Headless authentication via `WANDB_API_KEY`, documented for a queued job.
- Graceful degradation: a W&B failure warns and falls through to `CSVLogger`; it never raises into the training loop.

**Explicitly out of scope:**

- Sample-crop image panels, gradient/parameter histograms, sweeps, artifact/model registry. Add only when a real question needs them.
- Replacing `CSVLogger`. It stays — R2's tests assert on `metrics.csv`, and a file the test suite can read is something W&B cannot provide.
- Any change to `losses.py`, `model.py`, `dataset.py`, `sampling.py`, `splits.py`. The angular-error computation is reused as-is via `angular_error_degrees`.
- Offline mode (`WANDB_MODE=offline`) as a *supported config path*. It is documented as an operator escape hatch (§FR20) but adds no code and no config key.
- Changing the training objective. The radian `angular_loss` remains what is optimized; degrees are logged only.

## Functional Requirements

### Gaze-target inverse

**FR1.** `unit_to_spherical(g)` is added to `src/eyenet/gaze_target.py`. It accepts a `torch.Tensor` of shape `(B, 3)` or `(3,)` and returns a `torch.Tensor` of shape `(B, 2)` or `(2,)` respectively, dtype matching the input's floating dtype, on the input's device. Column 0 is `theta`, column 1 is `phi`, both in radians.

**FR2.** `unit_to_spherical` implements the MPIIGaze convention, the exact inverse of `spherical_to_unit`:
```
theta = arcsin(-g_y)
phi   = arctan2(-g_x, -g_z)
```
`g_y` is clamped to `[-1 + EPS, 1 - EPS]` with `EPS = 1e-7` before `arcsin`, reusing the constant already exported by `src/eyenet/losses.py`. This mirrors the `losses.py` rationale: `arcsin`' derivative diverges at `±1`, and a unit vector pointing straight down is representable in float32. The clamp is not cosmetic and must not be removed.

**FR3.** `unit_to_spherical` does **not** normalize its input. It documents that it assumes a unit vector, which is guaranteed by `GazeResNet18.forward`'s `F.normalize` and by `spherical_to_unit`'s construction. Passing a non-unit vector is a caller error, not a handled case.

**FR4.** `unit_to_spherical` raises `ValueError` if the input's last dimension is not 3, or if it has more than 2 dimensions. The message names the received shape.

**FR5.** The existing numpy `spherical_to_unit` is **not modified**. `gaze_target.py` therefore holds one numpy function and one torch function; this is deliberate — `spherical_to_unit` runs once per sample in the numpy-side data pipeline, `unit_to_spherical` runs per batch on the training device, and forcing either into the other's backend would add a per-batch host/device sync for no benefit.

### Angular error in degrees, train and validation (metric 1)

**FR6.** `training_step` logs `train/angular_error_deg` with `on_step=False, on_epoch=True`. The value is the batch mean of `angular_error_degrees(pred, target)`. R2's `_step` already computes this quantity and discards it in `training_step`; this requirement only logs it.

**FR7.** `val/angular_error_deg` continues to be logged exactly as R2 logs it. No change.

**FR8.** `train/loss` continues to be logged with `on_step=True, on_epoch=True`. R2's tests assert on `metrics.csv`'s `train/loss_step` / `train/loss_epoch` columns; changing the `on_step`/`on_epoch` flags would rename those columns and break them.

### Output variance (metric 2)

**FR9.** The module accumulates predicted unit vectors over each epoch and logs, at epoch end, the per-component variance of the predictions:
- `train/pred_var_x`, `train/pred_var_y`, `train/pred_var_z`
- `val/pred_var_x`, `val/pred_var_y`, `val/pred_var_z`

Each is a scalar float. Variance is computed over all samples seen in that epoch's split (unbiased, `torch.var` default `correction=1`), per component of the 3-vector, not as a single pooled number — a collapse can be axis-specific, and R2's subset run showed exactly that shape of failure (`theta` flat, `phi` learning).

**FR10.** Variance is computed from the **model's raw output** in the canonical (flipped) frame the network sees — the same vectors the loss is computed against. No unflipping. Unflipping is R4's export-time concern; applying it here would mix two frames in one statistic.

**FR11.** If an epoch accumulates fewer than 2 samples, the variance metrics are **not logged** for that epoch (`torch.var` with `correction=1` on a single sample is `nan`). No exception is raised. This is reachable in tests via `limit_val_batches` with a batch size of 1.

### Error split by eye (metric 3)

**FR12.** The module accumulates per-sample angular error in degrees together with the sample's `patch` string over each epoch, and logs at epoch end:
- `train/angular_error_deg_left`, `train/angular_error_deg_right`
- `val/angular_error_deg_left`, `val/angular_error_deg_right`

Each is the mean over all samples of that patch seen in the epoch.

**FR13.** These metrics aggregate over the **epoch**, never per batch. A batch may contain only one patch, which would make a per-batch left/right comparison pure noise. Accumulation happens in `on_*_batch_end`-equivalent code inside the step method; emission happens in `on_*_epoch_end`.

**FR14.** `patch` arrives as a **tuple of `str`**, not a tensor — this is `default_collate`'s confirmed R2 behavior. The metric code masks with a list comprehension or a one-time conversion to a boolean tensor. It must not assume tensor indexing on `patch`.

**FR15.** If an epoch contains **zero** samples of a patch, that patch's metric is **not logged** for that epoch. No `nan` is emitted and no exception is raised. This is the normal case for a synthetic single-patch test batch.

**FR16.** If the batch has no `patch` element (a synthetic 2-tuple batch, which R2's `_step` explicitly tolerates), the per-eye accumulation is **skipped silently** and the rest of the step proceeds. `tests/test_lightning_module.py` drives the module on 2-tuple batches and must keep passing unchanged.

**FR17.** A persistent left/right gap is the signature of an F-FLIP convention break. The metric's job is to make that legible during a run; interpreting it is not this feature's concern and no threshold or alert is implemented.

### Error split by axis (metric 4)

**FR18.** The module logs, at epoch end, mean absolute per-axis error in degrees:
- `train/theta_error_deg`, `train/phi_error_deg`
- `val/theta_error_deg`, `val/phi_error_deg`

computed by converting both `pred` and `target` through `unit_to_spherical` and taking the mean absolute difference per component, converted to degrees.

**FR19.** The `phi` difference is **wrapped to `[-pi, pi]`** before the absolute value is taken:
```
d_phi = torch.atan2(torch.sin(phi_p - phi_t), torch.cos(phi_p - phi_t))
```
`phi` comes from `arctan2` and is discontinuous at `±pi`. Two nearly identical gaze directions straddling that branch cut would otherwise report a ~360° error and poison the epoch mean. `theta` comes from `arcsin` and is confined to `[-pi/2, pi/2]`, so it needs no wrapping.

### Configuration and headless operation

**FR20.** A `logging:` block is added to the config schema:
```yaml
logging:
  wandb:
    enabled: false
    project: eyenet
    entity: null        # null => W&B default entity for the API key
    run_name: null      # null => W&B generates one
    tags: []
```
`enabled: false` is the default and is what any config the test suite reaches must carry.

**FR21.** When `logging.wandb.enabled` is `false` (or the `logging` block is absent entirely), `scripts/train.py` must **not import `wandb` or `WandbLogger`**, and must make no network call. The import is local to the enabled branch, not at module top level. `pl.Trainer` receives `logger=[CSVLogger(...)]` only. This is what keeps `tests/test_train_script.py` runnable offline with no account.

**FR22.** A missing `logging` block is equivalent to `logging.wandb.enabled: false`. `configs/baseline.yaml` — the R2 config the tests drive — is therefore valid unchanged, though FR27 adds the explicit block to it anyway for discoverability.

**FR23.** When `logging.wandb.enabled` is `true`, `scripts/train.py` composes a `WandbLogger` and passes `logger=[CSVLogger(...), WandbLogger(...)]` to `pl.Trainer`, in that order. `CSVLogger` first is load-bearing: Lightning takes `trainer.log_dir` and the `ModelCheckpoint` default directory from `logger[0]`, and `output.dir` / `runs/<name>/csv/version_*/metrics.csv` must not move.

**FR24.** Authentication is by the **`WANDB_API_KEY` environment variable**. `scripts/train.py` never calls `wandb.login()` interactively and never prompts. The key is read by `wandb` itself from the environment; the script does not read, log, print, or otherwise handle the key's value.

**FR25.** If `logging.wandb.enabled` is `true` but `WANDB_API_KEY` is unset, the script emits a `warnings.warn` naming the variable and continues **with `CSVLogger` only**. It does not raise. Rationale: a queued R3 full-split run costs hours of wall clock; losing it to a missing dashboard credential is a strictly worse outcome than losing the dashboard. The training result is the deliverable; the dashboard is instrumentation.

**FR26.** If constructing the `WandbLogger` raises for any reason (network unreachable, key rejected, `wandb` not installed), the exception is caught, `warnings.warn` reports it with the original message, and training proceeds with `CSVLogger` only. No W&B failure propagates into `trainer.fit`. `metrics.csv` is written in every case, so no run is ever unrecoverable.

**FR27.** `configs/baseline.yaml` gains an explicit `logging.wandb.enabled: false` block. Its behavior is unchanged (FR22 makes the block redundant), but the R2 config is the file a reader copies, and a silently-absent key is worse documentation than an explicit `false`.

**FR28.** A new `configs/r3_full.yaml` is **not** part of this feature. R3 owns its own config; F-WANDB only makes the `logging:` block available to it.

### Dependency

**FR29.** `requirements.txt` gains `wandb`. It is imported only on the `enabled: true` path (FR21), so the test suite does not depend on it being importable — but it is a declared dependency, not an optional extra, because R3's runs require it.

## Public API Summary

```python
# src/eyenet/gaze_target.py  (added; existing spherical_to_unit unchanged)

def unit_to_spherical(g: torch.Tensor) -> torch.Tensor:
    """Inverse of spherical_to_unit, MPIIGaze convention.

    g: (B, 3) or (3,) unit vector tensor. Assumed unit-norm; not normalized here.
    Returns (B, 2) or (2,): column 0 = theta, column 1 = phi, radians.
        theta = arcsin(-g_y)   (g_y clamped to [-1+EPS, 1-EPS])
        phi   = arctan2(-g_x, -g_z)
    Raises ValueError if g.shape[-1] != 3 or g.ndim > 2.
    """


# src/eyenet/lightning_module.py  (signature unchanged; internals extended)

class GazeEstimationModule(pl.LightningModule):
    def __init__(self, pretrained: bool = True, lr: float = 1e-4,
                 weight_decay: float = 0.0) -> None: ...

    # new epoch-aggregation hooks
    def on_train_epoch_start(self) -> None: ...
    def on_train_epoch_end(self) -> None: ...
    def on_validation_epoch_start(self) -> None: ...
    def on_validation_epoch_end(self) -> None: ...


# scripts/train.py  (signature unchanged)

def main(config_path: str) -> None: ...
def build_loggers(cfg: dict, out: Path) -> list:
    """CSVLogger always first (FR23); WandbLogger appended when enabled and
    constructible. Never raises on a W&B failure (FR25, FR26)."""
```

### Logged metric names

| Name | Split | When | Source |
|---|---|---|---|
| `train/loss_step`, `train/loss_epoch` | train | R2, unchanged | `angular_loss`, radians |
| `val/loss` | val | R2, unchanged | `angular_loss`, radians |
| `val/angular_error_deg` | val | R2, unchanged | `angular_error_degrees` |
| `test/angular_error_deg` | test | R2, unchanged | `angular_error_degrees` |
| `train/angular_error_deg` | train | **new**, epoch | `angular_error_degrees` (FR6) |
| `{train,val}/pred_var_{x,y,z}` | both | **new**, epoch | per-component `torch.var` (FR9) |
| `{train,val}/angular_error_deg_{left,right}` | both | **new**, epoch | patch-masked mean (FR12) |
| `{train,val}/{theta,phi}_error_deg` | both | **new**, epoch | `unit_to_spherical` diff (FR18) |

## Dependencies

| This feature | Reads from | Writes to |
|---|---|---|
| `unit_to_spherical` | `losses.EPS` | — |
| `GazeEstimationModule` | `losses.angular_error_degrees`, `gaze_target.unit_to_spherical`, `batch[4]` (`patch` tuple) | Lightning loggers (`self.log`) |
| `scripts/train.py` | `cfg["logging"]["wandb"]`, `os.environ["WANDB_API_KEY"]` | `pl.Trainer(logger=[...])` |
| `configs/baseline.yaml` | — | `logging.wandb.enabled: false` |
| `requirements.txt` | — | `wandb` |

**Unmodified by this feature:** `losses.py`, `model.py`, `dataset.py`, `sampling.py`, `splits.py`, `eye_norm.py`.

**Consumes for the first time:** `batch[4]` (`patch`). R2's `training_step` deliberately ignores it; R4's export is its only other consumer.

## Running unattended in a job queue

The queued-job constraint drives FR24–FR26. The operator contract:

1. **Get the key once**, interactively, from a machine with a browser: `https://wandb.ai/authorize` (or `wandb login` on any workstation, which writes `~/.netrc`).
2. **Make it available to the job.** Preferred — put it in the submit script, not in the repo:
   ```bash
   #!/bin/bash
   #SBATCH --job-name=eyenet-r3
   export WANDB_API_KEY=<key>
   py scripts/train.py --config configs/r3_full.yaml
   ```
   Or export it in `~/.bashrc` / `~/.bash_profile` on the server so every job inherits it. `wandb` reads the variable itself; nothing is passed on the command line, where it would be visible in the queue's process listing.
3. **Never commit the key.** It is a credential. It belongs in the environment or in `~/.netrc` (mode `600`), not in `configs/*.yaml`, which is version-controlled.
4. **If the compute nodes have no outbound network**, the escape hatch is `export WANDB_MODE=offline` in the same submit script. `wandb` then buffers the run under `<output.dir>/wandb/offline-run-*`, and `wandb sync <path>` from a networked login node uploads it afterwards. This requires **no code and no config change** (FR20) — it is entirely an environment variable that `wandb` honors on its own, which is why it is out of scope as a config path.
5. **A W&B failure never costs you the run.** Per FR25/FR26 a missing key or an unreachable server downgrades to `CSVLogger` with a warning, and `metrics.csv` is written regardless. Check the job's stderr for the warning if a dashboard does not appear.
