# F-WANDB — Implementation Plan

## Context and Design Decisions

### Why these four metrics and not a generic dashboard

The Roadmap's R2 note is unusually specific about why the obvious metric fails on this data: EVE's labels are strongly biased toward the mean (everyone looks at a screen; spread ≈4.5°), so **a constant mean-gaze predictor scores ≈4.5°** — competitive with published baselines. The pooled angular error therefore cannot, on its own, separate a working model from a degenerate one here. Each of the four required metrics exists to catch a specific failure the pooled scalar hides:

| Metric | Failure it catches | Why the scalar misses it |
|---|---|---|
| Degrees, train+val on one chart | Overfitting | Visible only by comparing two curves |
| Per-component output variance | **Collapse to the mean** | A collapsed model's loss *falls*, and scores well |
| Error by eye (left/right) | **F-FLIP convention break** | Desyncs half the dataset; invisible in the pooled mean |
| Error by axis (theta/phi) | Axis-specific non-learning | R2's run had `phi` at r≈+0.68 and `theta` at r≈0; pooled error showed nothing |

The notebook needed a per-sample correlation plot to see the collapse question *after the fact*. Variance is the cheap version of that signal, available live. This is why the feature is worth doing before R3 rather than after.

### Why the module aggregates instead of using per-batch means

FR13. A batch may contain only one patch — the R1 sample index interleaves `(exp_key, frame, patch)` rows and the DataLoader shuffles, so a batch of 32 can easily be lopsided or single-patch. A per-batch left/right mean, averaged over the epoch, weights each batch equally regardless of how many samples of that patch it held, which turns the left-vs-right comparison into noise precisely when it matters. The module accumulates raw per-sample values in a plain Python list and reduces once per epoch. The same argument applies to variance, which is not even a mean and cannot be averaged over batches at all.

Buffers are plain lists of detached CPU tensors, cleared in `on_*_epoch_start`. This is deliberately not `torchmetrics`: the reductions are a mean and a variance, the module already avoids extra dependencies, and a `torchmetrics.Metric` would add DDP-sync semantics this project does not currently need (single-device runs) and would obscure the `nan`-guard behavior FR11/FR15 require.

Memory: an epoch of the full split is on the order of 10⁵–10⁶ samples; the buffers hold a `(3,)` float and two scalars per sample, i.e. tens of MB at the top end. Acceptable. If R3's full split makes this uncomfortable, the fix is a running-moments accumulator, not a per-batch mean — but do not pre-optimize it.

### Why `unit_to_spherical` is torch while `spherical_to_unit` stays numpy

FR5. They live in the same module and use different backends, which looks inconsistent and is not. `spherical_to_unit` runs once per sample inside the numpy-side data pipeline (`EyeGazeDataset.__getitem__`), where torch would be alien. `unit_to_spherical` runs once per batch on the training device; a numpy implementation would force a `.cpu()` sync per validation batch. Neither belongs in the other's backend, and duplicating either into both is more surface to keep in sync for no gain. The **round-trip test** (`spherical_to_unit(unit_to_spherical(g)) == g`, bridging the two backends) is what holds the pair honest, and it is mandatory — the Roadmap flags this as a prime sign-bug candidate, same class as F-FLIP.

### Why a W&B failure must not raise

FR25/FR26, and the constraint that R3's runs go through a job queue unattended. `scripts/train.py` already has a fail-fast pattern — bad `bundle_dir`/`crops_root` raise `FileNotFoundError` before the Trainer is built — and this feature deliberately does **not** follow it. The distinction is what the failure costs. A bad `bundle_dir` means the run cannot produce its deliverable, so failing in the first second saves the queue slot. A missing `WANDB_API_KEY` means the run produces its deliverable and you cannot watch it live; `metrics.csv` still lands, the checkpoint still lands, and the notebook still works. Killing an hours-long queued job over instrumentation inverts the cost. Warn loudly, continue.

### Why `CSVLogger` stays and stays first

FR23 and the Roadmap's explicit "R2's tests assert on `metrics.csv`". `CSVLogger` is a *testable* logger: `tests/test_train_script.py` reads the file and asserts on the loss column, which W&B fundamentally cannot support without a network and an account. Beyond testing, Lightning derives `trainer.log_dir` from `logger[0]`, so reordering the list would silently move the run artifacts documented in TechStack §Run artifacts. Order is load-bearing, not stylistic.

### Constitution constraints that bind here

- **Mission §Data Quality 1 (code correctness):** the `theta`/`phi` decode is exactly the class of sign/axis-convention bug the Mission calls out. Round-trip and hand-computed tests, not mocks.
- **Roadmap F-WANDB §Requirements:** no change to `losses.py`, `model.py`, `dataset.py`, `sampling.py`, `splits.py`. `angular_error_degrees` is reused as-is.
- **Roadmap F-WANDB:** `patch` reaches the metric code as a **tuple of `str`** (confirmed R2 behavior); do not assume tensor indexing.
- **Roadmap R2:** the `EPS` clamp pattern in `losses.py` is load-bearing and is extended, not reinvented, by FR2's `arcsin` clamp.
- **Mission §3 (no positional coupling):** not engaged. This feature writes no `exp_key`-addressed artifact; it reads `patch` for masking only. That rule binds at R4.

---

## Step 1 — `unit_to_spherical` in `src/eyenet/gaze_target.py`

**File:** `src/eyenet/gaze_target.py` (modify — add only; `spherical_to_unit` untouched)

Add the torch import and the function. Import `EPS` from `eyenet.losses` rather than redefining it — one constant, one definition, matching the `_cos` single-code-path discipline R2 established.

```python
import torch
from eyenet.losses import EPS


def unit_to_spherical(g: torch.Tensor) -> torch.Tensor:
    """Inverse of spherical_to_unit, MPIIGaze convention.

    g: (B, 3) or (3,) unit-norm tensor. NOT normalized here -- callers pass
    either GazeResNet18 output (F.normalize'd) or a spherical_to_unit target.
    Returns (B, 2) or (2,): [theta, phi] in radians, input dtype and device.
    """
    if g.ndim > 2 or g.shape[-1] != 3:
        raise ValueError(f"expected (B, 3) or (3,), got {tuple(g.shape)}")
    # arcsin' diverges at +/-1 and a straight-down gaze reaches it in float32 --
    # same failure mode as the arccos clamp in losses.py, same EPS.
    theta = torch.asin(torch.clamp(-g[..., 1], -1.0 + EPS, 1.0 - EPS))
    phi = torch.atan2(-g[..., 0], -g[..., 2])
    return torch.stack([theta, phi], dim=-1)
```

Note `g[..., 1]` and the `dim=-1` stack handle the `(3,)` and `(B,3)` cases with one code path; the `(3,)` case returns `(2,)` naturally.

**Verify before proceeding:** Step 3 depends on this. Run `tests/test_gaze_target.py` (Step 2) and get it green before touching the module.

---

## Step 2 — Tests for `unit_to_spherical`

**File:** `tests/test_gaze_target.py` (modify — add to the existing file, matching its style)

- **Round-trip, numpy↔torch (the load-bearing one).** For a grid of `(theta, phi)` covering `theta ∈ [-0.6, 0.6]`, `phi ∈ [-1.2, 1.2]`: `spherical_to_unit(theta, phi)` → torch → `unit_to_spherical` → compare to the original `(theta, phi)`, `atol=1e-5`. This is the test that catches a sign flip in either direction.
- **Round-trip, vector→spherical→vector.** Random unit vectors (seeded, `g_z < 0` — the forward-facing half-space the convention covers), through `unit_to_spherical` then `spherical_to_unit`, recovers `g` to `atol=1e-5`.
- **Hand-computed cases.** `g = [0, 0, -1]` → `(0, 0)`. `g = [-1, 0, 0]` → `theta=0, phi=pi/2`. `g = [0, -1, 0]` → `theta=pi/2` (`phi` unconstrained at the pole — assert `theta` only).
- **Shape/dtype/device.** `(B,3)` → `(B,2)`; `(3,)` → `(2,)`; float32 in → float32 out.
- **Errors.** `(B,2)` raises `ValueError`; `(B,3,3)` raises `ValueError`; message contains the shape.
- **No NaN at the pole.** `g = [0, -1, 0]` (exactly, float32) produces finite `theta` — pins the FR2 clamp.

---

## Step 3 — Epoch aggregation in `src/eyenet/lightning_module.py`

**File:** `src/eyenet/lightning_module.py` (modify)

This is the substantive step. `__init__`'s signature does not change (FR-Public API), so nothing that constructs the module — including R2's checkpoint round-trip test — is affected.

**3a. Refactor `_step` to return the per-sample tensors the buffers need.** R2's `_step` returns `(loss, deg_mean)`; it must now also surface `pred` and the per-sample `(B,)` degree tensor. Keep the `batch[0], batch[1]` indexing — R2's tolerance of synthetic 2-tuple batches is relied on by `tests/test_lightning_module.py` and must survive (FR16).

```python
def _step(self, batch):
    image, target = batch[0], batch[1]
    pred = self(image)
    per_sample_deg = angular_error_degrees(pred, target)   # (B,)
    return angular_loss(pred, target), per_sample_deg, pred, target
```
Update `training_step` / `validation_step` / `test_step` call sites. `test_step` keeps logging only `test/angular_error_deg` — this feature adds no test-split metrics.

**3b. Buffer lifecycle.** Add a small helper so train and val share one implementation rather than two near-copies that drift:

```python
def _reset_buffers(self, stage: str) -> None:
    self._buf[stage] = {"pred": [], "deg": [], "patch": [], "theta_err": [], "phi_err": []}
```
Initialize `self._buf = {}` in `__init__` and call `_reset_buffers` from `on_train_epoch_start` / `on_validation_epoch_start`. Resetting at **start**, not end, means a mid-epoch interruption cannot leak stale samples into the next epoch's statistics.

**3c. Accumulate (called from `training_step` and `validation_step`).**

```python
def _accumulate(self, stage, pred, target, per_sample_deg, batch):
    b = self._buf[stage]
    b["pred"].append(pred.detach().cpu())
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
    d_phi = torch.atan2(torch.sin(sp[:, 1] - st[:, 1]), torch.cos(sp[:, 1] - st[:, 1]))
    b["theta_err"].append(torch.rad2deg(d_theta.abs()).cpu())
    b["phi_err"].append(torch.rad2deg(d_phi.abs()).cpu())
```

**3d. Emit (called from `on_train_epoch_end` / `on_validation_epoch_end`).**

```python
def _emit(self, stage: str) -> None:
    b = self._buf.get(stage)
    if not b or not b["deg"]:
        return
    pred = torch.cat(b["pred"])          # (N, 3)
    deg = torch.cat(b["deg"])            # (N,)
    prefix = "train" if stage == "train" else "val"

    if stage == "train":                  # FR6; val/angular_error_deg already logged per-step
        self.log(f"{prefix}/angular_error_deg", deg.mean())

    # FR9/FR11: var(correction=1) on <2 samples is nan -- skip rather than log nan.
    if pred.shape[0] >= 2:
        var = pred.var(dim=0)            # (3,) per-component, NOT pooled
        for i, axis in enumerate("xyz"):
            self.log(f"{prefix}/pred_var_{axis}", var[i])

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
```

The `mask.any()` guard is FR15 and the `if b["patch"]` guard is FR16 — both are what let the existing synthetic-batch tests pass untouched.

**3e. Logging call sites.** `training_step` keeps `self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)` **exactly as-is** (FR8) — changing those flags renames `metrics.csv`'s columns and breaks R2's tests. All new epoch-end metrics are logged from `on_*_epoch_end`, where Lightning treats them as epoch metrics by default; do not pass `on_step`/`on_epoch` there.

---

## Step 4 — Tests for the aggregation

**File:** `tests/test_lightning_module.py` (modify — add; existing 7 tests must pass unchanged)

- **Per-eye aggregation, hand-computed.** Drive `validation_step` over a synthetic 5-tuple epoch with known per-sample errors and a lopsided patch split (e.g. batch 1 = 3 left, batch 2 = 1 left + 3 right). Assert the logged `val/angular_error_deg_left` / `_right` equal the hand-computed **per-sample** means — and specifically that `_left` is *not* the mean of the two batches' left-means, which is what pins FR13's epoch-level (not per-batch) requirement.
- **Per-axis decomposition.** Build a target and a prediction differing in `theta` only (same `phi`): assert `phi_error_deg ≈ 0` (`atol=1e-3`) and `theta_error_deg ≈` the known offset. Then the mirror case. This is the test that catches a swapped `theta`/`phi`.
- **`phi` wraparound.** Two vectors straddling the `atan2` branch cut, ~1° apart in true angle: assert `phi_error_deg ≈ 1`, not ≈359. Pins FR19.
- **Variance detects collapse.** A synthetic epoch of a constant prediction yields `pred_var_* ≈ 0`; a spread of predictions yields clearly non-zero. The point is the contrast, not a threshold.
- **Single-sample epoch logs no variance** (FR11): assert the variance keys are absent from `trainer.logged_metrics`, and that no exception was raised.
- **Single-patch epoch logs only that patch** (FR15): all-left epoch ⇒ `_left` present, `_right` absent.
- **2-tuple batches still work** (FR16): the existing synthetic-batch tests pass with no modification, and no per-eye key is logged.

---

## Step 5 — `build_loggers` in `scripts/train.py`

**File:** `scripts/train.py` (modify)

Extract logger construction into a testable function rather than inlining it in `main` — the enabled/disabled/failure branches are exactly what Step 6 needs to drive without a Trainer.

```python
import os
import warnings

def build_loggers(cfg: dict, out: Path) -> list:
    # FR23: CSVLogger FIRST. Lightning derives trainer.log_dir from logger[0];
    # reordering silently moves every run artifact.
    loggers = [CSVLogger(save_dir=str(out), name="csv")]

    wandb_cfg = (cfg.get("logging") or {}).get("wandb") or {}   # FR22
    if not wandb_cfg.get("enabled", False):
        return loggers                                          # FR21: no wandb import

    # FR25: instrumentation must never cost a queued run. Warn, degrade to CSV.
    if not os.environ.get("WANDB_API_KEY"):
        warnings.warn(
            "logging.wandb.enabled is true but WANDB_API_KEY is unset; "
            "continuing with CSVLogger only. Export WANDB_API_KEY in your job "
            "script (see spec requirements.md, 'Running unattended in a job queue')."
        )
        return loggers

    try:
        from pytorch_lightning.loggers import WandbLogger   # FR21: local import
        loggers.append(WandbLogger(
            project=wandb_cfg.get("project", "eyenet"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name"),
            tags=wandb_cfg.get("tags") or [],
            save_dir=str(out),
        ))
    except Exception as e:                                   # FR26
        warnings.warn(f"W&B logging disabled ({type(e).__name__}: {e}); "
                      "continuing with CSVLogger only.")
    return loggers
```

The broad `except Exception` is deliberate and is the FR26 requirement, not laziness: `wandb` surfaces auth, network, and version failures as several unrelated exception types, and the whole point is that *none* of them reach `trainer.fit`. Note `save_dir=str(out)` puts any offline-mode buffer under `<output.dir>/wandb/`, which is what the §Running unattended step-4 `wandb sync` path expects.

In `main`, replace the inline logger with:
```python
trainer = pl.Trainer(logger=build_loggers(cfg, out), callbacks=[...], **cfg["trainer"])
```
Leave the existing `FileNotFoundError` path checks ahead of it untouched — those stay fail-fast (see §Why a W&B failure must not raise for the distinction).

---

## Step 6 — Tests for `build_loggers`

**File:** `tests/test_train_script.py` (modify — add; existing 6 tests must pass unchanged and offline)

- **Disabled ⇒ CSV only, no wandb import.** `build_loggers({}, tmp_path)` and `{"logging": {"wandb": {"enabled": False}}}` both return a 1-element list of `CSVLogger`. Assert `"wandb" not in sys.modules` afterwards (FR21) — guard for the case where another test already imported it.
- **Missing `logging` block ⇒ disabled** (FR22).
- **Enabled + no `WANDB_API_KEY` ⇒ warns, CSV only** (FR25). `monkeypatch.delenv("WANDB_API_KEY", raising=False)`; `pytest.warns(UserWarning, match="WANDB_API_KEY")`; result length 1; **no exception**.
- **Enabled + constructor raises ⇒ warns, CSV only** (FR26). `monkeypatch.setenv("WANDB_API_KEY", "fake")` and monkeypatch `WandbLogger` to raise; assert length 1 and a warning. No network is touched.
- **CSVLogger is first** (FR23) — assert `isinstance(loggers[0], CSVLogger)` in every case.
- **The existing end-to-end 2-batch run still passes** with `configs/baseline.yaml`, offline, no `WANDB_API_KEY` (FR21). This is the regression gate for the whole feature.

No test constructs a real `WandbLogger` or touches the network. TechStack §Key Libraries already records `wandb` as "never exercised in tests".

---

## Step 7 — Config and dependency

**Files:** `configs/baseline.yaml`, `requirements.txt` (both modify)

`configs/baseline.yaml` — append (FR27; behavior unchanged, this is documentation for the reader who copies this file):
```yaml
logging:
  wandb:
    enabled: false      # R2 baseline stays offline; R3 flips this
    project: eyenet
    entity: null        # null => default entity for WANDB_API_KEY
    run_name: null      # null => W&B generates one
    tags: []
```

`requirements.txt` — add `wandb` (FR29).

---

## Step 8 — Documentation

**Files:** `spec/constitution/Roadmap.md`, `spec/constitution/TechStack.md` (both modify)

- Roadmap: mark F-WANDB done; record the implemented metric names, the round-trip test result, and any spec corrections found during implementation (R2 established this convention and it is what makes the Roadmap trustworthy as a record — do not skip it).
- TechStack §New Modules: add a `New Modules (F-WANDB)` table row for `unit_to_spherical` and the `build_loggers` surface; note `logging:` in the config schema and the `WANDB_API_KEY` contract.
- TechStack §Run artifacts: note that `<output.dir>/wandb/` appears when W&B is enabled, and that `csv/version_*/metrics.csv` is unmoved because `CSVLogger` stays `logger[0]`.

---

## Implementation Order

1. **Step 1** — `unit_to_spherical` in `gaze_target.py`.
2. **Step 2** — its tests, incl. the numpy↔torch round-trip. *Green before proceeding — Step 3 depends on it.*
3. **Step 3** — `lightning_module.py`: `_step` refactor, buffers, `_accumulate`, `_emit`.
4. **Step 4** — aggregation tests, incl. the epoch-vs-per-batch and `phi`-wraparound pins.
5. **Step 5** — `build_loggers` in `train.py`.
6. **Step 6** — logger tests; confirm the existing offline end-to-end run is unbroken.
7. **Step 7** — `configs/baseline.yaml` `logging:` block, `requirements.txt` `wandb`.
8. **Step 8** — Roadmap + TechStack updates.
9. **Validation** — full suite green, then a real enabled run per `validation.md` §Data Validity.
