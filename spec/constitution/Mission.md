# Mission

## Core Objective

Train, analyze, and export a gaze-estimation model that predicts 3D unit gaze direction from single-eye 128×128 crops. The trained model's predictions — a per-frame gaze signal in normalized space — are the intended input to a downstream denoiser model (out of scope for this repo).

## Problem Statement

The sibling project `EveDataset` produces, per experiment frame, a face crop plus the normalized-space ground-truth gaze orientation (Zhang et al. 2018 normalization: spherical `(theta, phi)`, roll-removed, in a per-frame rotated camera frame defined by `R`). This repo is the modeling counterpart: given an eye crop, regress the gaze direction that a denoiser model will later consume as its noisy-but-structured input signal.

This project does not concern itself with:
- *Deriving* the normalization geometry — head-pose estimation, the per-frame rotation `R`, eye origin `o`, and ground-truth gaze — that is `EveDataset`'s responsibility, consumed here only through its installable package interface. This repo *applies* that geometry (reusing `R`/`o`) to warp the eye image into the matching normalized frame; it does not recompute it.
- The denoiser model itself — this repo's job ends at producing and persisting predicted gaze vectors.

## Solution Approach

- **Input (X):** a single-eye crop, 128×128, produced by **Zhang et al. 2018 data normalization** (perspective-warp that aligns the head coordinate system's x-axis and scales to a canonical eye-to-camera distance — see TechStack §Eye-Crop Extraction / Data Normalization), then normalized via the ImageNet mean/std convention (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`, RGB, scaled to `[0,1]` before normalization). The geometric warp is what puts the input image in the **same normalized frame as the target vector** (see below) — a raw bounding-box crop would leave input and target in inconsistent coordinate frames.
- **Target (Y):** the ground-truth gaze direction for that eye, converted from EveDataset's roll-removed spherical `(theta, phi)` representation into a 3D unit vector in normalized camera space, using the MPIIGaze decode convention already used by EveDataset: `g = [-cos(theta)·sin(phi), -sin(theta), -cos(theta)·cos(phi)]`.
- **Model:** ResNet18 (ImageNet-pretrained backbone), regression head producing a 3-vector, normalized to unit length before loss computation.
- **Loss:** computed over unit vectors in normalized space (e.g. cosine/angular loss between predicted and ground-truth unit vectors), not over raw spherical angles.
- **Output:** predicted gaze vectors persisted to disk as a dataset (not just a checkpoint), addressed by the same `exp_key`/frame/patch keying convention EveDataset uses, so the denoiser project can consume them without positional coupling.

## Data Source and Interface

All data — face crops and normalized-space ground truth — is provided by the `evedataset` Python package (built by a sibling session/repo: `EveDataset`, installed as a dependency, not vendored). This repo consumes it exclusively through `EveBundle`'s public, `exp_key`-addressed API (`get_face_crop`, `get_normalized_gaze`, `get_frame_validity`, etc.) — never by reading EveDataset's internal HDF5 caches directly. If the accessor's API doesn't yet expose something this repo needs, that is raised as a request to the EveDataset session, not worked around by reaching into its internals.

`EveBundle` returns whole 512×512 face crops only — it does not deliver per-eye crops, and the face crop it delivers is **not** eye-normalized. **This repo owns eye-crop extraction and eye-image normalization**: producing left/right 128×128 eye crops via the **Zhang et al. 2018 data-normalization warp** (not a plain bounding-box cut), mirroring the geometry-helper pattern EveDataset uses internally (`face_crop_tools.py`). This step needs its own pure, unit-tested geometry module before any dataset/training code depends on it.

**Frame-consistency requirement (critical):** EVE already ran the data-normalization procedure when the dataset was built and **stores its outputs per patch, per frame** — including `{left,right}_W`, the full perspective transform matrix (`left`/`right` = eye patch, not camera — EVE's four cameras are `basler`, `webcam_l`, `webcam_c`, `webcam_r`; each camera's HDF stores its own `left_W`/`right_W`) (`DATASET.md` §HDF file format). So this repo **never re-derives the normalization geometry**: no `solvePnP`, no rebuilding `R`, no assembling `W`. It fetches the stored `W` and applies it. Re-deriving any part of it would produce a subtly different frame from the one the target `g_tobii` was normalized into — a silent desync between input and label.

Two consequences the implementation must respect (detailed in TechStack):
1. `W` is defined to warp the **original camera frame** (it embeds `inv(cam)`). Our eye crops were cut *without* normalization, so they sit in a different pixel frame; `W` must be **composed with the crop→camera transform** before it can be applied to them. Applying `W` to a raw pre-cut crop is geometrically invalid and fails silently.
2. `EveBundle` does not currently expose `W` or `camera_matrix`. Per the Data Source rule above, that is a **request to the EveDataset session**, not a workaround here.

## Data Quality and Correctness Standard

Mirrors `EveDataset`'s standard, applied to the modeling side:

1. **Code correctness** — training loop, loss computation (spherical→unit-vector conversion, angular error), and data loading must have unit/integration tests against real or realistic samples, not synthetic mocks that could hide a sign or axis-convention bug in the gaze decode.
2. **Model validity** — trained model performance (mean angular error, degrees) must be checked against the accepted range for appearance-based gaze estimation on comparable benchmarks (e.g. MPIIGaze/EVE-reported baselines), to catch silent regressions from a normalization or convention mismatch.
3. **Data architecture integrity** — every prediction persisted to disk must be addressable by `exp_key` (+ frame index + patch), never by array position, per `EveDataset`'s Mission.md §3 rationale: positional coupling fails silently across independently regenerated artifacts.

## Success Criteria

- A ResNet18 model trained end-to-end on EveDataset-sourced eye crops, achieving angular error competitive with published appearance-based gaze baselines.
- A reproducible export pipeline that runs inference over a full split and writes a keyed, versioned dataset of predicted gaze vectors, ready for the denoiser project to consume.
- Every non-trivial transform (image normalization, spherical→unit-vector conversion, loss) has a test proving correctness against a hand-computed or reference example.
- No silent positional coupling between this repo's outputs and EveDataset's inputs.

## Resolved Design Decisions

1. **Eye-crop extraction is owned by this repo** (see above) — `evedataset` delivers face crops only.
2. **Left/right eye handling — shared-weight model with a flip convention.** One ResNet18 is trained over both eyes. Right-eye crops (and their target vectors' x-component) are mirrored to the left-eye convention before the forward pass, so the network only ever sees one canonical eye orientation. The flip must be applied consistently to both the image (horizontal flip) and the corresponding ground-truth unit vector (negate the x-component in normalized camera space) — this pairing is a prime candidate for a silent sign-bug, so it gets a dedicated unit test (flip-of-flip = identity; flipped vector stays unit-norm).
3. **Validity policy — strictest available gate.** A frame/patch is included only if `get_frame_validity(exp_key)[frame]` is `True` AND `get_normalized_gaze(exp_key, patch)["validity"][frame]` is `True` for that specific patch (`left`/`right`). No partial-validity relaxation; a sample is either fully valid for its patch or excluded.
4. **Export format — HDF5, `exp_key`-addressed.** The exported prediction dataset stores, per row: `exp_key`, `frame` index, `patch` (`"left"`/`"right"`), predicted unit gaze vector `(3,)`, and validity — enough to reconstruct exactly which sample/frame/eye a prediction corresponds to without any positional assumption. Layout detailed in TechStack.md.
