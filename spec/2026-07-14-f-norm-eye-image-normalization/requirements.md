# F-NORM: Eye-Image Data Normalization — Requirements

## Goal

Produce a 128×128 RGB eye patch by applying the Zhang et al. 2018 perspective warp stored in EveDataset to a face crop, placing the eye image in the exact same normalized camera frame as the ground-truth target `g_tobii`. This warp removes head-pose variation and scales to a canonical eye-to-camera distance; without it, the input image and the training label live in inconsistent coordinate frames, causing a silent but fundamental misalignment. The feature implements two pure geometry functions — `compose_warp` and `normalize_eye` — and validates them against EVE's own pre-computed normalized eye video where available.

## Scope

**In scope:**
- `compose_warp(W, x0, y0)` — compose the stored per-frame `W` (3×3, defined on the original 1920×1080 frame) with the crop-to-frame translation derived from the crop origin `(x0, y0)`, yielding a single homography ready to pass to `cv2.warpPerspective`.
- `normalize_eye(crop, H_crop, out_size)` — apply the composed homography to a face crop, warping straight to the 128×128 output.
- Unit tests for both functions (pure math, no H5/accessor I/O in tests).
- Integration test verifying the patch is non-trivial and correctly shaped for at least one real valid `(exp_key, frame, patch)`.
- Visual ground-truth check in a notebook (FR7).

**Out of scope:**
- The left/right flip convention (`flip_for_canonical_eye`) — that is F-FLIP, a separate feature.
- Any re-derivation of `W`, `R`, or head-pose (`solvePnP`) — strictly forbidden per Mission.md.
- ImageNet mean/std normalization (applied later in the data pipeline, R1).
- Reading HDF5 or calling EveBundle accessors from inside `eye_norm.py` — the module is pure geometry, caller-supplied arrays only.
- Training, data pipeline, or DataModule code.
- Face crop extraction (already delivered by EveDataset F6 via `get_face_crop`).

## Functional Requirements

**FR1 — Coordinate-frame composition.**
`compose_warp(W, x0, y0)` must compose the stored homography `W` (which maps original 1920×1080 frame pixels → normalized-patch pixels) with the inverse crop translation `T_inv`:

```
T_inv = [[1, 0, x0],
         [0, 1, y0],
         [0, 0, 1 ]]

H_crop = W @ T_inv
```

where `(x0, y0)` is the integer top-left corner of the 512×512 face-crop window in the original 1920×1080 frame, as returned by `bundle.get_crop_origin(exp_key)[t]`. The result `H_crop` maps face-crop pixels → normalized-patch pixels and is suitable as the `M` argument to `cv2.warpPerspective`.

**FR2 — Single-step output: warp directly to 128×128.**
`normalize_eye` warps the face crop straight to `out_size` (default `(128, 128)`). Rationale: by the Zhang construction the rotation `R` puts the eye centre on the optical axis, so it projects **exactly onto `cam_norm`'s principal point** — independent of head pose, gaze, or focal length. Measured across 40 experiments, EVE's principal point is **≈ (63, 61)** (stable to ~1.4 px) and its native eye patch is ~128 px, so warping direct-to-128 lands the eye ~3 px from centre. No intermediate canvas, no center-crop, no intrinsics rescale, and EVE's `focal_norm`/`roiSize` are never needed.

(Historical note: an earlier design warped to a 256×256 canvas and center-cropped `[64:192]`, assuming the eye sat at (128,128); it actually sits at ~(63,61), so that crop landed on the cheek. Superseded by direct-to-128.)

**FR3 — `normalize_eye` function.**
`normalize_eye(crop, H_crop, out_size=(128, 128))` must:
- Accept `crop` as a `(512, 512, 3)` uint8 RGB ndarray (the face crop from `get_face_crop`).
- Return `cv2.warpPerspective(crop, H_crop, (out_size[1], out_size[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))` — a `(128, 128, 3)` uint8 RGB ndarray.
- Not call `cv2.equalizeHist` — ImageNet normalization is applied downstream.
- Not convert to grayscale — RGB output is required.

**FR4 — Pure functions, no I/O.**
Both `compose_warp` and `normalize_eye` must be pure functions: no HDF5 reads, no `EveBundle` calls, no global state, no side effects. All inputs are caller-supplied numpy arrays.

**FR5 — Input shapes and dtypes.**
`compose_warp` inputs:
- `W`: `(3, 3)` float32 — single-frame warp matrix (caller indexes `W_batch[t]`).
- `x0`, `y0`: Python int or int32 scalar — crop origin in the 1920×1080 frame.
- Returns: `(3, 3)` float64.

`normalize_eye` inputs:
- `crop`: `(512, 512, 3)` uint8 ndarray.
- `H_crop`: `(3, 3)` float64 or float32 ndarray (result of `compose_warp`).
- `out_size`: `(int, int)` tuple `(H, W)`, default `(128, 128)`.
- Returns: `(out_H, out_W, 3)` uint8.

**FR6 — Validity gating (caller responsibility).**
The module does not check validity. Callers must gate on:
- `bundle.get_frame_validity(exp_key)[t] == True`
- `bundle.get_warp_matrix(exp_key, patch)["validity"][t] == True`

These two are the same underlying boolean array (per EveDataset F9 — no separate W-validity field exists), so either check alone is equivalent. Both are documented here for clarity.

**FR7 — No re-derivation of W.**
The implementation must never call `solvePnP`, compute `R` from head pose, or assemble `W` from `cam_norm`, `S`, `R`, `inv(cam)`. Only the stored `W` from `bundle.get_warp_matrix(exp_key, patch)["W"][t]` may be used. Violation would silently desync the eye patch from its label.

**FR8 — Ground-truth visual check (validation, not runtime).**
For at least one real valid `(exp_key, frame, patch)`, the normalized patch produced by `normalize_eye` must visually show a frontalized eye (brow at top, iris centered, no perspective skew). If EVE's `{camera}_eyes.mp4` is accessible, compare pixel-wise: mean absolute error ≤ 10 per channel. This check runs once in a notebook, not in CI.

## Public API Summary

```python
# src/eye_norm.py

def compose_warp(
    W: np.ndarray,   # (3, 3) float32 — per-frame warp matrix from EveBundle
    x0: int,         # crop window left edge in original 1920×1080 frame
    y0: int,         # crop window top edge in original 1920×1080 frame
) -> np.ndarray:     # (3, 3) float64 — composed homography for warpPerspective
    ...

def normalize_eye(
    crop: np.ndarray,               # (512, 512, 3) uint8 RGB — face crop from get_face_crop()
    H_crop: np.ndarray,             # (3, 3) — result of compose_warp()
    out_size: tuple[int, int] = (128, 128),   # (H, W) output; warp lands here directly
) -> np.ndarray:                    # (128, 128, 3) uint8 RGB — normalized eye patch
    ...
```

## Dependencies

| What | Source | How consumed |
|---|---|---|
| `W[t]` — per-frame 3×3 warp matrix | `bundle.get_warp_matrix(exp_key, patch)["W"][t]` (EveDataset F9) | Passed by caller as `W` argument |
| `(x0, y0)` — crop origin | `bundle.get_crop_origin(exp_key)[t]` (EveDataset F9) | Passed by caller as `x0`, `y0` |
| `face_crop` — 512×512 RGB | `bundle.get_face_crop(exp_key, frame, crops_root)` (EveDataset F6/F8) | Passed by caller as `crop` |
| Frame validity | `bundle.get_frame_validity(exp_key)[t]` (EveDataset F8) | Caller gates before calling |
| Warp/gaze validity | `bundle.get_warp_matrix(exp_key, patch)["validity"][t]` (EveDataset F9) | Caller gates before calling |
| `cv2.warpPerspective` | `opencv-python` | Called inside `normalize_eye` |
