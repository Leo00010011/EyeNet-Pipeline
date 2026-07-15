# F-NORM: Eye-Image Data Normalization — Implementation Plan

## Context and Design Decisions

### Why not re-derive W
Per Mission.md and TechStack.md §Eye-Crop Extraction, EVE already ran the Zhang et al. 2018 normalization when the dataset was built and stored the output matrices per frame. Re-deriving any part of W (building R from head pose, assembling `cam_norm · S · R · inv(cam)`) would produce a subtly different transform from the one used to compute the stored `g_tobii` target — a silent but fundamental frame desync between input and label. The constraint is absolute: fetch W from `bundle.get_warp_matrix(exp_key, patch)["W"][t]`, apply it, nothing else.

### Why we compose W with T_inv instead of applying it to the face crop directly
W is defined as `cam_norm · S · R · inv(cam)` — it maps pixel coordinates in the **original 1920×1080 camera frame** to the normalized patch. Our input is a 512×512 face crop extracted at offset `(x0, y0)` from that frame; its pixels sit in a shifted coordinate system. Feeding the crop directly to `warpPerspective(crop, W, ...)` would silently produce garbage — `W` would be applied at the wrong pixel origin. The fix is to prepend the inverse crop translation `T_inv = [[1,0,x0],[0,1,y0],[0,0,1]]`, giving `H_crop = W @ T_inv`. This composed homography correctly maps face-crop pixels → original-frame pixels → normalized-patch pixels in one pass.

### Why warp directly to 128×128
The Zhang et al. normalization warp is defined so that the eye lands at the principal point of `cam_norm`. By construction `R` puts the eye centre on the optical axis, so it projects onto `(cx, cy)` regardless of head pose, gaze, or focal length. Measured across 40 experiments, EVE's principal point is **≈ (63, 61)** (stable to ~1.4 px) and its native eye patch is ~128 px — so warping `H_crop` straight to a 128×128 output lands the eye ~3 px from centre, fully captured, with no intermediate canvas, no center-crop, no intrinsics rescale, and no need to know EVE's `focal_norm`/`roiSize`.

> An earlier revision warped to a 256×256 canvas then center-cropped `[64:192, 64:192]`, assuming the eye sat at (128,128). It actually sits at ~(63,61) — the top-left quadrant of that canvas — so the crop landed on the cheek. The direct-to-128 warp replaces it.

### This module is pure geometry — no I/O coupling
`eye_norm.py` receives arrays as arguments, returns arrays, and has no knowledge of EveBundle, HDF5, or file paths. This mirrors EveDataset's `face_crop_tools.py` pattern. Tests therefore need no real bundle fixture — synthetic arrays suffice for unit tests, and real bundle data is used only in integration tests.

### F-NORM and F-FLIP are independent features
The flip convention (horizontal flip for right-eye crops, negate x in target vector) is F-FLIP and is not implemented here. Any caller of `normalize_eye` that needs the flip must apply it afterward. This separation avoids conflating two distinct coordinate operations and makes each independently testable.

---

## Step 1 — Create `src/eye_norm.py`

New file. Pure functions, no imports from `evedataset` or `src/`.

```python
import cv2
import numpy as np


def compose_warp(
    W: np.ndarray,
    x0: int,
    y0: int,
) -> np.ndarray:
    """
    Compose stored W (maps original 1920x1080 frame → normalized patch) with
    T_inv (maps face-crop pixels → original frame pixels).

    H_crop = W @ T_inv,  T_inv = [[1, 0, x0], [0, 1, y0], [0, 0, 1]]

    Returns (3, 3) float64 homography for cv2.warpPerspective.
    """
    T_inv = np.array([[1, 0, x0],
                      [0, 1, y0],
                      [0, 0, 1 ]], dtype=np.float64)
    return W.astype(np.float64) @ T_inv


def normalize_eye(
    crop: np.ndarray,
    H_crop: np.ndarray,
    out_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """
    Warp a 512x512 RGB face crop straight into the 128x128 normalized eye patch.

    The eye lands at EVE's principal point ~(63,61) — near the 128-patch centre —
    by the Zhang construction, so no intermediate canvas or center-crop is needed.

    Returns (out_H, out_W, 3) uint8 RGB.
    """
    out_H, out_W = out_size
    return cv2.warpPerspective(
        crop, H_crop, (out_W, out_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
```

---

## Step 2 — Create `tests/test_eye_norm.py` (unit tests, no real bundle)

New file. All tests use synthetic arrays; no EveBundle fixture needed.

**Test group 1 — `compose_warp` correctness:**

```python
def test_compose_warp_identity_origin():
    """(x0=0, y0=0) → H_crop == W cast to float64"""
    W = np.eye(3, dtype=np.float32)
    H = compose_warp(W, 0, 0)
    assert H == approx(np.eye(3))

def test_compose_warp_translation_offsets():
    """T_inv correctly shifts pixel coords by (x0, y0)"""
    W = np.eye(3, dtype=np.float32)
    H = compose_warp(W, x0=100, y0=200)
    pt_crop = np.array([0, 0, 1], dtype=np.float64)
    pt_out = H @ pt_crop
    assert pt_out[0] / pt_out[2] == approx(100.0)
    assert pt_out[1] / pt_out[2] == approx(200.0)

def test_compose_warp_output_dtype_and_shape():
    H = compose_warp(np.eye(3, dtype=np.float32), 10, 20)
    assert H.dtype == np.float64
    assert H.shape == (3, 3)

def test_compose_warp_equals_W_at_T_inv():
    """Result equals W @ T_inv exactly"""
    W = np.random.rand(3, 3).astype(np.float32)
    x0, y0 = 137, 254
    H = compose_warp(W, x0, y0)
    T_inv = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64)
    assert H == approx(W.astype(np.float64) @ T_inv, abs=1e-10)
```

**Test group 2 — `normalize_eye` shape, dtype, and content:**

```python
def test_normalize_eye_output_shape():
    crop = np.zeros((512, 512, 3), dtype=np.uint8)
    H = compose_warp(np.eye(3, dtype=np.float32), 0, 0)
    out = normalize_eye(crop, H)
    assert out.shape == (128, 128, 3)
    assert out.dtype == np.uint8

def test_normalize_eye_custom_out_size():
    crop = np.zeros((512, 512, 3), dtype=np.uint8)
    H = compose_warp(np.eye(3, dtype=np.float32), 0, 0)
    out = normalize_eye(crop, H, out_size=(64, 64))
    assert out.shape == (64, 64, 3)

def test_normalize_eye_uniform_gray_no_equalization():
    """Uniform gray crop → uniform gray output (confirms no equalizeHist)"""
    crop = np.full((512, 512, 3), 128, dtype=np.uint8)
    H = compose_warp(np.eye(3, dtype=np.float32), 0, 0)
    out = normalize_eye(crop, H)
    assert out.mean() == approx(128.0, abs=2.0)

def test_normalize_eye_rgb_channels_independent():
    crop = np.zeros((512, 512, 3), dtype=np.uint8)
    crop[:, :, 0] = 200  # red channel only
    H = compose_warp(np.eye(3, dtype=np.float32), 0, 0)
    out = normalize_eye(crop, H)
    assert out[:, :, 0].mean() > 100
    assert out[:, :, 1].mean() < 5
    assert out[:, :, 2].mean() < 5

def test_normalize_eye_does_not_mutate_crop():
    crop = np.full((512, 512, 3), 77, dtype=np.uint8)
    original = crop.copy()
    H = compose_warp(np.eye(3, dtype=np.float32), 0, 0)
    normalize_eye(crop, H)
    assert np.array_equal(crop, original)
```

**Test group 3 — warp landing:**

```python
def test_warp_lands_at_output_center():
    """
    normalize_eye warps directly to out_size, so a W that maps the crop origin
    to output (64,64) must place that bright pixel at the 128x128 patch centre.
    """
    # W translates crop origin (0,0) → output centre (64, 64)
    W = np.array([[1, 0, 64],
                  [0, 1, 64],
                  [0, 0, 1 ]], dtype=np.float32)
    crop = np.zeros((512, 512, 3), dtype=np.uint8)
    crop[0, 0] = (255, 255, 255)  # bright pixel at crop origin

    H = compose_warp(W, x0=0, y0=0)
    out = normalize_eye(crop, H)

    assert out[64, 64, 0] > 200, "bright pixel not at expected center position"
```

---

## Step 3 — Integration test against real EveBundle data (`tests/test_eye_norm_integration.py`)

New file. Uses the `sample_bundle` and `face_crops_root` fixtures from `tests/conftest.py`.

```python
def test_normalize_eye_produces_valid_patch(sample_bundle, face_crops_root, gaze_covered_exp_key):
    exp_key = gaze_covered_exp_key
    warp = sample_bundle.get_warp_matrix(exp_key, "left")
    combined = warp["validity"] & sample_bundle.get_frame_validity(exp_key)
    assert combined.any(), "no valid frame in test exp_key"

    t = int(np.argmax(combined))
    W_t = warp["W"][t]
    x0, y0 = sample_bundle.get_crop_origin(exp_key)[t]

    crop = sample_bundle.get_face_crop(exp_key, t, face_crops_root)
    H_crop = compose_warp(W_t, int(x0), int(y0))
    patch = normalize_eye(crop, H_crop)

    assert patch.shape == (128, 128, 3)
    assert patch.dtype == np.uint8
    assert patch.mean() > 5.0, "patch is all-black — warp maps outside crop bounds"

def test_right_eye_patch(sample_bundle, face_crops_root, gaze_covered_exp_key):
    exp_key = gaze_covered_exp_key
    warp = sample_bundle.get_warp_matrix(exp_key, "right")
    t = int(np.argmax(warp["validity"] & sample_bundle.get_frame_validity(exp_key)))
    W_t = warp["W"][t]
    x0, y0 = sample_bundle.get_crop_origin(exp_key)[t]
    crop = sample_bundle.get_face_crop(exp_key, t, face_crops_root)
    patch = normalize_eye(crop, compose_warp(W_t, int(x0), int(y0)))
    assert patch.shape == (128, 128, 3)
    assert patch.mean() > 5.0
```

---

## Step 4 — Visual ground-truth check (notebook, not pytest)

Add cells to `notebooks/inspect_eye_norm.ipynb`:

1. Pick the first valid `(exp_key, t, "left")` from `gaze_covered_exp_key`.
2. Produce the normalized patch with `compose_warp` + `normalize_eye`.
3. Display: raw face crop | 256 context view (annotated with the 128 output window + eye-centre landing point) | final 128×128 patch side-by-side.
4. **Check:** eye should be visible and centred in the 128×128 output (brow at top, iris centred). The printed eye-centre landing point should be near (64, 64).
5. If `{camera}_eyes.mp4` is accessible: extract the matching frame and display alongside for pixel comparison (tolerance ≤ 10 MAE per channel).

This is FR8's ground-truth consistency check; it runs once manually, not in CI.

---

## Implementation Order

1. **Create `src/eye_norm.py`** (Step 1) — `compose_warp`, `normalize_eye`. No prerequisites.
2. **Create `tests/test_eye_norm.py`** (Step 2) — unit tests, run immediately after Step 1.
3. **Create `tests/test_eye_norm_integration.py`** (Step 3) — real bundle, depends on Step 1.
4. **Visual notebook check** (Step 4) — run manually after Step 3 passes; confirm the eye is centred (landing point near (64,64)).
