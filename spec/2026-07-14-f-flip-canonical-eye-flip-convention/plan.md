# Plan — F-FLIP: Canonical-Eye Flip Convention

## Context and Design Decisions

**Self-inverse operation, one function.** Negating x is its own inverse, and `np.fliplr` is its own inverse. A single function therefore covers training (flip both), export (flip vector only via `image=None`), and the roundtrip identity test. No separate `unflip_*` function needed.

**`eye` parameter keys off the W-patch name, never eye-coords.** The constitution is explicit: `get_eye_coords_in_crop`'s `left`/`right` labels are the *opposite* convention from the W-patch name. The `eye` parameter receives the patch string that was passed to `get_warp_matrix` — the caller is responsible for threading it correctly. This function does not call any accessor.

**`image=None` for export path.** At inference/export time, no image is available (we have a model prediction, not a crop). Accepting `None` and skipping the image flip makes the same function work in both contexts without branching in the caller. The return type is consistently `(image_out | None, vector_out)`.

**Pure function, no mutation.** Matches `compose_warp`/`normalize_eye` philosophy. Both branches copy the vector so the caller's array is never modified. `np.fliplr` returns a view, so `np.ascontiguousarray` is used to materialise a fresh buffer.

**Strict uint8 for image.** The flip occurs before ImageNet normalization in the training pipeline. Accepting float images would mask a wrong pipeline order. Dtype enforcement is done implicitly (the function does nothing dtype-specific, but the contract is documented and tests verify the returned dtype matches the input).

**Constitution constraint respected:** this function is the *only* place in the codebase where eye identity determines a transform. All downstream code (Dataset, DataLoader, export loop) must pass the patch name from `get_warp_matrix` directly to `eye` — never derive it from pixel coordinates.

---

## Step 1 — Add `flip_for_canonical_eye` to `src/eye_norm.py`

**File:** `src/eye_norm.py`

Append after the existing `normalize_eye` function. No imports needed beyond what F-NORM already brings (`numpy`).

```python
def flip_for_canonical_eye(
    image: np.ndarray | None,
    gaze_vector: np.ndarray,
    eye: str,
) -> tuple[np.ndarray | None, np.ndarray]:
    if eye not in ("left", "right"):
        raise ValueError(f"eye must be 'left' or 'right', got {eye!r}")

    if eye == "left":
        return image, gaze_vector.copy()

    # right-eye: horizontally flip image (if present), negate x-component of vector
    flipped_image = (
        np.ascontiguousarray(np.fliplr(image)) if image is not None else None
    )
    flipped_vector = gaze_vector.copy()
    flipped_vector[0] = -flipped_vector[0]
    return flipped_image, flipped_vector
```

Key points:
- `np.fliplr` on a `(H,W,C)` array flips along axis=1 (horizontal). Returns a view; `np.ascontiguousarray` makes it a fresh owned buffer with no shared memory with the input.
- `gaze_vector.copy()` is called in both branches — FR7 (no in-place mutation) holds unconditionally.
- `flipped_vector[0] = -flipped_vector[0]` preserves y and z exactly.

---

## Step 2 — Add unit tests to `tests/test_eye_norm.py`

**File:** `tests/test_eye_norm.py`

Append the following test functions after the existing F-NORM tests. No new fixtures needed — tests use synthetic numpy arrays.

```python
# ── F-FLIP tests ──────────────────────────────────────────────────────────────

import numpy as np
from src.eye_norm import flip_for_canonical_eye

def _make_image():
    """128×128×3 uint8 with unique pixel pattern for roundtrip checks."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)

def _unit_vec(x, y, z, dtype=np.float32):
    v = np.array([x, y, z], dtype=dtype)
    return v / np.linalg.norm(v)


def test_flip_left_is_identity():
    img = _make_image()
    vec = _unit_vec(0.6, 0.8, 0.0)
    out_img, out_vec = flip_for_canonical_eye(img, vec, "left")
    assert np.array_equal(out_img, img)
    assert np.array_equal(out_vec, vec)

def test_flip_right_image_horizontal():
    img = _make_image()
    vec = _unit_vec(0.6, 0.8, 0.0)
    out_img, _ = flip_for_canonical_eye(img, vec, "right")
    assert out_img.shape == (128, 128, 3)
    assert out_img.dtype == np.uint8
    assert np.array_equal(out_img, img[:, ::-1, :])

def test_flip_right_vector_negates_x():
    img = _make_image()
    vec = _unit_vec(0.6, 0.8, 0.0)
    _, out_vec = flip_for_canonical_eye(img, vec, "right")
    assert np.isclose(out_vec[0], -vec[0], atol=1e-7)
    assert out_vec[1] == vec[1]
    assert out_vec[2] == vec[2]

def test_flip_right_none_image():
    vec = _unit_vec(0.6, 0.8, 0.0)
    out_img, out_vec = flip_for_canonical_eye(None, vec, "right")
    assert out_img is None
    assert np.isclose(out_vec[0], -vec[0], atol=1e-7)

def test_flip_invalid_eye_raises():
    img = _make_image()
    vec = _unit_vec(0.6, 0.8, 0.0)
    with pytest.raises(ValueError):
        flip_for_canonical_eye(img, vec, "face")
    with pytest.raises(ValueError):
        flip_for_canonical_eye(img, vec, "LEFT")
    with pytest.raises(ValueError):
        flip_for_canonical_eye(img, vec, "Right")

def test_flip_then_flip_is_identity():
    img = _make_image()
    vec = _unit_vec(0.6, 0.8, 0.0)
    img2, vec2 = flip_for_canonical_eye(img, vec, "right")
    img3, vec3 = flip_for_canonical_eye(img2, vec2, "right")
    assert np.array_equal(img3, img), "pixel roundtrip failed"
    assert np.array_equal(vec3, vec), "vector roundtrip failed"

def test_flip_then_flip_none_image_is_identity():
    vec = _unit_vec(0.6, 0.8, 0.0)
    _, vec2 = flip_for_canonical_eye(None, vec, "right")
    _, vec3 = flip_for_canonical_eye(None, vec2, "right")
    assert np.array_equal(vec3, vec)

def test_flip_preserves_unit_norm():
    for x, y, z in [(0.6, 0.8, 0.0), (-0.577, 0.577, 0.577), (1.0, 0.0, 0.0)]:
        vec = _unit_vec(x, y, z)
        _, out_vec = flip_for_canonical_eye(None, vec, "right")
        assert abs(np.linalg.norm(out_vec) - 1.0) < 1e-7, f"unit-norm violated for {(x,y,z)}"

def test_flip_pure_x_vector():
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    _, out_vec = flip_for_canonical_eye(None, vec, "right")
    np.testing.assert_array_equal(out_vec, np.array([-1.0, 0.0, 0.0], dtype=np.float32))

def test_flip_does_not_mutate_input():
    img = _make_image()
    img_orig = img.copy()
    vec = _unit_vec(0.6, 0.8, 0.0)
    vec_orig = vec.copy()
    flip_for_canonical_eye(img, vec, "right")
    assert np.array_equal(img, img_orig), "input image was mutated"
    assert np.array_equal(vec, vec_orig), "input vector was mutated"

def test_flip_preserves_dtype_float32():
    vec = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    _, out_vec = flip_for_canonical_eye(None, vec, "right")
    assert out_vec.dtype == np.float32

def test_flip_preserves_dtype_float64():
    vec = np.array([0.6, 0.8, 0.0], dtype=np.float64)
    _, out_vec = flip_for_canonical_eye(None, vec, "right")
    assert out_vec.dtype == np.float64

def test_flip_image_is_contiguous():
    img = _make_image()
    out_img, _ = flip_for_canonical_eye(img, np.zeros(3, dtype=np.float32), "right")
    assert out_img.flags["C_CONTIGUOUS"]
```

---

## Implementation Order

1. **Step 1** — `flip_for_canonical_eye` in `src/eye_norm.py`
2. **Step 2** — unit tests in `tests/test_eye_norm.py`

No other files need to change for this feature. R1 (DataLoader) and R4 (export pipeline) will import and use `flip_for_canonical_eye` when they are implemented.
