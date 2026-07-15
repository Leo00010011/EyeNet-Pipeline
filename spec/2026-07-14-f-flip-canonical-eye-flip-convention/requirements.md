# Requirements — F-FLIP: Canonical-Eye Flip Convention

## Goal

Implement `flip_for_canonical_eye` in `src/eye_norm.py` to establish the single canonical eye orientation the shared-weight ResNet18 always receives. Right-eye crops (patch name `"right"` from `get_warp_matrix`) and their corresponding ground-truth unit gaze vectors are mirrored to match the left-eye convention before any forward pass. The flip is self-inverse (negating x twice is identity), so the same function covers both the training path (flip image + vector) and the export/inference path (unflip vector only, `image=None`). Correct pairing of image and vector transforms is critical — a sign error here silently trains the model on mismatched inputs and labels.

## Scope

**In scope:**
- `flip_for_canonical_eye(image, gaze_vector, eye)` added to `src/eye_norm.py`.
- Horizontal flip of `(128,128,3)` uint8 image for `eye="right"`.
- Negation of x-component of `(3,)` gaze vector for `eye="right"`.
- Identity pass-through for `eye="left"`.
- `image=None` accepted to support the vector-only export/unflip path.
- Unit tests added to `tests/test_eye_norm.py`.

**Out of scope:**
- Batch/vectorised processing over multiple samples.
- Float image inputs (flip occurs pre-ImageNet-normalization; float images are out of scope to guard against pipeline-order bugs).
- Changes to `compose_warp` or `normalize_eye`.
- Dataset / DataLoader / Lightning DataModule integration (R1).
- Export pipeline integration (R4).
- Any re-derivation of which eye appears on which image side — eye identity comes from the `W`-patch name only.

## Functional Requirements

**FR1.** `flip_for_canonical_eye(image, gaze_vector, eye)` exists in `src/eye_norm.py` and is importable from that module.

**FR2.** `eye` must be one of the strings `"left"` or `"right"` (case-sensitive). Any other value raises `ValueError`.

**FR3.** When `eye == "left"`: returns `(image_copy, vector_copy)` — both pixel values and vector values are identical to the inputs; no mutation of the inputs.

**FR4.** When `eye == "right"` and `image is not None`: the returned image is `image` horizontally flipped — `out_image[r, c] == image[r, W-1-c]` for all `r ∈ [0,128)`, `c ∈ [0,128)`. Output is `(128,128,3)` uint8 and contiguous.

**FR5.** When `eye == "right"` and `image is None`: the image component of the return value is `None`.

**FR6.** When `eye == "right"`: the returned gaze vector has its x-component negated: `out_vector[0] = -gaze_vector[0]`, `out_vector[1] = gaze_vector[1]`, `out_vector[2] = gaze_vector[2]`.

**FR7.** Neither the input `image` array nor the input `gaze_vector` array is mutated in-place; the caller's originals are always unchanged after the call.

**FR8.** Return type is a 2-tuple `(image_out, vector_out)`. `image_out` is `None` when `image` was `None`, otherwise a new `np.ndarray`.

**FR9.** The dtype and shape of the returned image match the input exactly when not `None`: shape `(128,128,3)`, dtype `uint8`.

**FR10.** The dtype of the returned `gaze_vector` matches the input dtype (float32 in → float32 out; float64 in → float64 out).

**FR11 (Roundtrip identity).** Flip-then-flip is the exact identity on both image and vector:
```python
img2, vec2 = flip_for_canonical_eye(img, vec, "right")
img3, vec3 = flip_for_canonical_eye(img2, vec2, "right")
assert np.array_equal(img3, img)   # pixel-exact
assert np.array_equal(vec3, vec)   # value-exact
```

**FR12 (Unit-norm preservation).** If `|gaze_vector| == 1.0`, then `|out_vector| == 1.0` within float32 precision (absolute error ≤ 1e-7).

**FR13 (Canonical sign check).** A pure x-direction vector `[1, 0, 0]` with `eye="right"` yields `[-1, 0, 0]`.

**FR14 (Error path).** `flip_for_canonical_eye(img, vec, "face")` raises `ValueError`. Same for `"LEFT"`, `"Right"`, or any string not in `{"left", "right"}`.

## Public API Summary

```python
# src/eye_norm.py

def flip_for_canonical_eye(
    image: np.ndarray | None,   # (128,128,3) uint8, or None for vector-only export path
    gaze_vector: np.ndarray,    # (3,) float32 or float64 — unit gaze vector in normalized camera space
    eye: str,                   # "left" | "right"  (W-patch name from get_warp_matrix)
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Apply the canonical-eye flip convention.

    right-eye: horizontally flip image (if not None), negate gaze_vector x-component.
    left-eye:  identity — both image and vector returned unchanged.

    image=None is the export/unflip path: only the vector transform is applied.
    Raises ValueError if eye is not "left" or "right".
    """
```

## Dependencies

| Dependency | Direction | What is consumed or produced |
|---|---|---|
| `src/eye_norm.py` (F-NORM, existing) | Extended | `flip_for_canonical_eye` is added to this file alongside `compose_warp` / `normalize_eye` |
| `tests/test_eye_norm.py` (existing) | Extended | New F-FLIP test cases added alongside existing F-NORM tests |
| `EveBundle.get_warp_matrix(exp_key, patch)` | Reads (caller) | The patch key (`"left"` / `"right"`) is the `eye` argument — **sole** source of eye identity |
| `EveBundle.get_normalized_gaze(exp_key, patch)` | Reads (caller) | `g_tobii` spherical → decoded 3D unit vector is the `gaze_vector` argument |
| F-NORM `normalize_eye` output | Upstream | The `(128,128,3)` uint8 image produced by `normalize_eye` is the `image` argument at training time |
