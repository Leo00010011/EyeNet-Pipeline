# Validation — F-FLIP: Canonical-Eye Flip Convention

## Code Correctness

### Group 1 — Identity (Left-Eye Pass-through)

- [ ] `flip_for_canonical_eye(img, vec, "left")` returns an image with pixel values identical to `img` (`np.array_equal(out_img, img) == True`).
- [ ] `flip_for_canonical_eye(img, vec, "left")` returns a vector with values identical to `vec` (`np.array_equal(out_vec, vec) == True`).
- [ ] Input image is not mutated after a left-eye call.
- [ ] Input vector is not mutated after a left-eye call.

### Group 2 — Right-Eye Image Flip

- [ ] `flip_for_canonical_eye(img, vec, "right")` returns `out_img[r, c] == img[r, 127-c]` for all valid `r`, `c` (pixel-exact horizontal flip).
- [ ] Output image shape is `(128, 128, 3)`.
- [ ] Output image dtype is `uint8`.
- [ ] Output image is C-contiguous in memory (`out_img.flags["C_CONTIGUOUS"] == True`).
- [ ] Output image shares no memory with the input (`np.shares_memory(out_img, img) == False`).

### Group 3 — Right-Eye Vector Negation

- [ ] `out_vec[0] == -vec[0]` within atol=1e-7 after a right-eye call.
- [ ] `out_vec[1] == vec[1]` exactly (y-component unchanged).
- [ ] `out_vec[2] == vec[2]` exactly (z-component unchanged).
- [ ] Pure x-direction: `vec=[1,0,0]`, `eye="right"` → `out_vec == [-1, 0, 0]` (`np.testing.assert_array_equal`).
- [ ] Pure y-direction: `vec=[0,1,0]`, `eye="right"` → `out_vec == [0, 1, 0]` (unchanged).
- [ ] Pure z-direction: `vec=[0,0,1]`, `eye="right"` → `out_vec == [0, 0, 1]` (unchanged).

### Group 4 — image=None (Export / Unflip Path)

- [ ] `flip_for_canonical_eye(None, vec, "right")` returns `(None, flipped_vec)` — image component is exactly `None`.
- [ ] `flip_for_canonical_eye(None, vec, "left")` returns `(None, vec_copy)` — identity on the vector.
- [ ] Vector transform for `image=None` is identical to the vector transform when image is provided.

### Group 5 — Flip-then-Flip Roundtrip (Critical)

- [ ] `img2, vec2 = flip(img, vec, "right")` then `img3, vec3 = flip(img2, vec2, "right")` → `np.array_equal(img3, img)` is `True` (pixel-exact).
- [ ] Same roundtrip: `np.array_equal(vec3, vec)` is `True` (value-exact, no float drift).
- [ ] Roundtrip with `image=None`: `_, vec2 = flip(None, vec, "right")`, `_, vec3 = flip(None, vec2, "right")` → `np.array_equal(vec3, vec)`.
- [ ] Roundtrip for left-eye is trivially identity (same checks pass).

### Group 6 — Unit-Norm Preservation

- [ ] `vec = [0.6, 0.8, 0.0]` float32: `abs(|flip(vec, "right")| - 1.0) < 1e-7`.
- [ ] `vec = [-0.577, 0.577, 0.577]` float32 (≈unit-norm): `abs(|flip(vec, "right")| - 1.0) < 1e-7`.
- [ ] `vec = [1.0, 0.0, 0.0]` float32: `|flip(vec, "right")| == 1.0` exactly.
- [ ] All three cases above also hold for float64 input.

### Group 7 — Input Mutation Guard

- [ ] Input `image` array is byte-identical before and after a right-eye call (`np.array_equal(img, img_before_call)`).
- [ ] Input `gaze_vector` array is value-identical before and after a right-eye call.
- [ ] Same for a left-eye call.

### Group 8 — Dtype Preservation

- [ ] `image` dtype `uint8` in → `uint8` out (right-eye).
- [ ] `gaze_vector` dtype `float32` in → `float32` out (right-eye).
- [ ] `gaze_vector` dtype `float64` in → `float64` out (right-eye).

### Group 9 — Error Handling

- [ ] `flip_for_canonical_eye(img, vec, "face")` raises `ValueError`.
- [ ] `flip_for_canonical_eye(img, vec, "LEFT")` raises `ValueError` (case-sensitive).
- [ ] `flip_for_canonical_eye(img, vec, "Right")` raises `ValueError`.
- [ ] `flip_for_canonical_eye(img, vec, "")` raises `ValueError`.

---

## Data Validity

These checks operate on real data from the sample bundle and should be run in a notebook or a manual integration test.

- [ ] **Visual spot-check (right-eye):** For a real `(exp_key, frame, patch="right")` from the sample bundle, produce the normalized eye via `compose_warp` + `normalize_eye`, then call `flip_for_canonical_eye`. Plot side-by-side: original normalized eye vs. flipped output. The flipped eye should appear as a mirror image (horizontally reversed). Expected: visually recognisable as the same eye, but flipped.

- [ ] **Visual spot-check (left-eye):** Same procedure for `patch="left"`. Expected: output is pixel-identical to input (no visual difference).

- [ ] **Gaze vector direction check:** For a real right-eye frame where `g_tobii` indicates gaze pointing slightly to the image-right (positive x in decoded 3D space), the flipped vector should point slightly to the image-left (negative x). Confirm sign matches intuition: the network "sees" a left-eye-like crop with gaze pointing left.

- [ ] **Roundtrip on real data (10 frames, right-eye):** For 10 consecutive valid right-eye frames of one `exp_key`, confirm that flip-then-flip produces exact pixel identity (`max_pixel_diff == 0`) and exact vector identity (`max_vec_diff < 1e-7`).

---

## Data Architecture Integrity

- [ ] **Eye identity sourced from W-patch name only:** Any integration test or notebook that calls `flip_for_canonical_eye` in the real data path must thread the `eye` argument from the key passed to `get_warp_matrix` — never from `get_eye_coords_in_crop`. Assert in code review that no call site passes `eye` derived from `get_eye_coords_in_crop`.

- [ ] **Referential transparency:** Calling `flip_for_canonical_eye` twice with identical inputs produces byte-identical outputs. No global state, caching, or side effects.

- [ ] **No persistence:** `flip_for_canonical_eye` must not write to HDF5, disk, or any cache. Confirm via code inspection and a test that the function has no file I/O.

- [ ] **Correct patch convention propagated to R1:** When the R1 DataLoader is implemented, confirm in its integration test that `patch` (the string `"left"` or `"right"`) passed to `get_warp_matrix` is the same value passed to `flip_for_canonical_eye` — not re-derived from any other source. This is a keying invariant: the image and its label must be flipped under the same convention.
