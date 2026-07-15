# F-NORM: Eye-Image Data Normalization — Validation

## Code Correctness

### Group 1 — `compose_warp` Unit Tests

- [ ] `test_compose_warp_identity_origin`: With `W = I` and `(x0=0, y0=0)`, `compose_warp` returns a matrix `==` identity (within float64 tolerance 1e-10). Failure: any element differs by > 1e-10.
- [ ] `test_compose_warp_translation_offsets`: With `W = I`, `x0=100`, `y0=200`, mapping crop-pixel `(0, 0, 1)` through `H_crop` yields `(100.0, 200.0)` in homogeneous coordinates. Failure: coordinates differ by > 1e-6.
- [ ] `test_compose_warp_arbitrary_offset`: With `x0=204`, `y0=312` and a non-identity `W` (constructed to map `(x0, y0) → (64, 64)`), `H_crop @ [0, 0, 1]` yields `(64.0, 64.0)`. Failure: off by > 1e-6.
- [ ] `test_compose_warp_output_dtype`: Return dtype is `np.float64`, shape is `(3, 3)`. Failure: wrong dtype or shape.
- [ ] `test_compose_warp_matrix_form`: The result of `compose_warp(W, x0, y0)` equals `W.astype(float64) @ T_inv` exactly. Failure: matrix comparison fails at 1e-10.

### Group 2 — `normalize_eye` Shape and Dtype Tests

- [ ] `test_normalize_eye_output_shape`: `normalize_eye(zeros_crop, I_H, (128,128))` returns shape `(128, 128, 3)`. Failure: any dimension wrong.
- [ ] `test_normalize_eye_output_dtype`: Return dtype is `np.uint8`. Failure: wrong dtype.
- [ ] `test_normalize_eye_uniform_gray`: Uniform gray crop (value 128) + identity H_crop → output mean within 2.0 of 128.0. Confirms no histogram equalization is applied. Failure: mean < 125 or > 131.
- [ ] `test_normalize_eye_rgb_channels_independent`: Crop with only red channel set to 100; output red-channel mean > 50, green and blue means < 5. Failure: channel cross-contamination detected.
- [ ] `test_normalize_eye_custom_out_size`: `normalize_eye(crop, H, out_size=(64, 64))` returns shape `(64, 64, 3)`. Failure: wrong shape.
- [ ] `test_normalize_eye_no_equalizeHist`: Compare output of `normalize_eye` vs. `cv2.equalizeHist` applied after: they must differ on a real crop with non-uniform content. Confirms histogram equalization is absent. Failure: outputs are identical.

### Group 3 — Composition End-to-End Tests (pure, no bundle)

- [ ] `test_warp_lands_at_output_center`: Synthetic W maps crop origin `(0, 0)` to output centre `(64, 64)`; after `compose_warp(W, 0, 0)` and `normalize_eye(crop, H_crop)` (direct-to-128, no center-crop), a bright pixel at crop-pixel `(0, 0)` lands at output `(64, 64)`. Failure: bright region not at expected location (> 10 px off).
- [ ] `test_warp_does_not_mutate_crop`: The input `crop` array is unchanged after `normalize_eye`. Failure: any element of original crop differs from its initial value.
- [ ] `test_compose_warp_associativity`: `compose_warp(W2 @ W1, x0, y0)` vs `W2 @ compose_warp(W1, x0, y0)` — should produce the same matrix (matrix multiplication is associative). Failure: element-wise diff > 1e-8.

### Group 4 — Integration Tests Against Real EveBundle

- [ ] `test_normalize_eye_produces_valid_128x128_patch`: For `gaze_covered_exp_key`, left-eye patch; output shape `(128, 128, 3)`, dtype uint8, mean > 5.0 (not all-black). Failure: wrong shape, dtype, or all-black output.
- [ ] `test_right_eye_patch_shape`: Same as above for "right" patch. Failure: shape mismatch.
- [ ] `test_validity_gate_respected`: Calling `normalize_eye` on frames where `validity[t] == False` does not raise but may produce a degenerate patch (NaN-derived W). This confirms the caller must gate on validity before calling. Failure: unexpected exception raised on invalid frame.
- [ ] `test_crop_origin_consistency`: `get_crop_origin(exp_key)[t]` for two different valid frames of the same exp_key yields two different `(x0, y0)` values (or the same — no assertion on change, but confirms the field is frame-varying, not constant). Verified by reading at least 5 frames and checking the values are non-constant across the experiment. Failure: all frames return identical (0, 0) — indicates migration bug.
- [ ] `test_warp_matrix_shape`: `get_warp_matrix(exp_key, "left")["W"]` shape is `(90, 3, 3)` float32; `"validity"` shape is `(90,)` bool. Failure: wrong shape or dtype.

---

## Data Validity

These checks run as notebook cells or a standalone script against the real migrated bundle (`bundle.h5` built with `include_gaze_vector_data=True`). Each states the expected outcome.

**DV1 — Warp matrix determinant (spot check, 50 experiments):**
For all valid frames across a 50-exp sample, `np.linalg.det(W[t])` > 0 and finite for all three patches. Expected range per EveDataset F9 spec: face ~0.52–0.77, left/right ~1.18–1.79. Failure: any negative determinant → degenerate or reflected warp; any NaN/Inf → migration error.

**DV2 — Crop origin clamping (full set, 2487 experiments):**
For all valid frames, `x0 + 512 <= 1920` and `y0 + 512 <= 1080`. Expected: 100% compliance (per EveDataset F9 production run). Failure: any out-of-bounds origin → face crop window overflows the original frame.

**DV3 — Crop origin + eye corners recovery (30-exp spot check):**
For each experiment in the sample, `get_crop_origin(exp_key)[t] + get_eye_coords_in_crop(exp_key, t)["left"]` should recover the landmark positions in the original frame within 1 px of the raw landmark. Expected: 0 px diff (per EveDataset F9 verification). Failure: any diff > 1 px → crop-origin or eye-corner coordinate is misaligned.

**DV4 — Normalized patch is non-trivial (10-exp visual spot check):**
For 10 experiments, for the first valid left-eye frame, compute the normalized 128×128 patch and verify: mean > 10 (not all-black), mean < 245 (not all-white/overexposed), at least 30% of pixels differ from the mean by > 10 (not uniform). Expected: all 10 experiments pass. Failure: a uniform or blank patch indicates W @ T_inv maps outside the crop bounds.

**DV5 — W validity equals gaze-norm validity (3-exp cross-check):**
For 3 experiments, assert `bundle.get_warp_matrix(exp_key, "left")["validity"]` and `bundle.get_normalized_gaze(exp_key, "left")["validity"]` are element-wise identical (same underlying data). Expected: all True. Failure: any difference → API contract violation; update validity gating logic.

**DV7 — Eye lands at the principal point (regression guard, 40-exp spot check):**
For 40 experiments, warp each patch's paired eye-corner coordinates through `compose_warp(W, x0, y0)` (pairing: `left_W` ↔ `eye_coords['right']`, `right_W` ↔ `eye_coords['left']`) and average to the eye centre. Expected: landing point within **(56–70, 56–66)** output px for every sample (empirically (60,62) for `left_W`, (66,60) for `right_W`, std ~1.4). This is the property that makes direct-to-128 correct; a regression here (e.g. re-introducing a center-crop, or swapping the eye-coords pairing) moves the eye off-patch. Failure: any landing point outside the band, or the two patches converging on the same eye.

**DV6 — Ground-truth pixel comparison (FR8, manual, 1+ experiment):**
If `{camera}_eyes.mp4` (EVE's pre-computed normalized eye video) is accessible: extract frame `t` for a known valid `(exp_key, t, "left")`, compare pixel-wise to `normalize_eye` output. Expected: mean absolute error ≤ 10 per channel. If not accessible: visually confirm the patch shows a frontalized eye (brow at top, iris centered, no perspective skew). Failure: large systematic offset → W composition is wrong; all-black → crop origin is wrong.

---

## Data Architecture Integrity

These checks verify the keying invariants that ensure no positional coupling between this module's outputs and EveDataset's data.

- [ ] **W keyed by exp_key, not position:** `bundle.get_warp_matrix(exp_keys[i], "left")["W"]` accessed in a shuffled order returns the same result as in sorted order. The bundle must not rely on array position for correctness. Failure: results differ when order of access changes.
- [ ] **Crop origin keyed by exp_key, not position:** Same check for `bundle.get_crop_origin(exp_key)`. Failure: results differ → positional coupling present.
- [ ] **No phantom exp_keys in warp data:** Every `exp_key` for which `bundle.has_gaze_norm(exp_key)` is True must also return a valid `get_warp_matrix` without KeyError. Conversely, calling `get_warp_matrix` on an exp_key where `has_gaze_norm` is False must raise KeyError. Failure: either direction violated → coverage check broken.
- [ ] **No phantom exp_keys in crop origin data:** Same check for `has_face_crops` / `get_crop_origin`. Failure: either direction violated.
- [ ] **Coverage parity (2487/2487):** The count of exp_keys for which both `has_gaze_norm` and `has_face_crops` return True equals 2487 (per EveDataset F9 production run). If the bundle under test was built from the migrated production cache, this must hold exactly. Failure: count < 2487 → migration incomplete; count > 2487 → phantom keys present.
