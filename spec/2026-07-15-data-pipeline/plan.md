# Plan — Data Pipeline (R1)

## Context and Design Decisions

**Why the test split is EVE's `val`, not EVE's `test`.** EVE's official `test` subjects are a genuine held-out benchmark set with no usable ground truth — training against them, or even measuring against them, isn't possible for this project. Rather than inventing a separate held-out carve-out, EVE's `val` subjects (which *do* have ground truth) become this project's `test` split, and this project's own train/val distinction is carved out of EVE's `train` subjects instead. This keeps every split subject-grounded in EVE's own partition rather than introducing a second, independent partitioning scheme layered on top — consistent with Mission's "never recompute what EveDataset owns" principle applied one level up: we don't recompute *which subjects are held out for benchmarking*, only how to subdivide the ones EVE already handed us for training.

**Why the train/val split is by subject, seeded, and persisted.** Splitting by frame or `exp_key` within a subject would leak near-identical frames (same subject, same session, adjacent frames) across train and val, inflating apparent generalization. Subject-level partitioning avoids this and mirrors how EVE itself partitions `train`/`val`/`test`. A pure `random.seed()`-at-import approach isn't good enough because "which subjects landed in val" must be reproducible across the R2 training run, the R3 evaluation run, and any later resumed experiment — hence a persisted JSON manifest (`{seed, val_fraction, assignment}`) that can be regenerated fresh or reloaded verbatim.

**Why F-FLIP is a trusted input rather than re-verified here.** `flip_for_canonical_eye` in `src/eyenet/geometry.py` already implements the flip convention and negates the vector's x-component for right-eye samples; Roadmap.md's checkbox for F-FLIP being unchecked reflects the box not yet being ticked, not that the implementation is wrong. This spec does not re-derive or re-test flip correctness — R1 only wires it into the per-item pipeline, keyed off the `patch` argument passed to `get_warp_matrix`/`get_normalized_gaze` (never off `get_eye_coords_in_crop`, whose left/right labels are the opposite convention — TechStack §Left/Right Flip Convention).

**What's cached vs. computed on demand.** The sample index (FR3) and split assignment (FR4/FR5) are computed once per `EyeGazeDataModule.setup()` call and held in memory as a DataFrame/dict — small (subject count × ~90 frames × 2 patches), no need to persist beyond the split manifest itself. Per-item image warp/flip/preprocessing (FR6) is computed on every `__getitem__` call, not cached — this mirrors F-NORM's design (pure functions, no intermediate H5 cache) and keeps the dataset stateless aside from the two small index structures.

**Out of scope, and why.** Augmentation beyond the fixed pipeline, the loss function, and the model are explicitly R2+ per Roadmap.md — adding them here would blur R1's boundary (producing correct, shaped tensors) with R2's boundary (training on them). The export pipeline's HDF5 schema (R4) is unrelated to this feature's read-side concerns.

## Step 1 — `src/eyenet/gaze_target.py`: spherical→unit vector conversion

```python
def spherical_to_unit(theta, phi):
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    g = np.stack([
        -np.cos(theta) * np.sin(phi),
        -np.sin(theta),
        -np.cos(theta) * np.cos(phi),
    ], axis=-1)
    return g.astype(np.float32)
```
No dependencies on other new modules. Unit-testable in isolation against hand-computed examples (e.g. `theta=phi=0` → `(0,0,-1)`).

## Step 2 — `src/eyenet/preprocessing.py`: ImageNet normalization

```python
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_eye_crop(image: np.ndarray) -> torch.Tensor:
    if image.shape != (128, 128, 3) or image.dtype != np.uint8:
        raise ValueError(f"expected (128,128,3) uint8, got {image.shape} {image.dtype}")
    x = image.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())  # (3,128,128)
```
Depends only on numpy/torch — no dependency on Step 1.

## Step 3 — `src/eyenet/sampling.py`: validity-gated sample index

```python
def build_sample_index(bundle, exp_keys) -> pd.DataFrame:
    rows = []
    for exp_key in exp_keys:
        if not (bundle.has_gaze_norm(exp_key) and bundle.has_face_crops(exp_key)):
            continue
        frame_valid = bundle.get_frame_validity(exp_key)  # (90,) bool
        for patch in ("left", "right"):
            gaze = bundle.get_normalized_gaze(exp_key, patch)
            patch_valid = gaze["validity"]  # (90,) bool
            for frame in np.nonzero(frame_valid & patch_valid)[0]:
                rows.append({"exp_key": exp_key, "frame": int(frame), "patch": patch})
    return pd.DataFrame(rows, columns=["exp_key", "frame", "patch"])
```
Depends on `EveBundle` only (F-NORM's existing accessor surface, no new API needed).

## Step 4 — `src/eyenet/splits.py`: split assignment and persistence

```python
def assign_splits(samples_df: pd.DataFrame) -> dict[str, str]:
    # samples_df: EveBundle.samples_df with columns exp_key, subject, set, valid
    result = {}
    for _, row in samples_df[["subject", "set"]].drop_duplicates().iterrows():
        if row["set"] == "val":
            result[row["subject"]] = "test"
        # "train" subjects are resolved by make_train_val_split, not here
        # "test" subjects are omitted entirely
    return result

def make_train_val_split(train_subjects, val_fraction, seed) -> dict[str, str]:
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be in (0, 1)")
    if not train_subjects:
        raise ValueError("train_subjects must be non-empty")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sorted(train_subjects))  # sorted() first for determinism across platforms
    n_val = round(val_fraction * len(shuffled))
    return {**{s: "val" for s in shuffled[:n_val]}, **{s: "train" for s in shuffled[n_val:]}}

def save_split(path, split, seed, val_fraction) -> None:
    with open(path, "w") as f:
        json.dump({"seed": seed, "val_fraction": val_fraction, "assignment": split}, f, indent=2)

def load_split(path) -> dict[str, str]:
    with open(path) as f:  # raises FileNotFoundError naturally
        data = json.load(f)
    if "assignment" not in data or any(v not in ("train", "val") for v in data["assignment"].values()):
        raise ValueError(f"malformed split file: {path}")
    return data["assignment"]
```
The full pipeline combines `assign_splits` (EVE `val`→our `test`) with `make_train_val_split` or `load_split` (EVE `train`→our `train`/`val`) into one merged `dict[subject, "train"|"val"|"test"]` — this merge happens in `EyeGazeDataModule.setup()` (Step 6), not inside `splits.py`, keeping `splits.py` free of any `EveBundle` dependency (pure dict/DataFrame logic, easiest to unit test).

## Step 5 — `src/eyenet/dataset.py`: `EyeGazeDataset`

```python
class EyeGazeDataset(Dataset):
    def __init__(self, bundle, crops_root, sample_index, split_assignment, target_split):
        if target_split not in ("train", "val", "test"):
            raise KeyError(target_split)
        subject_of = dict(zip(bundle.samples_df["exp_key"], bundle.samples_df["subject"]))
        mask = sample_index["exp_key"].map(subject_of).map(split_assignment).eq(target_split)
        self._index = sample_index[mask].reset_index(drop=True)
        self._bundle, self._crops_root = bundle, crops_root

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        exp_key, frame, patch = self._index.iloc[i][["exp_key", "frame", "patch"]]
        crop = self._bundle.get_face_crop(exp_key, frame, self._crops_root)
        W = self._bundle.get_warp_matrix(exp_key, patch)["W"][frame]
        x0, y0 = self._bundle.get_crop_origin(exp_key)[frame]
        H_crop = compose_warp(W, x0, y0)
        eye = normalize_eye(crop, H_crop)                      # F-NORM
        gaze = self._bundle.get_normalized_gaze(exp_key, patch)
        theta, phi = gaze["g_tobii"][frame]
        target = spherical_to_unit(theta, phi)
        eye, target = flip_for_canonical_eye(eye, target, patch)  # F-FLIP
        return preprocess_eye_crop(eye), torch.from_numpy(target), exp_key, int(frame), patch
```
Depends on Steps 1–4 plus the existing `src/eye_norm.py` (F-NORM) and `src/eyenet/geometry.py::flip_for_canonical_eye` (F-FLIP).

## Step 6 — `src/eyenet/dataset.py`: `EyeGazeDataModule`

```python
class EyeGazeDataModule(pl.LightningDataModule):
    def __init__(self, bundle, crops_root, split_source, batch_size=32, num_workers=4):
        super().__init__()
        self.bundle, self.crops_root = bundle, crops_root
        self.split_source = split_source  # {"seed":.., "val_fraction":..} or {"path":..}
        self.batch_size, self.num_workers = batch_size, num_workers

    def setup(self, stage=None):
        merged = assign_splits(self.bundle.samples_df)
        train_subjects = self.bundle.samples_df.loc[
            self.bundle.samples_df["set"] == "train", "subject"
        ].unique().tolist()
        if "path" in self.split_source:
            merged.update(load_split(self.split_source["path"]))
        else:
            merged.update(make_train_val_split(
                train_subjects, self.split_source["val_fraction"], self.split_source["seed"]))
        index = build_sample_index(self.bundle, self.bundle.samples_df["exp_key"].unique())
        self.train_ds = EyeGazeDataset(self.bundle, self.crops_root, index, merged, "train")
        self.val_ds   = EyeGazeDataset(self.bundle, self.crops_root, index, merged, "val")
        self.test_ds  = EyeGazeDataset(self.bundle, self.crops_root, index, merged, "test")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
```
Depends on everything above; this is the integration point R2's training script will import.

## Implementation Order

1. `src/eyenet/gaze_target.py` — `spherical_to_unit` (Step 1)
2. `src/eyenet/preprocessing.py` — `preprocess_eye_crop` (Step 2)
3. `src/eyenet/sampling.py` — `build_sample_index` (Step 3)
4. `src/eyenet/splits.py` — `assign_splits`, `make_train_val_split`, `save_split`, `load_split` (Step 4)
5. `src/eyenet/dataset.py::EyeGazeDataset` (Step 5)
6. `src/eyenet/dataset.py::EyeGazeDataModule` (Step 6)
