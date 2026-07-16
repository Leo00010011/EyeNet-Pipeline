# Requirements — Data Pipeline (R1)

## Goal

Wrap `EveBundle` in a PyTorch `Dataset`/`DataModule` that yields ready-to-train `(eye_crop, target_unit_vector, exp_key, frame, patch)` samples: for every valid `(exp_key, frame, patch)`, fetch the face crop, produce the Zhang-normalized + canonically-flipped 128×128 eye crop (R0's `eye_norm`/`geometry` modules), convert EveDataset's spherical `(theta, phi)` ground truth to a 3D unit vector (MPIIGaze convention), and apply ImageNet image preprocessing. This is the last data-side step before R2 (model/training loop) can begin — every array shape, dtype, and value range downstream training code will assume is fixed here.

## Scope

In scope:
- `spherical_to_unit(theta, phi)` — isolated, unit-tested MPIIGaze spherical→unit-vector conversion.
- ImageNet preprocessing of the 128×128 uint8 eye crop into a normalized float32 tensor.
- A sample index builder: enumerate every `(exp_key, frame, patch)` passing the strictest validity gate (frame validity AND per-patch validity).
- **Split policy** on top of EveDataset's official per-subject `set` column (`"train"/"val"/"test"`, `SampleTable.VALID_SETS`):
  - EVE's official `test` subjects are held out with no usable ground truth for this project's purposes and are **excluded entirely** from this pipeline.
  - EVE's official `val` subjects become this project's **test** split (untouched, no further subdivision).
  - EVE's official `train` subjects are randomly partitioned, **by subject** (never by frame or exp_key — a subject never spans two of our splits), into this project's **train** and **val** splits.
  - The random train/val partition must be **reproducible and inspectable**: a split assignment is persisted to disk (subject → `"train"`/`"val"`, plus the seed used), and a caller can either generate a new random split or load a previously persisted one by path.
- `EyeGazeDataset` (PyTorch `Dataset`) and `EyeGazeDataModule` (PyTorch Lightning `DataModule`) built on top of the above.

Out of scope (deferred to later roadmap items):
- The model, training loop, and loss function (R2).
- The export pipeline (R4).
- Data augmentation beyond the fixed normalization/flip/ImageNet steps already specified.
- Changing or re-deriving `set`, `validity`, `W`, or any other EveDataset-owned field.
- F-FLIP's own correctness (`flip_for_canonical_eye` in `src/eyenet/geometry.py`) — treated here as a finished, trusted dependency; this spec only wires it into the pipeline, and does not re-verify or redesign it. `compute_eye_crop_window`/`crop_eye` in the same module are legacy bounding-box helpers pre-dating F-NORM and are **not** used by this pipeline.

## Functional Requirements

**FR1 — `spherical_to_unit(theta, phi)`.**
Input: `theta`, `phi` as Python floats or `(N,)` float32/float64 arrays (roll-removed spherical gaze, EveDataset's `g_tobii` convention). Output: unit vector(s) of shape `(3,)` or `(N, 3)`, same float dtype as input (default float32 for scalar input), computed as `g = [-cos(theta)·sin(phi), -sin(theta), -cos(theta)·cos(phi)]`. Must satisfy `‖g‖ = 1 ± 1e-5` for every row. Pure function, no I/O.

**FR2 — ImageNet preprocessing.**
`preprocess_eye_crop(image: np.ndarray) -> torch.Tensor`. Input `(128, 128, 3)` uint8 RGB. Output `(3, 128, 128)` float32 tensor, values scaled to `[0, 1]` then normalized with `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` per channel (channel order preserved as RGB — no BGR conversion). Raises `ValueError` if input shape is not `(128, 128, 3)` or dtype is not `uint8`.

**FR3 — Validity-gated sample index.**
`build_sample_index(bundle, exp_keys) -> list[tuple[str, int, str]]` (or equivalent DataFrame). For each `exp_key` in `exp_keys` and each `patch in ("left", "right")`: include `(exp_key, frame, patch)` iff `bundle.get_frame_validity(exp_key)[frame] is True` AND `bundle.get_normalized_gaze(exp_key, patch)["validity"][frame] is True`. No partial-validity relaxation. An `exp_key` with `has_gaze_norm(exp_key) is False` or `has_face_crops(exp_key) is False` contributes zero samples (skipped, not an error).

**FR4 — Split assignment.**
`assign_splits(samples_df) -> dict[str, str]` — maps every `subject` whose `set == "test"` to nothing (excluded); every `subject` whose `set == "val"` to `"test"`; every `subject` whose `set == "train"` is a candidate for random `"train"`/`"val"` partitioning (FR5). Result covers only `"train"`, `"val"`, `"test"` as final split labels — EVE's raw `set` value is never surfaced downstream of this function.

**FR5 — Reproducible random train/val partition.**
`make_train_val_split(train_subjects: list[str], val_fraction: float, seed: int) -> dict[str, str]` — deterministically shuffles `train_subjects` using `seed` (e.g. `numpy.random.default_rng(seed)`), assigns the first `round(val_fraction * len(train_subjects))` to `"val"`, the rest to `"train"`. Raises `ValueError` if `val_fraction` is not in `(0, 1)` or `train_subjects` is empty.
`save_split(path, split: dict[str, str], seed: int) -> None` — persists to a JSON file: `{"seed": seed, "val_fraction": ..., "assignment": {subject: "train"|"val"}}`.
`load_split(path) -> dict[str, str]` — loads and returns the `assignment` dict; raises `FileNotFoundError` if `path` does not exist, `ValueError` if the file is malformed (missing `"assignment"` key, or a value outside `{"train","val"}`).
Callers choose fresh-random-split vs. load-existing explicitly (no implicit fallback from one to the other).

**FR6 — `EyeGazeDataset`.**
`EyeGazeDataset(bundle, crops_root, sample_index, split_assignment, target_split)` — a `torch.utils.data.Dataset` whose `__len__` is the count of `(exp_key, frame, patch)` entries in `sample_index` whose subject maps to `target_split` (via `split_assignment`, or directly to `"test"` for held-out-val subjects). `__getitem__(i)` returns a tuple:
`(eye_crop: torch.FloatTensor (3,128,128), target: torch.FloatTensor (3,), exp_key: str, frame: int, patch: str)`.
Per-item pipeline: `get_face_crop` → `compose_warp`+`normalize_eye` (F-NORM) → `flip_for_canonical_eye` (F-FLIP, keyed off `patch`, not `get_eye_coords_in_crop`) → `preprocess_eye_crop` (FR2) for the image; `get_normalized_gaze(...)["g_tobii"][frame]` → `spherical_to_unit` (FR1) → flip-consistent x-negation from the same `flip_for_canonical_eye` call for the target. Raises `KeyError` if `target_split` is not one of `"train"`, `"val"`, `"test"`.

**FR7 — `EyeGazeDataModule`.**
Lightning `DataModule` wrapping FR3–FR6: constructor takes the bundle, crops root, a split source (either a `seed`+`val_fraction` to generate fresh via FR5, or a `path` to load via FR5's `load_split`), and standard `batch_size`/`num_workers`. `setup()` builds the three `EyeGazeDataset` instances (`train`, `val`, `test`); `train_dataloader()`/`val_dataloader()`/`test_dataloader()` return standard `DataLoader`s (`shuffle=True` for train only).

## Public API Summary

```python
# src/eyenet/gaze_target.py
def spherical_to_unit(theta: float | np.ndarray, phi: float | np.ndarray) -> np.ndarray: ...

# src/eyenet/preprocessing.py
def preprocess_eye_crop(image: np.ndarray) -> torch.Tensor: ...

# src/eyenet/sampling.py
def build_sample_index(bundle: "EveBundle", exp_keys: list[str]) -> pd.DataFrame: ...

# src/eyenet/splits.py
def assign_splits(samples_df: pd.DataFrame) -> dict[str, str]: ...
def make_train_val_split(train_subjects: list[str], val_fraction: float, seed: int) -> dict[str, str]: ...
def save_split(path: str, split: dict[str, str], seed: int, val_fraction: float) -> None: ...
def load_split(path: str) -> dict[str, str]: ...

# src/eyenet/dataset.py
class EyeGazeDataset(torch.utils.data.Dataset):
    def __init__(self, bundle, crops_root, sample_index, split_assignment, target_split: str): ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, str, int, str]: ...

class EyeGazeDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, bundle, crops_root, split_source: dict, batch_size: int = 32, num_workers: int = 4): ...
    def setup(self, stage: str | None = None) -> None: ...
    def train_dataloader(self) -> torch.utils.data.DataLoader: ...
    def val_dataloader(self) -> torch.utils.data.DataLoader: ...
    def test_dataloader(self) -> torch.utils.data.DataLoader: ...
```

## Dependencies

| Reads from | Writes to |
|---|---|
| `EveBundle.samples_df` (`exp_key`, `subject`, `set`, `valid`) | Split manifest JSON file (path chosen by caller, e.g. `splits/split_<seed>.json`) |
| `EveBundle.get_frame_validity(exp_key)` → `(90,) bool` | — |
| `EveBundle.get_normalized_gaze(exp_key, patch)` → `g_tobii`, `validity` | — |
| `EveBundle.get_face_crop(exp_key, frame, crops_root)` → `(512,512,3)` uint8 | — |
| `EveBundle.get_warp_matrix` / `get_crop_origin` (via `src/eye_norm.py`) | — |
| `src/eyenet/geometry.py::flip_for_canonical_eye` (F-FLIP, trusted as-is) | — |
