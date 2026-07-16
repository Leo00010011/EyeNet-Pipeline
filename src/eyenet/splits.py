"""Split assignment and persistence, on top of EveDataset's per-subject `set` column.

Pure dict/DataFrame logic — no EveBundle dependency, easiest to unit test.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def assign_splits(samples_df: pd.DataFrame) -> dict[str, str]:
    """Map subjects whose EVE `split == "val"` to our "test" split.

    Subjects with `split == "test"` are excluded entirely (no usable ground truth).
    Subjects with `split == "train"` are left unassigned here; they are resolved
    by make_train_val_split/load_split.
    """
    result = {}
    for _, row in samples_df[["subject", "split"]].drop_duplicates().iterrows():
        if row["split"] == "val":
            result[row["subject"]] = "test"
    return result


def make_train_val_split(train_subjects: list[str], val_fraction: float, seed: int) -> dict[str, str]:
    """Deterministically shuffle train_subjects and split into our "train"/"val"."""
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be in (0, 1)")
    if not train_subjects:
        raise ValueError("train_subjects must be non-empty")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sorted(train_subjects))
    n_val = round(val_fraction * len(shuffled))
    return {**{s: "val" for s in shuffled[:n_val]}, **{s: "train" for s in shuffled[n_val:]}}


def save_split(path, split: dict[str, str], seed: int, val_fraction: float) -> None:
    """Persist {seed, val_fraction, assignment} to a JSON file at path."""
    with open(path, "w") as f:
        json.dump({"seed": seed, "val_fraction": val_fraction, "assignment": split}, f, indent=2)


def load_split(path) -> dict[str, str]:
    """Load a split manifest's assignment dict.

    Raises FileNotFoundError if path doesn't exist, ValueError if malformed
    (missing "assignment" key, or a value outside {"train","val"}).
    """
    with open(path) as f:
        data = json.load(f)
    if "assignment" not in data or any(v not in ("train", "val") for v in data["assignment"].values()):
        raise ValueError(f"malformed split file: {path}")
    return data["assignment"]
