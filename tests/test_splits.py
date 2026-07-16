import json

import pandas as pd
import pytest

from eyenet.splits import assign_splits, load_split, make_train_val_split, save_split


def _synthetic_samples_df():
    return pd.DataFrame({
        "subject": ["s1", "s2", "s3", "s4"],
        "split": ["train", "val", "test", "train"],
    })


def test_assign_splits():
    result = assign_splits(_synthetic_samples_df())
    assert result == {"s2": "test"}


def test_make_train_val_split_counts_and_determinism():
    subjects = [f"s{i}" for i in range(10)]
    split_a = make_train_val_split(subjects, val_fraction=0.2, seed=0)
    assert list(split_a.values()).count("val") == 2
    assert list(split_a.values()).count("train") == 8

    split_b = make_train_val_split(subjects, val_fraction=0.2, seed=0)
    assert split_a == split_b

    split_c = make_train_val_split(subjects, val_fraction=0.2, seed=1)
    assert split_a != split_c


@pytest.mark.parametrize("val_fraction", [0, 1.5])
def test_invalid_val_fraction_raises(val_fraction):
    with pytest.raises(ValueError):
        make_train_val_split(["s1", "s2"], val_fraction=val_fraction, seed=0)


def test_empty_train_subjects_raises():
    with pytest.raises(ValueError):
        make_train_val_split([], val_fraction=0.2, seed=0)


def test_save_load_round_trip(tmp_path):
    split = {"s1": "train", "s2": "val"}
    path = tmp_path / "split.json"
    save_split(path, split, seed=0, val_fraction=0.2)
    loaded = load_split(path)
    assert loaded == split


def test_load_split_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path / "nonexistent.json")


def test_load_split_missing_assignment_key_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"seed": 0, "val_fraction": 0.2}))
    with pytest.raises(ValueError):
        load_split(path)


def test_load_split_invalid_value_raises(tmp_path):
    path = tmp_path / "bad2.json"
    path.write_text(json.dumps({"seed": 0, "val_fraction": 0.2, "assignment": {"s1": "test"}}))
    with pytest.raises(ValueError):
        load_split(path)
