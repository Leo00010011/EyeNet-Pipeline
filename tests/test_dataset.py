import numpy as np
import pytest
import torch

from eye_norm import compose_warp, normalize_eye
from eyenet.dataset import EyeGazeDataModule, EyeGazeDataset
from eyenet.gaze_target import spherical_to_unit
from eyenet.sampling import build_sample_index
from eyenet.splits import assign_splits, load_split, make_train_val_split


def _full_split_assignment(bundle):
    merged = assign_splits(bundle.samples_df)
    train_subjects = bundle.samples_df.loc[bundle.samples_df["split"] == "train", "subject"].unique().tolist()
    merged.update(make_train_val_split(train_subjects, val_fraction=0.2, seed=0))
    return merged


def test_bogus_split_raises_keyerror(sample_bundle, face_crops_root):
    index = build_sample_index(sample_bundle, sample_bundle.samples_df["exp_key"].unique())
    merged = _full_split_assignment(sample_bundle)
    with pytest.raises(KeyError):
        EyeGazeDataset(sample_bundle, face_crops_root, index, merged, "bogus")


def test_split_lengths_sum_to_total(sample_bundle, face_crops_root):
    index = build_sample_index(sample_bundle, sample_bundle.samples_df["exp_key"].unique())
    merged = _full_split_assignment(sample_bundle)
    total = 0
    for split in ("train", "val", "test"):
        ds = EyeGazeDataset(sample_bundle, face_crops_root, index, merged, split)
        total += len(ds)
    assert total == len(index)


def _find_sample(sample_bundle, face_crops_root, patch):
    index = build_sample_index(sample_bundle, sample_bundle.samples_df["exp_key"].unique())
    merged = _full_split_assignment(sample_bundle)
    subset = index[index["patch"] == patch]
    if len(subset) == 0:
        pytest.skip(f"no {patch}-eye sample in sample bundle")
    return index, merged, subset.iloc[0]


def test_getitem_shapes(sample_bundle, face_crops_root):
    index = build_sample_index(sample_bundle, sample_bundle.samples_df["exp_key"].unique())
    merged = _full_split_assignment(sample_bundle)
    for split in ("train", "val", "test"):
        ds = EyeGazeDataset(sample_bundle, face_crops_root, index, merged, split)
        if len(ds) == 0:
            continue
        image, target, exp_key, frame, patch = ds[0]
        assert image.shape == (3, 128, 128)
        assert image.dtype == torch.float32
        assert target.shape == (3,)
        assert target.dtype == torch.float32
        assert abs(torch.linalg.norm(target).item() - 1.0) <= 1e-4
        return
    pytest.skip("no non-empty split available")


def test_right_eye_flip_wiring(sample_bundle, face_crops_root):
    index, merged, row = _find_sample(sample_bundle, face_crops_root, "right")
    ds = EyeGazeDataset(sample_bundle, face_crops_root, index, merged, "train")
    subject_of = dict(zip(sample_bundle.samples_df["exp_key"], sample_bundle.samples_df["subject"]))
    split = merged.get(subject_of[row["exp_key"]])
    ds = EyeGazeDataset(sample_bundle, face_crops_root, index, merged, split)
    match = ds._index[
        (ds._index["exp_key"] == row["exp_key"])
        & (ds._index["frame"] == row["frame"])
        & (ds._index["patch"] == row["patch"])
    ]
    i = match.index[0]
    image, target, exp_key, frame, patch = ds[i]

    crop = sample_bundle.get_face_crop(row["exp_key"], row["frame"], face_crops_root)
    W = sample_bundle.get_warp_matrix(row["exp_key"], row["patch"])["W"][row["frame"]]
    x0, y0 = sample_bundle.get_crop_origin(row["exp_key"])[row["frame"]]
    H_crop = compose_warp(W, x0, y0)
    unflipped_eye = normalize_eye(crop, H_crop)
    expected_flipped = np.ascontiguousarray(unflipped_eye[:, ::-1])

    gaze = sample_bundle.get_normalized_gaze(row["exp_key"], row["patch"])
    theta, phi = gaze["g_tobii"][row["frame"]]
    unflipped_target = spherical_to_unit(theta, phi)

    from eyenet.preprocessing import preprocess_eye_crop
    np.testing.assert_array_equal(
        preprocess_eye_crop(expected_flipped).numpy(), image.numpy()
    )
    assert target.numpy()[0] == pytest.approx(-unflipped_target[0], abs=1e-6)


def test_left_eye_unchanged_wiring(sample_bundle, face_crops_root):
    index, merged, row = _find_sample(sample_bundle, face_crops_root, "left")
    subject_of = dict(zip(sample_bundle.samples_df["exp_key"], sample_bundle.samples_df["subject"]))
    split = merged.get(subject_of[row["exp_key"]])
    ds = EyeGazeDataset(sample_bundle, face_crops_root, index, merged, split)
    match = ds._index[
        (ds._index["exp_key"] == row["exp_key"])
        & (ds._index["frame"] == row["frame"])
        & (ds._index["patch"] == row["patch"])
    ]
    i = match.index[0]
    image, target, exp_key, frame, patch = ds[i]

    crop = sample_bundle.get_face_crop(row["exp_key"], row["frame"], face_crops_root)
    W = sample_bundle.get_warp_matrix(row["exp_key"], row["patch"])["W"][row["frame"]]
    x0, y0 = sample_bundle.get_crop_origin(row["exp_key"])[row["frame"]]
    H_crop = compose_warp(W, x0, y0)
    unflipped_eye = normalize_eye(crop, H_crop)

    gaze = sample_bundle.get_normalized_gaze(row["exp_key"], row["patch"])
    theta, phi = gaze["g_tobii"][row["frame"]]
    unflipped_target = spherical_to_unit(theta, phi)

    from eyenet.preprocessing import preprocess_eye_crop
    np.testing.assert_array_equal(preprocess_eye_crop(unflipped_eye).numpy(), image.numpy())
    np.testing.assert_allclose(target.numpy(), unflipped_target, atol=1e-6)


def test_datamodule_fresh_split(sample_bundle, face_crops_root):
    dm = EyeGazeDataModule(sample_bundle, face_crops_root, {"seed": 0, "val_fraction": 0.2}, batch_size=4, num_workers=0)
    dm.setup()
    for loader_fn in (dm.train_dataloader, dm.val_dataloader, dm.test_dataloader):
        loader = loader_fn()
        batch = next(iter(loader))
        images, targets = batch[0], batch[1]
        assert images.shape[1:] == (3, 128, 128)
        assert targets.shape[1:] == (3,)


def test_datamodule_reload_split_reproduces_membership(sample_bundle, face_crops_root, tmp_path):
    from eyenet.splits import save_split

    dm1 = EyeGazeDataModule(sample_bundle, face_crops_root, {"seed": 0, "val_fraction": 0.2}, num_workers=0)
    dm1.setup()
    merged = _full_split_assignment(sample_bundle)
    path = tmp_path / "split.json"
    save_split(path, {k: v for k, v in merged.items() if v in ("train", "val")}, seed=0, val_fraction=0.2)

    dm2 = EyeGazeDataModule(sample_bundle, face_crops_root, {"path": str(path)}, num_workers=0)
    dm2.setup()

    def keys(ds):
        return set(zip(ds._index["exp_key"], ds._index["frame"], ds._index["patch"]))

    assert keys(dm1.train_ds) == keys(dm2.train_ds)
    assert keys(dm1.val_ds) == keys(dm2.val_ds)


def test_no_eve_test_subjects_leak(sample_bundle, face_crops_root):
    dm = EyeGazeDataModule(sample_bundle, face_crops_root, {"seed": 0, "val_fraction": 0.2}, num_workers=0)
    dm.setup()
    eve_test_subjects = set(
        sample_bundle.samples_df.loc[sample_bundle.samples_df["split"] == "test", "subject"]
    )
    subject_of = dict(zip(sample_bundle.samples_df["exp_key"], sample_bundle.samples_df["subject"]))
    for ds in (dm.train_ds, dm.val_ds, dm.test_ds):
        subjects = {subject_of[ek] for ek in ds._index["exp_key"]}
        assert subjects.isdisjoint(eve_test_subjects)
