"""PyTorch Dataset/DataModule wrapping EveBundle for gaze-estimation training.

Per-item pipeline: get_face_crop -> compose_warp+normalize_eye (F-NORM) ->
flip_for_canonical_eye (F-FLIP) -> preprocess_eye_crop, for the image; and
get_normalized_gaze -> spherical_to_unit -> flip-consistent x-negation, for
the target. Sample index and split assignment are computed once in
setup() and held in memory; per-item work is not cached.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from eye_norm import compose_warp, normalize_eye
from eyenet.gaze_target import spherical_to_unit
from eyenet.geometry import flip_for_canonical_eye
from eyenet.preprocessing import preprocess_eye_crop
from eyenet.sampling import build_sample_index
from eyenet.splits import assign_splits, load_split, make_train_val_split


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
        frame = int(frame)
        crop = self._bundle.get_face_crop(exp_key, frame, self._crops_root)
        W = self._bundle.get_warp_matrix(exp_key, patch)["W"][frame]
        x0, y0 = self._bundle.get_crop_origin(exp_key)[frame]
        H_crop = compose_warp(W, x0, y0)
        eye = normalize_eye(crop, H_crop)
        gaze = self._bundle.get_normalized_gaze(exp_key, patch)
        theta, phi = gaze["g_tobii"][frame]
        target = spherical_to_unit(theta, phi)
        eye, target = flip_for_canonical_eye(eye, target, patch)
        return preprocess_eye_crop(eye), torch.from_numpy(target), exp_key, frame, patch


class EyeGazeDataModule(pl.LightningDataModule):
    def __init__(self, bundle, crops_root, split_source, batch_size=32, num_workers=4):
        super().__init__()
        self.bundle, self.crops_root = bundle, crops_root
        self.split_source = split_source
        self.batch_size, self.num_workers = batch_size, num_workers

    def setup(self, stage=None):
        merged = assign_splits(self.bundle.samples_df)
        train_subjects = self.bundle.samples_df.loc[
            self.bundle.samples_df["split"] == "train", "subject"
        ].unique().tolist()
        if "path" in self.split_source:
            merged.update(load_split(self.split_source["path"]))
        else:
            merged.update(make_train_val_split(
                train_subjects, self.split_source["val_fraction"], self.split_source["seed"]))
        index = build_sample_index(self.bundle, self.bundle.samples_df["exp_key"].unique())
        self.train_ds = EyeGazeDataset(self.bundle, self.crops_root, index, merged, "train")
        self.val_ds = EyeGazeDataset(self.bundle, self.crops_root, index, merged, "val")
        self.test_ds = EyeGazeDataset(self.bundle, self.crops_root, index, merged, "test")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
