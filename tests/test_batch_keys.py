"""Validation.md 'Data Architecture Integrity' for R2.

R2 persists no exp_key-addressed dataset -- checkpoints are weights. What it
must guarantee is that R1's (exp_key, frame, patch) key path survives the
training loop intact, so R4's export inherits it working rather than
discovering it broken.
"""

import torch

from eyenet.dataset import EyeGazeDataModule
from eyenet.lightning_module import GazeEstimationModule


def _datamodule(sample_bundle, face_crops_root):
    dm = EyeGazeDataModule(
        sample_bundle,
        face_crops_root,
        {"seed": 0, "val_fraction": 0.2},
        batch_size=4,
        num_workers=0,
    )
    dm.setup()
    return dm


def test_key_metadata_survives_collation(sample_bundle, face_crops_root):
    dm = _datamodule(sample_bundle, face_crops_root)
    image, target, exp_key, frame, patch = next(iter(dm.val_dataloader()))
    batch_size = image.shape[0]

    # default_collate yields str metadata as a tuple (validation.md says "list";
    # the container type is incidental -- what matters is one key per row).
    assert isinstance(exp_key, (list, tuple)) and len(exp_key) == batch_size
    assert all(isinstance(k, str) for k in exp_key)
    assert isinstance(frame, torch.Tensor) and frame.shape == (batch_size,)
    assert frame.dtype in (torch.int32, torch.int64)
    assert isinstance(patch, (list, tuple)) and len(patch) == batch_size
    assert all(p in ("left", "right") for p in patch)


def test_training_step_does_not_alter_key_metadata(sample_bundle, face_crops_root):
    dm = _datamodule(sample_bundle, face_crops_root)
    batch = next(iter(dm.val_dataloader()))
    _, _, exp_key, frame, patch = batch
    before = (list(exp_key), frame.clone(), list(patch))

    module = GazeEstimationModule(pretrained=False)
    module.training_step(batch, 0)

    assert list(exp_key) == before[0]
    assert torch.equal(frame, before[1])
    assert list(patch) == before[2]


def test_batch_row_i_matches_index_row_i(sample_bundle, face_crops_root):
    """Prediction i belongs to key i -- the invariant R4's export depends on."""
    dm = _datamodule(sample_bundle, face_crops_root)
    dataset = dm.val_ds
    image, target, exp_key, frame, patch = next(iter(dm.val_dataloader()))

    for i in range(image.shape[0]):
        row = dataset._index.iloc[i]
        assert exp_key[i] == row["exp_key"]
        assert int(frame[i]) == int(row["frame"])
        assert patch[i] == row["patch"]

        # The image at row i re-derives byte-identically from that triple.
        item_image, item_target = dataset[i][0], dataset[i][1]
        assert torch.equal(item_image, image[i])
        assert torch.equal(item_target, target[i])


def test_limit_batches_does_not_alter_key_to_row_mapping(sample_bundle, face_crops_root):
    """limit_* truncates how many batches are drawn, never which key sits in which row."""
    dm = _datamodule(sample_bundle, face_crops_root)

    def triples(n_batches):
        out = []
        for i, batch in enumerate(dm.val_dataloader()):
            if i >= n_batches:
                break
            _, _, exp_key, frame, patch = batch
            out += list(zip(exp_key, [int(f) for f in frame], patch))
        return out

    limited = triples(2)
    unlimited = triples(4)
    assert len(limited) > 0
    assert unlimited[: len(limited)] == limited
