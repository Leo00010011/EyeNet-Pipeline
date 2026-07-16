import numpy as np
import pytest
import torch

from eyenet.preprocessing import _MEAN, _STD, preprocess_eye_crop


def test_zeros():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    out = preprocess_eye_crop(img)
    assert out.shape == (3, 128, 128)
    assert out.dtype == torch.float32
    for c in range(3):
        expected = (0.0 - _MEAN[c]) / _STD[c]
        np.testing.assert_allclose(out[c].numpy(), expected, atol=1e-6)


def test_full_255():
    img = np.full((128, 128, 3), 255, dtype=np.uint8)
    out = preprocess_eye_crop(img)
    for c in range(3):
        expected = (1.0 - _MEAN[c]) / _STD[c]
        np.testing.assert_allclose(out[c].numpy(), expected, atol=1e-6)


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        preprocess_eye_crop(np.zeros((64, 64, 3), dtype=np.uint8))


def test_wrong_dtype_raises():
    with pytest.raises(ValueError):
        preprocess_eye_crop(np.zeros((128, 128, 3), dtype=np.float32))
