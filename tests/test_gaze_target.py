import math

import numpy as np
import pytest
import torch

from eyenet.gaze_target import spherical_to_unit, unit_to_spherical


def test_zero_zero():
    g = spherical_to_unit(0.0, 0.0)
    np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-5)
    assert abs(np.linalg.norm(g) - 1.0) <= 1e-5


def test_theta0_phi_halfpi():
    g = spherical_to_unit(0.0, np.pi / 2)
    np.testing.assert_allclose(g, [-1.0, 0.0, 0.0], atol=1e-5)


def test_theta_halfpi_phi0():
    g = spherical_to_unit(np.pi / 2, 0.0)
    np.testing.assert_allclose(g, [0.0, -1.0, 0.0], atol=1e-5)


def test_vectorized():
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi / 2, np.pi / 2, size=50)
    phi = rng.uniform(-np.pi / 2, np.pi / 2, size=50)
    g = spherical_to_unit(theta, phi)
    assert g.shape == (50, 3)
    norms = np.linalg.norm(g, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_dtype_float32():
    assert spherical_to_unit(0.0, 0.0).dtype == np.float32
    assert spherical_to_unit(np.zeros(5), np.zeros(5)).dtype == np.float32


def test_unit_to_spherical_roundtrip_grid():
    theta = np.linspace(-0.6, 0.6, 13)
    phi = np.linspace(-1.2, 1.2, 13)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    tt, pp = tt.ravel(), pp.ravel()
    g = spherical_to_unit(tt, pp)
    out = unit_to_spherical(torch.from_numpy(g))
    np.testing.assert_allclose(out[:, 0].numpy(), tt, atol=1e-5)
    np.testing.assert_allclose(out[:, 1].numpy(), pp, atol=1e-5)


def test_unit_to_spherical_roundtrip_vector():
    rng = np.random.default_rng(0)
    n = 200
    g = rng.normal(size=(n, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    g[:, 2] = -np.abs(g[:, 2])  # g_z < 0, forward-facing half-space
    g = g.astype(np.float32)
    sph = unit_to_spherical(torch.from_numpy(g))
    back = spherical_to_unit(sph[:, 0].numpy(), sph[:, 1].numpy())
    np.testing.assert_allclose(back, g, atol=1e-5)


def test_unit_to_spherical_hand_computed_straight_ahead():
    out = unit_to_spherical(torch.tensor([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(out.numpy(), [0.0, 0.0], atol=1e-6)


def test_unit_to_spherical_hand_computed_horizontal():
    out = unit_to_spherical(torch.tensor([-1.0, 0.0, 0.0]))
    np.testing.assert_allclose(out.numpy(), [0.0, math.pi / 2], atol=1e-6)


def test_unit_to_spherical_hand_computed_vertical():
    out = unit_to_spherical(torch.tensor([0.0, -1.0, 0.0]))
    assert abs(out[0].item() - math.pi / 2) <= 1e-3


def test_unit_to_spherical_shapes():
    batched = unit_to_spherical(torch.zeros(5, 3) + torch.tensor([0.0, 0.0, -1.0]))
    assert batched.shape == (5, 2)
    single = unit_to_spherical(torch.tensor([0.0, 0.0, -1.0]))
    assert single.shape == (2,)


def test_unit_to_spherical_dtype_device():
    g = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    out = unit_to_spherical(g)
    assert out.dtype == torch.float32
    assert out.device == g.device


def test_unit_to_spherical_errors():
    with pytest.raises(ValueError, match=r"\(4, 2\)"):
        unit_to_spherical(torch.zeros(4, 2))
    with pytest.raises(ValueError, match=r"\(2, 3, 3\)"):
        unit_to_spherical(torch.zeros(2, 3, 3))


def test_unit_to_spherical_no_nan_at_pole():
    out = unit_to_spherical(torch.tensor([[0.0, -1.0, 0.0]]))
    assert torch.isfinite(out).all()
