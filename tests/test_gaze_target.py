import numpy as np

from eyenet.gaze_target import spherical_to_unit


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
