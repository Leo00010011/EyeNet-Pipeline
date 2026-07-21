"""Tests for gaze estimation metrics (distinct from loss functions)."""

import math

import pytest
import torch

from eyenet.metrics import angular_variance

SQRT3_2 = math.sqrt(3.0) / 2.0


def t(rows):
    return torch.tensor(rows, dtype=torch.float32)


# --- Group 1: hand-computed cases -------------------------------------------


def test_identical_vectors_zero_variance():
    """All vectors identical -> variance ≈ 0 (up to EPS clamp in arccos)."""
    vectors = t([[1.0, 0.0, 0.0]] * 10)
    var = angular_variance(vectors)
    # Perfect prediction is clamped at arccos(1-1e-7) ≈ 0.026°
    assert var.item() < 0.05


def test_two_perpendicular_vectors_45_degrees():
    """Two perpendicular vectors -> mean points between them -> ~45° variance."""
    vectors = t([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    var = angular_variance(vectors)
    # Mean direction is at 45° to each, so variance should be ~45°
    assert var.item() == pytest.approx(45.0, abs=1.0)


def test_two_opposite_vectors_90_degrees():
    """Two opposite vectors -> mean is ambiguous -> high variance."""
    vectors = t([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    var = angular_variance(vectors)
    # Two opposite vectors: mean can point either way, but after normalization
    # it's uncertain. Each vector is ~90° from the mean direction.
    assert var.item() > 80.0


def test_single_vector_returns_zero():
    """Single vector has no variance by definition."""
    vectors = t([[1.0, 0.0, 0.0]])
    var = angular_variance(vectors)
    assert var.item() == 0.0


def test_three_vectors_at_120_degrees():
    """Three vectors equally spaced at 120° apart -> mean ≈ origin -> high variance."""
    # Vectors at 0°, 120°, 240° in x-y plane
    vectors = t(
        [
            [1.0, 0.0, 0.0],
            [-0.5, SQRT3_2, 0.0],
            [-0.5, -SQRT3_2, 0.0],
        ]
    )
    var = angular_variance(vectors)
    # Three equally-spaced vectors sum to near-zero. After normalization,
    # each is ~60-120° from the mean direction, averaging to ~90°.
    assert var.item() == pytest.approx(90.0, abs=1.0)


def test_narrow_cone_around_z():
    """Multiple vectors tightly clustered around z-axis."""
    torch.manual_seed(42)
    base = t([[0.0, 0.0, 1.0]])
    # Small perturbations (0.1 scale noise)
    noise = torch.randn(10, 3) * 0.1
    vectors = (base + noise)
    vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    var = angular_variance(vectors)
    # Tight cluster -> low variance (0.1 noise -> ~9° spread)
    assert var.item() < 15.0


def test_non_unit_input_is_normalized_internally():
    """Non-unit vectors should be normalized before variance computation."""
    vectors_unit = t([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vectors_scaled = t([[3.0, 0.0, 0.0], [0.0, 7.0, 0.0]])
    var_unit = angular_variance(vectors_unit)
    var_scaled = angular_variance(vectors_scaled)
    assert var_unit.item() == pytest.approx(var_scaled.item(), abs=1e-4)


# --- Group 2: error paths ---------------------------------------------------


def test_wrong_last_dim_raises():
    """Input must be (N, 3), not (N, 4) or (N, 2)."""
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        angular_variance(torch.zeros(10, 4))


def test_unbatched_input_raises():
    """Input must be 2D (N, 3), not 1D (3,)."""
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        angular_variance(torch.zeros(3))


def test_3d_input_raises():
    """Input must be 2D, not 3D."""
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        angular_variance(torch.zeros(5, 3, 3))
