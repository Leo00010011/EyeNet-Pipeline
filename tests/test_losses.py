"""Hand-computed correctness and numerical-safety tests for the angular loss.

Expected values are written by hand, never derived from a second copy of the
formula under test.
"""

import math

import pytest
import torch

from eyenet.losses import EPS, angular_error_degrees, angular_loss

SQRT3_2 = math.sqrt(3.0) / 2.0


def t(rows):
    return torch.tensor(rows, dtype=torch.float32)


# --- Group 1: hand-computed cases -------------------------------------------


def test_identical_vectors_zero_loss():
    loss = angular_loss(t([[0.0, 0.0, -1.0]]), t([[0.0, 0.0, -1.0]]))
    assert loss.item() < 1e-3  # EPS clamp floors this at arccos(1-1e-7) ~ 4.5e-4 rad


def test_orthogonal_loss_is_half_pi():
    loss = angular_loss(t([[1.0, 0.0, 0.0]]), t([[0.0, 1.0, 0.0]]))
    assert loss.item() == pytest.approx(math.pi / 2, abs=1e-5)


def test_orthogonal_degrees_is_90():
    deg = angular_error_degrees(t([[1.0, 0.0, 0.0]]), t([[0.0, 1.0, 0.0]]))
    assert deg.shape == (1,)
    assert deg.item() == pytest.approx(90.0, abs=1e-3)


def test_sixty_degree_case():
    deg = angular_error_degrees(t([[1.0, 0.0, 0.0]]), t([[0.5, SQRT3_2, 0.0]]))
    assert deg.item() == pytest.approx(60.0, abs=1e-3)


def test_opposed_degrees_is_180():
    deg = angular_error_degrees(t([[1.0, 0.0, 0.0]]), t([[-1.0, 0.0, 0.0]]))
    # The EPS clamp costs ~0.028 deg at the endpoint, so atol must exceed that.
    assert deg.item() == pytest.approx(180.0, abs=5e-2)


def test_non_unit_input_is_normalized_internally():
    deg = angular_error_degrees(t([[2.0, 0.0, 0.0]]), t([[0.0, 5.0, 0.0]]))
    assert deg.item() == pytest.approx(90.0, abs=1e-3)


def test_batch_shape_and_values():
    pred = t([[1.0, 0.0, 0.0]] * 4)
    target = t(
        [
            [1.0, 0.0, 0.0],  # 0 deg
            [0.5, SQRT3_2, 0.0],  # 60 deg
            [0.0, 1.0, 0.0],  # 90 deg
            [-1.0, 0.0, 0.0],  # 180 deg
        ]
    )
    deg = angular_error_degrees(pred, target)
    assert deg.shape == (4,)
    assert deg.dtype == torch.float32
    # 0 deg and 180 deg rows sit on the EPS clamp, worth ~0.028 deg each.
    expected = torch.tensor([0.0, 60.0, 90.0, 180.0])
    assert torch.allclose(deg, expected, atol=5e-2)

    loss = angular_loss(pred, target)
    assert loss.item() == pytest.approx(
        torch.deg2rad(expected).mean().item(), abs=1e-4
    )


# --- Group 2: numerical safety (the critical group) -------------------------


def test_no_nan_gradient_at_cos_one():
    """Without the EPS clamp this yields NaN. This test is why EPS exists."""
    pred = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    angular_loss(pred, t([[1.0, 0.0, 0.0]])).backward()
    assert torch.isfinite(pred.grad).all()


def test_no_nan_gradient_at_cos_minus_one():
    pred = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    angular_loss(pred, t([[-1.0, 0.0, 0.0]])).backward()
    assert torch.isfinite(pred.grad).all()


def test_gradient_magnitude_is_bounded_at_cos_one():
    pred = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    angular_loss(pred, t([[1.0, 0.0, 0.0]])).backward()
    assert pred.grad.abs().max().item() < 1e4


def test_finite_over_random_batch_including_near_zero_rows():
    torch.manual_seed(0)
    pred = torch.randn(128, 3)
    target = torch.randn(128, 3)
    pred[0] = 0.0
    target[1] = 0.0
    pred[2] = torch.tensor([1e-12, 0.0, 0.0])

    loss = angular_loss(pred, target)
    deg = angular_error_degrees(pred, target)
    assert torch.isfinite(loss)
    assert torch.isfinite(deg).all()
    assert ((deg >= 0.0) & (deg <= 180.0)).all()


def test_eps_constant_value():
    assert EPS == 1e-7


# --- Group 3: error paths ---------------------------------------------------


def test_unbatched_input_raises_naming_b3():
    with pytest.raises(ValueError, match=r"\(B, 3\)"):
        angular_loss(torch.zeros(3), torch.zeros(3))


def test_last_dim_not_three_raises():
    with pytest.raises(ValueError, match=r"\(B, 3\)"):
        angular_loss(torch.zeros(2, 4), torch.zeros(2, 4))


def test_shape_mismatch_raises_naming_both_shapes():
    with pytest.raises(ValueError) as excinfo:
        angular_loss(torch.zeros(2, 3), torch.zeros(3, 3))
    assert "(2, 3)" in str(excinfo.value) and "(3, 3)" in str(excinfo.value)


def test_error_paths_apply_to_degrees_too():
    with pytest.raises(ValueError, match=r"\(B, 3\)"):
        angular_error_degrees(torch.zeros(3), torch.zeros(3))
