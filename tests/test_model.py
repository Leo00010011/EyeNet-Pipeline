"""GazeResNet18 shape/norm/gradient tests.

All cases use pretrained=False so the suite is offline-deterministic: these
tests are about tensor shapes, norms and gradient flow, where ImageNet weights
add nothing and a weight download adds a failure mode.
"""

import pytest
import torch
import torch.nn as nn

from eyenet.model import GazeResNet18


@pytest.fixture(scope="module")
def model():
    return GazeResNet18(pretrained=False)


def test_forward_shape_and_dtype(model):
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 3)
    assert out.dtype == torch.float32


def test_output_rows_are_unit_norm(model):
    out = model(torch.randn(2, 3, 128, 128))
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_128_input_needs_no_resize(model):
    model(torch.randn(1, 3, 128, 128))  # completes without error
    assert not any(isinstance(m, nn.Upsample) for m in model.modules())
    assert isinstance(model.backbone.fc, nn.Sequential)
    linears = [m for m in model.backbone.fc if isinstance(m, nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(512, 256), (256, 3)]
    assert any(isinstance(m, nn.Dropout) and m.p == 0.5 for m in model.backbone.fc)


def test_head_out_features_is_three(model):
    linears = [m for m in GazeResNet18(pretrained=False).backbone.fc if isinstance(m, nn.Linear)]
    assert linears[-1].out_features == 3


def test_gradients_flow_to_the_head():
    m = GazeResNet18(pretrained=False)
    m(torch.randn(2, 3, 128, 128)).sum().backward()
    last_linear = [layer for layer in m.backbone.fc if isinstance(layer, nn.Linear)][-1]
    grad = last_linear.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_zero_input_is_safe(model):
    out = model(torch.zeros(1, 3, 128, 128))
    assert torch.isfinite(out).all()


# --- F-OPTUNA: independent head dropouts (Group 3) ---


def _dropouts(**kw):
    m = GazeResNet18(pretrained=False, **kw)
    return [layer.p for layer in m.backbone.fc if isinstance(layer, nn.Dropout)]


def test_independent_dropouts_apply_in_head_order():
    assert _dropouts(dropout1=0.1, dropout2=0.3) == [0.1, 0.3]


def test_dropout_shim_preserves_r2_behavior():
    """A caller passing only `dropout` must still get both layers set to it."""
    assert _dropouts(dropout=0.4) == [0.4, 0.4]


def test_explicit_dropout1_wins_and_shim_fills_the_rest():
    assert _dropouts(dropout=0.4, dropout1=0.1) == [0.1, 0.4]


def test_forward_still_unit_norm_under_new_signature():
    m = GazeResNet18(pretrained=False, dropout1=0.1, dropout2=0.3)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 3)
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)
