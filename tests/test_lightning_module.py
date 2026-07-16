"""GazeEstimationModule tests against synthetic batches only.

Per FR14 this module must not depend on EveBundle: nothing here imports
evedataset or any conftest bundle fixture, and that is itself part of the
contract being tested.
"""

import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from eyenet.lightning_module import GazeEstimationModule


def synthetic_batch(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    image = torch.randn(n, 3, 128, 128, generator=g)
    target = F.normalize(torch.randn(n, 3, generator=g), p=2, dim=1)
    return image, target


def module(**kw):
    kw.setdefault("pretrained", False)
    return GazeEstimationModule(**kw)


class LossHistory(pl.Callback):
    """Captures epoch-level train/loss without needing a logger backend."""

    def __init__(self):
        self.epochs = []

    def on_train_epoch_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get("train/loss_epoch")
        if value is not None:
            self.epochs.append(float(value))


def test_training_step_returns_finite_differentiable_scalar():
    m = module()
    loss = m.training_step(synthetic_batch(), 0)
    assert loss.ndim == 0
    assert loss.requires_grad
    assert torch.isfinite(loss)


def test_five_tuple_r1_batch_is_accepted_and_metadata_ignored():
    m = module()
    torch.manual_seed(0)
    image, target = synthetic_batch()
    exp_key, frame, patch = ["k1"] * 4, torch.tensor([0, 1, 2, 3]), ["left"] * 4

    m.eval()
    with torch.no_grad():
        two = m.training_step((image, target), 0)
        five = m.training_step((image, target, exp_key, frame, patch), 0)
    assert torch.allclose(two, five)

    # Metadata survives the call untouched -- R4's export key path.
    assert exp_key == ["k1"] * 4
    assert torch.equal(frame, torch.tensor([0, 1, 2, 3]))
    assert patch == ["left"] * 4


def _overfit_run(tmp_path, max_epochs=30):
    image, target = synthetic_batch()
    loader = DataLoader(TensorDataset(image, target), batch_size=4)
    m = module(lr=1e-3)
    history = LossHistory()
    trainer = pl.Trainer(
        overfit_batches=1,
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[history],
        default_root_dir=str(tmp_path),
    )
    trainer.fit(m, loader)
    return m, trainer, history


def test_loss_decreases_on_overfit_batch(tmp_path):
    _, _, history = _overfit_run(tmp_path)
    assert len(history.epochs) >= 2
    assert history.epochs[-1] < 0.5 * history.epochs[0]


def test_no_nans_in_parameters_after_optimizer_steps(tmp_path):
    m, _, _ = _overfit_run(tmp_path)
    assert all(torch.isfinite(p).all() for p in m.parameters())


def test_checkpoint_round_trip(tmp_path):
    m, trainer, _ = _overfit_run(tmp_path, max_epochs=1)
    path = tmp_path / "rt.ckpt"
    trainer.save_checkpoint(path)

    fixed = torch.randn(2, 3, 128, 128, generator=torch.Generator().manual_seed(7))
    m.eval()
    with torch.no_grad():
        before = m(fixed)

    loaded = GazeEstimationModule.load_from_checkpoint(path)
    assert loaded.hparams.lr == m.hparams.lr
    loaded.eval()
    with torch.no_grad():
        after = loaded(fixed)
    assert torch.allclose(before, after, atol=1e-6)


def test_configure_optimizers_is_adam_with_configured_lr():
    opt = module(lr=3e-4).configure_optimizers()
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == 3e-4


def test_save_hyperparameters_records_constructor_args():
    m = module(lr=3e-4, weight_decay=1e-5)
    assert m.hparams.lr == 3e-4
    assert m.hparams.weight_decay == 1e-5
    assert m.hparams.pretrained is False


def _capturing_module(**kw):
    """A module whose self.log calls are captured into a plain dict instead of
    requiring an attached Trainer."""
    m = module(**kw)
    captured = {}

    def fake_log(name, value, *args, **kwargs):
        captured[name] = float(value) if hasattr(value, "item") else value

    m.log = fake_log
    return m, captured


def _rotated_vector(deg, n):
    """n copies of a unit vector `deg` degrees away from [0, 0, -1] (pure x rotation)."""
    rad = math.radians(deg)
    v = torch.tensor([math.sin(rad), 0.0, -math.cos(rad)], dtype=torch.float32)
    return v.unsqueeze(0).repeat(n, 1)


def test_per_eye_hand_computed_means_epoch_level():
    from eyenet.losses import angular_error_degrees

    m, captured = _capturing_module()
    m.on_validation_epoch_start()

    # batch1: 3 left samples, 2 deg error each
    pred1 = _rotated_vector(2.0, 3)
    target1 = torch.tensor([[0.0, 0.0, -1.0]] * 3)
    image1 = torch.zeros(3, 3, 128, 128)
    m.forward = lambda x: pred1
    m.validation_step((image1, target1, ["k"] * 3, torch.arange(3), ["left", "left", "left"]), 0)

    # batch2: 1 left (6 deg) + 3 right (10 deg)
    pred2 = torch.cat([_rotated_vector(6.0, 1), _rotated_vector(10.0, 3)], dim=0)
    target2 = torch.tensor([[0.0, 0.0, -1.0]] * 4)
    image2 = torch.zeros(4, 3, 128, 128)
    m.forward = lambda x: pred2
    m.validation_step(
        (image2, target2, ["k"] * 4, torch.arange(4), ["left", "right", "right", "right"]), 0
    )

    m.on_validation_epoch_end()

    left_errs = torch.cat([
        angular_error_degrees(pred1, target1),
        angular_error_degrees(pred2[:1], target2[:1]),
    ])
    right_errs = angular_error_degrees(pred2[1:], target2[1:])
    expected_left_mean = left_errs.mean().item()
    expected_right_mean = right_errs.mean().item()

    assert abs(captured["val/angular_error_deg_left"] - expected_left_mean) < 1e-3
    assert abs(captured["val/angular_error_deg_right"] - expected_right_mean) < 1e-3

    # FR13 pin: epoch-level mean != naive average of per-batch left-means.
    batch1_left_mean = angular_error_degrees(pred1, target1).mean().item()
    batch2_left_mean = angular_error_degrees(pred2[:1], target2[:1]).mean().item()
    naive_avg = (batch1_left_mean + batch2_left_mean) / 2
    assert abs(expected_left_mean - naive_avg) > 1e-3


def test_per_axis_theta_only_offset():
    m, captured = _capturing_module()
    m.on_validation_epoch_start()

    target = torch.tensor([[0.0, 0.0, -1.0]])
    theta_deg = 3.0
    rad = math.radians(theta_deg)
    pred = torch.tensor([[0.0, -math.sin(rad), -math.cos(rad)]])  # theta offset only
    image = torch.zeros(1, 3, 128, 128)
    m.forward = lambda x: pred
    m.validation_step((image, target), 0)
    m.on_validation_epoch_end()

    assert abs(captured["val/theta_error_deg"] - theta_deg) < 1e-3
    assert abs(captured["val/phi_error_deg"] - 0.0) < 1e-3


def test_per_axis_phi_only_offset():
    m, captured = _capturing_module()
    m.on_validation_epoch_start()

    target = torch.tensor([[0.0, 0.0, -1.0]])
    phi_deg = 3.0
    pred = _rotated_vector(phi_deg, 1)
    image = torch.zeros(1, 3, 128, 128)
    m.forward = lambda x: pred
    m.validation_step((image, target), 0)
    m.on_validation_epoch_end()

    assert abs(captured["val/phi_error_deg"] - phi_deg) < 1e-3
    assert abs(captured["val/theta_error_deg"] - 0.0) < 1e-3


def test_phi_wraparound_not_360():
    from eyenet.gaze_target import spherical_to_unit

    m, captured = _capturing_module()
    m.on_validation_epoch_start()

    phi_a = math.pi - math.radians(0.5)
    phi_b = -math.pi + math.radians(0.5)
    target = torch.from_numpy(spherical_to_unit(0.0, phi_a)).unsqueeze(0)
    pred = torch.from_numpy(spherical_to_unit(0.0, phi_b)).unsqueeze(0)
    image = torch.zeros(1, 3, 128, 128)
    m.forward = lambda x: pred
    m.validation_step((image, target), 0)
    m.on_validation_epoch_end()

    assert abs(captured["val/phi_error_deg"] - 1.0) < 0.1


def test_variance_collapse_vs_spread():
    m, captured = _capturing_module()
    m.on_validation_epoch_start()
    const_pred = torch.tensor([[0.0, 0.0, -1.0]] * 4)
    target = const_pred.clone()
    image = torch.zeros(4, 3, 128, 128)
    m.forward = lambda x: const_pred
    m.validation_step((image, target), 0)
    m.on_validation_epoch_end()
    assert captured["val/pred_var_x"] < 1e-6
    assert captured["val/pred_var_y"] < 1e-6
    assert captured["val/pred_var_z"] < 1e-6

    m2, captured2 = _capturing_module()
    m2.on_validation_epoch_start()
    spread_pred = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0, -1.0], [-1.0, 0.5, -1.0], [0.0, -1.0, -0.5], [0.5, 0.5, -1.0]]),
        p=2,
        dim=-1,
    )
    target2 = torch.tensor([[0.0, 0.0, -1.0]] * 4)
    m2.forward = lambda x: spread_pred
    m2.validation_step((image, target2), 0)
    m2.on_validation_epoch_end()
    assert captured2["val/pred_var_x"] > 1e-3
    assert captured2["val/pred_var_y"] > 1e-3
    assert captured2["val/pred_var_z"] > 1e-3


def test_variance_per_component_not_pooled():
    m, captured = _capturing_module()
    m.on_validation_epoch_start()
    # vary only x; y, z constant
    xs = torch.tensor([0.3, -0.3, 0.5, -0.5])
    pred = torch.stack([xs, torch.zeros(4), -torch.sqrt(1 - xs**2)], dim=1)
    target = torch.tensor([[0.0, 0.0, -1.0]] * 4)
    image = torch.zeros(4, 3, 128, 128)
    m.forward = lambda x: pred
    m.validation_step((image, target), 0)
    m.on_validation_epoch_end()
    assert captured["val/pred_var_x"] > 1e-3
    assert captured["val/pred_var_y"] < 1e-6
    # z = -sqrt(1-x^2) is not perfectly constant as x varies (unit-norm coupling),
    # but its variance stays an order of magnitude below x's by construction.
    assert captured["val/pred_var_z"] < captured["val/pred_var_x"] / 10


def test_single_sample_epoch_logs_no_variance():
    m, captured = _capturing_module()
    m.on_validation_epoch_start()
    pred = torch.tensor([[0.0, 0.0, -1.0]])
    target = pred.clone()
    image = torch.zeros(1, 3, 128, 128)
    m.forward = lambda x: pred
    m.validation_step((image, target), 0)
    m.on_validation_epoch_end()
    assert "val/pred_var_x" not in captured
    assert "val/pred_var_y" not in captured
    assert "val/pred_var_z" not in captured


def test_single_patch_epoch_logs_only_that_patch():
    m, captured = _capturing_module()
    m.on_validation_epoch_start()
    pred = _rotated_vector(2.0, 4)
    target = torch.tensor([[0.0, 0.0, -1.0]] * 4)
    image = torch.zeros(4, 3, 128, 128)
    m.forward = lambda x: pred
    m.validation_step((image, target, ["k"] * 4, torch.arange(4), ["left"] * 4), 0)
    m.on_validation_epoch_end()
    assert "val/angular_error_deg_left" in captured
    assert "val/angular_error_deg_right" not in captured


def test_two_tuple_batches_unaffected_no_per_eye_key():
    m, captured = _capturing_module()
    m.on_validation_epoch_start()
    image, target = synthetic_batch()
    m.validation_step((image, target), 0)
    m.on_validation_epoch_end()
    assert "val/angular_error_deg_left" not in captured
    assert "val/angular_error_deg_right" not in captured


def test_buffers_reset_between_epochs():
    m, captured = _capturing_module()

    m.on_validation_epoch_start()
    pred1 = _rotated_vector(2.0, 4)
    target1 = torch.tensor([[0.0, 0.0, -1.0]] * 4)
    image = torch.zeros(4, 3, 128, 128)
    m.forward = lambda x: pred1
    m.validation_step((image, target1), 0)
    m.on_validation_epoch_end()
    epoch1_deg = captured["val/angular_error_deg"] if "val/angular_error_deg" in captured else None

    m.on_validation_epoch_start()
    pred2 = _rotated_vector(8.0, 4)
    target2 = torch.tensor([[0.0, 0.0, -1.0]] * 4)
    m.forward = lambda x: pred2
    m.validation_step((image, target2), 0)
    m.on_validation_epoch_end()
    epoch2_theta = captured["val/theta_error_deg"]

    # epoch2's theta/phi metrics must come only from epoch2 samples.
    assert abs(epoch2_theta - 0.0) < 1e-3  # pure-phi rotation => theta stays 0
    assert epoch1_deg is None or abs(epoch1_deg - 2.0) < abs(epoch1_deg - 8.0)
