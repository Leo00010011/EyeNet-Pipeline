"""suggest_params / sampler / pruner tests -- no study, no bundle, no network.

FixedTrial lets the suggest dispatch be checked without running an optimization,
mirroring the R2 philosophy of testing each seam in isolation.
"""

import optuna
import pytest

from eyenet.hpo import build_pruner, build_sampler, suggest_params

SEARCH_SPACE = {
    "dropout1": {"type": "float", "low": 0.0, "high": 0.7},
    "weight_decay": {"type": "float", "low": 1.0e-6, "high": 1.0e-2, "log": True},
    "hidden_dim": {"type": "categorical", "choices": [128, 256]},
    "loss": {"type": "categorical", "choices": ["angular", "cosine"]},
}


def test_suggest_params_dispatches_each_type():
    fixed = {"dropout1": 0.25, "weight_decay": 1.0e-4, "hidden_dim": 128, "loss": "cosine"}
    out = suggest_params(optuna.trial.FixedTrial(fixed), SEARCH_SPACE)

    assert set(out) == set(SEARCH_SPACE)
    assert isinstance(out["dropout1"], float) and out["dropout1"] == pytest.approx(0.25)
    assert isinstance(out["weight_decay"], float)
    assert out["hidden_dim"] in {128, 256}
    assert out["loss"] in {"angular", "cosine"}


def test_suggest_params_int_type_is_supported():
    space = {"hidden_dim": {"type": "int", "low": 64, "high": 256, "step": 64}}
    out = suggest_params(optuna.trial.FixedTrial({"hidden_dim": 128}), space)
    assert out["hidden_dim"] == 128


def test_unknown_search_type_raises_naming_the_dimension():
    space = {"dropout1": {"type": "bogus", "low": 0.0, "high": 1.0}}
    with pytest.raises(ValueError, match="dropout1"):
        suggest_params(optuna.trial.FixedTrial({}), space)


@pytest.mark.parametrize("space", [{}, None])
def test_empty_search_space_raises(space):
    """FR19: a search with nothing to search is a config error, not a silent
    single-config run."""
    with pytest.raises(ValueError, match="search_space"):
        suggest_params(optuna.trial.FixedTrial({}), space)


# --- sampler ---


def _cfg(**optuna_block):
    return {"optuna": optuna_block}


def test_build_sampler_names():
    assert isinstance(build_sampler(_cfg(sampler={"name": "tpe"})), optuna.samplers.TPESampler)
    assert isinstance(
        build_sampler(_cfg(sampler={"name": "random"})), optuna.samplers.RandomSampler
    )


def test_build_sampler_defaults_to_tpe():
    assert isinstance(build_sampler(_cfg()), optuna.samplers.TPESampler)


def test_build_sampler_unknown_raises():
    with pytest.raises(ValueError, match="grid"):
        build_sampler(_cfg(sampler={"name": "grid"}))


def test_sampler_seed_is_honored():
    """Two identically-seeded samplers must produce the same first suggestion --
    otherwise a 'reproducible' study silently is not."""
    def first(seed):
        study = optuna.create_study(sampler=build_sampler(_cfg(sampler={"name": "random", "seed": seed})))
        study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)
        return study.trials[0].params["x"]

    assert first(42) == first(42)


# --- pruner ---


@pytest.mark.parametrize(
    "name,cls",
    [
        ("median", optuna.pruners.MedianPruner),
        ("asha", optuna.pruners.SuccessiveHalvingPruner),
        ("none", optuna.pruners.NopPruner),
    ],
)
def test_build_pruner_names(name, cls):
    assert isinstance(build_pruner(_cfg(pruner={"name": name})), cls)


def test_build_pruner_defaults_to_median():
    assert isinstance(build_pruner(_cfg()), optuna.pruners.MedianPruner)


def test_build_pruner_unknown_raises():
    with pytest.raises(ValueError, match="hyperband"):
        build_pruner(_cfg(pruner={"name": "hyperband"}))


# --- pruning callback (reimplemented against pytorch_lightning; see hpo.py) ---


class _FakeTrainer:
    def __init__(self, metrics, sanity_checking=False):
        self.callback_metrics = metrics
        self.sanity_checking = sanity_checking


class _FakeModule:
    current_epoch = 3


class _RecordingTrial:
    def __init__(self, prune):
        self._prune, self.reported = prune, []

    def report(self, value, step):
        self.reported.append((value, step))

    def should_prune(self):
        return self._prune


def test_pruning_callback_reports_and_prunes():
    import torch

    from eyenet.hpo import PruningCallback

    trial = _RecordingTrial(prune=True)
    cb = PruningCallback(trial, monitor="val/angular_error_deg")
    trainer = _FakeTrainer({"val/angular_error_deg": torch.tensor(12.5)})
    with pytest.raises(optuna.TrialPruned):
        cb.on_validation_end(trainer, _FakeModule())
    assert trial.reported == [(12.5, 3)]


def test_pruning_callback_does_not_prune_when_trial_says_no():
    import torch

    from eyenet.hpo import PruningCallback

    trial = _RecordingTrial(prune=False)
    cb = PruningCallback(trial, monitor="val/angular_error_deg")
    cb.on_validation_end(_FakeTrainer({"val/angular_error_deg": torch.tensor(4.0)}), _FakeModule())
    assert trial.reported == [(4.0, 3)]


def test_pruning_callback_ignores_the_sanity_check_pass():
    """Lightning validates once before epoch 0; reporting there would
    double-report step 0 and skew the pruner's history."""
    import torch

    from eyenet.hpo import PruningCallback

    trial = _RecordingTrial(prune=True)
    cb = PruningCallback(trial, monitor="val/angular_error_deg")
    cb.on_validation_end(
        _FakeTrainer({"val/angular_error_deg": torch.tensor(1.0)}, sanity_checking=True),
        _FakeModule(),
    )
    assert trial.reported == []


def test_pruning_callback_warns_on_missing_metric_without_raising():
    from eyenet.hpo import PruningCallback

    trial = _RecordingTrial(prune=True)
    cb = PruningCallback(trial, monitor="val/angular_error_deg")
    with pytest.warns(UserWarning, match="angular_error_deg"):
        cb.on_validation_end(_FakeTrainer({}), _FakeModule())
    assert trial.reported == []
