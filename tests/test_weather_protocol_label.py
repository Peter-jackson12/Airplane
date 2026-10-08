"""The weather comparison runner must record the protocol it actually ran.

It trains with ``run_fold_nested_grid`` (nested n_estimators grid, no early-stopping
callback), whereas ``CVConfig.describe()`` hard-codes ``inner_early_stopping=True``.
"""
from __future__ import annotations

from dataclasses import replace

import rerun_all_phases as runner
from notebooks import run_weather_model_comparison as comparison
from src.cv import CVConfig


def test_weather_protocol_says_nested_grid_without_early_stopping():
    proto = comparison.protocol_description()
    assert proto["inner_early_stopping"] is False
    assert proto["tree_selection"] == "nested_grid"
    assert proto["stopping_rounds_used"] is False
    assert proto["n_estimators_grid"] == list(runner.N_ESTIMATORS_GRID)
    assert proto["nested_threshold"] is True


def test_weather_protocol_follows_per_run_seed_and_matches_runner_label():
    cfg = replace(runner.CFG, seed=7)
    proto = comparison.protocol_description(cfg)
    assert proto["seed"] == 7
    base = runner.protocol_description()
    assert {**proto, "seed": base["seed"]} == {
        **base, "n_estimators_grid": list(base["n_estimators_grid"])}


def test_cvconfig_describe_is_untouched_legacy_label():
    # Guard: the fix lives in the runner; CVConfig.describe() feeds run fingerprints.
    assert CVConfig().describe()["inner_early_stopping"] is True
