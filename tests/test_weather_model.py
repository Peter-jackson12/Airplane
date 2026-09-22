"""Synthetic contracts for the frozen submission weather-model comparison."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from notebooks import run_weather_model_comparison as comparison
from src.cv import CVConfig, make_folds
from src.weather_model import (
    HEADLINE_LATENCY_MINUTES,
    SEEDS,
    WEATHER_MODEL_FEATURES,
    WEATHER_NUMERIC_FIELDS,
    align_raw_to_join,
    attach_weather_features,
    extract_weather_features,
    fold_fingerprint,
    joined_usecols,
    labeled_mask,
    stable_json_hash,
)


def joined_frame():
    data = {"ID": ["C", "A", "B", "D"]}
    for role in ("origin", "destination"):
        data[f"{role}_observed_at"] = [
            "2019-01-01T11:00:00Z", "2019-01-01T11:30:00Z", pd.NA, pd.NA
        ]
        data[f"{role}_weather_age_minutes"] = [60.0, 30.0, np.nan, np.nan]
        for j, field in enumerate(WEATHER_NUMERIC_FIELDS):
            data[f"{role}_{field}"] = [10.0 + j, np.nan if field == "p01i" else 20.0 + j,
                                       np.nan, np.nan]
    return pd.DataFrame(data)


def test_feature_contract_is_frozen_minimal_and_headline_is_ten_minutes():
    assert HEADLINE_LATENCY_MINUTES == 10
    assert SEEDS == (42, 1, 7)
    assert WEATHER_NUMERIC_FIELDS == ("tmpf", "dwpf", "sknt", "vsby", "p01i")
    assert len(WEATHER_MODEL_FEATURES) == 14
    assert "weather_origin_gust" not in WEATHER_MODEL_FEATURES
    assert "weather_destination_relh" not in WEATHER_MODEL_FEATURES
    assert set(joined_usecols()).issuperset({"ID", "origin_observed_at",
                                              "destination_weather_age_minutes"})


def test_extract_weather_preserves_rows_and_missingness_without_imputation():
    joined = joined_frame()
    out = extract_weather_features(joined)
    assert out.ID.tolist() == joined.ID.tolist()
    assert list(out.columns[1:]) == list(WEATHER_MODEL_FEATURES)
    assert out.weather_origin_matched.tolist() == [1, 1, 0, 0]
    assert out.weather_destination_matched.tolist() == [1, 1, 0, 0]
    assert pd.isna(out.loc[1, "weather_origin_p01i"])
    assert pd.isna(out.loc[2, "weather_origin_tmpf"])
    assert out.loc[0, "weather_origin_age_minutes"] == 60


def test_extract_weather_rejects_target_actuals_duplicate_id_age_and_nonfinite():
    base = joined_frame()
    cases = []
    target = base.copy()
    target["Delay"] = ["Delayed"] * len(target)
    cases.append(target)
    duplicate = base.copy()
    duplicate.loc[1, "ID"] = duplicate.loc[0, "ID"]
    cases.append(duplicate)
    age = base.copy()
    age.loc[2, "origin_weather_age_minutes"] = 5.0
    cases.append(age)
    infinite = base.copy()
    infinite.loc[0, "origin_tmpf"] = np.inf
    cases.append(infinite)
    for frame in cases:
        with pytest.raises(ValueError):
            extract_weather_features(frame)


def test_align_raw_uses_join_order_and_never_drops_unmatched_weather_rows():
    raw = pd.DataFrame({
        "ID": ["A", "B", "C", "D", "E"],
        "Delay": ["Delayed", None, "Not_Delayed", "", "Delayed"],
        "x": [1, 2, 3, 4, 5],
    })
    ids = ["C", "A", "B", "D"]
    out = align_raw_to_join(raw, ids)
    assert out.ID.tolist() == ids
    assert out.x.tolist() == [3, 1, 2, 4]
    assert labeled_mask(out).tolist() == [True, True, False, False]


def test_align_raw_rejects_missing_or_duplicate_identity():
    raw = pd.DataFrame({"ID": ["A", "B"], "Delay": [None, None]})
    with pytest.raises(ValueError, match="absent"):
        align_raw_to_join(raw, ["A", "C"])
    bad = pd.concat([raw, raw.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        align_raw_to_join(bad, ["A"])


def test_attach_weather_uses_exact_label_split_and_preserves_base_attrs():
    raw = pd.DataFrame({
        "ID": ["C", "A", "B", "D"],
        "Delay": ["Not_Delayed", "Delayed", None, ""],
    })
    weather = extract_weather_features(joined_frame())
    X_lab = pd.DataFrame({"base": [30, 10]})
    X_unlab = pd.DataFrame({"base": [20, 40]})
    X_lab.attrs["missing_category_codes"] = {"Airline": 9}
    on_lab, on_unlab = attach_weather_features(X_lab, X_unlab, raw, weather)
    assert on_lab.base.tolist() == [30, 10]
    assert on_unlab.base.tolist() == [20, 40]
    assert on_lab.weather_origin_matched.tolist() == [1, 1]
    assert on_unlab.weather_origin_matched.tolist() == [0, 0]
    assert on_lab.attrs["missing_category_codes"] == {"Airline": 9}


def test_attach_weather_rejects_order_or_split_mismatch():
    raw = pd.DataFrame({"ID": ["C", "A", "B", "D"],
                        "Delay": ["Not_Delayed", "Delayed", None, ""]})
    weather = extract_weather_features(joined_frame())
    with pytest.raises(ValueError, match="ID order"):
        attach_weather_features(pd.DataFrame({"x": [1, 2]}),
                                pd.DataFrame({"x": [3, 4]}),
                                raw, weather.iloc[::-1].reset_index(drop=True))
    with pytest.raises(ValueError, match="label split"):
        attach_weather_features(pd.DataFrame({"x": [1]}),
                                pd.DataFrame({"x": [2, 3, 4]}),
                                raw, weather)


def test_fold_fingerprint_is_deterministic_and_seed_sensitive():
    y = pd.Series([0, 1] * 40)
    a = make_folds(y, CVConfig(seed=42))
    b = make_folds(y, CVConfig(seed=42))
    c = make_folds(y, CVConfig(seed=7))
    assert fold_fingerprint(a, len(y)) == fold_fingerprint(b, len(y))
    assert fold_fingerprint(a, len(y)) != fold_fingerprint(c, len(y))


class ProbeModel:
    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

    def fit(self, X, y):
        self.feature_name_ = list(X.columns)
        self.mean_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        x = pd.to_numeric(X["signal"], errors="coerce").fillna(0).to_numpy(float)
        p = np.clip(0.2 + 0.6 * x, 0.01, 0.99)
        return np.column_stack([1 - p, p])


def test_evaluate_condition_uses_repository_nested_selector_without_outer_fold_drift(monkeypatch):
    n = 100
    y = pd.Series([0, 1] * (n // 2), dtype=int)
    X = pd.DataFrame({"signal": np.tile([0.1, 0.9], n // 2)})
    monkeypatch.setattr(comparison.runner, "model_factory", ProbeModel)
    monkeypatch.setattr(comparison.runner, "N_ESTIMATORS_GRID", [1, 2])
    monkeypatch.setattr(comparison.runner, "CFG",
                        CVConfig(n_splits=5, seed=42, threshold_grid=(0.2, 0.8, 0.1)))
    spec = SimpleNamespace(cat_cols=(), inner_splits=0, te_drop_original=False)
    first = comparison.evaluate_condition(X, y, seed=42, spec=spec)
    second = comparison.evaluate_condition(X.assign(extra=1.0), y, seed=42, spec=spec)
    assert first["n_rows"] == second["n_rows"] == n
    assert first["fold_fingerprint"] == second["fold_fingerprint"]
    assert len(json.loads(first["per_fold_thresholds"])) == 5
    assert 0 <= first["macro_f1_nested"] <= 1


def test_summary_is_paired_by_seed_and_rejects_fold_mismatch():
    rows = []
    for seed in SEEDS:
        for condition, shift in (("weather_off", 0.0), ("weather_on", 0.01)):
            rows.append({
                "seed": seed, "condition": condition, "n_rows": 50,
                "fold_fingerprint": f"same-{seed}",
                "macro_f1_nested": 0.5 + shift,
                "log_loss": 0.4 - shift,
                "roc_auc": 0.6 + shift,
                "f1_at_050": 0.45 + shift,
                "recall": 0.3 + shift,
            })
    paired, summary = comparison.summarize(pd.DataFrame(rows))
    assert paired.macro_f1_nested_delta_on_minus_off.tolist() == pytest.approx([0.01] * 3)
    assert summary["paired_deltas_on_minus_off"]["log_loss"]["mean"] == pytest.approx(-0.01)
    bad = pd.DataFrame(rows)
    bad.loc[(bad.seed == 42) & (bad.condition == "weather_on"), "fold_fingerprint"] = "different"
    with pytest.raises(ValueError, match="different outer folds"):
        comparison.summarize(bad)


def test_checkpoint_identity_is_strict_and_resume_cells_unique(tmp_path):
    identity = {"join": "abc", "features": list(WEATHER_MODEL_FEATURES)}
    path = tmp_path / "checkpoint.json"
    rows = [{"seed": 42, "condition": "weather_off"}]
    comparison.save_checkpoint(path, identity, rows)
    assert comparison.load_checkpoint(path, identity) == rows
    with pytest.raises(ValueError, match="different experiment"):
        comparison.load_checkpoint(path, {**identity, "join": "changed"})
    payload = {
        "identity_sha256": stable_json_hash(identity),
        "identity": identity,
        "rows": rows + rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        comparison.load_checkpoint(path, identity)


def test_validate_inputs_only_builds_contract_but_fits_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    (tmp_path / "output").mkdir()
    adopted = pd.DataFrame({
        "ID": ["A", "B", "C"],
        "Delay": ["Delayed", "Not_Delayed", None],
    })
    weather = pd.DataFrame({"ID": ["A", "B", "C"]})
    for name in WEATHER_MODEL_FEATURES:
        weather[name] = 1.0
    weather["weather_origin_matched"] = pd.Series([1, 1, 0], dtype="int8")
    weather["weather_destination_matched"] = pd.Series([1, 0, 0], dtype="int8")
    manifest = {
        "outputs": {"10": {"sha256": "a" * 64}},
        "sources": {"raw_train": {"sha256": "b" * 64}},
    }
    scenario = {
        "origin_matched": 2, "destination_matched": 1,
        "both_matched": 1, "one_matched": 1, "none_matched": 1,
    }
    X_off = pd.DataFrame({"base": [1.0, 2.0]})
    X_on = pd.concat([
        X_off,
        weather.loc[:1, list(WEATHER_MODEL_FEATURES)].reset_index(drop=True)
    ], axis=1)
    y = pd.Series([1, 0], dtype=int)
    spec = SimpleNamespace()
    monkeypatch.setattr(comparison, "load_inputs",
                        lambda _: (adopted, weather, manifest, scenario, {}))
    monkeypatch.setattr(comparison, "build_pair",
                        lambda *_: (X_off, X_on, y, spec))
    monkeypatch.setattr(comparison.subprocess, "check_output",
                        lambda args, **kwargs: "c" * 40 if args[1] == "rev-parse" else "")
    monkeypatch.setattr(comparison, "evaluate_condition",
                        lambda *a, **k: pytest.fail("validate-only must not fit models"))
    monkeypatch.setattr(sys, "argv", [
        "run_weather_model_comparison",
        "--name", "baseline_recovery_v2_validate_only",
        "--validate-inputs-only",
    ])
    comparison.main()
    assert not (tmp_path / "data").exists()
    assert not list((tmp_path / "output").iterdir())
