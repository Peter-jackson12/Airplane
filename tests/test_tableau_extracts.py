"""Synthetic-data tests for scripts/build_tableau_extracts.py aggregation logic.

No local data is needed; the real-data run is scripts/build_tableau_extracts.py itself.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.build_tableau_extracts import (
    bin_weather_values,
    class_metrics,
    confusion_long,
    delayed_flag,
    error_stability_table,
    fill_category,
    hour_bucket,
    matched_status,
    metrics_long,
    oof_error_table,
    rate_table,
    time_pattern,
    weather_bins_long,
    write_csv,
)
from src.weather_model import WEATHER_MODEL_FEATURES


def test_delayed_flag_rejects_unlabeled_and_maps_labels():
    assert delayed_flag(pd.Series(["Delayed", "Not_Delayed", " Delayed"])).tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        delayed_flag(pd.Series(["Delayed", None]))
    with pytest.raises(ValueError):
        delayed_flag(pd.Series(["maybe"]))


def test_rate_table_denominators_and_missing_bucket():
    df = pd.DataFrame({"airline": fill_category(pd.Series(["A", "A", None, "B", "B", "B"])),
                       "delayed": [1, 0, 1, 0, 0, 1]})
    t = rate_table(df, ["airline"], population="p")
    assert t.n_rows.sum() == 6 and t.n_delayed.sum() == 3
    assert set(t.airline) == {"A", "B", "(결측)"}
    a = t.set_index("airline")
    assert a.loc["A", "delay_rate"] == 0.5 and a.loc["(결측)", "n_rows"] == 1
    assert a.loc["B", "delay_rate"] == pytest.approx(1 / 3)
    assert (t.n_delayed + t.n_not_delayed == t.n_rows).all()
    assert t.share_of_population.sum() == pytest.approx(1.0)
    assert (t.population_rows == 6).all() and (t.population_delayed == 3).all()


def test_rate_table_rejects_nan_keys():
    df = pd.DataFrame({"k": ["a", None], "delayed": [0, 1]})
    with pytest.raises(ValueError):
        rate_table(df, ["k"], population="p")


def test_hour_bucket_keeps_missing_and_2400():
    h = hour_bucket(pd.Series([0, 59, 740.0, 2359, 2400, np.nan]))
    assert h.hour_order.tolist() == [0, 0, 7, 23, 24, -1]
    assert h.hour_label.tolist() == ["00", "00", "07", "23", "24 (2400 표기)", "결측(원본 시각 없음)"]


def test_time_pattern_matches_four_groups():
    dep = pd.Series([1.0, np.nan, 1.0, np.nan])
    arr = pd.Series([1.0, 1.0, np.nan, np.nan])
    assert time_pattern(dep, arr).tolist() == [
        "both_observed", "departure_missing", "arrival_missing", "both_missing"]


def test_precip_bins_cover_trace_zero_unmatched_and_field_missing():
    v = pd.Series([0.0, 0.0001, 0.01, 0.05, 0.2, np.nan, np.nan])
    m = pd.Series([1, 1, 1, 1, 1, 1, 0])
    b = bin_weather_values(v, m, "p01i")
    assert b.bin_kind.tolist() == ["special", "special", "value", "value", "value",
                                   "field_missing", "unmatched"]
    assert b.bin_label.iloc[1].startswith("미량")
    assert b.bin_label.iloc[2] == "(0.0001, 0.05) inch"
    assert b.bin_label.iloc[3] == "[0.05, 0.1) inch"
    assert b.bin_label.iloc[4] == "≥ 0.1 inch"


def test_unmatched_rows_with_values_are_rejected():
    with pytest.raises(ValueError):
        bin_weather_values(pd.Series([3.0]), pd.Series([0]), "tmpf")


def test_wind_zero_is_its_own_bin_and_temperature_negative_is_covered():
    b = bin_weather_values(pd.Series([0.0, 0.5, 97.19]), pd.Series([1, 1, 1]), "sknt")
    assert b.bin_label.tolist() == ["0 knot(무풍 보고)", "(0, 6) knot", "≥ 31 knot"]
    t = bin_weather_values(pd.Series([-40.0, 32.0, 120.0]), pd.Series([1, 1, 1]), "tmpf")
    assert t.bin_label.tolist() == ["< 32 °F", "[32, 50) °F", "≥ 86 °F"]


def _synthetic_weather(n=200, seed=0):
    rng = np.random.default_rng(seed)
    data = {"ID": [f"R{i}" for i in range(n)]}
    for role in ("origin", "destination"):
        matched = (rng.random(n) > 0.1).astype("int8")
        for field, lo, hi in (("tmpf", -30, 110), ("dwpf", -30, 80), ("sknt", 0, 40),
                              ("vsby", 0, 12), ("p01i", 0, 0.3)):
            vals = rng.uniform(lo, hi, n)
            if field == "p01i":
                vals = np.where(rng.random(n) < 0.5, 0.0, vals)
                vals = np.where(rng.random(n) < 0.1, 0.0001, vals)
            vals = np.where(rng.random(n) < 0.05, np.nan, vals)  # field missing
            data[f"weather_{role}_{field}"] = np.where(matched == 1, vals, np.nan)
        data[f"weather_{role}_age_minutes"] = np.where(matched == 1, rng.integers(10, 91, n), np.nan)
        data[f"weather_{role}_matched"] = matched
    frame = pd.DataFrame(data)
    assert set(WEATHER_MODEL_FEATURES) <= set(frame.columns)
    return frame, pd.Series(rng.integers(0, 2, n))


def test_weather_bins_long_every_role_feature_sums_to_population():
    w, y = _synthetic_weather()
    t = weather_bins_long(w, y)
    sums = t.groupby(["role", "feature"]).agg(n=("n_rows", "sum"), d=("n_delayed", "sum"))
    assert (sums.n == len(w)).all() and (sums.d == int(y.sum())).all()
    unmatched = t[(t.role == "origin") & (t.feature == "tmpf") & (t.bin_kind == "unmatched")]
    assert int(unmatched.n_rows.iloc[0]) == int((w.weather_origin_matched == 0).sum())
    assert (t.delay_rate == t.n_delayed / t.n_rows).all()
    both = t[t.feature == "matched_status"]
    assert both.n_rows.sum() == len(w)


def test_matched_status_four_states():
    s = matched_status(pd.Series([1, 1, 0, 0]), pd.Series([1, 0, 1, 0]))
    assert s.bin_label.tolist() == ["양쪽 공항 결합", "출발 공항만 결합", "도착 공항만 결합", "양쪽 미결합"]
    assert s.bin_kind.tolist() == ["value", "partial", "partial", "unmatched"]


def _runs():
    return pd.DataFrame({"seed": [42, 42], "condition": ["weather_off", "weather_on"],
                         "n_rows": [100, 100], "positive_rows": [20, 20],
                         "tn": [60, 65], "fp": [20, 15], "fn": [12, 11], "tp": [8, 9],
                         "macro_f1_nested": [0.5, 0.55], "log_loss": [0.4, 0.39]})


def test_confusion_long_equals_runs_and_sums():
    runs = _runs()
    c = confusion_long(runs, experiment="x", variant_col="condition", population="p")
    assert len(c) == 8
    for _, r in runs.iterrows():
        part = c[c.variant == r.condition].set_index("cell").n
        assert part.to_dict() == {"TN": r.tn, "FP": r.fp, "FN": r.fn, "TP": r.tp}
        assert part.sum() == r.n_rows
    bad = runs.copy()
    bad.loc[0, "tn"] = 1
    with pytest.raises(ValueError):
        confusion_long(bad, experiment="x", variant_col="condition", population="p")


def test_class_metrics_and_metrics_long():
    m = class_metrics(60, 20, 12, 8)
    assert m["recall_delayed"] == pytest.approx(8 / 20)
    assert m["precision_delayed"] == pytest.approx(8 / 28)
    assert m["false_positive_rate"] == pytest.approx(20 / 80)
    assert m["macro_f1_from_counts"] == pytest.approx((16 / 48 + 120 / 152) / 2)
    ml = metrics_long(_runs(), experiment="x", variant_col="condition", population="p",
                      metrics=("macro_f1_nested", "log_loss"))
    assert set(ml.metric) >= {"macro_f1_nested", "recall_delayed", "f1_not_delayed"}


def test_oof_error_table_and_stability():
    frame = pd.DataFrame({
        "seed": [1, 1, 1, 1, 2, 2, 2, 2],
        "ID": ["a", "b", "c", "d"] * 2,
        "y_true": [1, 1, 0, 0] * 2,
        "prediction": [1, 0, 1, 0, 0, 0, 1, 0],
        "g": ["x", "x", "y", "y"] * 2,
    })
    t = oof_error_table(frame, "g", population="p")
    assert t.n_rows.sum() == 8
    s1 = t[(t.seed == 1)].set_index("group")
    assert s1.loc["x", ["tn", "fp", "fn", "tp"]].tolist() == [0, 0, 1, 1]
    assert s1.loc["x", "recall_delayed"] == 0.5
    assert s1.loc["y", "false_positive_rate"] == 0.5
    assert (t.seed_population_rows == 4).all()
    per_row = frame.assign(w=(frame.y_true != frame.prediction).astype(int)).groupby("ID").agg(
        error_seed_count=("w", "sum"), g=("g", "first"))
    st = error_stability_table(per_row.assign(error_seed_count=per_row.error_seed_count), "g",
                               population="p")
    assert st.n_rows.sum() == 4
    y = st.set_index("group").loc["y"]
    assert (y.n_wrong_2_of_3, y.n_wrong_0_of_3) == (1, 1)


def test_write_csv_is_lf_only(tmp_path):
    info = write_csv(pd.DataFrame({"a": [1, 2], "b": ["가", "나"]}), tmp_path / "x.csv")
    data = (tmp_path / "x.csv").read_bytes()
    assert b"\r" not in data and info["rows"] == 2


def test_age_last_bin_is_closed_and_infinite_edges_are_blank():
    a = bin_weather_values(pd.Series([10.0, 59.0, 90.0]), pd.Series([1, 1, 1]), "age_minutes")
    assert a.bin_label.tolist() == ["[0, 30) 분", "[30, 60) 분", "[60, 90] 분"]
    w, y = _synthetic_weather(n=50, seed=3)
    t = weather_bins_long(w, y)
    assert np.isfinite(t.bin_lower.dropna()).all() and np.isfinite(t.bin_upper.dropna()).all()
