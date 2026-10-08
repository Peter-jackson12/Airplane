"""Pure-logic tests for scripts/feedback_missing_audit.py on synthetic data.

The time example is the same worked example used in
docs/FEEDBACK_PREPROCESSING_KO.md (section 3).
"""
import numpy as np
import pandas as pd

from scripts.feedback_missing_audit import (
    airline_accounting,
    time_accounting,
    time_frames,
    traffic_accounting,
    vulnerable_group_from_groups_csv,
)


def test_airline_accounting_identity_and_residual_reasons():
    df = pd.DataFrame({
        "Carrier_Code(IATA)": ["A", "A", "B", "B", "B", "C", None],
        "Airline": ["one", None, "x", "y", None, None, None],
    })
    out = airline_accounting(df)
    legacy, clean = out["legacy_first"], out["clean_unique"]
    assert legacy["original_missing"] == clean["original_missing"] == 4
    # first: A and B fill, C (never observed with a name) and None stay missing
    assert (legacy["restored"], legacy["residual_missing"]) == (2, 2)
    # unique: only A fills; B is ambiguous (x / y)
    assert (clean["restored"], clean["residual_missing"]) == (1, 3)
    assert clean["residual_breakdown"] == {
        "key_missing": 1, "key_never_observed_with_value": 1, "key_maps_to_multiple_values": 1}
    assert out["legacy_minus_clean_restored"] == 1
    assert out["observed_values_changed"] == 0
    # counting a sub-population does not refit the mapping
    sub = airline_accounting(df, count_mask=[False, False, False, False, True, False, False])
    assert sub["population_rows"] == 1 and sub["fit_bundle_rows"] == 7
    assert sub["legacy_first"]["restored"] == 1 and sub["clean_unique"]["residual_missing"] == 1


def _worked_example():
    rows = [  # origin, dest, dep, arr
        ("A", "B", 1000, 1140),     # r0 gap 100
        ("A", "B", 1200, 1200),     # r1 gap 0 (kept in the median)
        ("A", "B", 800, 1010),      # r2 gap 130
        ("A", "B", 2330, np.nan),   # r3 arrival restored
        ("A", "B", np.nan, 40),     # r4 departure restored
        ("A", "B", np.nan, np.nan), # r5 both missing
        ("C", "D", 900, np.nan),    # r6 route without any valid gap -> global median
        ("E", "F", 100, 600),       # r7 gap 300
    ]
    return pd.DataFrame(rows, columns=["Origin_Airport", "Destination_Airport",
                                       "Estimated_Departure_Time", "Estimated_Arrival_Time"])


def test_time_restoration_worked_example():
    df = _worked_example()
    before, after = time_frames(df)
    gap = after["Estimated_Duration"]
    # route median of A->B uses 100, 0, 130 -> 100 (would be 115 without the 0)
    assert gap.iloc[3] == gap.iloc[4] == gap.iloc[5] == 100
    # global median is taken AFTER route fill: [0,100,100,100,100,130,300] -> 100,
    # not the observed-only median of [100,0,130,300] = 115
    assert before["Estimated_Duration"].median() == 115
    assert gap.iloc[6] == 100
    # one-side restoration with % 1440
    assert (after.Arr_Hour.iloc[3], after.Arr_Minute.iloc[3]) == (1, 10)   # 23:30 + 100 -> 01:10
    assert (after.Dep_Hour.iloc[4], after.Dep_Minute.iloc[4]) == (23, 0)   # 00:40 - 100 -> 23:00
    assert (after.Arr_Hour.iloc[6], after.Arr_Minute.iloc[6]) == (10, 40)  # 09:00 + 100
    # both missing: the gap is filled but the clock times stay unknown (-1)
    assert after.Dep_Hour.iloc[5] == -1 and after.Arr_Hour.iloc[5] == -1

    acc = time_accounting(df, frames=(before, after))
    assert acc["dep_hour"] == {"original_missing": 2, "restored": 1, "residual_missing": 1}
    assert acc["arr_hour"] == {"original_missing": 3, "restored": 2, "residual_missing": 1}
    g = acc["local_time_gap"]
    assert (g["original_missing"], g["restored"], g["residual_missing"]) == (4, 4, 0)
    assert (g["restored_by_route_median"], g["restored_by_global_median"]) == (3, 1)
    assert acc["valid_gap_zero_before_restore"] == 1
    assert acc["global_median_before_route_fill"] == 115
    assert acc["global_median_after_route_fill"] == 100


def test_midnight_wrap_gap():
    df = pd.DataFrame({"Origin_Airport": ["A"], "Destination_Airport": ["B"],
                       "Estimated_Departure_Time": [2300], "Estimated_Arrival_Time": [100]})
    before, _ = time_frames(df)
    assert before["Estimated_Duration"].iloc[0] == 120  # (60 - 1380) + 1440


def test_traffic_counts_self_row_excludes_unknown_hour_and_ignores_year():
    frame = pd.DataFrame({
        "Month": [1, 1, 1, 1], "Day_of_Month": [5, 5, 5, 5],
        "Origin_Airport": ["X", "X", "X", "X"], "Destination_Airport": ["Y", "Y", "Z", "Y"],
        "Dep_Hour": [8, 8, 9, -1], "Arr_Hour": [10, 10, 11, -1],
    })
    out = traffic_accounting(frame, years=[2018, 2019, 2018, 2019])
    o = out["Origin_Traffic"]
    assert o["nan_rows"] == 1 and o["min_value"] == 1.0
    assert o["rows_equal_1_only_self"] == 1        # the 09h row counts only itself
    assert o["rows_in_key_with_both_years"] == 2   # 2018 and 2019 rows share one key


def test_vulnerable_group_mean_of_seed_recalls():
    groups = pd.DataFrame({
        "phase_key": ["P6_clean"] * 3 + ["P6_fixed"],
        "dimension": ["raw_time_pattern"] * 4,
        "group": ["both_missing"] * 4,
        "seed": [42, 1, 7, 42],
        "n": [3031] * 4, "positives": [519] * 4,
        "tp": [55, 49, 52, 47], "fn": [464, 470, 467, 472],
        "recall": [55 / 519, 49 / 519, 52 / 519, 47 / 519],
    })
    out = vulnerable_group_from_groups_csv(groups)
    assert set(out["seeds"]) == {"1", "7", "42"}
    assert round(out["recall_mean_of_seeds"] * 100, 2) == 10.02
