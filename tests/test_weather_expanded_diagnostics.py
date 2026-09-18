import pandas as pd

from notebooks.diagnose_weather_expanded_cache import age_percentiles, classify_unmatched


def test_age_percentiles_empty_series_returns_none_fields_not_nan_or_error():
    result = age_percentiles(pd.Series([], dtype=float))
    assert result['n'] == 0
    assert result['median'] is None and result['max'] is None


def test_age_percentiles_ignores_missing_values():
    result = age_percentiles(pd.Series([10.0, 20.0, None, 30.0]))
    assert result['n'] == 3
    assert result['median'] == 20.0
    assert result['max'] == 30.0


def test_classify_unmatched_separates_mapping_unresolved_failed_and_no_report():
    idx = range(5)
    matched = pd.Series([True, False, False, False, False], index=idx)
    collectible = pd.Series([True, True, False, True, True], index=idx)
    unresolved = pd.Series([False, False, False, True, False], index=idx)
    station = pd.Series(['A', 'B', 'C', 'D', 'BAD'], index=idx)
    reason = classify_unmatched(matched, collectible, unresolved, failed_stations={'BAD'}, station_col=station)
    assert reason.tolist() == ['matched', 'no_report_within_max_age', 'not_collectible_mapping',
                               'unresolved_prediction_at', 'collection_failed_or_not_attempted']
