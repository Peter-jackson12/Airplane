import pandas as pd

from notebooks.diagnose_weather_expanded_cache import (age_percentiles, classify_unmatched,
                                                        diagnose_stale_vs_unavailable_vs_absent)


def test_age_percentiles_empty_series_returns_none_fields_not_nan_or_error():
    result = age_percentiles(pd.Series([], dtype=float))
    assert result['n'] == 0
    assert result['median'] is None and result['max'] is None


def test_age_percentiles_ignores_missing_values():
    result = age_percentiles(pd.Series([10.0, 20.0, None, 30.0]))
    assert result['n'] == 3
    assert result['median'] == 20.0
    assert result['max'] == 30.0


def empty_obs():
    return pd.DataFrame({'station': pd.Series(dtype=object), 'observed_at': pd.Series(dtype='datetime64[ns, UTC]')})


def test_classify_unmatched_separates_mapping_unresolved_and_failed():
    idx = range(5)
    matched = pd.Series([True, False, False, False, False], index=idx)
    collectible = pd.Series([True, True, False, True, True], index=idx)
    unresolved = pd.Series([False, False, False, True, False], index=idx)
    station = pd.Series(['A', 'B', 'C', 'D', 'BAD'], index=idx)
    day = pd.Series(['2019-06-01'] * 5, index=idx)
    prediction_at = pd.Series([pd.Timestamp('2019-06-01T09:00Z')] * 5, index=idx)
    reason = classify_unmatched(matched, collectible, unresolved, failed_keys={('BAD', '2019-06-01')},
                                station_col=station, day_col=day, prediction_at=prediction_at,
                                obs=empty_obs(), latency_minutes=10)
    assert reason.tolist() == ['matched', 'no_report_observed_before_prediction_time',
                               'not_collectible_mapping', 'unresolved_prediction_at',
                               'collection_failed_or_not_attempted']


def test_classify_unmatched_does_not_propagate_a_failure_to_a_different_day_same_station():
    """A (station, day) failure must not blanket-mark every row that shares
    the same station on a DIFFERENT, successfully-collected day."""
    idx = range(2)
    matched = pd.Series([False, False], index=idx)
    collectible = pd.Series([True, True], index=idx)
    unresolved = pd.Series([False, False], index=idx)
    station = pd.Series(['A', 'A'], index=idx)
    day = pd.Series(['2019-06-01', '2019-06-02'], index=idx)  # only day 1 failed
    prediction_at = pd.Series([pd.Timestamp('2019-06-01T09:00Z'), pd.Timestamp('2019-06-02T09:00Z')], index=idx)
    reason = classify_unmatched(matched, collectible, unresolved, failed_keys={('A', '2019-06-01')},
                                station_col=station, day_col=day, prediction_at=prediction_at,
                                obs=empty_obs(), latency_minutes=10)
    assert reason.iloc[0] == 'collection_failed_or_not_attempted'
    assert reason.iloc[1] != 'collection_failed_or_not_attempted'


def test_diagnose_no_report_when_cache_has_nothing_before_prediction_time():
    station = pd.Series(['A'], index=[0])
    prediction_at = pd.Series([pd.Timestamp('2019-06-01T09:00Z')], index=[0])
    obs = pd.DataFrame({'station': ['A'], 'observed_at': [pd.Timestamp('2019-06-01T20:00Z')]})  # AFTER prediction
    reason = diagnose_stale_vs_unavailable_vs_absent(station, prediction_at, obs, latency_minutes=10)
    assert reason.iloc[0] == 'no_report_observed_before_prediction_time'


def test_diagnose_not_yet_available_under_latency_assumption():
    """A report exists shortly before prediction_at and is well within the
    max_age window, but a large assumed latency pushes its availability past
    prediction_at -- this must be distinguished from a genuine absence or
    genuine staleness."""
    station = pd.Series(['A'], index=[0])
    prediction_at = pd.Series([pd.Timestamp('2019-06-01T09:00Z')], index=[0])
    obs = pd.DataFrame({'station': ['A'], 'observed_at': [pd.Timestamp('2019-06-01T08:55Z')]})  # age=5min
    reason = diagnose_stale_vs_unavailable_vs_absent(station, prediction_at, obs, latency_minutes=60)
    assert reason.iloc[0] == 'not_yet_available_under_latency_assumption'


def test_diagnose_stale_beyond_max_age():
    station = pd.Series(['A'], index=[0])
    prediction_at = pd.Series([pd.Timestamp('2019-06-01T09:00Z')], index=[0])
    obs = pd.DataFrame({'station': ['A'], 'observed_at': [pd.Timestamp('2019-06-01T07:00Z')]})  # age=120min
    reason = diagnose_stale_vs_unavailable_vs_absent(station, prediction_at, obs, latency_minutes=0)
    assert reason.iloc[0] == 'stale_beyond_max_age'


def test_diagnose_a_would_have_matched_report_is_flagged_not_silently_absorbed():
    """If a candidate report is both available and within max_age, it should
    have matched in the join already; if classify_unmatched ever calls this
    for such a row it is a real bug, surfaced explicitly rather than mislabeled."""
    station = pd.Series(['A'], index=[0])
    prediction_at = pd.Series([pd.Timestamp('2019-06-01T09:00Z')], index=[0])
    obs = pd.DataFrame({'station': ['A'], 'observed_at': [pd.Timestamp('2019-06-01T08:50Z')]})  # age=10min
    reason = diagnose_stale_vs_unavailable_vs_absent(station, prediction_at, obs, latency_minutes=0)
    assert reason.iloc[0] == 'unexpected_unmatched_despite_available_report'
