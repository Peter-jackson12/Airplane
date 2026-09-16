"""Artificial timestamps here are structural tests, not flight evidence."""
import pandas as pd
import pytest

from src.weather import join_weather_asof, local_hhmm_to_utc


def test_timezones_dst_invalid_and_2400():
    dates = pd.Series(["2022-01-01", "2022-07-01", "2022-11-06", "2022-03-13", "2022-01-01", "2022-01-01"])
    actual = local_hhmm_to_utc(dates, pd.Series([1200, 1200, 130, 230, 2400, 1260]), "America/New_York")
    assert actual.iloc[0] == pd.Timestamp("2022-01-01 17:00Z")
    assert actual.iloc[1] == pd.Timestamp("2022-07-01 16:00Z")
    assert actual.iloc[2:4].isna().all()
    assert actual.iloc[4] == pd.Timestamp("2022-01-02 05:00Z")
    assert pd.isna(actual.iloc[5])


def frames():
    req = pd.DataFrame({"station": ["A", "B", "A", "A"], "prediction_at": pd.to_datetime(["2022-01-01 12:00Z", "2022-01-01 12:00Z", "2022-01-01 16:00Z", None], utc=True)}, index=[9, 4, 8, 1])
    obs = pd.DataFrame({"station": ["A", "A", "B"], "observed_at": pd.to_datetime(["2022-01-01 11:00Z", "2022-01-01 11:50Z", "2022-01-01 11:30Z"], utc=True), "available_at": pd.to_datetime(["2022-01-01 11:10Z", "2022-01-01 12:05Z", "2022-01-01 11:40Z"], utc=True), "wind": [10., 99., 20.]})
    return req, obs


def test_no_future_availability_cross_station_stale_or_row_growth():
    req, obs = frames()
    out = join_weather_asof(req, obs)
    assert out.index.equals(req.index)
    assert len(out) == len(req)
    assert out.wind.iloc[:2].tolist() == [10, 20]
    assert out.wind.iloc[2:].isna().all()
    assert (out.available_at.dropna() <= out.prediction_at.loc[out.available_at.notna()]).all()


def test_duplicates_rejected():
    req, obs = frames()
    with pytest.raises(ValueError, match="duplicate"):
        join_weather_asof(req, pd.concat([obs, obs.iloc[:1]]))


def test_naive_time_rejected():
    req, obs = frames()
    req["prediction_at"] = req.prediction_at.dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        join_weather_asof(req, obs)


def test_old_observation_with_recent_publication_still_expires():
    req, obs = frames()
    obs.loc[0, "observed_at"] = pd.Timestamp("2022-01-01 08:00Z")
    assert pd.isna(join_weather_asof(req, obs).wind.iloc[0])


def test_missing_station_and_empty_archive():
    req, obs = frames()
    req.loc[9, "station"] = None
    out = join_weather_asof(req, obs.iloc[:0])
    assert len(out) == len(req)
    assert out.wind.isna().all()
