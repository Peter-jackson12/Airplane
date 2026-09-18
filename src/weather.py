"""Weather alignment primitives; no guessed year or inferred arrival date.

Callers must establish flight dates and airport/station mappings independently.
Observation time is not publication time: available_at must be supplied explicitly.
Historical archives without receipt timestamps require a documented latency assumption
and do not establish operational point-in-time correctness by themselves.
"""
from __future__ import annotations

import pandas as pd


def local_hhmm_to_utc(dates: pd.Series, hhmm: pd.Series, timezone: str) -> pd.Series:
    """Convert explicit local dates; 2400 means next midnight, DST ambiguity -> NaT.

Missing/invalid times remain missing. Never use an imputed hour to claim an
observed flight timestamp. Arrival needs its own verified local calendar date.
"""
    if not dates.index.equals(hhmm.index):
        raise ValueError("dates and hhmm indexes must match")
    dates = pd.to_datetime(dates, errors="raise")
    if dates.dt.tz is not None:
        raise ValueError("dates must be timezone-naive local dates")
    if (dates.dropna() != dates.dropna().dt.normalize()).any():
        raise ValueError("dates must be midnight calendar dates")
    value = pd.to_numeric(hhmm, errors="coerce")
    valid = value.eq(value.round()) & (
        (value.ge(0) & value.lt(2400) & value.mod(100).lt(60)) | value.eq(2400)
    )
    minutes = (value.floordiv(100) * 60 + value.mod(100)).where(valid)
    naive = dates + pd.to_timedelta(minutes, unit="m")
    return naive.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def _utc(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{name} must be timezone-aware")
    # Normalized to a fixed (ns) resolution: pandas infers a datetime64 unit (s/us/ns) from the
    # INPUT values, so an all-missing or empty series (a legitimate "nothing available" edge case,
    # not an error) can end up a different unit than a populated one built the same way elsewhere,
    # and merge_asof's dtype check then rejects the two sides as mismatched types.
    return series.dt.tz_convert("UTC").astype("datetime64[ns, UTC]")


def join_weather_asof(
    requests: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    max_age: str = "90min",
) -> pd.DataFrame:
    """One result per request using last available report at prediction time.

Requests: station, prediction_at (aware). Observations: station, observed_at,
available_at (aware), plus features. Both observed_at and available_at must be
<= prediction_at, and observed_at age is capped. Duplicate availability keys
are rejected, requiring explicit upstream correction/version resolution.
Missing station/time requests remain unmatched. Input row order/index survives.
Same primitive handles origin and destination at the SAME prediction cutoff;
destination future observed arrival weather must not be used for predeparture
prediction. Forecast valid-time/issue-time selection is a separate contract.
"""
    req, obs = requests.copy(), observations.copy()
    if not {"station", "prediction_at"}.issubset(req):
        raise ValueError("requests require station and prediction_at")
    if not {"station", "observed_at", "available_at"}.issubset(obs):
        raise ValueError("observations require station, observed_at and available_at")
    if (set(req) & set(obs)) != {"station"}:
        raise ValueError("request/observation column collision")
    if "__row" in req or "__row" in obs or "weather_age_minutes" in req or "weather_age_minutes" in obs:
        raise ValueError("reserved output column")
    req["prediction_at"] = _utc(req.prediction_at, "prediction_at")
    for col in ("observed_at", "available_at"):
        obs[col] = _utc(obs[col], col)
    if obs[["station", "observed_at", "available_at"]].isna().any().any():
        raise ValueError("observation keys must not be missing")
    if obs.duplicated(["station", "available_at"]).any():
        raise ValueError("duplicate station/availability; resolve report versions first")
    if (obs.available_at < obs.observed_at).any():
        raise ValueError("availability cannot precede observation")
    age_limit = pd.Timedelta(max_age)
    if age_limit <= pd.Timedelta(0):
        raise ValueError("max_age must be positive")
    req["__row"] = range(len(req))
    eligible = req.station.notna() & req.prediction_at.notna()
    # An empty observation set (every collection attempt failed, or nothing was ever collectible) is a
    # legitimate "no data available" input, not an error condition -- but pandas can only infer the
    # nullable "string" dtype for a text column from actual values; a zero-row frame falls back to
    # plain "object", which merge_asof's by= key comparison then rejects as a dtype mismatch against a
    # differently-constructed (e.g. non-empty) station column. Both sides are normalized to plain
    # object dtype right before the merge so this stays a safe join-key type detail, never a
    # caller-visible crash on an edge case that is otherwise perfectly answerable ("nothing matches").
    left = req.loc[eligible].sort_values("prediction_at").assign(station=lambda d: d.station.astype(object))
    right = obs.sort_values("available_at").assign(station=lambda d: d.station.astype(object))
    matched = pd.merge_asof(
        left, right,
        by="station", left_on="prediction_at", right_on="available_at",
        direction="backward", tolerance=age_limit,
    )
    age = matched.prediction_at - matched.observed_at
    stale = age.gt(age_limit)
    weather_cols = [col for col in obs if col != "station"]
    matched.loc[stale, weather_cols] = None
    matched["weather_age_minutes"] = age.dt.total_seconds().div(60).where(~stale)
    result = pd.concat([matched, req.loc[~eligible]], ignore_index=True).sort_values("__row")
    result = result.drop(columns="__row")
    result.index = requests.index
    return result
