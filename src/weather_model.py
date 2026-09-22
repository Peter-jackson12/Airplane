"""Minimal, predeclared weather-model contract for the submission comparison.

This module does not fit a model and never chooses features from model scores.
The headline scenario is the already-declared 10-minute publication-latency
ASSUMPTION. 0/30/60-minute joins remain coverage sensitivity evidence only.

Weather-on adds exactly fourteen numeric features to the same P6_clean rows:
for origin and destination independently, tmpf/dwpf/sknt/vsby/p01i,
weather_age_minutes, and an explicit matched flag. LightGBM handles missing
numeric values natively; no weather-row filtering or weather imputation occurs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

HEADLINE_LATENCY_MINUTES = 10
BASE_PHASE_KEY = "P6_clean"
SEEDS = (42, 1, 7)
WEATHER_NUMERIC_FIELDS = ("tmpf", "dwpf", "sknt", "vsby", "p01i")
ROLES = ("origin", "destination")
FORBIDDEN_JOIN_COLUMNS = {
    "delay", "not_delayed", "delayed", "arrdelay", "depdelay",
    "actual_departure_time", "actual_arrival_time",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_feature_columns() -> list[str]:
    out: list[str] = []
    for role in ROLES:
        out.extend(f"weather_{role}_{field}" for field in WEATHER_NUMERIC_FIELDS)
        out.extend([f"weather_{role}_age_minutes", f"weather_{role}_matched"])
    return out


WEATHER_MODEL_FEATURES = tuple(model_feature_columns())


def joined_usecols() -> list[str]:
    cols = ["ID"]
    for role in ROLES:
        cols.extend(f"{role}_{field}" for field in WEATHER_NUMERIC_FIELDS)
        cols.extend([f"{role}_weather_age_minutes", f"{role}_observed_at"])
    return cols


def extract_weather_features(joined: pd.DataFrame) -> pd.DataFrame:
    """Convert a validated full-join frame into the frozen numeric model features.

    The caller should load the joined artifact with src.weather_full.read_joined_csv
    so the explicit <NA> contract is kept. This function rejects target/outcome
    columns even though it does not use them.
    """
    require("ID" in joined, "joined weather requires ID")
    require(joined.ID.notna().all() and joined.ID.is_unique,
            "joined weather IDs must be nonmissing and unique")
    folded = {str(c).casefold() for c in joined.columns}
    require(not (FORBIDDEN_JOIN_COLUMNS & folded),
            "target/actual-outcome columns must not enter weather feature extraction")
    required = set(joined_usecols())
    require(required.issubset(joined.columns),
            f"joined weather missing columns: {sorted(required - set(joined.columns))}")

    result = pd.DataFrame({"ID": joined.ID.astype(str).to_numpy()})
    for role in ROLES:
        observed = pd.to_datetime(joined[f"{role}_observed_at"], utc=True,
                                  errors="coerce", format="mixed")
        matched = observed.notna()
        age = pd.to_numeric(joined[f"{role}_weather_age_minutes"], errors="raise")
        require(matched.equals(age.notna()),
                f"{role} weather age/matched identity differs")
        require(age.dropna().between(0, 90, inclusive="both").all(),
                f"{role} weather age outside [0,90]")

        for field in WEATHER_NUMERIC_FIELDS:
            source = pd.to_numeric(joined[f"{role}_{field}"], errors="raise")
            finite = source.dropna().to_numpy(dtype=float)
            require(np.isfinite(finite).all(), f"{role}_{field} contains nonfinite values")
            result[f"weather_{role}_{field}"] = source.to_numpy()
        result[f"weather_{role}_age_minutes"] = age.to_numpy()
        result[f"weather_{role}_matched"] = matched.astype("int8").to_numpy()

    require(list(result.columns[1:]) == list(WEATHER_MODEL_FEATURES),
            "weather model feature order drift")
    return result


def align_raw_to_join(raw: pd.DataFrame, joined_ids: Iterable[str]) -> pd.DataFrame:
    """Return raw rows in full-join ID order without changing the denominator."""
    require("ID" in raw and "Delay" in raw, "raw train requires ID and Delay")
    require(raw.ID.notna().all() and raw.ID.is_unique,
            "raw train IDs must be nonmissing and unique")
    ids = pd.Series(list(joined_ids), dtype="string")
    require(ids.notna().all() and ids.is_unique, "joined IDs must be unique")
    indexed = raw.copy()
    indexed["ID"] = indexed.ID.astype(str)
    indexed = indexed.set_index("ID", drop=False)
    missing = ids[~ids.isin(indexed.index)]
    require(missing.empty, f"joined IDs absent from raw train: {missing.iloc[:5].tolist()}")
    out = indexed.loc[ids.astype(str).tolist()].reset_index(drop=True)
    require(out.ID.astype(str).tolist() == ids.astype(str).tolist(),
            "raw/join row order mismatch")
    return out


def labeled_mask(raw: pd.DataFrame) -> pd.Series:
    target = raw["Delay"]
    return target.notna() & target.astype("string").str.strip().ne("")


def attach_weather_features(
    X_labeled: pd.DataFrame,
    X_unlabeled: pd.DataFrame,
    raw_adopted: pd.DataFrame,
    weather: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach weather by the exact label split used by src.features.split_labeled."""
    require(len(raw_adopted) == len(weather), "raw/weather denominator mismatch")
    require(raw_adopted.ID.astype(str).tolist() == weather.ID.astype(str).tolist(),
            "raw/weather ID order mismatch")
    mask = labeled_mask(raw_adopted)
    require(int(mask.sum()) == len(X_labeled) and int((~mask).sum()) == len(X_unlabeled),
            "weather/base label split mismatch")
    cols = list(WEATHER_MODEL_FEATURES)
    lab_weather = weather.loc[mask, cols].reset_index(drop=True)
    unlab_weather = weather.loc[~mask, cols].reset_index(drop=True)
    on_lab = pd.concat([X_labeled.reset_index(drop=True), lab_weather], axis=1)
    on_unlab = pd.concat([X_unlabeled.reset_index(drop=True), unlab_weather], axis=1)
    on_lab.attrs = dict(X_labeled.attrs)
    on_unlab.attrs = dict(X_unlabeled.attrs)
    require(not on_lab.columns.duplicated().any() and not on_unlab.columns.duplicated().any(),
            "weather feature column collision")
    return on_lab, on_unlab


def fold_fingerprint(folds: list[tuple[np.ndarray, np.ndarray]], n_rows: int) -> str:
    """Hash only validation assignment; both paired conditions must share it."""
    assignment = np.full(n_rows, -1, dtype=np.int16)
    counts = np.zeros(n_rows, dtype=np.int8)
    for fold, (_, valid) in enumerate(folds):
        valid = np.asarray(valid)
        require((valid >= 0).all() and (valid < n_rows).all(), "invalid fold indices")
        assignment[valid] = fold
        np.add.at(counts, valid, 1)
    require(np.all(counts == 1) and np.all(assignment >= 0),
            "each row must belong to exactly one outer validation fold")
    return hashlib.sha256(assignment.tobytes()).hexdigest()


def stable_json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
