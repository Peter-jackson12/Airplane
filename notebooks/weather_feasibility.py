"""Reproducible weather audit and bounded public archive download; no training.

Run: uv run --offline python -u notebooks/weather_feasibility.py
2022-12-23 is an archive connectivity probe, NOT a verified flight date.
"""
from pathlib import Path
import hashlib
import inspect
import io
import json
import sys
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output" / "weather_review"


def audit_existing():
    OUT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data/train.csv"
    if not path.is_file():
        raise FileNotFoundError("Real data/train.csv required")
    raw = pd.read_csv(path)
    old = pd.read_csv(ROOT / "data/train_with_weather.csv")
    assert len(raw) == len(old) and raw.ID.equals(old.ID)
    changed = [c for c in raw if not raw[c].equals(old[c])]
    weather_cols = [c for c in old if c.startswith("Weather_")]
    label = raw.Delay.notna()
    profile = pd.DataFrame({
        "column": weather_cols,
        "nonmissing": [int(old[c].notna().sum()) for c in weather_cols],
        "coverage_pct": [old[c].notna().mean() * 100 for c in weather_cols],
        "labeled_coverage_pct": [old.loc[label, c].notna().mean() * 100 for c in weather_cols],
    })
    profile.to_csv(OUT / "legacy_weather_coverage.csv", index=False)
    airport = old.assign(matched=old.Weather_Origin_wspd.notna()).groupby("Origin_Airport").agg(rows=("ID", "size"), matched=("matched", "sum"))
    airport.to_csv(OUT / "legacy_airport_coverage.csv")
    raw.Airline.value_counts(dropna=False).to_csv(OUT / "airline_inventory.csv")
    keys = ["Month", "Day_of_Month", "Tail_Number", "Origin_Airport", "Destination_Airport", "Estimated_Departure_Time", "Estimated_Arrival_Time", "Distance", "Carrier_ID(DOT)"]
    # No target or outcome fields in the fingerprint export.
    fp = raw.dropna(subset=keys).sort_values(keys).groupby("Month").head(20)
    fp[["ID"] + keys].to_csv(OUT / "year_verification_fingerprints.csv", index=False)
    meta = {
        "raw_rows": len(raw), "raw_columns": len(raw.columns),
        "legacy_columns": len(old.columns), "raw_columns_changed": changed,
        "origin_airports": int(raw.Origin_Airport.nunique()),
        "dest_airports": int(raw.Destination_Airport.nunique()),
        "origin_airports_with_weather": int(airport.matched.gt(0).sum()),
        "weather_columns": weather_cols,
        "year_columns": [c for c in raw if "year" in c.lower() or "date" in c.lower()],
        "feb29_rows": int(((raw.Month == 2) & (raw.Day_of_Month == 29)).sum()),
        "departure_missing": int(raw.Estimated_Departure_Time.isna().sum()),
        "candidate_year": 2022, "candidate_year_verified": False,
        "fingerprint_rows": len(fp), "flight_weather_join_performed": False,
        "raw_sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        "legacy_sha256": hashlib.file_digest((ROOT / "data/train_with_weather.csv").open("rb"), "sha256").hexdigest(),
        "executed_utc": datetime.now(timezone.utc).isoformat(),
    }
    from meteostat import Hourly
    meta["meteostat_hourly_signature"] = str(inspect.signature(Hourly))
    (OUT / "meteostat_time_source.txt").write_text(inspect.getsource(Hourly._set_time), encoding="utf-8")
    (OUT / "audit_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta, profile


def fetch_probe():
    OUT.mkdir(parents=True, exist_ok=True)
    params = {"station": ["ATL", "ORD", "JFK"], "data": "all",
              "sts": "2022-12-23T00:00:00Z", "ets": "2022-12-24T00:00:00Z",
              "tz": "UTC", "format": "onlycomma", "latlon": "yes",
              "missing": "M", "trace": "T", "report_type": [3, 4]}
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urlencode(params, doseq=True)
    cache = ROOT / "data/weather_probe/iem_20221223_ATL_ORD_JFK.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        with urlopen(Request(url, headers={"User-Agent": "Airplane-course-weather-audit/1.0"}), timeout=60) as r:
            body = r.read(10_000_001)
        if len(body) > 10_000_000:
            raise ValueError("Probe response exceeds bounded size")
        parsed = pd.read_csv(io.BytesIO(body), comment="#", na_values=["M"], keep_default_na=False)
        if not {"station", "valid", "metar", "vsby", "gust"}.issubset(parsed):
            raise ValueError("Unexpected archive schema; response not cached")
        cache.write_bytes(body)
    obs = pd.read_csv(cache, comment="#", na_values=["M"], keep_default_na=False)
    assert set(obs.station) == {"ATL", "ORD", "JFK"}
    fields = ["tmpf", "sknt", "gust", "vsby", "skyc1", "skyl1", "wxcodes", "p01i", "snowdepth", "metar"]
    rows = []
    for station, group in obs.groupby("station"):
        row = {"station": station, "reports": len(group)}
        row.update({f"{f}_reported": int(group[f].replace("", pd.NA).notna().sum()) for f in fields})
        rows.append(row)
    coverage = pd.DataFrame(rows)
    coverage.to_csv(OUT / "iem_probe_coverage.csv", index=False)
    provenance = {"url": url, "sha256": hashlib.file_digest(cache.open("rb"), "sha256").hexdigest(),
                  "cache": str(cache.relative_to(ROOT)), "rows": len(obs),
                  "probe_only_not_flight_year": True,
                  "checked_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "iem_probe_source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return obs, coverage


def check_alignment(obs):
    from src.weather import local_hhmm_to_utc, join_weather_asof
    demo = pd.DataFrame({"local_date": ["2022-01-15", "2022-07-15"], "hhmm": [1200, 1200]})
    demo["New_York_utc"] = local_hhmm_to_utc(demo.local_date, demo.hhmm, "America/New_York")
    demo["Chicago_utc"] = local_hhmm_to_utc(demo.local_date, demo.hhmm, "America/Chicago")
    demo.to_csv(OUT / "timezone_examples.csv", index=False)
    w = obs[["station", "valid", "sknt", "gust", "vsby"]].copy()
    w["observed_at"] = pd.to_datetime(w.pop("valid"), utc=True)
    # Probe assumption only. IEM valid is NOT original receipt/publication time.
    w["available_at"] = w.observed_at + pd.Timedelta("10min")
    if w.duplicated(["station", "available_at"]).any():
        raise ValueError("Probe has duplicate reports; inspect correction versions")
    req = pd.MultiIndex.from_product([["ATL", "ORD", "JFK"], pd.date_range("2022-12-23", periods=24, freq="h", tz="UTC")], names=["station", "prediction_at"]).to_frame(index=False)
    joined = join_weather_asof(req, w)
    available = joined.available_at.notna()
    assert (joined.loc[available, "available_at"] <= joined.loc[available, "prediction_at"]).all()
    joined.to_csv(OUT / "asof_probe.csv", index=False)
    result = {"requests": len(req), "matched": int(available.sum()), "assumed_latency_minutes": 10,
              "future_reports_selected": 0, "flight_rows_used": 0,
              "max_age_minutes": float(joined.weather_age_minutes.max())}
    (OUT / "asof_probe_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return demo, result


if __name__ == "__main__":
    meta, profile = audit_existing()
    print(json.dumps(meta, indent=2), flush=True)
    print(profile.to_string(index=False), flush=True)
    obs, coverage = fetch_probe()
    print(coverage.to_string(index=False), flush=True)
    print(check_alignment(obs)[1], flush=True)
