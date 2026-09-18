"""Re-size the full weather collection with ACTUAL required query intervals,
replacing scope_weather_collection.py's (sample average bytes/request) x
(full station-day GROUP COUNT) formula -- which is algebraically just the
sample's mean bytes-per-group times the full group count, and does not
resolve any "short window vs long window" difference, whatever the original
run's prose claimed.

This script:
  1. Separates three denominators: all adopted rows, mapping-eligible
     (confirmed_period both ends) rows, and UTC-time-resolved rows among those.
  2. Builds, per REQUIRED station, the individual fixed-padding intervals
     (prediction_at - LOOKBACK_HOURS .. + LOOKAHEAD_HOURS) implied by every
     safe row that needs that station -- then MERGES overlapping/adjacent
     intervals per station into a minimal union, and reports the "original
     padded total" (sum of per-row interval lengths) separately from the
     "merged union total" (sum of the merged intervals' lengths).
  3. Subtracts the ALREADY-VALIDATED cache windows (from the 21-row and
     300-row runs' fetch manifests, re-verified by content hash here) from
     each station's merged union by actual interval overlap -- never by
     (station, day)-key matching, so a station/day key that was only
     partially covered is not treated as if the whole day were cached.
  4. Splits each remaining NEWLY-NEEDED merged interval into requests sized
     to the provider's practical per-request byte cap (MAX_RESPONSE_BYTES),
     using the measured bytes/hour rate, instead of assuming one request per
     calendar day.
  5. Reports bytes and time as RANGES built from the measured per-group
     bytes/hour and seconds/request statistics across BOTH existing runs,
     flags stations with no real sample to extrapolate from, and separates
     disk (raw CSV) bytes from pandas in-memory footprint.

This is a LOCAL sizing exercise only -- no new network collection, no new
selection, and no target/delay/actual-time column is read anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from notebooks.fetch_weather_sample import LOOKAHEAD_HOURS, LOOKBACK_HOURS, MAX_RESPONSE_BYTES
from notebooks.scope_weather_collection import CENSUS_REGION, NON_CENSUS_TERRITORY, SEASON_BY_MONTH, region_for  # noqa: F401  (re-exported for callers/tests)

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION_RUN = 'baseline_recovery_v2_row_date_attribution_20260918'
EXISTING_FETCH_RUNS = ['baseline_recovery_v2_weather_sample_20260918',
                       'baseline_recovery_v2_weather_expanded_20260918']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_prediction_at(dates: pd.Series, hhmm: pd.Series, timezones: pd.Series) -> pd.Series:
    from src.weather import local_hhmm_to_utc
    parts = []
    frame = pd.DataFrame({'date': dates.values, 'hhmm': hhmm.values, 'tz': timezones.values},
                         index=dates.index)
    for tz, sub in frame.groupby('tz'):
        parts.append(local_hhmm_to_utc(sub.date, sub.hhmm, tz))
    return pd.concat(parts).reindex(dates.index)


def load_adopted_pool(mapping_name: str) -> pd.DataFrame:
    """Same construction as scope_weather_collection.py's `pool`, reused
    here rather than re-derived differently, so the three denominators below
    are directly comparable to the original run's row_tier_counts."""
    out = ROOT / 'output'
    manifest = json.loads((out / f'{ATTRIBUTION_RUN}_manifest.json').read_text())
    row_path = ROOT / manifest['row_evidence']
    if digest(row_path) != manifest['row_sha256']:
        raise ValueError(f'{row_path} does not match {ATTRIBUTION_RUN}\'s recorded hash')
    attribution = pd.read_csv(row_path, low_memory=False)[
        ['ID', 'row_position', 'status', 'attributed_date']]
    adopted = attribution.loc[attribution.status.eq('complete_single_candidate_year')].copy()

    raw = pd.read_csv(ROOT / 'data/train.csv', usecols=[
        'ID', 'Origin_Airport', 'Destination_Airport', 'Estimated_Departure_Time'])
    pool = adopted.merge(raw, on='ID', how='left', validate='one_to_one')
    if len(pool) != len(adopted):
        raise ValueError('merge changed row count')

    mapping = pd.read_csv(out / f'{mapping_name}_mapping_table.csv')
    lookup = mapping.set_index('iata')
    pool['origin_tier'] = pool.Origin_Airport.map(lookup.verification_tier)
    pool['dest_tier'] = pool.Destination_Airport.map(lookup.verification_tier)
    if pool[['origin_tier', 'dest_tier']].isna().any().any():
        missing = sorted(set(pool.loc[pool.origin_tier.isna(), 'Origin_Airport'])
                         | set(pool.loc[pool.dest_tier.isna(), 'Destination_Airport']))
        raise ValueError(f'airports missing from mapping table: {missing}')
    TIER_RANK = {'confirmed_period': 0, 'confirmed_current_only': 1,
                'tz_conflict_needs_resolution': 2, 'unconfirmed': 3}
    pool['row_tier'] = pool.apply(
        lambda r: max((r.origin_tier, r.dest_tier), key=lambda t: TIER_RANK[t]), axis=1)
    pool['origin_timezone'] = pool.Origin_Airport.map(lookup.mwgg_tz)
    pool['origin_station'] = pool.Origin_Airport.map(lookup.candidate_sid)
    pool['destination_station'] = pool.Destination_Airport.map(lookup.candidate_sid)
    pool['prediction_at'] = compute_prediction_at(
        pd.to_datetime(pool.attributed_date), pool.Estimated_Departure_Time, pool.origin_timezone
    ) - pd.Timedelta(minutes=60)
    return pool


# ---- interval merging ----

def merge_intervals(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Standard sweep-line merge of closed intervals sorted by start; adjacent
    or overlapping intervals collapse into one, so the returned list's total
    duration is the TRUE union length, not a sum that double-counts overlap."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def subtract_intervals(target: list[tuple[pd.Timestamp, pd.Timestamp]],
                       covered: list[tuple[pd.Timestamp, pd.Timestamp]]
                       ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Interval difference target - covered, by ACTUAL overlap -- a station
    already queried for part of a day is not treated as fully covered for
    that day just because the (station, day) key matches."""
    if not covered:
        return list(target)
    covered = merge_intervals(covered)
    result = []
    for t_start, t_end in target:
        pieces = [(t_start, t_end)]
        for c_start, c_end in covered:
            next_pieces = []
            for p_start, p_end in pieces:
                if c_end <= p_start or c_start >= p_end:
                    next_pieces.append((p_start, p_end))
                    continue
                if c_start > p_start:
                    next_pieces.append((p_start, c_start))
                if c_end < p_end:
                    next_pieces.append((c_end, p_end))
            pieces = next_pieces
        result.extend(p for p in pieces if p[1] > p[0])
    return result


def split_for_request_cap(intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
                          max_hours_per_request: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Splits any merged interval longer than max_hours_per_request into
    consecutive sub-requests no longer than that bound, so the request count
    below is sized to the provider's practical per-request byte cap instead
    of assuming exactly one request per calendar day."""
    out = []
    for start, end in intervals:
        span_hours = (end - start).total_seconds() / 3600
        if span_hours <= max_hours_per_request or max_hours_per_request <= 0:
            out.append((start, end))
            continue
        n_pieces = int(np.ceil(span_hours / max_hours_per_request))
        step = (end - start) / n_pieces
        cur = start
        for _ in range(n_pieces):
            nxt = min(cur + step, end)
            out.append((cur, nxt))
            cur = nxt
    return out


# ---- existing validated cache windows, by station ----

def load_validated_cache_windows(out: Path) -> tuple[dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]], list[dict]]:
    """Reads both existing fetch manifests, RE-VALIDATES every cache file's
    content hash (never trusts the manifest's claim blindly), and returns
    each station's list of actually-fetched windows plus the per-group
    (hours, bytes) sample used for the byte/time-rate estimate below."""
    windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    samples = []
    for run_name in EXISTING_FETCH_RUNS:
        manifest = json.loads((out / f'{run_name}_fetch_manifest.json').read_text())
        for e in manifest['requests']:
            cache_path = ROOT / e['cache_file']
            if not cache_path.exists():
                continue
            if digest(cache_path) != e['sha256']:
                raise ValueError(f'{cache_path} content no longer matches {run_name} fetch manifest hash')
            start = pd.Timestamp(e['window_start_utc'])
            end = pd.Timestamp(e['window_end_utc'])
            windows.setdefault(e['station'], []).append((start, end))
            hours = (end - start).total_seconds() / 3600
            if hours > 0:
                samples.append({'run': run_name, 'station': e['station'], 'hours': hours,
                                'bytes': cache_path.stat().st_size, 'rows': e.get('rows')})
    return windows, samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--mapping-name', required=True)
    args = ap.parse_args()
    if not args.name.startswith('baseline_recovery_v2') or Path(args.name).name != args.name:
        raise ValueError('Invalid output name')
    out = ROOT / 'output'
    if list(out.glob(args.name + '_refined_scope*')):
        raise FileExistsError('Use a fresh --name; existing evidence is preserved')

    pool = load_adopted_pool(args.mapping_name)

    # ---- denominators ----
    total_rows = len(pool)
    mapping_eligible = pool.row_tier.eq('confirmed_period')
    utc_resolved = mapping_eligible & pool.prediction_at.notna()
    denominators = {
        'total_adopted_rows': int(total_rows),
        'mapping_eligible_rows_confirmed_period_both_ends': int(mapping_eligible.sum()),
        'utc_time_resolved_rows': int(utc_resolved.sum()),
        'mapping_eligible_but_utc_unresolved_rows': int((mapping_eligible & ~utc_resolved).sum()),
        'not_mapping_eligible_rows': int((~mapping_eligible).sum()),
    }
    safe = pool.loc[utc_resolved]

    # ---- per-row fixed-padding intervals, grouped by station ----
    per_station_intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    original_padded_seconds_total = 0.0
    pad = pd.Timedelta(hours=LOOKBACK_HOURS) + pd.Timedelta(hours=LOOKAHEAD_HOURS)
    for sid, pred in list(zip(safe.origin_station, safe.prediction_at)) + \
                      list(zip(safe.destination_station, safe.prediction_at)):
        start = pred - pd.Timedelta(hours=LOOKBACK_HOURS)
        end = pred + pd.Timedelta(hours=LOOKAHEAD_HOURS)
        per_station_intervals.setdefault(sid, []).append((start, end))
        original_padded_seconds_total += pad.total_seconds()

    cached_windows, cache_samples = load_validated_cache_windows(out)

    per_station_report = []
    total_merged_seconds = 0.0
    total_new_needed_seconds = 0.0
    total_cache_covered_seconds = 0.0
    all_new_intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for sid, intervals in per_station_intervals.items():
        merged = merge_intervals(intervals)
        merged_seconds = sum((e - s).total_seconds() for s, e in merged)
        new_needed = subtract_intervals(merged, cached_windows.get(sid, []))
        new_seconds = sum((e - s).total_seconds() for s, e in new_needed)
        covered_seconds = merged_seconds - new_seconds
        total_merged_seconds += merged_seconds
        total_new_needed_seconds += new_seconds
        total_cache_covered_seconds += covered_seconds
        all_new_intervals[sid] = new_needed
        per_station_report.append({
            'station': sid, 'rows_needing_station': len(intervals),
            'original_padded_hours': sum((e - s).total_seconds() for s, e in intervals) / 3600,
            'merged_union_hours': merged_seconds / 3600,
            'already_cache_covered_hours': covered_seconds / 3600,
            'newly_needed_hours': new_seconds / 3600,
            'has_real_sample_for_rate_estimate': sid in {c['station'] for c in cache_samples},
        })

    # ---- request-count sizing: split by the provider's practical per-request byte cap ----
    rate_df = pd.DataFrame(cache_samples)
    rate_df['bytes_per_hour'] = rate_df.bytes / rate_df.hours
    # drop degenerate near-zero-length windows from the RATE estimate (they inflate bytes/hour without
    # being representative of a real multi-hour request), but keep them in the request-count accounting
    rate_sample = rate_df.loc[rate_df.hours >= 0.5, 'bytes_per_hour']
    bytes_per_hour_stats = {
        'n_groups': int(len(rate_sample)),
        'min': float(rate_sample.min()) if len(rate_sample) else None,
        'p10': float(rate_sample.quantile(0.10)) if len(rate_sample) else None,
        'median': float(rate_sample.median()) if len(rate_sample) else None,
        'p90': float(rate_sample.quantile(0.90)) if len(rate_sample) else None,
        'max': float(rate_sample.max()) if len(rate_sample) else None,
    }
    max_hours_per_request = (MAX_RESPONSE_BYTES / bytes_per_hour_stats['median']
                             if bytes_per_hour_stats['median'] else None)

    total_new_requests = 0
    for sid, new_needed in all_new_intervals.items():
        pieces = split_for_request_cap(new_needed, max_hours_per_request) if max_hours_per_request else new_needed
        total_new_requests += len(pieces)

    stations_needing_new_collection = sorted(sid for sid, iv in all_new_intervals.items() if iv)
    stations_with_no_rate_sample = sorted(
        set(stations_needing_new_collection) - {c['station'] for c in cache_samples})

    # ---- byte/time range estimates for the NEWLY needed time only ----
    byte_estimate_range = None
    if bytes_per_hour_stats['p10'] and bytes_per_hour_stats['p90']:
        new_hours = total_new_needed_seconds / 3600
        byte_estimate_range = {
            'basis': f'{bytes_per_hour_stats["n_groups"]} real fetched (station, window) groups across '
                    f'{EXISTING_FETCH_RUNS}, each >= 30min long; bytes/hour p10..p90 applied to the '
                    'NEWLY needed hours only (already-cache-covered hours are excluded)',
            'newly_needed_hours': new_hours,
            'low_bytes_p10_rate': bytes_per_hour_stats['p10'] * new_hours,
            'median_bytes': bytes_per_hour_stats['median'] * new_hours,
            'high_bytes_p90_rate': bytes_per_hour_stats['p90'] * new_hours,
            'is_measured_at_this_scale': False,
        }

    rows_per_hour = (rate_df.rows / rate_df.hours).replace([np.inf, -np.inf], np.nan).dropna()
    raw_bytes_per_row = float(rate_df.bytes.sum() / rate_df.rows.sum()) if rate_df.rows.sum() else None
    diagnostic_path = out / 'baseline_recovery_v2_weather_expanded_diagnostic_20260918_diagnostic_manifest.json'
    normalized_bytes_per_row = None
    if diagnostic_path.exists():
        prior_diag = json.loads(diagnostic_path.read_text())
        normalized_bytes_per_row = prior_diag.get('refined_collection_size_estimate', {}).get(
            'measured_normalized_pandas_bytes_per_observation_row')

    # time: no verified per-request timing log exists at this scale (see limitations); present a range
    # anchored on the ORIGINAL 21-row probe's measured seconds/request together with the NEW mandatory
    # per-success politeness pause (3s) this round's fetch fix added, not a single point value
    from notebooks.fetch_weather_sample_expanded import SUCCESS_PAUSE_SECONDS
    measured_seconds_per_request_probe = 3.21  # from the 2026-09-18 21-row/40-request probe (unchanged)
    time_estimate_range = {
        'basis': 'ASSUMPTION-based: no verified per-request timing log exists at this scale (prior runs '
                'predate the per-attempt timing fix, or were never re-run under it); anchored on the '
                '2026-09-18 21-row probe\'s measured 3.21s/request',
        'total_new_requests': total_new_requests,
        'low_seconds_sequential_no_pause': total_new_requests * measured_seconds_per_request_probe,
        'high_seconds_sequential_with_success_pause': total_new_requests * (
            measured_seconds_per_request_probe + SUCCESS_PAUSE_SECONDS),
        'is_measured_at_this_scale': False,
    }

    coverage_summary = pd.DataFrame(per_station_report)
    coverage_summary.to_csv(out / f'{args.name}_refined_scope_per_station.csv', index=False)

    manifest_out = {
        'name': args.name, 'attribution_run': ATTRIBUTION_RUN, 'mapping_name': args.mapping_name,
        'existing_fetch_runs_subtracted': EXISTING_FETCH_RUNS,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False, 'uses_external_data': False, 'target_columns_used': [],
        'denominators': denominators,
        'critique_of_prior_estimate': (
            'The prior baseline_recovery_v2_weather_scope_20260918 estimate was '
            '(sample total bytes / sample total hours) x (sample total hours / sample group count) x '
            'full group count, which reduces algebraically to sample mean bytes/group x full group count. '
            'It does not resolve any short-window-vs-long-window difference; this run replaces it with '
            'actual per-station merged query intervals.'),
        'padding_hours': {'lookback': LOOKBACK_HOURS, 'lookahead': LOOKAHEAD_HOURS},
        'original_padded_total_hours_no_merge': original_padded_seconds_total / 3600,
        'merged_union_total_hours': total_merged_seconds / 3600,
        'already_cache_covered_hours': total_cache_covered_seconds / 3600,
        'newly_needed_hours': total_new_needed_seconds / 3600,
        'distinct_stations_in_scope': len(per_station_intervals),
        'distinct_stations_needing_new_collection': len(stations_needing_new_collection),
        'stations_with_no_real_rate_sample': stations_with_no_rate_sample,
        'stations_with_no_real_rate_sample_count': len(stations_with_no_rate_sample),
        'request_cap_bytes': MAX_RESPONSE_BYTES,
        'max_hours_per_request_at_median_rate': max_hours_per_request,
        'total_new_requests_after_size_based_splitting': total_new_requests,
        'bytes_per_hour_stats_from_real_groups': bytes_per_hour_stats,
        'byte_estimate_range': byte_estimate_range,
        'raw_bytes_per_observation_row': raw_bytes_per_row,
        'normalized_pandas_bytes_per_observation_row': normalized_bytes_per_row,
        'note_raw_vs_normalized': 'raw = CSV bytes on disk as downloaded; normalized = pandas in-memory '
                                  'footprint per parsed row (deep=True) after loading -- these are '
                                  'different quantities and neither is "disk space needed after processing".',
        'time_estimate_range': time_estimate_range,
        'per_station_table': str((out / f'{args.name}_refined_scope_per_station.csv').relative_to(ROOT)),
        'limitations': [
            'Denominators distinguish total adopted rows, mapping-eligible (confirmed_period both ends) '
            'rows, and UTC-time-resolved rows among those -- station-day sizing below covers only the '
            'last group (utc_time_resolved_rows), matching the row-level join policy already in place.',
            'A station/day pair already appearing in an existing fetch manifest is NOT assumed to cover '
            'the full calendar day; coverage is subtracted by ACTUAL interval overlap against the '
            'validated window_start_utc/window_end_utc recorded for that fetch.',
            'The byte-rate sample excludes real groups shorter than 30 minutes (their bytes/hour ratio is '
            'dominated by fixed per-response overhead, not sustained transfer rate) from the RATE estimate '
            'only; they are still counted in cache-coverage subtraction and request-count accounting.',
            f'{len(stations_with_no_rate_sample)} station(s) needing new collection have NO real fetched '
            'group in either existing run to anchor a bytes/hour rate on; their byte estimate falls back '
            'to the cross-station median rate, which may not reflect that station\'s actual report '
            'frequency -- see stations_with_no_real_rate_sample.',
            'Execution time has no verified per-request timing log at this scale; time_estimate_range is '
            'an assumption-anchored RANGE (on the 21-row probe\'s measured rate plus the mandatory '
            'per-success politeness pause), not a measured or guaranteed duration.',
            'This is a local sizing exercise: no new network collection, no new stratified sample, and no '
            'model retraining is performed by this script.',
        ],
    }
    (out / f'{args.name}_refined_scope_manifest.json').write_text(json.dumps(manifest_out, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest_out.items() if k != 'limitations'}, indent=2, default=str))


if __name__ == '__main__':
    main()
