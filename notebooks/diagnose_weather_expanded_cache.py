"""Re-diagnose the ALREADY-COLLECTED 300-row expanded sample using only its
existing cache and join outputs (baseline_recovery_v2_weather_expanded_20260918)
-- no new network calls, no new fetch, no re-selection. This exists to answer
the questions the original run's summary CSVs did not: fixed-300 vs
collectible-only denominators side by side, per-role/year/season/airport
match rates and observation-age distributions, WHY an unmatched row is
unmatched (mapping vs never-attempted-collection vs no-report-at-that-time vs
stale), field-level missing/trace/quality flags on the actually-downloaded
observations, and a refined full-population collection-size estimate that
uses each REAL group's own window length (hours) instead of assuming every
station-day request is the same size as the 21-row probe's short windows.

Reads only: the 300-row run's selection_with_prediction_at.csv, its fetch
manifest, its four joined_latency{0,10,30,60}min.csv files, and (for the
refined size estimate) the row-level date attribution + mapping table that
scope_weather_collection.py already used. No target/delay/actual-time column
is read anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION_RUN = 'baseline_recovery_v2_row_date_attribution_20260918'
LOOKBACK_HOURS = 6
LOOKAHEAD_HOURS = 2
LATENCIES = [0, 10, 30, 60]
WEATHER_FIELDS = ['tmpf', 'dwpf', 'relh', 'sknt', 'gust', 'vsby', 'p01i', 'skyc1', 'wxcodes', 'snowdepth']
NUMERIC_WEATHER_FIELDS = {'tmpf', 'dwpf', 'relh', 'sknt', 'gust', 'vsby', 'p01i', 'snowdepth'}
CATEGORICAL_WEATHER_FIELDS = {'skyc1', 'wxcodes'}  # free-text codes -- a numeric-parse check is meaningless here


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def age_percentiles(series: pd.Series) -> dict:
    s = series.dropna()
    if not len(s):
        return {'n': 0, 'p10': None, 'p25': None, 'median': None, 'p75': None, 'p90': None, 'max': None}
    q = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return {'n': int(len(s)), 'p10': float(q[0.10]), 'p25': float(q[0.25]), 'median': float(q[0.50]),
           'p75': float(q[0.75]), 'p90': float(q[0.90]), 'max': float(s.max())}


def diagnose_stale_vs_unavailable_vs_absent(station_col: pd.Series, prediction_at: pd.Series,
                                            obs: pd.DataFrame, latency_minutes: float,
                                            max_age_minutes: float = 90) -> pd.Series:
    """For rows that are collectible, resolved, not a collection failure, and
    still unmatched, distinguishes WHY by looking at the actual cached
    observations for that station -- ignoring the join's own availability and
    max-age filters, to find the most recent report with observed_at <=
    prediction_at:
      - no report exists with observed_at <= prediction_at at all in the
        cached window -> a genuine absence, not a latency or staleness issue.
      - such a report exists but (observed_at + latency_minutes) >
        prediction_at -> it exists and is not stale, but is not yet
        AVAILABLE under this specific latency ASSUMPTION (a different, larger
        assumed latency would still exclude it; a smaller one might not).
      - such a report exists and is available under this latency, but its
        age (prediction_at - observed_at) exceeds max_age_minutes -> genuinely
        stale/aged out.
    Lumping these into one "no report" bucket would blur an assumption
    artifact (latency) with a real data-absence fact and a real staleness fact.
    """
    reason = pd.Series('no_report_observed_before_prediction_time', index=station_col.index, dtype=object)
    valid = station_col.notna() & prediction_at.notna()
    if not valid.any() or not len(obs):
        return reason

    # pd.merge_asof (like a regular merge) does NOT preserve the left frame's index in its output -- the
    # original row labels are carried through as an explicit column instead, then restored via set_index
    # after the merge, so alignment back into `reason` never depends on row order surviving the merge/sort.
    req = pd.DataFrame({'station': station_col.loc[valid].astype(object),
                        'prediction_at': prediction_at.loc[valid]}).reset_index(names='__orig_index')
    obs_sorted = obs[['station', 'observed_at']].dropna().assign(
        station=lambda d: d.station.astype(object)).sort_values('observed_at')
    left = req.sort_values('prediction_at')
    matched = pd.merge_asof(left, obs_sorted, by='station', left_on='prediction_at',
                            right_on='observed_at', direction='backward').set_index('__orig_index')

    has_candidate = matched.observed_at.notna()
    observed_at = pd.to_datetime(matched.observed_at, utc=True)
    pred = pd.to_datetime(matched.prediction_at, utc=True)
    available_at = observed_at + pd.Timedelta(minutes=latency_minutes)
    age_minutes = (pred - observed_at).dt.total_seconds() / 60
    not_yet_available = has_candidate & (available_at > pred)
    stale = has_candidate & ~not_yet_available & (age_minutes > max_age_minutes)
    would_have_matched = has_candidate & ~not_yet_available & ~stale

    out = pd.Series('no_report_observed_before_prediction_time', index=matched.index, dtype=object)
    out[not_yet_available] = 'not_yet_available_under_latency_assumption'
    out[stale] = 'stale_beyond_max_age'
    # should never occur -- a candidate that is both available and within max_age should have matched
    # in the join itself; surfaced explicitly rather than silently mislabeled as an absence
    out[would_have_matched] = 'unexpected_unmatched_despite_available_report'
    reason.loc[out.index] = out
    return reason


def classify_unmatched(row_matched: pd.Series, collectible: pd.Series, unresolved: pd.Series,
                       failed_keys: set, station_col: pd.Series, day_col: pd.Series,
                       prediction_at: pd.Series, obs: pd.DataFrame, latency_minutes: float) -> pd.Series:
    """Reasoned category for every request, not just a match/no-match bit.
    A collection failure is connected to the SPECIFIC (station, day) that
    failed, not propagated to every row that happens to share the same
    station on a different, successfully-collected day."""
    reason = pd.Series('matched', index=row_matched.index)
    reason[~collectible] = 'not_collectible_mapping'
    reason[unresolved & collectible] = 'unresolved_prediction_at'
    station_day = pd.Series(list(zip(station_col, day_col)), index=row_matched.index)
    never_attempted = collectible & ~unresolved & station_day.isin(failed_keys) & ~row_matched
    reason[never_attempted] = 'collection_failed_or_not_attempted'
    remaining = collectible & ~unresolved & ~row_matched & ~never_attempted
    if remaining.any():
        detailed = diagnose_stale_vs_unavailable_vs_absent(
            station_col.loc[remaining], prediction_at.loc[remaining], obs, latency_minutes)
        reason.loc[remaining] = detailed
    reason[row_matched] = 'matched'
    return reason


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='the ALREADY-COLLECTED run to re-diagnose '
                    '(e.g. baseline_recovery_v2_weather_expanded_20260918)')
    ap.add_argument('--out-name', required=True, help='output name for this diagnostic pass; must be fresh')
    ap.add_argument('--mapping-name', required=True)
    args = ap.parse_args()
    out = ROOT / 'output'
    if list(out.glob(args.out_name + '_diagnostic*')):
        raise FileExistsError('Use a fresh --out-name; existing evidence is preserved')

    selection = pd.read_csv(out / f'{args.name}_selection_with_prediction_at.csv')
    fetch_manifest = json.loads((out / f'{args.name}_fetch_manifest.json').read_text())
    # connected to the SPECIFIC (station, day) that failed, not the station alone -- a failure on one
    # day must not be propagated to every other, successfully-collected day for that same station
    failed_keys = {(f['station'], f['day']) for f in fetch_manifest.get('failed', [])}
    unresolved = selection.prediction_at.isna()
    collectible = selection.collectible.astype(bool)

    # ---- 0. load the ACTUAL cached observations once, reused for both the unmatched-reason
    # diagnosis below and the field-quality section further down ----
    cache_files = sorted({ROOT / e['cache_file'] for e in fetch_manifest['requests']})
    obs_frames = [pd.read_csv(p, comment='#', na_values=['M'], keep_default_na=False) for p in cache_files]
    obs = pd.concat(obs_frames, ignore_index=True) if obs_frames else pd.DataFrame(columns=WEATHER_FIELDS)
    if {'station', 'valid'}.issubset(obs.columns):
        obs_for_diag = obs.assign(observed_at=pd.to_datetime(obs['valid'], utc=True))[['station', 'observed_at']]
    else:
        obs_for_diag = pd.DataFrame(columns=['station', 'observed_at'])

    # ---- 1. denominators: fixed 300 vs collectible-only, side by side ----
    denominators = pd.DataFrame([{
        'denominator': 'fixed_300_all_selected_rows', 'rows': int(len(selection))},
        {'denominator': 'collectible_confirmed_period_both_ends', 'rows': int(collectible.sum())},
        {'denominator': 'not_collectible_mapping', 'rows': int((~collectible).sum())},
        {'denominator': 'unresolved_prediction_at_dst', 'rows': int(unresolved.sum())},
    ])
    denominators.to_csv(out / f'{args.out_name}_diagnostic_denominators.csv', index=False)

    # ---- 2/3. per-scenario per-role match rates, age distributions, unmatched-reason breakdown ----
    match_rows, age_rows, reason_rows, year_season_airport_rows = [], [], [], []
    for latency in LATENCIES:
        joined = pd.read_csv(ROOT / 'data/weather_probe' / f'{args.name}_joined' / f'joined_latency{latency}min.csv')
        prediction_at = pd.to_datetime(joined.prediction_at, utc=True)
        day_col = prediction_at.dt.strftime('%Y-%m-%d')
        for role in ('origin', 'destination'):
            matched = joined[f'{role}_observed_at'].notna()
            station_col = joined[f'{role}_station'] if f'{role}_station' in joined else pd.Series(
                [None] * len(joined))
            match_rows.append({
                'latency_minutes': latency, 'role': role,
                'rows_fixed_300': int(len(joined)), 'collectible_rows': int(collectible.sum()),
                'matched_rows': int(matched.sum()),
                'match_rate_of_fixed_300': float(matched.mean()),
                'match_rate_of_collectible': float(matched.sum() / collectible.sum()) if collectible.sum() else None,
            })
            age_rows.append({'latency_minutes': latency, 'role': role,
                             **age_percentiles(joined[f'{role}_weather_age_minutes'])})
            reason = classify_unmatched(matched, collectible, unresolved, failed_keys, station_col, day_col,
                                        prediction_at, obs_for_diag, latency)
            for cat, count in reason.value_counts().items():
                reason_rows.append({'latency_minutes': latency, 'role': role, 'reason': cat, 'rows': int(count)})
        both_matched = joined.origin_observed_at.notna() & joined.destination_observed_at.notna()
        grp = joined.assign(matched_both=both_matched).groupby(
            ['attributed_year', 'season', 'Origin_Airport'], dropna=False).agg(
            rows=('ID', 'size'), collectible=('collectible', 'sum'), matched_both=('matched_both', 'sum'))
        grp['latency_minutes'] = latency
        year_season_airport_rows.append(grp.reset_index())

    pd.DataFrame(match_rows).to_csv(out / f'{args.out_name}_diagnostic_match_rates.csv', index=False)
    pd.DataFrame(age_rows).to_csv(out / f'{args.out_name}_diagnostic_age_distribution.csv', index=False)
    pd.DataFrame(reason_rows).to_csv(out / f'{args.out_name}_diagnostic_unmatched_reasons.csv', index=False)
    pd.concat(year_season_airport_rows, ignore_index=True).to_csv(
        out / f'{args.out_name}_diagnostic_year_season_airport.csv', index=False)

    # ---- 4. field-level missing/trace/quality flags on the ACTUAL cached observations ----
    field_rows = []
    for field in WEATHER_FIELDS:
        if field not in obs.columns:
            continue
        col = obs[field]
        missing_m = col.isna()  # na_values=['M'] already turned literal "M" into NaN on read
        is_categorical = field in CATEGORICAL_WEATHER_FIELDS
        is_trace = (col.astype(str).str.strip().eq('T') if not is_categorical
                   else pd.Series(False, index=col.index))  # trace precip ("T") only applies to p01i
        other_non_numeric = None
        if not is_categorical:
            numeric = pd.to_numeric(col, errors='coerce')
            other_non_numeric = int((numeric.isna() & ~missing_m & ~is_trace).sum())
        field_rows.append({
            'field': field, 'field_type': 'categorical_code' if is_categorical else 'numeric',
            'total_observation_rows': int(len(col)),
            'missing_M': int(missing_m.sum()), 'missing_M_rate': float(missing_m.mean()),
            'trace_T_values': int(is_trace.sum()) if not is_categorical else None,
            'other_non_numeric_values': other_non_numeric,
        })
    field_quality = pd.DataFrame(field_rows)
    field_quality.to_csv(out / f'{args.out_name}_diagnostic_field_quality.csv', index=False)

    # ---- 5. ID/row-count/order integrity across scenarios ----
    # A failure here is a data-integrity precondition for every table already written above being
    # comparable across scenarios; it is raised immediately, not recorded as a boolean and left for the
    # manifest reader to notice a normal-looking exit.
    base_ids = selection.ID.tolist()
    for latency in LATENCIES:
        joined = pd.read_csv(ROOT / 'data/weather_probe' / f'{args.name}_joined' / f'joined_latency{latency}min.csv')
        if joined.ID.tolist() != base_ids:
            raise ValueError(f'joined_latency{latency}min.csv row ID/order does not match the selection; '
                             'refusing to report comparable-looking tables across scenarios')
        if len(joined) != len(selection):
            raise ValueError(f'joined_latency{latency}min.csv has {len(joined)} rows, expected {len(selection)}')
    integrity_ok = True

    # ---- 6. refined collection-size estimate using REAL per-group window hours, not a flat per-request average ----
    manifest_scope = json.loads((out / f'{args.mapping_name}_scope_manifest.json').read_text()) \
        if (out / f'{args.mapping_name}_scope_manifest.json').exists() else None
    real_groups = []
    for e in fetch_manifest['requests']:
        start = pd.Timestamp(e['window_start_utc'])
        end = pd.Timestamp(e['window_end_utc'])
        hours = (end - start).total_seconds() / 3600
        path = ROOT / e['cache_file']
        real_groups.append({'hours': hours, 'bytes': path.stat().st_size if path.exists() else None,
                            'rows': e.get('rows')})
    real_groups = pd.DataFrame(real_groups).dropna(subset=['bytes'])
    total_hours = real_groups.hours.sum()
    total_bytes = real_groups.bytes.sum()
    bytes_per_hour = float(total_bytes / total_hours) if total_hours else None
    raw_bytes_per_obs_row = float(real_groups.bytes.sum() / real_groups.rows.sum()) if real_groups.rows.sum() else None
    normalized_bytes_per_obs_row = float(obs.memory_usage(deep=True).sum() / len(obs)) if len(obs) else None

    refined_estimate = {
        'basis': f'{len(real_groups)} REAL station-day groups actually collected in {args.name} '
                 '(both the 21-row probe scale and this 300-row expanded run), using EACH group\'s own '
                 'recorded window length in hours -- not a single flat bytes-per-request average applied '
                 'uniformly to short (21-row) and long (full-population) windows alike.',
        'measured_groups': int(len(real_groups)),
        'measured_total_window_hours': float(total_hours),
        'measured_total_raw_bytes': int(total_bytes),
        'measured_bytes_per_window_hour': bytes_per_hour,
        'measured_raw_bytes_per_observation_row': raw_bytes_per_obs_row,
        'measured_normalized_pandas_bytes_per_observation_row': normalized_bytes_per_obs_row,
        'note_raw_vs_normalized': 'raw = bytes of the downloaded CSV as stored; normalized = in-memory '
                                  'pandas footprint per parsed observation row (deep=True) -- these differ '
                                  'and neither should be read as "storage needed on disk after processing".',
    }
    if manifest_scope is not None:
        full = manifest_scope['full_collection_estimate_confirmed_period_only']
        station_days_full = full['station_days_for_confirmed_period_rows_only']
        # Without re-deriving every one of the ~142,574 full-population group windows, this reports the
        # SAME group-count from the scope run but multiplies by the OBSERVED average hours/group from the
        # 300-row run (which spans far more airports/seasons than the 21-row probe) instead of a flat
        # bytes-per-request constant -- narrower than a full re-run, but no longer conflating short and
        # long windows into one number.
        avg_hours_per_group = float(real_groups.hours.mean())
        est_total_hours_full = station_days_full * avg_hours_per_group
        refined_estimate['full_population_reestimate'] = {
            'station_day_groups_from_scope_run': station_days_full,
            'assumption': 'full-population groups have the SAME average window-hours as the 300-row '
                          'run\'s real groups (not re-derived per-row here); this is a documented '
                          'assumption, not a new measurement at full scale.',
            'avg_hours_per_group_from_300_row_run': avg_hours_per_group,
            'estimated_total_hours_full_population': est_total_hours_full,
            'estimated_bytes_full_population': (bytes_per_hour * est_total_hours_full
                                                if bytes_per_hour else None),
            'previous_scope_run_flat_estimate_bytes': full.get('extrapolated_bytes'),
        }
    else:
        refined_estimate['full_population_reestimate'] = None

    # ---- 7. existing-cache coverage of the full population's required (station, day) set ----
    all_cached_station_days = set()
    for name in (args.name, 'baseline_recovery_v2_weather_sample_20260918'):
        fm_path = out / f'{name}_fetch_manifest.json'
        if fm_path.exists():
            fm = json.loads(fm_path.read_text())
            for e in fm.get('requests', []):
                if 'day' in e:
                    all_cached_station_days.add((e['station'], e['day']))
    coverage_note = {
        'distinct_station_day_groups_already_cached_locally': len(all_cached_station_days),
        'source_runs_checked': [args.name, 'baseline_recovery_v2_weather_sample_20260918'],
        'limitation': 'This counts groups already cached from prior VALIDATION runs only (21-row + '
                      '300-row); it is not a claim about coverage of the full 142,574-group population, '
                      'which was never attempted at that scale.',
    }

    manifest_out = {
        'name': args.out_name, 'source_run': args.name, 'mapping_name': args.mapping_name,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False, 'uses_external_data': False, 'target_columns_used': [],
        'id_row_count_order_preserved_across_all_scenarios': integrity_ok,
        'denominators_table': str((out / f'{args.out_name}_diagnostic_denominators.csv').relative_to(ROOT)),
        'match_rates_table': str((out / f'{args.out_name}_diagnostic_match_rates.csv').relative_to(ROOT)),
        'age_distribution_table': str((out / f'{args.out_name}_diagnostic_age_distribution.csv').relative_to(ROOT)),
        'unmatched_reasons_table': str((out / f'{args.out_name}_diagnostic_unmatched_reasons.csv').relative_to(ROOT)),
        'year_season_airport_table': str(
            (out / f'{args.out_name}_diagnostic_year_season_airport.csv').relative_to(ROOT)),
        'field_quality_table': str((out / f'{args.out_name}_diagnostic_field_quality.csv').relative_to(ROOT)),
        'refined_collection_size_estimate': refined_estimate,
        'existing_cache_coverage': coverage_note,
        'limitations': [
            'This is a re-diagnosis of ALREADY-COLLECTED data (no new fetch, no new selection); it '
            'cannot recover information about requests that were never made.',
            'unmatched_reasons splits collectible-and-resolved-but-unmatched rows into '
            '"collection_failed_or_not_attempted" (that SPECIFIC (station, day) appears in the fetch '
            'manifest\'s failed list -- a failure is never propagated to other, successfully-collected '
            'days for the same station), then, among the rest, three further categories computed by '
            'looking at the actual cached observations rather than only the join output: '
            '"no_report_observed_before_prediction_time" (no report at all with observed_at <= '
            'prediction_at exists in the cache for that station), '
            '"not_yet_available_under_latency_assumption" (a report DOES exist and is not stale, but '
            'observed_at + this scenario\'s assumed latency is still after prediction_at -- an artifact '
            'of the latency ASSUMPTION, not a fact about data availability or age), and '
            '"stale_beyond_max_age" (a report exists and is available under this latency, but its age '
            'exceeds the 90-minute cap). For this particular run, failed was empty, so no unmatched row '
            'here is a collection failure.',
            'trace_T_values counts IEM\'s literal "T" (trace precipitation) encoding in p01i, distinct '
            'from "M" (missing) -- neither is a numeric 0, and a feature pipeline treating "T" as missing '
            'or as 0.0 would be making two different, unverified modeling choices.',
        ],
    }
    (out / f'{args.out_name}_diagnostic_manifest.json').write_text(json.dumps(manifest_out, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest_out.items() if k != 'limitations'}, indent=2, default=str))


if __name__ == '__main__':
    main()
