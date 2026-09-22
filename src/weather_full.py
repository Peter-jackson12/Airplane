"""Cache-only, bounded-memory full weather join. No target or model code.

Each bulk CSV is read/hash-checked/parsed once. UTC months retain a 90-minute
boundary tail plus the last report per station (for stale/absent diagnosis).
All four assumed latencies reuse that month's observations. Sorted disk
partitions are heap-merged back into original flight order, never all loaded
into RAM. A returned result is not a claim of measured publication latency.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import csv
import gzip
import hashlib
import heapq
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.weather import join_weather_asof

WEATHER_FIELDS = ['tmpf', 'dwpf', 'relh', 'sknt', 'gust', 'vsby', 'p01i',
                  'skyc1', 'wxcodes', 'snowdepth', 'metar']
NUMERIC_FIELDS = ['tmpf', 'dwpf', 'relh', 'sknt', 'gust', 'vsby', 'p01i', 'snowdepth']
LATENCIES = (0, 10, 30, 60)
MAX_AGE_MINUTES = 90
ROLES = {'origin': 'Origin_Airport', 'destination': 'Destination_Airport'}
SOURCE_FIELDS = ['source_request_id', 'source_window_start_utc', 'source_window_end_utc']
OBS_COLUMNS = ['station', 'observed_at', *WEATHER_FIELDS, *SOURCE_FIELDS]
FORBIDDEN = {'delay', 'delayed', 'not_delayed', 'arrdelay', 'depdelay',
             'actual_departure_time', 'actual_arrival_time'}


def require(condition: bool, message: str) -> None:
    # Runtime safety must survive python -O.
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative_path(root: Path, value: str) -> Path:
    require(isinstance(value, str) and bool(value), 'missing relative path')
    p = Path(value.replace('\\', '/'))
    require(not p.is_absolute() and '..' not in p.parts and ':' not in value,
            'unsafe relative path')
    resolved = (root / p).resolve()
    require(resolved.is_relative_to(root.resolve()), 'path escapes repository root')
    return resolved


def empty_observations() -> pd.DataFrame:
    out = pd.DataFrame({c: pd.Series(dtype=object) for c in OBS_COLUMNS})
    for c in ['observed_at', 'source_window_start_utc', 'source_window_end_utc']:
        out[c] = pd.Series(dtype='datetime64[ns, UTC]')
    for c in NUMERIC_FIELDS:
        out[c] = pd.Series(dtype=float)
    return out


def month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(month + '-01T00:00:00Z')
    require(start.strftime('%Y-%m') == month, 'invalid UTC month')
    return start, start + pd.offsets.MonthBegin(1)


def validate_entries(entries: list[dict], root: Path) -> dict[str, list[dict]]:
    by_month: dict[str, list[dict]] = {}
    ids, paths, pairs, networks = set(), set(), set(), {}
    for e in entries:
        rid = e['request_id']
        path = relative_path(root, e['cache_file'])
        require(rid not in ids and path not in paths, 'duplicate request/cache identity')
        ids.add(rid)
        paths.add(path)
        start, end = month_bounds(e['month'])
        require(pd.Timestamp(e['window_start_utc']) == start and
                pd.Timestamp(e['window_end_utc']) == end, 'non-month request window')
        stations = e['stations']
        require(bool(stations) and len(set(stations)) == len(stations), 'invalid station batch')
        for sid in stations:
            require(isinstance(sid, str) and bool(sid.strip()), 'invalid station identity')
            pair = (sid, e['month'])
            require(pair not in pairs, 'duplicate station/month batch')
            pairs.add(pair)
            require(sid not in networks or networks[sid] == e['network'],
                    'station identifier spans multiple networks')
            networks[sid] = e['network']
        require(isinstance(e['sha256'], str) and len(e['sha256']) == 64, 'invalid cache SHA')
        require(int(e['rows']) >= 0, 'invalid cache row count')
        by_month.setdefault(e['month'], []).append(e)
    return {m: sorted(es, key=lambda e: e['request_id']) for m, es in by_month.items()}


def read_cache(entry: dict, root: Path) -> tuple[pd.DataFrame, int]:
    path = relative_path(root, entry['cache_file'])
    require(path.stat().st_size <= 20_000_000, 'cache exceeds fixed transport byte cap')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == entry['sha256'],
            f'cache SHA mismatch: {entry["request_id"]}')
    raw = pd.read_csv(io.BytesIO(data), comment='#', dtype='string',
                      na_values=['M'], keep_default_na=False)
    require({'station', 'valid', *WEATHER_FIELDS}.issubset(raw.columns), 'missing IEM columns')
    require(len(raw) == int(entry['rows']), 'cache row count mismatch')
    require(raw.station.notna().all() and raw.station.isin(entry['stations']).all(),
            'cache station mismatch')
    raw['observed_at'] = pd.to_datetime(raw['valid'], utc=True, errors='raise').astype('datetime64[ns, UTC]')
    require(raw.observed_at.notna().all(), 'missing observation time')
    start, end = pd.Timestamp(entry['window_start_utc']), pd.Timestamp(entry['window_end_utc'])
    require(raw.observed_at.between(start, end, inclusive='both').all(),
            'out-of-window observation')
    for field in NUMERIC_FIELDS:
        raw[field] = pd.to_numeric(raw[field].replace('', pd.NA), errors='raise').astype(float)
        require(np.isfinite(raw[field].dropna()).all(), 'nonfinite weather feature')
    raw['station'] = raw.station.astype(object)
    raw['source_request_id'] = entry['request_id']
    raw['source_window_start_utc'] = start
    raw['source_window_end_utc'] = end
    for col in ['source_window_start_utc', 'source_window_end_utc']:
        raw[col] = raw[col].astype('datetime64[ns, UTC]')
    return raw[OBS_COLUMNS], len(data)


def deduplicate(obs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # A,A,B,B must fail too. Same METAR with conflicting parsed fields also fails.
    distinct = obs.drop_duplicates(['station', 'observed_at', *WEATHER_FIELDS])
    require(not distinct.duplicated(['station', 'observed_at']).any(),
            'conflicting reports at the same station/time')
    out = distinct.sort_values(['observed_at', 'station'], kind='stable').reset_index(drop=True)
    return out, len(obs) - len(out)


def prepare_pool(pool: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require({'ID', 'row_position', 'Origin_Airport', 'Destination_Airport',
             'row_tier', 'prediction_at'}.issubset(pool), 'missing pool columns')
    require(not (FORBIDDEN & {str(c).casefold() for c in pool}), 'target/outcome columns forbidden')
    require('_join_ordinal' not in pool, 'reserved ordinal column')
    require(pool.ID.notna().all() and pool.ID.is_unique, 'ID must be nonmissing and unique')
    position = pd.to_numeric(pool.row_position, errors='raise')
    require(position.notna().all() and position.is_unique and
            position.ge(0).all() and position.eq(position.round()).all(), 'invalid row_position')
    require({'iata', 'candidate_sid', 'matched_network', 'verification_tier'}.issubset(mapping),
            'missing mapping columns')
    require(mapping.iata.notna().all() and mapping.iata.is_unique, 'mapping IATA must be unique')
    lookup = mapping.set_index('iata')
    pool = pool.copy().reset_index(drop=True)
    require(isinstance(pool.prediction_at.dtype, pd.DatetimeTZDtype), 'prediction_at must be aware')
    pool['prediction_at'] = pool.prediction_at.dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
    tiers = [pool[airport].map(lookup.verification_tier) for airport in ROLES.values()]
    require(all(t.notna().all() for t in tiers), 'airport missing from mapping')
    eligible = tiers[0].eq('confirmed_period') & tiers[1].eq('confirmed_period')
    require(eligible.equals(pool.row_tier.eq('confirmed_period')), 'pool/mapping eligibility mismatch')
    active = mapping.loc[mapping.verification_tier.eq('confirmed_period')]
    require(active[['candidate_sid', 'matched_network']].notna().all().all(), 'missing confirmed station')
    require(active.groupby('candidate_sid').matched_network.nunique().le(1).all(),
            'mapping station identifier spans multiple networks')
    pool = pool.drop(columns=['origin_station', 'destination_station'], errors='ignore')
    reserved = {f'{role}_{col}' for role in ROLES
                for col in [*OBS_COLUMNS, 'available_at', 'weather_age_minutes', 'reason']}
    require(not (reserved & set(pool)), 'reserved weather output columns')
    pool.insert(0, '_join_ordinal', range(len(pool)))
    return pool, lookup


def validate_demand(pool: pd.DataFrame, lookup: pd.DataFrame, entries: list[dict]) -> None:
    covered = {(sid, e['month']): e['network'] for e in entries for sid in e['stations']}
    safe = pool.loc[pool.row_tier.eq('confirmed_period') & pool.prediction_at.notna()]
    for airport in ROLES.values():
        station = safe[airport].map(lookup.candidate_sid)
        network = safe[airport].map(lookup.matched_network)
        for times in [safe.prediction_at, safe.prediction_at - pd.Timedelta(minutes=90)]:
            demand = pd.DataFrame({'station': station, 'network': network,
                                   'month': times.dt.strftime('%Y-%m')}).drop_duplicates()
            for sid, net, month in demand.itertuples(index=False, name=None):
                require(covered.get((sid, month)) == net,
                        f'missing/mismatched station-month demand: {sid} {month}')


def latest_before(req: pd.DataFrame, obs: pd.DataFrame) -> pd.Series:
    answer = pd.Series(pd.NaT, index=req.index, dtype='datetime64[ns, UTC]')
    valid = req.station.notna() & req.prediction_at.notna()
    if not valid.any() or obs.empty:
        return answer
    left = req.loc[valid].assign(_row=lambda d: d.index).sort_values('prediction_at')
    right = obs[['station', 'observed_at']].sort_values('observed_at')
    joined = pd.merge_asof(left, right, by='station', left_on='prediction_at',
                           right_on='observed_at', direction='backward').set_index('_row')
    answer.loc[joined.index] = joined.observed_at
    return answer


def join_block(pool: pd.DataFrame, lookup: pd.DataFrame, raw: pd.DataFrame):
    eligible = pool.row_tier.eq('confirmed_period')
    resolved = pool.prediction_at.notna()
    requests, latest = {}, {}
    for role, airport in ROLES.items():
        req = pool[['prediction_at']].copy()
        req['station'] = pool[airport].map(lookup.candidate_sid).where(eligible & resolved).astype(object)
        requests[role] = req
        latest[role] = latest_before(req, raw)
    for latency in LATENCIES:
        obs = raw.assign(available_at=raw.observed_at + pd.Timedelta(minutes=latency))
        result = pool.copy()
        for role, airport in ROLES.items():
            joined = join_weather_asof(requests[role], obs, max_age='90min')
            matched = joined.observed_at.notna()
            pred = pool.prediction_at
            require((~matched | (eligible & resolved)).all(), 'ineligible weather match')
            require(joined.loc[matched, 'station'].equals(
                requests[role].loc[matched, 'station']), 'station identity changed')
            require((joined.loc[matched, 'observed_at'] <= pred[matched]).all(), 'future observation')
            require((joined.loc[matched, 'available_at'] <= pred[matched]).all(), 'future availability')
            require(joined.loc[matched, 'weather_age_minutes'].between(0, 90).all(), 'invalid observation age')
            require(joined.loc[matched, 'observed_at'].between(
                joined.loc[matched, 'source_window_start_utc'],
                joined.loc[matched, 'source_window_end_utc']).all(), 'matched out-of-window observation')
            age = (pred - latest[role]).dt.total_seconds().div(60)
            reason = pd.Series('matched', index=pool.index, dtype=object)
            reason.loc[~matched & eligible & resolved & latest[role].isna()] = 'no_report_before_prediction_in_collected_windows'
            reason.loc[~matched & eligible & resolved & age.gt(90)] = 'stale_beyond_90_minutes'
            reason.loc[~matched & eligible & resolved & age.le(90)] = 'not_yet_available_under_latency_assumption'
            reason.loc[eligible & ~resolved] = 'unresolved_prediction_at'
            reason.loc[~eligible] = 'not_mapping_eligible'
            require(not (~matched & reason.eq('matched')).any(), 'unclassified unmatched row')
            require(not (~matched & eligible & resolved & age.between(latency, 90)).any(),
                    'unexpected unmatched despite available report')
            joined = joined.drop(columns='prediction_at').add_prefix(role + '_')
            joined[role + '_reason'] = reason
            result = pd.concat([result, joined], axis=1)
        require(result.ID.tolist() == pool.ID.tolist(), 'block row identity changed')
        yield latency, result


class Diagnostics:
    def __init__(self):
        self.counts = {lat: Counter() for lat in LATENCIES}
        self.roles = {(lat, role): {'reasons': Counter(), 'missing': Counter(),
                                   'matched_missing': Counter(), 'blank': Counter(), 'ages': []}
                      for lat in LATENCIES for role in ROLES}

    def add(self, latency: int, frame: pd.DataFrame) -> None:
        masks = {role: frame[f'{role}_observed_at'].notna() for role in ROLES}
        origin, dest = masks['origin'], masks['destination']
        self.counts[latency].update(rows=len(frame), origin_matched=int(origin.sum()),
                                   destination_matched=int(dest.sum()), both_matched=int((origin & dest).sum()),
                                   one_matched=int((origin ^ dest).sum()), none_matched=int((~origin & ~dest).sum()))
        for role, matched in masks.items():
            rec = self.roles[latency, role]
            rec['reasons'].update(frame[f'{role}_reason'].value_counts().to_dict())
            rec['ages'].append(frame.loc[matched, f'{role}_weather_age_minutes'].to_numpy(dtype=float))
            for field in WEATHER_FIELDS:
                values = frame[f'{role}_{field}']
                rec['missing'][field] += int(values.isna().sum())
                rec['matched_missing'][field] += int((values.isna() & matched).sum())
                rec['blank'][field] += int((values.eq('').fillna(False) & matched).sum())

    def finish(self) -> list[dict]:
        results = []
        previous = None
        for lat in LATENCIES:
            counts = dict(self.counts[lat])
            require(counts['both_matched'] + counts['one_matched'] + counts['none_matched'] == counts['rows'],
                    'coverage denominator mismatch')
            if previous is not None:
                require(all(counts[f'{r}_matched'] <= previous[f'{r}_matched'] for r in ROLES),
                        'coverage increased with greater fixed latency')
            detail = {}
            for role in ROLES:
                rec = self.roles[lat, role]
                age = np.concatenate(rec['ages']) if rec['ages'] else np.array([])
                quantiles = [0, .1, .25, .5, .75, .9, .95, .99, 1]
                require(sum(rec['reasons'].values()) == counts['rows'], 'reason denominator mismatch')
                detail[role] = {
                    'reasons': dict(sorted(rec['reasons'].items())),
                    'match_rate_all_rows': counts[f'{role}_matched'] / counts['rows'] if counts['rows'] else None,
                    'age_quantiles_minutes': {str(q): float(v) for q, v in zip(quantiles, np.quantile(age, quantiles))} if len(age) else {},
                    'feature_missing_all_rows': dict(rec['missing']),
                    'feature_missing_matched_rows': dict(rec['matched_missing']),
                    'feature_blank_matched_rows': dict(rec['blank']),
                }
            results.append({'latency_minutes': lat, **counts, 'roles': detail})
            previous = counts
        return results


def merge_parts(parts: list[Path], target: Path, expected_ids: list[str]) -> dict:
    with ExitStack() as stack:
        readers = [csv.reader(stack.enter_context(p.open(encoding='utf-8', newline=''))) for p in parts]
        headers = [next(reader) for reader in readers]
        require(bool(headers) and all(h == headers[0] for h in headers), 'partition schema mismatch')
        header = headers[0]
        require(header[0] == '_join_ordinal' and len(set(header)) == len(header), 'invalid partition schema')
        id_col = header.index('ID')
        stream = stack.enter_context(target.with_suffix(target.suffix + '.part').open('xb'))
        zipped = stack.enter_context(gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0))
        text = stack.enter_context(io.TextIOWrapper(zipped, encoding='utf-8', newline=''))
        writer = csv.writer(text, lineterminator='\n')
        writer.writerow(header[1:])
        count = 0
        for ordinal, row in enumerate(heapq.merge(*readers, key=lambda r: int(r[0]))):
            require(ordinal < len(expected_ids) and int(row[0]) == ordinal and row[id_col] == expected_ids[ordinal],
                    'output row identity/order/uniqueness mismatch')
            writer.writerow(row[1:])
            count += 1
        require(count == len(expected_ids), 'output denominator mismatch')
    target.with_suffix(target.suffix + '.part').replace(target)
    return {'rows': count, 'sha256': digest(target), 'bytes': target.stat().st_size}


def run_partitioned(pool: pd.DataFrame, mapping: pd.DataFrame, entries: list[dict],
                    *, root: Path, local: Path) -> dict:
    pool, lookup = prepare_pool(pool, mapping)
    by_month = validate_entries(entries, root)
    validate_demand(pool, lookup, entries)
    local.mkdir(parents=True, exist_ok=False)
    parts_dir = local / 'parts'
    parts_dir.mkdir()
    labels = pool.prediction_at.dt.strftime('%Y-%m')
    carry = empty_observations()
    parts = {lat: [] for lat in LATENCIES}
    diagnostics = Diagnostics()
    stats = {'cache_files_read': 0, 'cache_rows_read': 0, 'cache_bytes_read': 0,
             'dropped_exact_duplicate_reports': 0, 'peak_partition_observations': 0,
             'peak_partition_observation_memory_bytes': 0}
    months = sorted(set(by_month) | set(labels.dropna()))
    for block_no, month in enumerate([*months, None]):
        if month is None:
            rows, raw = pool.loc[labels.isna()], empty_observations()
        else:
            _, end = month_bounds(month)
            frames = [carry]
            for entry in by_month.get(month, []):
                loaded, size = read_cache(entry, root)
                frames.append(loaded)
                stats['cache_files_read'] += 1
                stats['cache_rows_read'] += len(loaded)
                stats['cache_bytes_read'] += size
            raw, dropped = deduplicate(pd.concat(frames, ignore_index=True))
            stats['dropped_exact_duplicate_reports'] += dropped
            stats['peak_partition_observations'] = max(stats['peak_partition_observations'], len(raw))
            stats['peak_partition_observation_memory_bytes'] = max(
                stats['peak_partition_observation_memory_bytes'], int(raw.memory_usage(deep=True).sum()))
            # Keep an old last report for diagnosis, never as permission to bypass max_age.
            tail = raw.observed_at.ge(end - pd.Timedelta(minutes=90))
            last = ~raw.station.duplicated(keep='last')
            carry = raw.loc[tail | last].copy()
            rows = pool.loc[labels.eq(month)]
            del frames
        # Empty final block supplies headers for even an all-empty/all-unresolved pool.
        for latency, joined in join_block(rows, lookup, raw):
            diagnostics.add(latency, joined)
            path = parts_dir / f'{block_no:04d}_latency{latency}.csv'
            joined.sort_values('_join_ordinal').to_csv(path, index=False, lineterminator='\n', float_format='%.17g')
            parts[latency].append(path)
        print(json.dumps({'month': month, 'flight_rows': len(rows), **stats}), flush=True)
    require(stats['cache_files_read'] == len(entries), 'not all cache files were validated')
    outputs = {}
    ids = pool.ID.astype(str).tolist()
    for latency in LATENCIES:
        target = local / f'joined_latency{latency}min.csv.gz'
        outputs[str(latency)] = {'path': target.relative_to(root).as_posix(),
                                 **merge_parts(parts[latency], target, ids)}
    return {'outputs': outputs, 'scenarios': diagnostics.finish(), 'cache_validation': stats,
            'ordered_id_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
            'invariants': dict.fromkeys(['future_observation_uses', 'future_availability_uses',
                                       'station_mismatches', 'out_of_window_uses', 'duplicate_output_ids',
                                       'row_order_mismatches', 'ineligible_matches'], 0)}
