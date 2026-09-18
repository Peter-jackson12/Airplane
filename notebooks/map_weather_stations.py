"""Build and verify an airport -> IANA timezone -> IEM ASOS station mapping
for EVERY airport that appears (as origin or destination) among the adopted
(complete_single_candidate_year) rows -- not just the 8 hand-picked airports
used by the first 21-row sample.

Sources, in priority order:
  1. IEM's own network metadata (mesonet.agron.iastate.edu/geojson/network/*),
     which gives the station id actually usable in asos.py requests, its
     IANA tzname, and its archive_begin/archive_end -- i.e. official period
     validity, not just current existence.
  2. The mwgg/Airports dataset (already cached and hash-pinned for the first
     sample) for IATA/ICAO/state/country/lat-lon, used to build the CANDIDATE
     station id and to cross-check IEM's tzname independently.

A hash-pinned modern airport list is NOT evidence that a mapping held in
2018-2019; that period check is exactly what archive_begin/archive_end give.
No target, delay, or weather value is read anywhere here -- only airport
codes already used by the row-level date attribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION_RUN = 'baseline_recovery_v2_row_date_attribution_20260918'
MWGG_CACHE = ROOT / 'data/weather_probe/airports_mwgg.json'
MWGG_EXPECTED_SHA256 = 'f369eaa1c2944280d9678a96d5b477cedea0417f3c12a333c39834bf1c035739'
NETWORK_CACHE_DIR = ROOT / 'data/weather_probe/networks'

# Full US state/DC name -> 2-letter postal code, used only to build IEM's
# per-state ASOS network name (e.g. "GA_ASOS"). This is a standard postal
# abbreviation table, not a guessed or fitted mapping.
STATE_POSTAL = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
    'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
    'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
}
NON_CONUS_US_STATES = {'Alaska', 'Hawaii'}
# mwgg's `state` text is inconsistent for Pacific/Caribbean territories (city
# or district names, not the territory), so the `country` field is used
# there instead. IEM groups CNMI (Saipan/Rota) under GU_ASOS, not a separate
# MP_ASOS (confirmed empirically 2026-09-18: MP_ASOS has 0 features).
TERRITORY_NETWORKS = {
    'PR': ['PR_ASOS'], 'VI': ['VI_ASOS'], 'GU': ['GU_ASOS'], 'MP': ['GU_ASOS'], 'AS': ['AS_ASOS'],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_mwgg() -> dict:
    if digest(MWGG_CACHE) != MWGG_EXPECTED_SHA256:
        raise ValueError(f'{MWGG_CACHE} does not match the pinned hash; re-verify before use')
    return json.loads(MWGG_CACHE.read_text(encoding='utf-8'))


def candidate_network_and_sid(icao: str, iata: str, state: str, country: str) -> tuple[list[str], str]:
    if country == 'US' and state in STATE_POSTAL and state not in NON_CONUS_US_STATES:
        return [f'{STATE_POSTAL[state]}_ASOS'], iata
    if state == 'Alaska':
        return ['AK_ASOS'], icao
    if state == 'Hawaii':
        return ['HI_ASOS'], icao
    if country in TERRITORY_NETWORKS:
        return TERRITORY_NETWORKS[country], icao
    return [], icao  # no candidate network identified; stays unconfirmed


def fetch_network(network: str) -> dict:
    NETWORK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = NETWORK_CACHE_DIR / f'{network}.geojson'
    url = f'https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson'
    if not cache.exists():
        body = None
        for attempt in range(5):
            try:
                with urlopen(Request(url, headers={'User-Agent': 'Airplane-course-weather-mapping/1.0'}),
                            timeout=30) as r:
                    body = r.read(5_000_001)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    cache.write_text('{"features": [], "http_status": 404}')
                    body = None
                    break
                if exc.code != 429 or attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
            except (urllib.error.URLError, ConnectionResetError, TimeoutError):
                if attempt == 4:
                    raise
                time.sleep(3 * (attempt + 1))
        if body is not None:
            if len(body) > 5_000_000:
                raise ValueError('Network metadata response exceeds bounded size')
            cache.write_bytes(body)
        time.sleep(1)  # be polite between metadata requests
    data = json.loads(cache.read_text(encoding='utf-8'))
    by_sid = {}
    for f in data.get('features', []):
        props = dict(f['properties'])
        coords = (f.get('geometry') or {}).get('coordinates')
        if coords and len(coords) == 2:
            props['iem_lon'], props['iem_lat'] = coords[0], coords[1]
        by_sid[f['id']] = props
    return {'url': url, 'cache_file': str(cache.relative_to(ROOT)), 'sha256': digest(cache),
            'stations': by_sid}


PERIOD_START = pd.Timestamp('2018-01-01')
PERIOD_END = pd.Timestamp('2019-12-31')

# What verification_tier actually checks -- and, just as importantly, what it does NOT. It is an
# IEM-metadata-only evidence level: station id present in the CURRENT network listing, that listing's
# own archive_begin/archive_end brackets the period, and the tzname STRING matches mwgg's. It says
# nothing about whether the airport was at the same physical location, was the same facility, or had
# no relocation/renaming history during 2018-2019 -- that is a separate axis (location_distance_km /
# historical_identity_note below), left explicitly unconfirmed unless this round specifically
# investigated it (see PRIORITY_INVESTIGATION_TIERS / the *_priority_investigation.csv output).
EVIDENCE_BASIS = ('IEM network-metadata lookup only: station id presence + archive period coverage + '
                  'tzname string match. Does NOT by itself confirm physical location, facility '
                  'identity, or absence of a mid-period relocation/rename.')
PRIORITY_INVESTIGATION_TIERS = {'tz_conflict_needs_resolution', 'unconfirmed', 'confirmed_current_only'}


def haversine_km(lat1, lon1, lat2, lon2) -> float | None:
    """Great-circle distance in km, or None if any coordinate is missing --
    used only as a location-identity SIGNAL (a large distance between mwgg's
    and IEM's coordinates for the same candidate station id is a red flag),
    never as proof of identity by itself."""
    if any(v is None or (isinstance(v, float) and pd.isna(v)) for v in (lat1, lon1, lat2, lon2)):
        return None
    from math import asin, cos, radians, sin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371.0088 * asin(sqrt(a))


def utc_offsets_match_over_period(tz_a: str, tz_b: str, start: pd.Timestamp, end: pd.Timestamp) -> bool | None:
    """Whether two IANA zone names produce the IDENTICAL UTC offset at every
    sampled wall-clock moment across [start, end] -- distinguishes a genuine
    UTC-offset disagreement (e.g. one side observes DST, the other does not)
    from two different IANA names that have been substantively equivalent
    (same offsets, same transition dates) throughout the period. A tzname
    STRING mismatch is never silently treated as harmless just because the
    strings differ; this only reports whether it ALSO differs in effect.
    Returns None if either name cannot be resolved (never assumed True/False).
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        za, zb = ZoneInfo(tz_a), ZoneInfo(tz_b)
    except (ZoneInfoNotFoundError, ValueError):
        return None
    day = start.normalize()
    end = end.normalize()
    while day <= end:
        for hour in (0, 6, 12, 18):  # sample across the day to catch transition-hour edge effects
            naive = datetime(day.year, day.month, day.day, hour)
            if naive.replace(tzinfo=za).utcoffset() != naive.replace(tzinfo=zb).utcoffset():
                return False
        day += timedelta(days=1)
    return True


def subperiod_overlap(begin, end, period_start: pd.Timestamp, period_end: pd.Timestamp):
    """Intersection of [begin, end] (either bound may be None/open-ended)
    with [period_start, period_end], as (start, end) date strings, or
    (None, None) if there is no overlap at all -- a station that does not
    cover the FULL 2018-2019 window may still be a valid mapping for
    individual rows whose own date falls inside the part it does cover;
    confirmed_current_only must not be read as 'never usable'.
    """
    lo = max(begin, period_start) if begin is not None else period_start
    hi = min(end, period_end) if end is not None else period_end
    if lo > hi:
        return None, None
    return str(lo.date()), str(hi.date())


def classify(entry: dict, network_result: dict | None) -> dict:
    base = {'evidence_basis': EVIDENCE_BASIS, 'location_distance_km': None,
           'valid_subperiod_start': None, 'valid_subperiod_end': None,
           'utc_offset_equivalent_2018_2019': None}
    if network_result is None:
        return {**base, 'found_in_iem_network': False, 'verification_tier': 'unconfirmed',
                'iem_tzname': None, 'archive_begin': None, 'archive_end': None,
                'period_covers_2018_2019': None, 'tz_matches_mwgg': None}
    props = network_result
    begin = pd.to_datetime(props.get('archive_begin')) if props.get('archive_begin') else None
    end = pd.to_datetime(props.get('archive_end')) if props.get('archive_end') else None
    covers = bool(begin is not None and begin <= PERIOD_START and (end is None or end >= PERIOD_END))
    tz_match = props.get('tzname') == entry['mwgg_tz']
    offset_equivalent = None
    if not tz_match and props.get('tzname') and entry.get('mwgg_tz'):
        # A different IANA NAME is not automatically a different UTC-offset SCHEDULE -- check both,
        # report both, and let the tier stay conservative (tz_conflict) either way (see EVIDENCE_BASIS).
        offset_equivalent = utc_offsets_match_over_period(props['tzname'], entry['mwgg_tz'],
                                                          PERIOD_START, PERIOD_END)
    if not tz_match:
        # The two sources disagree on which IANA zone applies (e.g. one has
        # DST, the other does not) -- this is a correctness risk, not a
        # cosmetic difference, so it is NOT allowed to pass as "confirmed"
        # regardless of archive coverage. Resolving it is a follow-up task,
        # regardless of whether utc_offset_equivalent_2018_2019 comes back True.
        tier = 'tz_conflict_needs_resolution'
    else:
        tier = 'confirmed_period' if covers else 'confirmed_current_only'
    distance = haversine_km(entry.get('lat'), entry.get('lon'), props.get('iem_lat'), props.get('iem_lon'))
    sub_start, sub_end = subperiod_overlap(begin, end, PERIOD_START, PERIOD_END)
    return {**base, 'found_in_iem_network': True, 'verification_tier': tier,
            'iem_tzname': props.get('tzname'), 'archive_begin': props.get('archive_begin'),
            'archive_end': props.get('archive_end'), 'period_covers_2018_2019': covers,
            'tz_matches_mwgg': tz_match, 'location_distance_km': distance,
            'valid_subperiod_start': sub_start, 'valid_subperiod_end': sub_end,
            'utc_offset_equivalent_2018_2019': offset_equivalent}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    if not args.name.startswith('baseline_recovery_v2') or Path(args.name).name != args.name:
        raise ValueError('Invalid output name')
    out = ROOT / 'output'
    if list(out.glob(args.name + '_mapping*')):
        raise FileExistsError('Use a fresh name; existing evidence is preserved')

    manifest = json.loads((out / f'{ATTRIBUTION_RUN}_manifest.json').read_text())
    row_path = ROOT / manifest['row_evidence']
    if digest(row_path) != manifest['row_sha256']:
        raise ValueError(f'{row_path} does not match {ATTRIBUTION_RUN}\'s recorded hash')
    attribution = pd.read_csv(row_path, low_memory=False)[['ID', 'status']]
    adopted = attribution.loc[attribution.status.eq('complete_single_candidate_year'), ['ID']]

    raw = pd.read_csv(ROOT / 'data/train.csv', usecols=['ID', 'Origin_Airport', 'Destination_Airport'])
    pool = adopted.merge(raw, on='ID', how='left', validate='one_to_one')
    if pool[['Origin_Airport', 'Destination_Airport']].isna().any().any():
        raise ValueError('adopted rows must have both airports observed')
    airports = sorted(set(pool.Origin_Airport) | set(pool.Destination_Airport))

    mwgg = load_mwgg()
    iata_to_icao = {}
    for icao, entry in mwgg.items():
        if entry.get('iata'):
            iata_to_icao.setdefault(entry['iata'], icao)

    rows = []
    fetched_networks: dict[str, dict] = {}
    for iata in airports:
        icao = iata_to_icao.get(iata)
        base = {'iata': iata, 'icao': icao}
        if icao is None:
            rows.append({**base, 'state': None, 'country': None, 'candidate_networks': None,
                        'candidate_sid': None, 'mwgg_tz': None, 'lat': None, 'lon': None,
                        **classify({'mwgg_tz': None}, None),
                        'note': 'not found in mwgg/Airports dataset'})
            continue
        entry = mwgg[icao]
        state, country = entry.get('state'), entry.get('country')
        networks, candidate_sid = candidate_network_and_sid(icao, iata, state, country)
        info = {**base, 'state': state, 'country': country,
                'candidate_networks': '+'.join(networks) if networks else None,
                'candidate_sid': candidate_sid, 'mwgg_tz': entry.get('tz'),
                'lat': entry.get('lat'), 'lon': entry.get('lon')}
        found_props = None
        tried = []
        for network in networks:
            if network not in fetched_networks:
                fetched_networks[network] = fetch_network(network)
            station_props = fetched_networks[network]['stations'].get(candidate_sid)
            tried.append(network)
            if station_props is not None:
                found_props = station_props
                info['matched_network'] = network
                break
        classification = classify(info, found_props)
        note = '' if networks else 'no candidate network identified for this state/country'
        if networks and found_props is None:
            note = f'candidate sid {candidate_sid} not found in {"/".join(tried)}'
        rows.append({**info, **classification, 'note': note})

    table = pd.DataFrame(rows).sort_values('iata').reset_index(drop=True)
    table.to_csv(out / f'{args.name}_mapping_table.csv', index=False)

    tier_counts = table.verification_tier.value_counts().to_dict()
    tz_mismatches = table.loc[table.tz_matches_mwgg.eq(False), ['iata', 'iem_tzname', 'mwgg_tz']]
    tz_mismatches.to_csv(out / f'{args.name}_mapping_tz_mismatches.csv', index=False)

    # Priority investigation subset (task-flagged): the 8 tz_conflict + 9 unconfirmed + 3
    # confirmed_current_only airports from the previous round, plus any new ones this round finds in
    # the same tiers. This is NOT a claim that historical identity/location is now confirmed for all
    # 375 airports -- it surfaces exactly what location_distance_km / utc_offset_equivalent_2018_2019 /
    # valid_subperiod_* could establish from already-cached IEM+mwgg metadata alone, with no new
    # official source consulted, so historical_identity_confirmed stays explicitly unconfirmed (None).
    priority_cols = ['iata', 'icao', 'state', 'country', 'verification_tier', 'iem_tzname', 'mwgg_tz',
                     'tz_matches_mwgg', 'utc_offset_equivalent_2018_2019', 'archive_begin', 'archive_end',
                     'valid_subperiod_start', 'valid_subperiod_end', 'location_distance_km', 'note']
    priority = table.loc[table.verification_tier.isin(PRIORITY_INVESTIGATION_TIERS), priority_cols]
    priority = priority.assign(historical_identity_confirmed=None)  # explicit, separate, unconfirmed axis
    priority.to_csv(out / f'{args.name}_mapping_priority_investigation.csv', index=False)

    network_sources = [{'network': n, 'url': r['url'], 'cache_file': r['cache_file'],
                        'sha256': r['sha256'], 'stations_in_network': len(r['stations'])}
                       for n, r in sorted(fetched_networks.items())]

    manifest_out = {
        'name': args.name,
        'attribution_run': ATTRIBUTION_RUN, 'attribution_row_sha256': manifest['row_sha256'],
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False, 'uses_external_data': True, 'target_columns_used': [],
        'mwgg_source_sha256': MWGG_EXPECTED_SHA256,
        'airports_covered': len(airports), 'networks_fetched': len(fetched_networks),
        'network_sources': network_sources,
        'verification_tier_counts': tier_counts,
        'tz_mismatches': int(len(tz_mismatches)),
        'mapping_table': str((out / f'{args.name}_mapping_table.csv').relative_to(ROOT)),
        'priority_investigation_table': str(
            (out / f'{args.name}_mapping_priority_investigation.csv').relative_to(ROOT)),
        'period_checked': [str(PERIOD_START.date()), str(PERIOD_END.date())],
        'evidence_basis': EVIDENCE_BASIS,
        'limitations': [
            'verification_tier is an IEM-metadata-only evidence level (station id presence + archive '
            'period + tzname string match). It does NOT confirm the airport\'s physical location, '
            'facility identity, or relocation/rename history at the time -- those are separate, largely '
            'uninvestigated axes for the full 375-airport set. Do not read "confirmed_period" as '
            '"historically verified identical airport"; read it as the narrower claim EVIDENCE_BASIS '
            'states.',
            'location_distance_km is a great-circle distance between mwgg\'s and IEM\'s coordinates for '
            'the SAME candidate station id; a large value is a red flag worth investigating, a small '
            'value is a supporting signal, and neither is proof of identity by itself. It is populated '
            'only when both sources report coordinates.',
            'utc_offset_equivalent_2018_2019 distinguishes a genuine UTC-offset disagreement between two '
            'differently-named IANA zones from two names that happened to be substantively equivalent '
            '(identical offsets/transitions) throughout 2018-2019; it does NOT reclassify a tz_conflict '
            'row out of that tier -- a name mismatch still blocks confirmed status regardless of this '
            'value, per the tz_conflict_needs_resolution policy.',
            'valid_subperiod_start/end is the overlap between a station\'s own archive window and '
            '2018-01-01..2019-12-31; a confirmed_current_only station whose archive window partially '
            'overlaps the period may still be a valid mapping for individual rows whose OWN date falls '
            'inside that overlap -- confirmed_current_only should not be read as "never usable".',
            'historical_identity_confirmed (in the priority-investigation output) is deliberately left '
            'unconfirmed (None) for every row this round: no new official per-airport source (e.g. FAA '
            'historical facility records) was consulted, only already-cached IEM network metadata and '
            'the hash-pinned mwgg dataset. Completing full 355-airport historical-identity verification '
            'was explicitly out of scope for this round.',
            'confirmed_current_only means the station exists in IEM\'s CURRENT network listing with a '
            'timezone match, but its recorded archive window does not (or is not known to) cover '
            '2018-2019 -- this is a present-day mapping, not evidence it held at the time.',
            'confirmed_period means IEM\'s own archive_begin/archive_end brackets 2018-01-01..2019-12-31.',
            'unconfirmed airports are NOT silently assigned a nearby station; they stay excluded from '
            'any weather request until resolved.',
            'IEM groups Saipan/Rota (CNMI) under the GU_ASOS network, not a separate MP_ASOS (verified '
            'empirically: MP_ASOS returns 0 stations as of 2026-09-18).',
        ],
    }
    (out / f'{args.name}_mapping_manifest.json').write_text(json.dumps(manifest_out, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest_out.items() if k not in {'network_sources', 'limitations'}},
                     indent=2, default=str))


if __name__ == '__main__':
    main()
