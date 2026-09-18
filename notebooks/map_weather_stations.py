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
    by_sid = {f['id']: f['properties'] for f in data.get('features', [])}
    return {'url': url, 'cache_file': str(cache.relative_to(ROOT)), 'sha256': digest(cache),
            'stations': by_sid}


PERIOD_START = pd.Timestamp('2018-01-01')
PERIOD_END = pd.Timestamp('2019-12-31')


def classify(entry: dict, network_result: dict | None) -> dict:
    if network_result is None:
        return {'found_in_iem_network': False, 'verification_tier': 'unconfirmed',
                'iem_tzname': None, 'archive_begin': None, 'archive_end': None,
                'period_covers_2018_2019': None, 'tz_matches_mwgg': None}
    props = network_result
    begin = pd.to_datetime(props.get('archive_begin')) if props.get('archive_begin') else None
    end = pd.to_datetime(props.get('archive_end')) if props.get('archive_end') else None
    covers = bool(begin is not None and begin <= PERIOD_START and (end is None or end >= PERIOD_END))
    tz_match = props.get('tzname') == entry['mwgg_tz']
    if not tz_match:
        # The two sources disagree on which IANA zone applies (e.g. one has
        # DST, the other does not) -- this is a correctness risk, not a
        # cosmetic difference, so it is NOT allowed to pass as "confirmed"
        # regardless of archive coverage. Resolving it is a follow-up task.
        tier = 'tz_conflict_needs_resolution'
    else:
        tier = 'confirmed_period' if covers else 'confirmed_current_only'
    return {'found_in_iem_network': True, 'verification_tier': tier,
            'iem_tzname': props.get('tzname'), 'archive_begin': props.get('archive_begin'),
            'archive_end': props.get('archive_end'), 'period_covers_2018_2019': covers,
            'tz_matches_mwgg': tz_match}


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
                        'found_in_iem_network': False, 'verification_tier': 'unconfirmed',
                        'iem_tzname': None, 'archive_begin': None, 'archive_end': None,
                        'period_covers_2018_2019': None, 'tz_matches_mwgg': None,
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
        'period_checked': [str(PERIOD_START.date()), str(PERIOD_END.date())],
        'limitations': [
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
