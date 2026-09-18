"""Cross-check the 20 priority airports flagged by map_weather_stations.py's
PRIORITY_INVESTIGATION_TIERS output (tz_conflict_needs_resolution=8, unconfirmed=9,
confirmed_current_only=3) against an official source that goes beyond IEM's CURRENT
network metadata: NOAA/NCEI's Historical Observing Metadata Repository (HOMR), which
tracks station identifiers, renames, relocations, and period-of-record across a
station's full history.

Scope is fixed to exactly the 20 iata codes already flagged by that prior round --
this script asserts an exact match and refuses to run otherwise. It does not select
a new stratafix sample, collect any weather time series, or touch the production
mapping_table.csv / selection CSVs / recombination outputs.

What this script automates: fetching HOMR's station-history JSON for each airport
(one query per airport, id type chosen per HOMR_QUERY below) and saving the raw
response with its SHA-256, plus cross-referencing the already-cached IEM per-state
ASOS network listings for an alternate station id when the airport's own IATA code
was not found there.

What this script does NOT automate: reading a HOMR remark in English and deciding
whether it describes a genuine relocation, a compatible equipment move, or a plain
identifier rename is an analyst judgment call, not a deterministic classification --
same as this project's prior manual reading of BTS non-reporting-carrier remarks.
Those per-airport determinations are recorded in FINDINGS below as literal,
human-authored text tied to the fetched evidence, not derived by string matching.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.error
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PRIORITY_CSV = ROOT / 'output/baseline_recovery_v2_weather_scope_fix_20260918_mapping_priority_investigation.csv'
MAPPING_TABLE_CSV = ROOT / 'output/baseline_recovery_v2_weather_scope_fix_20260918_mapping_table.csv'
STRATAFIX_SELECTION_CSV = ROOT / 'output/baseline_recovery_v2_weather_expanded_stratafix_20260918_selection.csv'
HOMR_BASE = 'https://www.ncei.noaa.gov/access/homr/services/station/search'

# HOMR query used per airport: (id_type, id_value). ICAO is used wherever the
# airport's own ICAO code carries useful HOMR history; FAA is used for YUM and PBI
# specifically because the interesting HOMR record is reached by the FAA identifier
# actually used by NOAA (see FINDINGS['YUM'] / FINDINGS['PBI'] for why).
HOMR_QUERY = {
    'ISN': ('ICAO', 'KISN'), 'XWA': ('ICAO', 'KXWA'), 'YUM': ('FAA', 'YUM'),
    'STT': ('ICAO', 'TIST'), 'STX': ('ICAO', 'TISX'),
    'AZA': ('ICAO', 'KIWA'), 'BKG': ('ICAO', 'KBBG'), 'FCA': ('ICAO', 'KGPI'),
    'HHH': ('ICAO', 'KHXD'), 'MQT': ('ICAO', 'KSAW'), 'PBI': ('FAA', 'PBI'),
    'SCE': ('ICAO', 'KUNV'), 'USA': ('ICAO', 'KJQF'), 'SPN': ('ICAO', 'PGSN'),
    'IMT': ('ICAO', 'KIMT'), 'KTN': ('ICAO', 'PAKT'), 'PSG': ('ICAO', 'PAPG'),
    'SDF': ('ICAO', 'KSDF'), 'SIT': ('ICAO', 'PASI'), 'WRG': ('ICAO', 'PAWG'),
}

ISSUE_GROUP = {
    'ISN': 'period_relocation', 'XWA': 'period_relocation', 'YUM': 'period_relocation',
    'STT': 'tz_conflict', 'STX': 'tz_conflict',
    'AZA': 'identifier_mismatch', 'BKG': 'identifier_mismatch', 'FCA': 'identifier_mismatch',
    'HHH': 'identifier_mismatch', 'MQT': 'identifier_mismatch', 'PBI': 'identifier_mismatch',
    'SCE': 'identifier_mismatch', 'USA': 'identifier_mismatch', 'SPN': 'identifier_mismatch',
    'IMT': 'tz_alias', 'KTN': 'tz_alias', 'PSG': 'tz_alias', 'SDF': 'tz_alias',
    'SIT': 'tz_alias', 'WRG': 'tz_alias',
}

# Human-authored determinations, one entry per airport, each tied to the HOMR query
# above and (where used) to the cached IEM network file named in `iem_cross_check`.
# identity_determination is deliberately a free-form category, kept SEPARATE from
# the existing pipeline's verification_tier (see EVIDENCE_BASIS in
# map_weather_stations.py) -- this script never writes to verification_tier.
FINDINGS = {
    'ISN': dict(
        candidate_or_official_id='FAA ISN / ICAO KISN (Sloulin Field Intl, Williston ND)',
        iem_cross_check=None,
        facility_finding=(
            'NOAA HOMR (ICAO KISN, ncdcStnId 10007500): status CLOSED; remark "AIRPORT WAS '
            'RELOCATED 7.5 MILES NORTHWEST"; remark "THE LAST LCD PUBLICATION FOR ISN IS '
            'SEPTEMBER 2019. THE ASOS IS NOW LOCATED AT XWA." Matches IEM archive_end 2019-10-18.'),
        identity_determination='relocation_confirmed',
        valid_subperiod='2018-01-01 to 2019-10-18 (old Sloulin Field site only)',
        remaining_gap=(
            'Exact FAA facility closure date not independently cross-checked against an FAA 5010 '
            'record this round; IEM archive_end (2019-10-18) and HOMR\'s "last LCD Sept 2019" '
            'remark are consistent but not identical-precision sources.'),
    ),
    'XWA': dict(
        candidate_or_official_id='FAA XWA / ICAO KXWA (Williston Basin Intl, Williston ND)',
        iem_cross_check=None,
        facility_finding=(
            'NOAA HOMR (ICAO KXWA, ncdcStnId 30121192): remark "WILLISTON AIRPORT ASOS WAS '
            'RELOCATED 7.5 MILES TO THE NORTHWEST OF THE ORIGINAL AIRPORT SITE. THIS SITE WAS '
            'NOT CONSIDERED CLIMATOLOGICALLY COMPATIBLE WITH THE PREVIOUS SITE (ISN); THEREFORE, '
            'A NEW STATION AND NEW CLIMATE RECORD WILL BE STARTED." Explicitly a new, distinct '
            'station, not equipment continuity at the old site.'),
        identity_determination='relocation_confirmed_new_facility',
        valid_subperiod='2019-10-12 (IEM) / 2019-10-25 (NOAA HOMR) to 2019-12-31',
        remaining_gap=(
            'IEM archive_begin (2019-10-12) and NOAA HOMR POR begin (2019-10-25) differ by 13 '
            'days; not resolved this round. Rows dated in this 13-day gap should not be treated '
            'as confirmed under either source without further check.'),
    ),
    'YUM': dict(
        candidate_or_official_id=(
            'Pipeline used candidate_sid="YUM" (=IATA) in AZ_ASOS; actual airport ICAO is KNYL, '
            'official ASOS FAA id is NYL'),
        iem_cross_check='AZ_ASOS',
        facility_finding=(
            'NOAA HOMR confirms these are TWO DISTINCT physical stations: FAA "YUM" (preferredName '
            '"YUMA INTL AP", ICAO KYUM, WBAN 23195, COOP only, location "WEATHER SERVICE OFFICE '
            'YUMA within and 4 mi SSE of PO at Yuma AZ", POR 1946-01-01 to 2007-01-05, status '
            'CLOSED) versus FAA "NYL" (preferredName "YUMA MCAS - COOP, AZ", ICAO KNYL, WBAN '
            '03145, COOP+ASOS, lat/lon 32.65/-114.61667, POR 1960-07-01 to Present). Coordinates '
            'differ by roughly 2.3 km. The airport actually used by this project\'s flight records '
            '(ICAO KNYL per mwgg) is continuously covered by station NYL throughout 2018-2019; the '
            '"YUM" IEM entry the pipeline\'s candidate_sid search matched is a retired, unrelated '
            'Weather Service Office site, not the airport.'),
        identity_determination='identifier_mismatch_resolved_distinct_facility',
        valid_subperiod='2018-01-01 to 2019-12-31 under station NYL (not under the closed "YUM" id)',
        remaining_gap=(
            'Flagged as a candidate_network_and_sid() bug in notebooks/map_weather_stations.py '
            '(assumes IATA==IEM station sid for ordinary US states, false for Yuma); not changed '
            'automatically this round.'),
    ),
    'STT': dict(
        candidate_or_official_id='FAA STT / ICAO TIST (Cyril E. King Airport, St. Thomas VI)',
        iem_cross_check=None,
        facility_finding=(
            'NOAA HOMR (TIST): single continuous station since 1945-10-01, utcOffset field reported '
            'as a fixed -4 (no seasonal value given), consistent with a no-DST zone. No relocation '
            'or renaming remark found. US Virgin Islands does not observe DST (grouped with '
            'AZ/HI/AS/GU/MP/PR as an exception to the Uniform Time Act; corroborated by a '
            'secondary source, not an original federal document fetched this round) -- this favors '
            'mwgg\'s "America/St_Thomas" (fixed -4) over IEM\'s "Atlantic/Bermuda" (-4 standard / -3 '
            'DST, since Bermuda itself observes DST), matching the project\'s own '
            'utc_offset_equivalent_2018_2019=False finding for this pair.'),
        identity_determination='facility_continuous_tz_string_conflict_leans_mwgg',
        valid_subperiod='facility identity: 2018-01-01 to 2019-12-31 (continuous). tz conflict left unresolved.',
        remaining_gap=(
            'No primary US federal document (e.g. a Federal Register notice) fetched to state VI\'s '
            'DST exemption directly. tz_conflict_needs_resolution tier NOT upgraded by this finding, '
            'per existing project policy.'),
    ),
    'STX': dict(
        candidate_or_official_id='FAA STX / ICAO TISX (Henry E. Rohlsen Airport, St. Croix VI)',
        iem_cross_check=None,
        facility_finding=(
            'NOAA HOMR (TISX): single continuous station since 1945-10-01 (early history notes '
            '"Benedict Field renamed Alexander Hamilton Field", overlap between military/civilian '
            'obs 1945-1948, well before the study period), utcOffset fixed -4. Same reasoning as '
            'STT applies.'),
        identity_determination='facility_continuous_tz_string_conflict_leans_mwgg',
        valid_subperiod='facility identity: 2018-01-01 to 2019-12-31 (continuous). tz conflict left unresolved.',
        remaining_gap='Same as STT.',
    ),
}
IDENT = {
    'AZA': dict(real_id='IWA', icao='KIWA', net='AZ_ASOS',
                name='Phoenix-Mesa Gateway Airport (fmr Williams AFB / Chandler Muni)',
                remark='"CALL SIGN CHANGED FROM CHD TO IWA, DATE UNKNOWN." Continuous POR 1942-03-01 to Present.'),
    'BKG': dict(real_id='BBG', icao='KBBG', net='MO_ASOS', name='Branson Airport, MO',
                remark='No relocation remark; station opened with the airport in 2008-2009. POR 2008-10-01 to Present.'),
    'FCA': dict(real_id='GPI', icao='KGPI', net='MT_ASOS',
                name='Glacier Park International / Kalispell Glacier Airport, MT',
                remark='"STATION ID CHANGED FROM FCA TO GPI AT 14Z (7AM MST) ON OCTOBER 25, 2005." Continuous POR 1896-06-29 to Present.'),
    'HHH': dict(real_id='HXD', icao='KHXD', net='SC_ASOS', name='Hilton Head Airport, SC',
                remark='No relocation remark found. Continuous POR 1972-08-01 to Present.'),
    'MQT': dict(real_id='SAW', icao='KSAW', net='MI_ASOS',
                name='Sawyer International Airport (fmr K.I. Sawyer AFB), MI',
                remark='No relocation remark found (AFB->civilian transition 1995, before archive continuity). Continuous POR 1956-10-01 to Present.'),
    'PBI': dict(real_id='DJT (current); NWSLI retains PBI', icao='KDJT (current); was KPBI', net='FL_ASOS',
                name='West Palm Beach -> Donald J. Trump International Airport, FL',
                remark=('NWSLI field = "PBI" (retained); FAA id now "DJT"; single continuous ASOS/COOP '
                        'station since 1938-07-01, no relocation remark. Airport was officially "Palm Beach '
                        'International Airport" during 2018-2019; the FAA/NOAA identifier and legal name '
                        'changed to Donald J. Trump International Airport / DJT after the study period.')),
    'SCE': dict(real_id='UNV', icao='KUNV', net='PA_ASOS',
                name='State College Regional / University Park Airport, PA',
                remark='No relocation remark found. Continuous POR 1972-04-01 to Present.'),
    'USA': dict(real_id='JQF', icao='KJQF', net='NC_ASOS', name='Concord Regional Airport, NC',
                remark='No relocation remark found. Continuous POR 1995-09-01 to Present.'),
}
for _iata, _d in IDENT.items():
    FINDINGS[_iata] = dict(
        candidate_or_official_id=(
            f'Pipeline candidate_sid="{_iata}" (=IATA) not found in {_d["net"]}; actual IEM/FAA sid is '
            f'"{_d["real_id"]}" ({_d["icao"]}, {_d["name"]})'),
        iem_cross_check=_d['net'],
        facility_finding=f'NOAA HOMR ({_d["icao"]}): {_d["remark"]}',
        identity_determination='identifier_mismatch_resolved_same_facility',
        valid_subperiod='2018-01-01 to 2019-12-31 (continuous facility per official record)',
        remaining_gap=(
            'Flagged for a follow-up code check to candidate_network_and_sid() in '
            'notebooks/map_weather_stations.py, which currently assumes IATA==IEM station sid for '
            'ordinary US states; not changed automatically this round.'),
    )
FINDINGS['SPN'] = dict(
    candidate_or_official_id=(
        'Pipeline tried candidate_sid="PGSN" in GU_ASOS; actual FAA id per NOAA is "GSN"; not found '
        'in cached GU_ASOS listing under any tried id'),
    iem_cross_check='GU_ASOS',
    facility_finding=(
        'GU_ASOS cached network has 5 stations (PGRO Rota, PGUA Andersen AFB, PGUM Agana/Guam Intl, '
        'PGWT West Tinian, PWAK Wake Island) -- no Saipan station under any id. NOAA HOMR (ICAO PGSN) '
        'confirms an active ASOS at Saipan International Airport, FAA id "GSN", POR 2000-01-11 to '
        'Present, which would cover 2018-2019. This contradicts reading "unconfirmed" as "no station '
        'exists" -- one exists per NOAA. The gap is that this pipeline\'s cached IEM GU_ASOS response '
        'does not list it under PGSN, SPN, or GSN; whether IEM carries it under a different network '
        'name was not determined this round.'),
    identity_determination='unconfirmed_pipeline_coverage_gap',
    valid_subperiod='unresolved',
    remaining_gap=(
        'Needs a targeted IEM network query beyond GU_ASOS to find station GSN; out of scope for '
        'this read-only round (would be new data collection beyond the 20-airport metadata check).'),
)
ALIAS = {
    'IMT': dict(icao='KIMT', name='Ford Airport, Iron Mountain/Kingsford, MI',
                remark='No relocation remark found. Continuous POR 1949-12-01 to Present.'),
    'KTN': dict(icao='PAKT', name='Ketchikan International Airport, AK',
                remark=('"THIS WAS A COMPATIBLE STATION MOVE TO RELOCATE 50-4590 KETCHIKAN, TO THE ASOS '
                        'WHICH WAS COMMISSIONED MAY 23, 1997. STATION MOVED NW 5110 YARDS." -- an explicit '
                        '"compatible" equipment/site consolidation in 1997, well before 2018-2019, and '
                        'explicitly distinguished by NOAA from an incompatible relocation (contrast with '
                        'ISN/XWA above). Continuous POR 1973-10-03 to Present.')),
    'PSG': dict(icao='PAPG', name='Petersburg James A Johnson Airport, AK',
                remark=('COOP-side record shows a station-id rename PSG->APGA2 for the manual-observer COOP '
                        'program only, unrelated to the ASOS id this pipeline uses; no relocation remark '
                        'found. Continuous POR 1924-10-28 to Present.')),
    'SDF': dict(icao='KSDF', name='Louisville International / Standiford Field, KY',
                remark='No relocation remark found. Continuous POR 1947-11-15 to Present.'),
    'SIT': dict(icao='PASI', name='Sitka Rocky Gutierrez Airport, AK',
                remark='"REVIEWED. NO KNOWN CHANGES." Continuous POR 1930-08-01 to Present.'),
    'WRG': dict(icao='PAWG', name='Wrangell Airport, AK',
                remark=('"STATION INACTIVE SINCE 10/23/2012" applies to the COOP manual-observer program '
                        'only; the ASOS platform this pipeline uses is separately listed as active in IEM\'s '
                        'own cache. POR (COOP thread) 1984-11-15 to Present; IEM ASOS archive begins 1972.')),
}
for _iata, _d in ALIAS.items():
    FINDINGS[_iata] = dict(
        candidate_or_official_id=(
            f'IEM sid = {_iata} directly ({_d["icao"]}, {_d["name"]}); issue is tzname STRING mismatch '
            '(already utc_offset_equivalent_2018_2019=True in prior work), not identifier lookup'),
        iem_cross_check=None,
        facility_finding=f'NOAA HOMR ({_d["icao"]}): {_d["remark"]}',
        identity_determination='facility_continuity_confirmed_tz_string_conflict_unchanged',
        valid_subperiod='2018-01-01 to 2019-12-31 (continuous facility per official record)',
        remaining_gap=(
            'tzname STRING conflict vs mwgg intentionally left at tz_conflict_needs_resolution per '
            'existing policy; this round only adds facility-continuity confirmation.'),
    )

assert set(FINDINGS) == set(HOMR_QUERY) == set(ISSUE_GROUP), 'FINDINGS/HOMR_QUERY/ISSUE_GROUP must cover the same 20 airports'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_homr(id_type: str, id_value: str, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f'{id_type}_{id_value}.json'
    url = f'{HOMR_BASE}?qid={id_type}:{id_value}'
    if not cache.exists():
        body = None
        for attempt in range(5):
            try:
                with urlopen(Request(url, headers={'User-Agent': 'Airplane-course-station-identity/1.0'}),
                            timeout=30) as r:
                    body = r.read(5_000_001)
                break
            except (urllib.error.HTTPError, urllib.error.URLError, ConnectionResetError, TimeoutError):
                if attempt == 4:
                    raise
                time.sleep(3 * (attempt + 1))
        if len(body) > 5_000_000:
            raise ValueError('HOMR response exceeds bounded size')
        obj = json.loads(body.decode('utf-8'))
        cache.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
        time.sleep(1)  # be polite between HOMR requests
    return {'url': url, 'cache_file': str(cache.relative_to(ROOT)), 'sha256': digest(cache)}


def stratafix_membership() -> dict[str, bool]:
    airports = set()
    with open(STRATAFIX_SELECTION_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            airports.add(row['Origin_Airport'])
            airports.add(row['Destination_Airport'])
    return {iata: iata in airports for iata in HOMR_QUERY}


def prior_tiers() -> dict[str, str]:
    out = {}
    with open(PRIORITY_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out[row['iata']] = row['verification_tier']
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    if not args.name.startswith('baseline_recovery_v2') or Path(args.name).name != args.name:
        raise ValueError('Invalid output name')
    out = ROOT / 'output'
    existing = list(out.glob(args.name + '_evidence*')) + list(out.glob(args.name + '_manifest*'))
    if existing:
        raise FileExistsError('Use a fresh name; existing evidence is preserved')

    tiers = prior_tiers()
    if set(tiers) != set(HOMR_QUERY):
        raise ValueError(
            f'Target set mismatch vs {PRIORITY_CSV.name}: '
            f'only-in-prior={set(tiers) - set(HOMR_QUERY)} only-in-this-script={set(HOMR_QUERY) - set(tiers)}')

    strata = stratafix_membership()

    raw_dir = ROOT / 'data/weather_probe' / f'{args.name}_homr_raw'
    homr_results = {}
    for iata, (id_type, id_value) in sorted(HOMR_QUERY.items()):
        homr_results[iata] = fetch_homr(id_type, id_value, raw_dir)

    columns = [
        'iata', 'prior_verification_tier', 'issue_category', 'appears_in_stratafix_sample',
        'candidate_or_official_id', 'facility_finding', 'identity_determination',
        'valid_subperiod_evidence_basis', 'remaining_gap', 'evidence_agency', 'evidence_source_url',
        'evidence_record_ref', 'evidence_accessed_date', 'raw_evidence_file', 'raw_evidence_sha256',
        'iem_cross_check_network',
    ]
    evidence_path = out / f'{args.name}_evidence.csv'
    accessed = time.strftime('%Y-%m-%d')
    with open(evidence_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for iata in sorted(HOMR_QUERY):
            fnd = FINDINGS[iata]
            hr = homr_results[iata]
            id_type, id_value = HOMR_QUERY[iata]
            w.writerow({
                'iata': iata,
                'prior_verification_tier': tiers[iata],
                'issue_category': ISSUE_GROUP[iata],
                'appears_in_stratafix_sample': strata[iata],
                'candidate_or_official_id': fnd['candidate_or_official_id'],
                'facility_finding': fnd['facility_finding'],
                'identity_determination': fnd['identity_determination'],
                'valid_subperiod_evidence_basis': fnd['valid_subperiod'],
                'remaining_gap': fnd['remaining_gap'],
                'evidence_agency': 'NOAA/NCEI HOMR (Historical Observing Metadata Repository)',
                'evidence_source_url': hr['url'],
                'evidence_record_ref': f'{id_type}:{id_value}',
                'evidence_accessed_date': accessed,
                'raw_evidence_file': hr['cache_file'],
                'raw_evidence_sha256': hr['sha256'],
                'iem_cross_check_network': fnd['iem_cross_check'] or '',
            })

    manifest = {
        'name': args.name,
        'purpose': (
            'Official-source (NOAA/NCEI HOMR) cross-check of the 20 priority airports flagged by '
            f'{PRIORITY_CSV.relative_to(ROOT)}. Scope fixed to exactly these 20 airports.'),
        'fits_any_model': False, 'uses_external_data': True, 'target_columns_used': [],
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'priority_csv': {'path': str(PRIORITY_CSV.relative_to(ROOT)), 'sha256': digest(PRIORITY_CSV)},
        'stratafix_selection_csv': {
            'path': str(STRATAFIX_SELECTION_CSV.relative_to(ROOT)), 'sha256': digest(STRATAFIX_SELECTION_CSV)},
        'homr_base_url': HOMR_BASE,
        'homr_queries': {iata: f'{t}:{v}' for iata, (t, v) in sorted(HOMR_QUERY.items())},
        'evidence_table': str(evidence_path.relative_to(ROOT)),
        'evidence_table_sha256': digest(evidence_path),
        'row_count': len(HOMR_QUERY),
        'stratafix_membership_count': sum(strata.values()),
        'limitations': [
            'This script fetches and hashes NOAA/NCEI HOMR raw responses and assembles the evidence '
            'table deterministically, but the per-airport facility_finding / identity_determination '
            'text in FINDINGS is human-authored analysis of those responses (reading English free-text '
            'remarks is not a deterministic classification) -- re-running this script reproduces the '
            'same fetch and the same table only because FINDINGS is fixed source, not because the '
            'judgment itself is re-derived from the JSON.',
            'historical_identity_confirmed / verification_tier in the existing pipeline output '
            '(mapping_table.csv, mapping_priority_investigation.csv) are NOT modified by this script; '
            'identity_determination here is a separate field for a human or a future coding round to '
            'act on.',
            'SPN (Saipan) remains genuinely unconfirmed: NOAA HOMR shows an active ASOS (FAA id GSN) '
            'covering 2018-2019, but this project\'s cached IEM GU_ASOS network response does not list '
            'it under any tried id. Not resolved this round.',
            'The 8 identifier-mismatch findings (AZA/BKG/FCA/HHH/MQT/PBI/SCE/USA) and the YUM finding '
            'are candidate bugs in candidate_network_and_sid() in notebooks/map_weather_stations.py; '
            'that script was not modified by this investigation.',
            'FAA facility master records (e.g. a dated Form 5010) were not queried; only NOAA/NCEI HOMR '
            'and the already-cached IEM network metadata.',
            'No new weather time series collection, no stratafix real collection, no model retraining, '
            'and no changes to any baseline_recovery_v2_weather_* production output were made.',
        ],
    }
    manifest_path = out / f'{args.name}_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'limitations'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
