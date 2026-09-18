"""Cross-check the 20 priority airports flagged by map_weather_stations.py's
PRIORITY_INVESTIGATION_TIERS output (tz_conflict_needs_resolution=8, unconfirmed=9,
confirmed_current_only=3) against an official source that goes beyond IEM's CURRENT
network metadata: NOAA/NCEI's Historical Observing Metadata Repository (HOMR), which
tracks station identifiers, renames, relocations, and period-of-record across a
station's full history.

Scope is fixed to exactly these 20 iata codes -- this script asserts an exact match
against the prior investigation CSV and refuses to run otherwise. These 20 are the
confirmed_current_only(3) + tz_conflict_needs_resolution(8) + unconfirmed(9) tiers;
they do NOT overlap the 355 confirmed_period airports, and this script does not
investigate any of those 355. It does not select a new stratafix sample, collect any
weather time series, or touch the production mapping_table.csv / selection CSVs /
recombination outputs.

Second round (this revision). The first round (see git history) produced a single
free-text `facility_finding` per airport and blended four different kinds of claim
together: what the CURRENT identifier maps to, what directly supports the 2018-2019
window, how the observing PROGRAM (COOP/ASOS/AWOS/NEXRAD) record connects to the
location this pipeline actually uses, and outright inference/gaps. Reviewing that
text against the raw HOMR JSON surfaced two concrete over-claims this revision fixes:
  - WRG and PSG: the HOMR record this script fetches for both is COOP-only (no ASOS
    platform entry at all), yet the first round still assigned them the same
    "facility continuity confirmed" category as IMT/KTN/SDF/SIT, whose fetched HOMR
    records DO carry an ASOS platform with ASOS-specific remarks (e.g. "ASOS
    COMMISSIONED..." / "ACU LOCATION..."). Nothing in the WRG/PSG HOMR response
    connects that COOP record to the ASOS platform this pipeline actually reads from
    IEM. Both are downgraded here to `unconfirmed_program_linkage_insufficient`.
  - YUM: the first round treated NOAA HOMR's two FAA records (closed "YUM" vs active
    "NYL") as sufficient to declare "two distinct facilities" resolved. But this
    project's OWN cached IEM AZ_ASOS network listing (data/weather_probe/networks/
    AZ_ASOS.geojson) carries a THIRD, separate "YUM" feature at coordinates matching
    NYL/Yuma-MCAS (not HOMR's WSO-Yuma coordinates), sharing NYL's ncei91/climate_site
    codes. That is a real conflict between HOMR and this project's own IEM source
    about where "YUM" physically is, and this round does not resolve it -- it is
    downgraded to `identifier_mismatch_partially_resolved_source_conflict`. Separately,
    KNYL's ASOS platform entry was added to HOMR by an ad hoc metadata update dated
    2021-12-16 ("ADDING ASOS PLATFORM PER INFORMATION..."); that is a metadata-edit
    date, not a verified installation date, and the station's 1960-Present
    period-of-record is the COOP thread's POR, not a verified ASOS operating window.
    Both distinctions are now recorded as separate fields instead of folded into one
    "valid_subperiod" claim.

What this script automates: fetching HOMR's station-history JSON for each airport
(one primary query per airport, plus a secondary query for YUM per HOMR_EXTRA_QUERY
below) and saving the raw response with its SHA-256, cross-referencing the already
-cached IEM per-state ASOS/AWOS network GeoJSON files for the specific feature id
each finding relies on (recording that file's path, SHA-256, and the feature's own
archive_begin/archive_end/coordinates -- not just the network's name), and reproducing
the evidence table deterministically from a cache directory that reuses this round's
already-fetched raw responses without a new external query, unless --allow-network
is passed and a needed file is genuinely missing from that cache.

What this script does NOT automate: reading a HOMR remark in English and deciding
whether it describes a genuine relocation, a compatible equipment move, or a plain
identifier rename is an analyst judgment call, not a deterministic classification --
same as this project's prior manual reading of BTS non-reporting-carrier remarks.
Those per-airport determinations are recorded in FINDINGS below as literal,
human-authored text tied to the fetched evidence, not derived by string matching.
This script also does not prove that official sources are complete: an airport left
`unconfirmed` here may simply mean this round found no linking record, not that no
such record exists anywhere.

Third round (this revision). No new external data was fetched; only the existing
cache and the second round's own text were reviewed again. Two fixes:
  - Current-identifier vs. historical-continuity separation. The second round's
    `_TZ_ALIAS_CONFIRMED` group (IMT/KTN/SDF/SIT) gave all four the SAME
    `facility_continuity_confirmed_tz_string_conflict_unchanged` determination on the
    strength of a CURRENT ASOS platform entry and ACU-coordinate remark plus a POR
    that happens to say "to Present" -- none of which is dated evidence that the
    2018-2019 window specifically was unchanged. Only KTN's fetched record carries an
    explicitly dated pre-period event (the 1997 "compatible station move" remark);
    IMT/SDF/SIT carry no dated event at all, only current metadata. This round keeps
    KTN's determination (with the 1997 event named as what it actually is: dated but
    pre-period, not itself proof of the 2018-2019 window) and downgrades IMT/SDF/SIT
    to a determination that confirms the CURRENT facility/identifier mapping and
    ASOS program linkage, while leaving 2018-2019 continuity explicitly unconfirmed
    rather than folding it into "confirmed". The same current-vs-period distinction
    is now recorded (new `historical_continuity_2018_2019` field) for all 20 rows,
    including the identifier-mismatch group, where `inference_and_unresolved` no
    longer reads "none" for airports whose only basis is an unbroken POR plus the
    absence of a relocation remark -- that absence is now stated as what it is (an
    inference from silence, not a positive record), separately from the 2 airports
    (FCA, PBI) that do carry a dated or explicit record of the identifier's history.
  - Raw-record verification wired into the actual generation path, not just into
    tests reading the cache separately. `main()` now loads each fetched HOMR
    response's `stationCollection.stations`, selects the exact `ncdcStnId` each
    FINDINGS entry's judgment relies on (`EXPECTED_RECORD` below), and checks that
    record's `platforms` against what that judgment requires or forbids (e.g. IMT
    requires ASOS; WRG/PSG forbid ASOS/AWOS; AZA's two records are selected and
    checked separately by ncdcStnId so the AWOS record is never conflated with the
    NEXRAD one; YUM's FAA:YUM and ICAO:KNYL records are checked separately). It also
    requires every non-SPN `IEM_FEATURE_CHECK` entry to be `found=True` -- SPN's
    expected miss is the one allowed exception, matching FINDINGS['SPN'] already
    being `unconfirmed`. Any mismatch (missing station, wrong/missing platform,
    missing IEM feature) raises `RuntimeError` and aborts before any evidence file is
    written, instead of silently printing the fixed FINDINGS text against raw
    evidence that no longer supports it. This makes "the cache file exists" and "the
    cited record actually says what FINDINGS claims" two different, both-required
    conditions.

Fourth round (this revision). No new external data was fetched; only the existing
cache and this file's own text were reviewed again. Two fixes:
  - PBI's `_IDENTIFIER_MISMATCH['PBI']` remark and `continuity_basis` used to say the
    fetched HOMR record itself states the FAA/legal rename to Donald J. Trump
    International Airport happened AFTER 2018-2019. Re-reading the raw response
    (`data/weather_probe/.../FAA_PBI.json`) shows no such dated statement anywhere in
    its `remarks` -- only the record's CURRENT identifiers (FAA=DJT, ICAO=KDJT,
    NWSLI=PBI retained) and an undated ad hoc metadata update renaming the station per
    FAA/AIRNAV. The claim that the rename postdates the study window is real-world
    background knowledge, not something this fetched record says, and was blended in
    as if it were record-sourced -- the same kind of over-claim the second round
    already fixed for WRG/PSG and YUM. `continuity_basis` for PBI is now
    `rename_confirmed_by_current_ids_undated_in_record`, distinguished from FCA's
    actual dated 2005 rename remark and KTN's actual dated 1997 event, neither of
    which is touched by this fix.
  - `EXPECTED_RECORD`'s per-airport checks previously verified only `ncdcStnId`
    selection and `platforms`; a raw cache file whose `identifiers` were deleted or
    swapped to different FAA/ICAO/NWSLI/NEXRAD values, while `ncdcStnId` and
    `platforms` stayed untouched, would still pass and print the fixed FINDINGS text.
    A new `EXPECTED_IDENTIFIERS` table (keyed by `ncdcStnId`) and a matching check
    inside `verify_expected_record()` now require each judgment's cited identifiers
    to actually be present on the selected record. This does not force every
    `HOMR_QUERY` lookup key to also appear among the record's own current
    identifiers -- PBI is looked up by `FAA:PBI` (a key NOAA's search index still
    resolves) even though the record's own current FAA identifier is DJT; the lookup
    key and the record's present-day identifiers are allowed to differ.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PRIORITY_CSV = ROOT / 'output/baseline_recovery_v2_weather_scope_fix_20260918_mapping_priority_investigation.csv'
STRATAFIX_SELECTION_CSV = ROOT / 'output/baseline_recovery_v2_weather_expanded_stratafix_20260918_selection.csv'
DEFAULT_CACHE_DIR = ROOT / 'data/weather_probe/baseline_recovery_v2_station_identity_investigation_20260918_homr_raw'
NETWORKS_DIR = ROOT / 'data/weather_probe/networks'
HOMR_BASE = 'https://www.ncei.noaa.gov/access/homr/services/station/search'

# Primary HOMR query used per airport: (id_type, id_value). ICAO is used wherever the
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

# Secondary HOMR query fetched IN ADDITION to the primary one above, for airports
# where the judgment in FINDINGS relies on a second raw record. YUM's judgment reads
# BOTH the closed FAA:YUM record (via HOMR_QUERY) and the active airport's own ICAO
# record KNYL -- the first round fetched KNYL's JSON into the same cache directory
# but never linked its path/hash/record-ref into the evidence table or manifest.
HOMR_EXTRA_QUERY: dict[str, list[tuple[str, str]]] = {
    'YUM': [('ICAO', 'KNYL')],
}

# IEM network cross-checks this script actually reads a cached GeoJSON file for, one
# entry per (network, sid) pair referenced by a finding below. Recorded per airport
# so the evidence table names the exact file/feature used, not just the network name.
IEM_FEATURE_CHECK: dict[str, list[tuple[str, str]]] = {
    'YUM': [('AZ_ASOS', 'YUM'), ('AZ_ASOS', 'NYL')],
    'AZA': [('AZ_ASOS', 'IWA')],
    'BKG': [('MO_ASOS', 'BBG')],
    'FCA': [('MT_ASOS', 'GPI')],
    'HHH': [('SC_ASOS', 'HXD')],
    'MQT': [('MI_ASOS', 'SAW')],
    'PBI': [('FL_ASOS', 'DJT')],
    'SCE': [('PA_ASOS', 'UNV')],
    'USA': [('NC_ASOS', 'JQF')],
    'SPN': [('GU_ASOS', 'GSN'), ('GU_ASOS', 'SPN'), ('GU_ASOS', 'PGSN')],
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

# Human-authored determinations, one entry per airport. Each dict below separates
# four kinds of claim that the first round blended into one `facility_finding`
# string:
#   current_identifier_mapping   -- what the CURRENT official identifier/facility is
#   period_2018_2019_evidence    -- what directly speaks to the 2018-2019 window,
#                                    and explicitly where that evidence is indirect
#   program_location_linkage     -- how the observing PROGRAM record (COOP/ASOS/
#                                    AWOS/NEXRAD) connects to the location/platform
#                                    this pipeline actually reads from IEM
#   inference_and_unresolved     -- inference, conflicts, and open gaps, explicit
#   identity_determination       -- the resulting category, kept SEPARATE from the
#                                    existing pipeline's verification_tier (see
#                                    EVIDENCE_BASIS in map_weather_stations.py) --
#                                    this script never writes to verification_tier
#   remaining_gap                -- what a future round would still need to close
FINDINGS = {
    'ISN': dict(
        current_identifier_mapping='FAA ISN / ICAO KISN, NOAA HOMR ncdcStnId 10007500 (Sloulin Field Intl, Williston ND).',
        period_2018_2019_evidence=(
            'HOMR POR is 1954-09-15 to 2024-02-27 (the record itself stays open past the relocation; '
            'this end date is NOT the facility closure date, see inference_and_unresolved). Remark '
            '"THE LAST LCD PUBLICATION FOR ISN IS SEPTEMBER 2019. THE ASOS IS NOW LOCATED AT XWA" '
            'directly supports that the old Sloulin Field ASOS was still reporting for at least part '
            'of 2018-2019, up to a relocation in 2019.'),
        program_location_linkage=(
            'Platforms on this record: ASOS, PLCD. Remark "AIRPORT WAS RELOCATED 7.5 MILES NORTHWEST" '
            'ties the ASOS program directly to the old Sloulin Field site through the relocation date.'),
        inference_and_unresolved=(
            'IEM archive_end for ISN (2019-10-18, per prior mapping work) is a data-thread boundary, '
            'not necessarily the FAA facility closure date; HOMR record end date (2024-02-27) is a '
            'metadata-record boundary, also not a facility closure date. No FAA 5010 record was fetched '
            'to pin an exact closure date this round; the three dates (2019-09 LCD remark, 2019-10-18 '
            'IEM archive_end, 2024-02-27 HOMR record end) are not the same kind of date and are not '
            'reconciled here.'),
        identity_determination='relocation_confirmed_partial_subperiod',
        historical_continuity_2018_2019='direct_evidence_partial_window_dated_relocation',
        remaining_gap=(
            'Exact FAA facility closure date not independently cross-checked against an FAA 5010 '
            'record this round.'),
    ),
    'XWA': dict(
        current_identifier_mapping='FAA XWA / ICAO KXWA, NOAA HOMR ncdcStnId 30121192 (Williston Basin Intl, Williston ND).',
        period_2018_2019_evidence=(
            'HOMR POR begins 2019-10-25 (station opened partway through the study window). Remark '
            '"WILLISTON AIRPORT ASOS WAS RELOCATED... NOT CONSIDERED CLIMATOLOGICALLY COMPATIBLE WITH '
            'THE PREVIOUS SITE (ISN); THEREFORE, A NEW STATION AND NEW CLIMATE RECORD WILL BE STARTED" '
            'explicitly states this is a new station, not equipment continuity at the old site.'),
        program_location_linkage=(
            'Platforms on this record: COOP, ASOS. A separate remark gives the ASOS ACU coordinates '
            'directly (48.261572, -103.747943), tying the ASOS program to this specific site.'),
        inference_and_unresolved=(
            'IEM archive_begin (2019-10-12, per prior mapping work) and HOMR POR begin (2019-10-25) '
            'differ by 13 days; not resolved this round. Rows dated in this 13-day gap should not be '
            'treated as confirmed under either source without further check.'),
        identity_determination='relocation_confirmed_new_facility_partial_subperiod',
        historical_continuity_2018_2019='direct_evidence_partial_window_dated_new_station',
        remaining_gap='The 13-day IEM/HOMR boundary discrepancy above is unresolved.',
    ),
    'YUM': dict(
        current_identifier_mapping=(
            'Pipeline used candidate_sid="YUM" (=IATA) in AZ_ASOS. The airport this project\'s flight '
            'records actually use (ICAO KNYL per mwgg) maps to NOAA HOMR ncdcStnId 20000934, FAA id '
            'NYL, WBAN 03145.'),
        period_2018_2019_evidence=(
            'KNYL (ncdcStnId 20000934) HOMR POR is 1960-07-01 to Present -- but this POR covers the '
            'COOP thread; the ASOS PLATFORM ENTRY on this same record was added by an ad hoc metadata '
            'update dated 2021-12-16 ("ADDING ASOS PLATFORM PER INFORMATION IN ASOS LOCATION XLS '
            'PROVIDED BY NWS CM OFFICE"). That 2021-12-16 date is a metadata-edit date, not a verified '
            'ASOS installation date, and is NOT treated here as evidence the ASOS itself existed in '
            '2018-2019 -- nor is the 1960-Present POR treated as an ASOS operating window. Separately, '
            'this project\'s own cached IEM AZ_ASOS network file (data/weather_probe/networks/'
            'AZ_ASOS.geojson) lists feature "NYL" with archive_begin 1977-01-27 and archive_end null '
            '(still online), which does directly cover 2018-2019 and is the actual source this '
            'pipeline would read from if it used sid=NYL.'),
        program_location_linkage=(
            'NOAA HOMR\'s FAA:YUM record (ncdcStnId 20000933, COOP only, POR 1946-01-01 to 2007-01-05, '
            'preferredName "YUMA INTL AP", location "WEATHER SERVICE OFFICE YUMA within and 4 mi SSE '
            'of PO", coords 32.66667/-114.6) is a closed, COOP-only station distinct from KNYL/NYL '
            '(coords 32.65/-114.61667, ~2.3 km away, COOP+ASOS, POR 1960-Present).'),
        inference_and_unresolved=(
            'This project\'s OWN cached IEM AZ_ASOS network file also carries a THIRD record, feature '
            '"YUM" (archive_begin 1940-04-01, archive_end 2007-01-05, offline), at coordinates '
            '(-114.606, 32.6566) that match feature "NYL" in the same file, not HOMR\'s FAA:YUM '
            'coordinates -- and it shares NYL\'s ncei91/climate_site codes (USW00003145/CA1424). So '
            'HOMR and this project\'s own IEM cache disagree about where the closed "YUM" station '
            'physically was: HOMR places it at the old Weather Service Office (2.3 km from the '
            'airport); IEM\'s own geojson places its "YUM" feature at the airport (same point as NYL). '
            'This conflict is NOT resolved this round. What both sources agree on, independent of that '
            'conflict, is that "NYL" is the id that continuously covers 2018-2019 at the airport this '
            'pipeline flies into/out of.'),
        identity_determination='identifier_mismatch_partially_resolved_source_conflict',
        historical_continuity_2018_2019='unconfirmed_source_conflict_between_homr_and_iem',
        remaining_gap=(
            'The HOMR-vs-IEM coordinate conflict on the closed "YUM" identifier above is unresolved. '
            'Flagged as a candidate_network_and_sid() bug in notebooks/map_weather_stations.py '
            '(assumes IATA==IEM station sid for ordinary US states, false for Yuma); not changed '
            'automatically this round.'),
    ),
    'STT': dict(
        current_identifier_mapping='FAA STT / ICAO TIST, NOAA HOMR ncdcStnId 20024073 (Cyril E. King Airport, St. Thomas VI).',
        period_2018_2019_evidence=(
            'HOMR POR is 1945-10-01 to Present, covering 2018-2019 directly. utcOffset field reported '
            'as a fixed -4 (no seasonal value given), consistent with a no-DST zone.'),
        program_location_linkage='Platforms on this record: COOP, ASOS. No relocation or renaming remark found.',
        inference_and_unresolved=(
            'US Virgin Islands is widely reported as not observing DST (grouped with AZ/HI/AS/GU/MP/PR '
            'as an exception to the Uniform Time Act), corroborated here only by a secondary source '
            '(Wikipedia summary, DuckDuckGo search 2026-09-18), not an original federal document -- '
            'this favors mwgg\'s "America/St_Thomas" (fixed -4) over IEM\'s "Atlantic/Bermuda" (-4 '
            'standard / -3 DST, since Bermuda itself observes DST), matching the project\'s own '
            'utc_offset_equivalent_2018_2019=False finding for this pair. This inference does NOT '
            'upgrade the tz_conflict_needs_resolution tier.'),
        identity_determination='facility_continuous_tz_string_conflict_leans_mwgg',
        historical_continuity_2018_2019='unconfirmed_current_metadata_only',
        remaining_gap=(
            'No primary US federal document (e.g. a Federal Register notice) fetched to state VI\'s '
            'DST exemption directly. tz_conflict_needs_resolution tier NOT upgraded by this finding, '
            'per existing project policy. Per this round\'s current-vs-historical distinction, the POR '
            'reaching "Present" plus the absence of a relocation remark confirms the current facility, '
            'not specifically the 2018-2019 window.'),
    ),
    'STX': dict(
        current_identifier_mapping='FAA STX / ICAO TISX, NOAA HOMR ncdcStnId 10012328 (Henry E. Rohlsen Airport, St. Croix VI).',
        period_2018_2019_evidence='HOMR POR is 1945-10-01 to Present, covering 2018-2019 directly. utcOffset fixed -4.',
        program_location_linkage=(
            'Platforms on this record: COOP, ASOS. Early history notes "Benedict Field renamed '
            'Alexander Hamilton Field", overlap between military/civilian obs 1945-1948, well before '
            'the study period.'),
        inference_and_unresolved='Same secondary-source-only DST reasoning as STT applies; tier not upgraded.',
        identity_determination='facility_continuous_tz_string_conflict_leans_mwgg',
        historical_continuity_2018_2019='unconfirmed_current_metadata_only',
        remaining_gap='Same as STT, including the current-vs-historical caveat above.',
    ),
    'SPN': dict(
        current_identifier_mapping=(
            'Pipeline tried candidate_sid="PGSN" in GU_ASOS; NOAA HOMR (ICAO PGSN, ncdcStnId 30158921) '
            'gives the FAA id as "GSN", not PGSN or SPN.'),
        period_2018_2019_evidence=(
            'HOMR POR for GSN is 2000-01-11 to Present (platform ASOS), which covers 2018-2019 directly '
            'if this is the station the pipeline needs.'),
        program_location_linkage=(
            'This project\'s cached GU_ASOS network file (data/weather_probe/networks/GU_ASOS.geojson) '
            'has 5 features (PGRO Rota, PGUA Andersen AFB, PGUM Agana/Guam Intl, PGWT West Tinian, '
            'PWAK Wake Island) -- none with sid/id matching GSN, SPN, or PGSN. So the HOMR-confirmed '
            'ASOS at Saipan is not reachable through this pipeline\'s current network-file assumption '
            '(GU_ASOS) under any of the three ids tried.'),
        inference_and_unresolved=(
            'This is evidence of a coverage gap in this pipeline\'s cached network file / network-name '
            'assumption, not evidence that no station exists -- HOMR shows one does. Whether IEM '
            'serves this station under a different network name was not determined this round (would '
            'require a new IEM query, out of scope for this cache-only round).'),
        identity_determination='unconfirmed_pipeline_coverage_gap',
        historical_continuity_2018_2019='unconfirmed_network_coverage_gap',
        remaining_gap=(
            'Needs a targeted IEM network query beyond GU_ASOS to find station GSN; out of scope for '
            'this cache-only round (would be new data collection beyond the 20-airport metadata check).'),
    ),
}

_IDENTIFIER_MISMATCH = {
    'AZA': dict(real_id='IWA', icao='KIWA', net='AZ_ASOS', ncdc_stn_id='10000826', wban='23104',
                name='Phoenix-Mesa Gateway Airport (fmr Williams AFB / Chandler Muni)',
                platform='AWOS',
                remark='"CALL SIGN CHANGED FROM CHD TO IWA, DATE UNKNOWN." POR 1942-03-01 to Present.',
                other_records=(
                    'The same HOMR query (ICAO:KIWA) also returns a SECOND, unrelated record: '
                    'ncdcStnId 30001870, platform NEXRAD, POR 1994-04-27 to Present. That NEXRAD '
                    'record is not used for this finding -- only ncdcStnId 10000826 (AWOS) is.'),
                continuity_basis='undated_remark'),
    'BKG': dict(real_id='BBG', icao='KBBG', net='MO_ASOS', ncdc_stn_id='30083225', wban=None,
                name='Branson Airport, MO', platform='AWOS',
                remark='No relocation remark; station opened with the airport. POR 2008-10-01 to Present.',
                other_records=None, continuity_basis='absence_of_remark'),
    'FCA': dict(real_id='GPI', icao='KGPI', net='MT_ASOS', ncdc_stn_id='20012742', wban=None,
                name='Glacier Park International / Kalispell Glacier Airport, MT', platform='ASOS',
                remark='"STATION ID CHANGED FROM FCA TO GPI AT 14Z (7AM MST) ON OCTOBER 25, 2005." POR 1896-06-29 to Present.',
                other_records=None, continuity_basis='dated_remark_predates_window'),
    'HHH': dict(real_id='HXD', icao='KHXD', net='SC_ASOS', ncdc_stn_id='20017315', wban=None,
                name='Hilton Head Airport, SC', platform='AWOS',
                remark='No relocation remark found. POR 1972-08-01 to Present.', other_records=None,
                continuity_basis='absence_of_remark'),
    'MQT': dict(real_id='SAW', icao='KSAW', net='MI_ASOS', ncdc_stn_id='10005416', wban=None,
                name='Sawyer International Airport (fmr K.I. Sawyer AFB), MI', platform='AWOS',
                remark='No relocation remark found (AFB->civilian transition 1995, before archive continuity). POR 1956-10-01 to Present.',
                other_records=None, continuity_basis='absence_of_remark'),
    'PBI': dict(real_id='DJT (current); NWSLI retains PBI', icao='KDJT (current); was KPBI', net='FL_ASOS',
                ncdc_stn_id='20004306', wban=None,
                name='West Palm Beach -> Donald J. Trump International Airport, FL', platform='COOP+ASOS',
                remark=('NWSLI field = "PBI" (retained); current FAA id = "DJT", current ICAO id = "KDJT"; '
                        'single continuous ASOS/COOP station since 1938-07-01, no relocation remark. The '
                        'fetched record carries no remark stating WHEN the FAA identifier changed from PBI '
                        'to DJT, or that the airport used "PBI"/"Palm Beach International" specifically '
                        'throughout 2018-2019 -- the only identifier-related entry is an ad hoc metadata '
                        'update ("UPDATING STATION NAME PER FAA (AIRNAV)", enteredDate 2026-07-15), which is '
                        'a metadata-edit date, not a rename date. That the FAA/legal rename to Donald J. '
                        'Trump International Airport postdates the 2018-2019 study window is background '
                        'knowledge from outside this HOMR record, not a statement this remark makes.'),
                other_records=None, continuity_basis='rename_confirmed_by_current_ids_undated_in_record'),
    'SCE': dict(real_id='UNV', icao='KUNV', net='PA_ASOS', ncdc_stn_id='20016919', wban=None,
                name='State College Regional / University Park Airport, PA', platform='AWOS',
                remark='No relocation remark found. POR 1972-04-01 to Present.', other_records=None,
                continuity_basis='absence_of_remark'),
    'USA': dict(real_id='JQF', icao='KJQF', net='NC_ASOS', ncdc_stn_id='30002219', wban=None,
                name='Concord Regional Airport, NC', platform='AWOS',
                remark='No relocation remark found. POR 1995-09-01 to Present.', other_records=None,
                continuity_basis='absence_of_remark'),
}
# Third round: `inference_and_unresolved` used to read the same fixed 'none beyond
# remaining_gap below.' for all 8 airports, regardless of what kind of evidence
# each one actually has. `current_identifier_mapping`/`program_location_linkage`
# above only establish that the CURRENT alternate id maps to the same facility;
# separately from that, `continuity_basis` distinguishes airports with a dated (or
# explicitly time-bounded) HOMR remark (FCA, PBI) from those whose only basis is an
# unbroken POR plus the absence of a relocation remark (AZA -- whose own remark is
# undated -- and BKG/HHH/MQT/SCE/USA). Absence of a remark is evidence of silence,
# not a positive record of no change, and is now stated as such instead of left
# blank.
_CONTINUITY_TEXT = {
    'dated_remark_predates_window': (
        'The identifier-change remark above is explicitly DATED and predates 2018-2019, so the CURRENT '
        'id was already in place going into the study window. This does not independently confirm no '
        'further change happened between that date and 2018-2019 -- only that no such change is recorded '
        'in this fetched response.'),
    'rename_confirmed_by_current_ids_undated_in_record': (
        'The current identifiers (FAA=DJT, ICAO=KDJT, NWSLI retaining PBI) and the absence of a relocation '
        'remark are confirmed directly from this record -- but unlike FCA, this record carries no DATED '
        'remark for the identifier change itself, only an undated ad hoc metadata-name-update entry. That '
        'the rename postdates 2018-2019 is background/outside knowledge, not a claim this fetched HOMR '
        'response makes; do not attribute that timing claim to the record\'s own remarks.'),
    'undated_remark': (
        'The identifier-change remark above gives no date ("DATE UNKNOWN"), so it cannot bracket the '
        'change against 2018-2019 either way. Current-identifier resolution (AZA=IWA) does not by itself '
        'establish when that resolution became true.'),
    'absence_of_remark': (
        'No relocation or renaming remark was found in this record. That is an absence of contrary '
        'evidence, not a positive record confirming continuity specifically through 2018-2019 -- the POR '
        'reaching "Present" is a metadata boundary, not a per-year confirmation.'),
}
for _iata, _d in _IDENTIFIER_MISMATCH.items():
    FINDINGS[_iata] = dict(
        current_identifier_mapping=(
            f'Pipeline candidate_sid="{_iata}" (=IATA) not found in {_d["net"]}; actual IEM/FAA sid is '
            f'"{_d["real_id"]}" ({_d["icao"]}, {_d["name"]}, NOAA HOMR ncdcStnId {_d["ncdc_stn_id"]})'),
        period_2018_2019_evidence=f'NOAA HOMR ({_d["icao"]}): {_d["remark"]}',
        program_location_linkage=(
            f'Platform: {_d["platform"]}.' + (f' {_d["other_records"]}' if _d.get('other_records') else '')),
        inference_and_unresolved=_CONTINUITY_TEXT[_d['continuity_basis']],
        identity_determination='identifier_mismatch_resolved_same_facility',
        historical_continuity_2018_2019=_d['continuity_basis'],
        remaining_gap=(
            'Flagged for a follow-up code check to candidate_network_and_sid() in '
            'notebooks/map_weather_stations.py, which currently assumes IATA==IEM station sid for '
            'ordinary US states; not changed automatically this round. Current-identifier resolution is '
            'confirmed; 2018-2019 historical continuity beyond that, where continuity_basis is '
            '"absence_of_remark", "undated_remark", or "rename_confirmed_by_current_ids_undated_in_record" '
            '(PBI: current ids confirmed, but no dated rename remark in this record), is not independently '
            'confirmed this round.'),
    )

_TZ_ALIAS_CONFIRMED = {
    'IMT': dict(icao='KIMT', ncdc_stn_id='20010418', name='Ford Airport, Iron Mountain/Kingsford, MI',
                remark='No relocation remark found. ASOS platform on this record. POR 1949-12-01 to Present.'),
    'SDF': dict(icao='KSDF', ncdc_stn_id='10004692', name='Louisville International / Standiford Field, KY',
                remark='No relocation remark found. ASOS ACU-coordinate remark on this record. POR 1947-11-15 to Present.'),
    'SIT': dict(icao='PASI', ncdc_stn_id='20021837', name='Sitka Rocky Gutierrez Airport, AK',
                remark='"REVIEWED. NO KNOWN CHANGES." ASOS ACU-coordinate remark on this record. POR 1930-08-01 to Present.'),
}
# IMT/SDF/SIT: the fetched HOMR record confirms the CURRENT facility/identifier and
# that its platform list includes ASOS today, but carries no dated event at all --
# only current metadata (ACU coordinates, an undated "reviewed, no known changes"
# note, or the absence of a relocation remark). None of that is dated evidence that
# specifically the 2018-2019 window was unchanged; a POR ending "Present" is a
# metadata boundary, not a per-year confirmation (same caveat this script already
# applies to ISN/XWA's HOMR record-end dates). Third round: this is recorded as
# current-facility-confirmed with historical continuity for 2018-2019 left
# unconfirmed, instead of folded into one "facility continuity confirmed" category.
for _iata, _d in _TZ_ALIAS_CONFIRMED.items():
    FINDINGS[_iata] = dict(
        current_identifier_mapping=(
            f'IEM sid = {_iata} directly ({_d["icao"]}, {_d["name"]}, NOAA HOMR ncdcStnId '
            f'{_d["ncdc_stn_id"]}); issue is tzname STRING mismatch (already '
            'utc_offset_equivalent_2018_2019=True in prior work), not identifier lookup.'),
        period_2018_2019_evidence=(
            f'NOAA HOMR ({_d["icao"]}): {_d["remark"]} This is CURRENT metadata (an undated remark and/or '
            'a POR ending "Present"), not a dated event bracketing 2018-2019 specifically -- unlike KTN '
            'below, this record has no dated pre-period event to point to either.'),
        program_location_linkage=(
            'This record itself lists an ASOS platform with ASOS-specific remarks (ACU coordinates or an '
            'undated review note), directly linking the CURRENT observing program to this site -- unlike '
            'PSG/WRG below, whose fetched HOMR record carries no ASOS platform entry at all. This confirms '
            'program linkage today; it does not by itself confirm the linkage held throughout 2018-2019.'),
        inference_and_unresolved=(
            'Current ASOS platform + ACU coordinates/metadata + a POR that reaches "Present" do not, by '
            'themselves, confirm 2018-2019 continuity -- they confirm the current record, and the study '
            'window falls somewhere inside an open-ended POR with no dated remark either way. Unlike KTN, '
            'no dated event was found in this record to bracket the study period against.'),
        identity_determination='current_facility_confirmed_historical_continuity_unconfirmed_tz_string_conflict_unchanged',
        historical_continuity_2018_2019='unconfirmed_current_metadata_only',
        remaining_gap=(
            'tzname STRING conflict vs mwgg intentionally left at tz_conflict_needs_resolution per '
            'existing policy. This round confirms the current facility/identifier and program linkage, '
            'but explicitly leaves 2018-2019 historical continuity unconfirmed for lack of a dated record; '
            'a future round would need a dated remark or an FAA 5010 record covering that window.'),
    )

# KTN: unlike IMT/SDF/SIT above, this record carries an explicitly DATED event --
# the 1997 "compatible station move" that commissioned the ASOS -- kept as explicit
# historical evidence per this round's review. That date is still 21 years before
# 2018-2019, so it is not itself proof the 2018-2019 window was unchanged either;
# it only establishes that, as of 1997, this was a deliberate "compatible" (not
# climatologically disruptive) consolidation, distinguished by NOAA from ISN/XWA's
# incompatible relocation. Recorded as its own category rather than merged with
# IMT/SDF/SIT's current-metadata-only evidence or overstated as full-period proof.
FINDINGS['KTN'] = dict(
    current_identifier_mapping=(
        'IEM sid = KTN directly (PAKT, Ketchikan International Airport, AK, NOAA HOMR ncdcStnId '
        '10000202); issue is tzname STRING mismatch (already utc_offset_equivalent_2018_2019=True in '
        'prior work), not identifier lookup.'),
    period_2018_2019_evidence=(
        '"THIS WAS A COMPATIBLE STATION MOVE TO RELOCATE 50-4590 KETCHIKAN, TO THE ASOS WHICH WAS '
        'COMMISSIONED MAY 23, 1997. STATION MOVED NW 5110 YARDS." -- an explicit, DATED "compatible" '
        'equipment/site consolidation in 1997, 21 years before 2018-2019, explicitly distinguished by '
        'NOAA from an incompatible relocation (contrast with ISN/XWA above). This is dated evidence about '
        '1997, not direct evidence about 2018-2019 itself -- the gap between 1997 and the study window is '
        'still bridged only by the POR reaching "Present" and the absence of any later relocation remark.'),
    program_location_linkage=(
        '"ASOS COMMISSIONED MAY 23, 1997" remark and an ASOS ACU-coordinate remark are both on this '
        'record, tying the ASOS program to this specific site from 1997 onward. POR 1973-10-03 to Present.'),
    inference_and_unresolved=(
        'The 1997 event is real, dated evidence -- but it predates the study period by 21 years. Nothing '
        'in this record dates a change (or confirms no change) between 1997 and 2018-2019; that gap is '
        'bridged only by an open-ended POR and the absence of a later remark, the same limitation as '
        'IMT/SDF/SIT, just with one additional dated fact from before the window.'),
    identity_determination='facility_continuity_supported_by_dated_1997_event_tz_string_conflict_unchanged',
    historical_continuity_2018_2019='indirect_dated_pre_period_event_not_direct_2018_2019_confirmation',
    remaining_gap=(
        'tzname STRING conflict vs mwgg intentionally left at tz_conflict_needs_resolution per existing '
        'policy. The 1997 dated event is kept as explicit historical evidence, but does not by itself '
        'confirm the 2018-2019 window; a future round would still need a remark or record dated closer to '
        'or within 2018-2019 to close that gap.'),
)

# PSG and WRG: the HOMR record this script fetches for both is COOP-only -- no ASOS
# platform entry appears anywhere in the response. Downgraded from the confirmed
# category above: nothing in the fetched record links the ASOS platform this
# pipeline actually reads from IEM to this station's history.
_TZ_ALIAS_UNLINKED = {
    'PSG': dict(icao='PAPG', ncdc_stn_id='20021826', name='Petersburg James A Johnson Airport, AK',
                remark=('Platform on this record: COOP only. COOP-side record shows a station-id rename '
                        'PSG->APGA2 for the manual-observer COOP program, unrelated to the ASOS id this '
                        'pipeline uses; no relocation remark found. POR 1924-10-28 to Present (COOP thread).')),
    'WRG': dict(icao='PAWG', ncdc_stn_id='10000446', name='Wrangell Airport, AK',
                remark=('Platform on this record: COOP only. Remark "STATION INACTIVE SINCE 10/23/2012" '
                        '(the manual-observer COOP program specifically -- during the 2018-2019 study '
                        'window). A separate remark, "CORRECTED LAT AND LONG TO BE AT SRG AND NOT AT '
                        'AWOS", corrects the COOP rain-gauge (SRG) coordinates to distinguish them from '
                        'an AWOS at the same airport, but that remark does not itself carry the AWOS/ASOS '
                        'record\'s own identifiers, platform entry, or period of record -- it only '
                        'confirms an AWOS exists nearby, not that this COOP record\'s history applies to '
                        'it. POR (COOP thread) 1984-11-15 to Present.')),
}
for _iata, _d in _TZ_ALIAS_UNLINKED.items():
    FINDINGS[_iata] = dict(
        current_identifier_mapping=(
            f'IEM sid = {_iata} directly ({_d["icao"]}, {_d["name"]}); issue is tzname STRING mismatch '
            '(already utc_offset_equivalent_2018_2019=True in prior work), not identifier lookup.'),
        period_2018_2019_evidence=(
            f'NOAA HOMR ({_d["icao"]}, ncdcStnId {_d["ncdc_stn_id"]}): {_d["remark"]}'),
        program_location_linkage=(
            'No ASOS/AWOS platform entry appears anywhere in this HOMR record -- only COOP. This '
            'pipeline reads the automated (ASOS/AWOS) observation this airport\'s IEM sid actually '
            'serves, which this fetched record does not describe or link to.'),
        inference_and_unresolved=(
            'The COOP program\'s continuity (or, for WRG, its 2012 inactivation) says nothing directly '
            'about the separate automated platform this pipeline uses in 2018-2019. Treating this COOP '
            'record as confirming that automated platform\'s continuity would be an unsupported leap.'),
        identity_determination='unconfirmed_program_linkage_insufficient',
        historical_continuity_2018_2019='unconfirmed_program_linkage_insufficient',
        remaining_gap=(
            'Needs a HOMR query or other official record that actually returns this airport\'s ASOS/'
            'AWOS platform entry (this ICAO query only returned the COOP thread); not found this round.'),
    )

assert set(FINDINGS) == set(HOMR_QUERY) == set(ISSUE_GROUP), 'FINDINGS/HOMR_QUERY/ISSUE_GROUP must cover the same 20 airports'

# Structured record selection + program-linkage checks that main() actually runs
# against the fetched HOMR JSON, so a FINDINGS entry's claim about a specific
# ncdcStnId/platform is checked against the live response, not just asserted in
# prose. 'source' is 'primary' (the HOMR_QUERY cache file) or 'extra' (the
# HOMR_EXTRA_QUERY[iata][extra_index] cache file). 'requires'/'forbids' are the
# platform strings that selected station's `platforms` list must/must not contain.
EXPECTED_RECORD: dict[str, list[dict]] = {
    'ISN': [dict(source='primary', ncdc_stn_id='10007500', requires={'ASOS'}, forbids=set())],
    'XWA': [dict(source='primary', ncdc_stn_id='30121192', requires={'ASOS'}, forbids=set())],
    'YUM': [
        dict(source='primary', ncdc_stn_id='20000933', requires=set(), forbids={'ASOS', 'AWOS'}),
        dict(source='extra', extra_index=0, ncdc_stn_id='20000934', requires={'ASOS'}, forbids=set()),
    ],
    'STT': [dict(source='primary', ncdc_stn_id='20024073', requires={'ASOS'}, forbids=set())],
    'STX': [dict(source='primary', ncdc_stn_id='10012328', requires={'ASOS'}, forbids=set())],
    'SPN': [dict(source='primary', ncdc_stn_id='30158921', requires={'ASOS'}, forbids=set())],
    # AZA: ICAO:KIWA returns two unrelated records in the same response; both are
    # selected and checked separately by ncdcStnId so the AWOS record this finding
    # relies on is never conflated with the NEXRAD one.
    'AZA': [
        dict(source='primary', ncdc_stn_id='10000826', requires={'AWOS'}, forbids={'NEXRAD'}),
        dict(source='primary', ncdc_stn_id='30001870', requires={'NEXRAD'}, forbids={'AWOS', 'ASOS'}),
    ],
}
for _iata, _d in _IDENTIFIER_MISMATCH.items():
    if _iata == 'AZA':
        continue
    EXPECTED_RECORD[_iata] = [dict(
        source='primary', ncdc_stn_id=_d['ncdc_stn_id'], requires=set(_d['platform'].split('+')), forbids=set())]
for _iata, _d in _TZ_ALIAS_CONFIRMED.items():
    EXPECTED_RECORD[_iata] = [dict(source='primary', ncdc_stn_id=_d['ncdc_stn_id'], requires={'ASOS'}, forbids=set())]
EXPECTED_RECORD['KTN'] = [dict(source='primary', ncdc_stn_id='10000202', requires={'ASOS'}, forbids=set())]
for _iata, _d in _TZ_ALIAS_UNLINKED.items():
    EXPECTED_RECORD[_iata] = [dict(
        source='primary', ncdc_stn_id=_d['ncdc_stn_id'], requires=set(), forbids={'ASOS', 'AWOS'})]

assert set(EXPECTED_RECORD) == set(FINDINGS), 'EXPECTED_RECORD must cover the same 20 airports as FINDINGS'

# Fourth round (this revision): the platform-only checks above catch a record whose
# platforms list changes, but not one whose `identifiers` are deleted or swapped to a
# different FAA/ICAO/NWSLI/NEXRAD value while ncdcStnId and platforms stay untouched --
# every FINDINGS/EXPECTED_RECORD claim about "current identifier X" was, until now,
# asserted in prose only. Keyed by ncdcStnId (unique across all 22 checks above, per the
# assertion below). This intentionally does NOT require every HOMR_QUERY lookup key to
# also appear in the selected record's own identifiers -- PBI is queried by FAA:PBI (the
# key NOAA's search index still resolves), but the record's OWN current identifiers are
# FAA=DJT/ICAO=KDJT/NWSLI=PBI; the lookup key and the record's current identifiers are
# allowed to differ, and only the identifiers this pipeline's judgment actually depends
# on are checked here.
EXPECTED_IDENTIFIERS: dict[str, list[tuple[str, str]]] = {
    '10007500': [('FAA', 'ISN'), ('ICAO', 'KISN')],
    '30121192': [('FAA', 'XWA'), ('ICAO', 'KXWA'), ('NWSLI', 'XWA')],
    '20000933': [('FAA', 'YUM'), ('ICAO', 'KYUM')],
    '20000934': [('FAA', 'NYL'), ('ICAO', 'KNYL'), ('NWSLI', 'NYL')],
    '20024073': [('FAA', 'STT'), ('ICAO', 'TIST'), ('NWSLI', 'STT')],
    '10012328': [('FAA', 'STX'), ('ICAO', 'TISX'), ('NWSLI', 'STX')],
    '30158921': [('FAA', 'GSN'), ('ICAO', 'PGSN')],
    '10000826': [('FAA', 'IWA'), ('ICAO', 'KIWA')],
    '30001870': [('ICAO', 'KIWA'), ('NEXRAD', 'KIWA')],
    '30083225': [('FAA', 'BBG'), ('ICAO', 'KBBG')],
    '20012742': [('FAA', 'GPI'), ('ICAO', 'KGPI'), ('NWSLI', 'GPI')],
    '20017315': [('FAA', 'HXD'), ('ICAO', 'KHXD')],
    '10005416': [('FAA', 'SAW'), ('ICAO', 'KSAW')],
    '20004306': [('FAA', 'DJT'), ('ICAO', 'KDJT'), ('NWSLI', 'PBI')],
    '20016919': [('FAA', 'UNV'), ('ICAO', 'KUNV')],
    '30002219': [('FAA', 'JQF'), ('ICAO', 'KJQF')],
    '20010418': [('FAA', 'IMT'), ('ICAO', 'KIMT')],
    '10000202': [('FAA', 'KTN'), ('ICAO', 'PAKT'), ('NWSLI', 'KTN')],
    '20021826': [('FAA', 'PSG'), ('ICAO', 'PAPG'), ('NWSLI', 'APGA2')],
    '10004692': [('FAA', 'SDF'), ('ICAO', 'KSDF'), ('NWSLI', 'SDF')],
    '20021837': [('FAA', 'SIT'), ('ICAO', 'PASI'), ('NWSLI', 'SIT')],
    '10000446': [('FAA', 'WRG'), ('ICAO', 'PAWG')],
}
_all_expected_ncdc_stn_ids = {chk['ncdc_stn_id'] for checks in EXPECTED_RECORD.values() for chk in checks}
assert _all_expected_ncdc_stn_ids == set(EXPECTED_IDENTIFIERS), (
    'EXPECTED_IDENTIFIERS must cover exactly the ncdcStnIds referenced by EXPECTED_RECORD: '
    f'missing={_all_expected_ncdc_stn_ids - set(EXPECTED_IDENTIFIERS)} '
    f'extra={set(EXPECTED_IDENTIFIERS) - _all_expected_ncdc_stn_ids}')
for _checks in EXPECTED_RECORD.values():
    for _chk in _checks:
        _chk['expected_ids'] = EXPECTED_IDENTIFIERS[_chk['ncdc_stn_id']]

# Non-SPN entries in IEM_FEATURE_CHECK must be found=True for the corresponding
# FINDINGS determination to hold (e.g. YUM's judgment needs both the YUM and NYL
# AZ_ASOS features; AZA needs its IWA feature). SPN's expected miss is the one
# allowed exception -- FINDINGS['SPN'] is already 'unconfirmed_pipeline_coverage_gap'
# and does not depend on a found feature.
IEM_FEATURE_REQUIRED_FOUND = {iata for iata in IEM_FEATURE_CHECK if iata != 'SPN'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_homr(id_type: str, id_value: str, cache_dir: Path, allow_network: bool) -> dict:
    cache = cache_dir / f'{id_type}_{id_value}.json'
    if not cache.exists():
        if not allow_network:
            raise FileNotFoundError(
                f'cache-only mode: {cache} not found and --allow-network was not passed. '
                f'Refusing to fall back to a network fetch for {id_type}:{id_value}.')
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = f'{HOMR_BASE}?qid={id_type}:{id_value}'
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
    url = f'{HOMR_BASE}?qid={id_type}:{id_value}'
    fetch_dt = datetime.fromtimestamp(cache.stat().st_mtime, tz=timezone.utc)
    return {
        'url': url,
        'cache_file': str(cache.relative_to(ROOT)) if cache.is_relative_to(ROOT) else str(cache),
        'sha256': digest(cache),
        # The cache file's own filesystem mtime is the best available proxy for when
        # it was first fetched -- this project keeps no separate per-request fetch
        # log (same limitation noted elsewhere for weather-collection timing). It is
        # NOT a verified fetch log, so it is labeled as an estimate, and is reported
        # separately from evidence_reviewed_date (this run's assembly date) below.
        'fetch_date_estimate': fetch_dt.strftime('%Y-%m-%d'),
        'fetch_date_basis': 'filesystem_mtime_estimate_not_a_verified_fetch_log',
    }


def iem_feature_check(network: str, sid: str) -> dict:
    path = NETWORKS_DIR / f'{network}.geojson'
    if not path.exists():
        raise FileNotFoundError(f'IEM network cache not found: {path}')
    d = json.loads(path.read_text(encoding='utf-8'))
    sha = digest(path)
    match = [f for f in d['features'] if f.get('properties', {}).get('sid') == sid]
    rel = str(path.relative_to(ROOT))
    if not match:
        return {
            'network': network, 'sid': sid, 'file': rel, 'sha256': sha,
            'found': False, 'summary': 'not found in cached network file',
        }
    p = match[0]['properties']
    coords = match[0].get('geometry', {}).get('coordinates')
    summary = (
        f'archive_begin={p.get("archive_begin")} archive_end={p.get("archive_end")} '
        f'online={p.get("online")} coords={coords}')
    return {'network': network, 'sid': sid, 'file': rel, 'sha256': sha, 'found': True, 'summary': summary}


def load_station_collection(cache_file: Path) -> list[dict]:
    """Parse a fetched HOMR response and return its stations, or raise ValueError.

    Fetching and hashing a cache file (fetch_homr) is not the same as confirming the
    response actually contains a usable station record -- an empty cache ({} or a
    stationCollection with no stations) hashes and caches fine but carries nothing
    for select_expected_station() to select from.
    """
    obj = json.loads(cache_file.read_text(encoding='utf-8'))
    stations = obj.get('stationCollection', {}).get('stations')
    if not stations:
        raise ValueError(f'{cache_file}: stationCollection.stations is missing or empty')
    return stations


def select_expected_station(stations: list[dict], ncdc_stn_id: str) -> dict | None:
    for stn in stations:
        if str(stn.get('ncdcStnId')) == str(ncdc_stn_id):
            return stn
    return None


def verify_expected_record(iata: str, cache_files: dict[str, Path], checks: list[dict]) -> list[str]:
    """Check EXPECTED_RECORD[iata] against the actually fetched HOMR JSON.

    cache_files maps 'primary' and 'extra:<index>' to the cache file main() fetched
    for that source. Returns a list of failure strings (empty if every check in
    `checks` selected its ncdcStnId, that station's platforms satisfied
    requires/forbids, AND its identifiers included every pair in `expected_ids`);
    never raises for a normal mismatch, so callers can collect failures across all
    20 airports before deciding whether to abort.
    """
    failures = []
    for chk in checks:
        key = 'primary' if chk['source'] == 'primary' else f"extra:{chk['extra_index']}"
        cache_file = cache_files[key]
        try:
            stations = load_station_collection(cache_file)
        except ValueError as exc:
            failures.append(f'{iata}: {exc}')
            continue
        stn = select_expected_station(stations, chk['ncdc_stn_id'])
        if stn is None:
            failures.append(f"{iata}: ncdcStnId {chk['ncdc_stn_id']} not selectable in {cache_file.name}")
            continue
        platforms = {p.get('platform') for p in stn.get('platforms', [])}
        missing = chk['requires'] - platforms
        if missing:
            failures.append(
                f"{iata}: ncdcStnId {chk['ncdc_stn_id']} in {cache_file.name} is missing required "
                f'platform(s) {sorted(missing)}; record has {sorted(platforms)}')
        blocked = chk['forbids'] & platforms
        if blocked:
            failures.append(
                f"{iata}: ncdcStnId {chk['ncdc_stn_id']} in {cache_file.name} carries forbidden "
                f'platform(s) {sorted(blocked)}; record has {sorted(platforms)}')
        actual_ids = {(i.get('idType'), i.get('id')) for i in stn.get('identifiers', [])}
        missing_ids = [pair for pair in chk.get('expected_ids', []) if pair not in actual_ids]
        if missing_ids:
            failures.append(
                f"{iata}: ncdcStnId {chk['ncdc_stn_id']} in {cache_file.name} is missing required "
                f'identifier(s) {missing_ids}; record has identifiers {sorted(actual_ids)}')
    return failures


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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--cache-dir', default=None,
                     help='Existing HOMR raw-response cache directory to reuse (relative to repo root or '
                          'absolute). Defaults to the cache already checked in under data/weather_probe/ '
                          'from the first investigation round, so a fresh --name reproduces this round '
                          'without a new external query.')
    ap.add_argument('--allow-network', action='store_true',
                     help='Allow fetching from HOMR over the network for any query missing from '
                          '--cache-dir. Without this flag, a missing cache entry fails immediately '
                          'instead of silently falling back to the network.')
    args = ap.parse_args(argv)
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
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else DEFAULT_CACHE_DIR
    if not args.allow_network and not cache_dir.exists():
        raise FileNotFoundError(
            f'cache-only mode: cache directory {cache_dir} does not exist and --allow-network was not '
            'passed. This reproduction does not create a new cache from scratch over the network.')

    reviewed = time.strftime('%Y-%m-%d')
    homr_results: dict[str, dict] = {}
    extra_results: dict[str, list[dict]] = {}
    for iata, (id_type, id_value) in sorted(HOMR_QUERY.items()):
        homr_results[iata] = fetch_homr(id_type, id_value, cache_dir, args.allow_network)
        extra_results[iata] = [
            {**fetch_homr(t, v, cache_dir, args.allow_network), 'record_ref': f'{t}:{v}'}
            for t, v in HOMR_EXTRA_QUERY.get(iata, [])
        ]

    iem_results: dict[str, list[dict]] = {
        iata: [iem_feature_check(net, sid) for net, sid in checks]
        for iata, checks in IEM_FEATURE_CHECK.items()
    }

    # Connect raw-record verification to this actual generation path: select each
    # FINDINGS judgment's cited ncdcStnId from the JSON just fetched above and check
    # its platforms, instead of only hashing the cache file's bytes. A cache file
    # existing and hashing is not evidence that the record it contains still says
    # what FINDINGS claims; both are required, and record_verification_failures
    # below is empty only when both hold for every one of the 20 airports.
    record_verification_failures: list[str] = []
    record_verification_checks: dict[str, list[str]] = {}
    for iata in sorted(HOMR_QUERY):
        id_type, id_value = HOMR_QUERY[iata]
        cache_files = {'primary': cache_dir / f'{id_type}_{id_value}.json'}
        for i, (t, v) in enumerate(HOMR_EXTRA_QUERY.get(iata, [])):
            cache_files[f'extra:{i}'] = cache_dir / f'{t}_{v}.json'
        checks = EXPECTED_RECORD[iata]
        record_verification_failures.extend(verify_expected_record(iata, cache_files, checks))
        record_verification_checks[iata] = [
            f"ncdcStnId={c['ncdc_stn_id']} requires={sorted(c['requires'])} forbids={sorted(c['forbids'])} "
            f"expected_ids={sorted(c.get('expected_ids', []))}"
            for c in checks
        ]
    for iata in sorted(IEM_FEATURE_REQUIRED_FOUND):
        for r in iem_results[iata]:
            if not r['found']:
                record_verification_failures.append(
                    f"{iata}: required IEM feature {r['network']}:{r['sid']} not found in {r['file']} "
                    "(SPN is the only allowed miss)")
    if record_verification_failures:
        raise RuntimeError(
            'Raw-record verification failed against the fetched HOMR/IEM cache; refusing to write '
            'evidence that would print FINDINGS text the raw evidence no longer supports:\n  - '
            + '\n  - '.join(record_verification_failures))

    columns = [
        'iata', 'prior_verification_tier', 'issue_category', 'appears_in_stratafix_sample',
        'current_identifier_mapping', 'period_2018_2019_evidence', 'program_location_linkage',
        'historical_continuity_2018_2019', 'inference_and_unresolved', 'identity_determination',
        'remaining_gap', 'evidence_agency', 'evidence_source_url', 'evidence_record_ref',
        'raw_evidence_file', 'raw_evidence_sha256',
        'raw_evidence_fetch_date_estimate', 'raw_evidence_fetch_date_basis',
        'extra_raw_evidence', 'iem_cross_check', 'record_verification_checks', 'evidence_reviewed_date',
    ]
    evidence_path = out / f'{args.name}_evidence.csv'
    with open(evidence_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for iata in sorted(HOMR_QUERY):
            fnd = FINDINGS[iata]
            hr = homr_results[iata]
            id_type, id_value = HOMR_QUERY[iata]
            extra = '; '.join(
                f'{e["record_ref"]}|{e["cache_file"]}|{e["sha256"]}' for e in extra_results[iata])
            iem = '; '.join(
                f'{r["network"]}:{r["sid"]}|{r["file"]}|{r["sha256"]}|found={r["found"]}|{r["summary"]}'
                for r in iem_results.get(iata, []))
            w.writerow({
                'iata': iata,
                'prior_verification_tier': tiers[iata],
                'issue_category': ISSUE_GROUP[iata],
                'appears_in_stratafix_sample': strata[iata],
                'current_identifier_mapping': fnd['current_identifier_mapping'],
                'period_2018_2019_evidence': fnd['period_2018_2019_evidence'],
                'program_location_linkage': fnd['program_location_linkage'],
                'historical_continuity_2018_2019': fnd['historical_continuity_2018_2019'],
                'inference_and_unresolved': fnd['inference_and_unresolved'],
                'identity_determination': fnd['identity_determination'],
                'remaining_gap': fnd['remaining_gap'],
                'evidence_agency': 'NOAA/NCEI HOMR (Historical Observing Metadata Repository)',
                'evidence_source_url': hr['url'],
                'evidence_record_ref': f'{id_type}:{id_value}',
                'raw_evidence_file': hr['cache_file'],
                'raw_evidence_sha256': hr['sha256'],
                'raw_evidence_fetch_date_estimate': hr['fetch_date_estimate'],
                'raw_evidence_fetch_date_basis': hr['fetch_date_basis'],
                'extra_raw_evidence': extra,
                'iem_cross_check': iem,
                'record_verification_checks': '; '.join(record_verification_checks[iata]),
                'evidence_reviewed_date': reviewed,
            })

    manifest = {
        'name': args.name,
        'purpose': (
            'Official-source (NOAA/NCEI HOMR) cross-check of the 20 priority airports flagged by '
            f'{PRIORITY_CSV.relative_to(ROOT)}. Scope fixed to exactly these 20 airports; they are '
            'disjoint from the 355 confirmed_period airports, which this script does not touch.'),
        'fits_any_model': False, 'uses_external_data': bool(args.allow_network),
        'target_columns_used': [],
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'cache_mode': 'allow_network' if args.allow_network else 'cache_only',
        'cache_dir': str(cache_dir.relative_to(ROOT)) if cache_dir.is_relative_to(ROOT) else str(cache_dir),
        'priority_csv': {'path': str(PRIORITY_CSV.relative_to(ROOT)), 'sha256': digest(PRIORITY_CSV)},
        'stratafix_selection_csv': {
            'path': str(STRATAFIX_SELECTION_CSV.relative_to(ROOT)), 'sha256': digest(STRATAFIX_SELECTION_CSV)},
        'homr_base_url': HOMR_BASE,
        'homr_queries': {iata: f'{t}:{v}' for iata, (t, v) in sorted(HOMR_QUERY.items())},
        'homr_extra_queries': {iata: [f'{t}:{v}' for t, v in qs] for iata, qs in sorted(HOMR_EXTRA_QUERY.items())},
        'iem_feature_checks': {
            iata: [{'network': r['network'], 'sid': r['sid'], 'file': r['file'], 'sha256': r['sha256'],
                    'found': r['found']} for r in rows]
            for iata, rows in sorted(iem_results.items())
        },
        'evidence_table': str(evidence_path.relative_to(ROOT)),
        'evidence_table_sha256': digest(evidence_path),
        'record_verification': {
            'passed': True,
            'checks_run': sum(len(v) for v in record_verification_checks.values()),
            'iem_features_required_found': sorted(IEM_FEATURE_REQUIRED_FOUND),
            'note': (
                'True here means every EXPECTED_RECORD check for all 20 airports selected its configured '
                'ncdcStnId from the fetched response and matched its required/forbidden platforms AND its '
                'required identifiers (EXPECTED_IDENTIFIERS, fourth round), and every non-SPN '
                'IEM_FEATURE_CHECK entry was found=True -- main() raises RuntimeError before writing any '
                'evidence file if that is not the case, so a run that reached this point and wrote output '
                'always has passed=True.'),
        },
        'row_count': len(HOMR_QUERY),
        'stratafix_membership_count': sum(strata.values()),
        'population_note': (
            '20 rows here are confirmed_current_only(3) + tz_conflict_needs_resolution(8) + '
            'unconfirmed(9) from the prior mapping round; they do not overlap the 355 '
            'confirmed_period airports, which remain uninvestigated by this script.'),
        'limitations': [
            'This script fetches and hashes NOAA/NCEI HOMR raw responses and cross-checks cached IEM '
            'network GeoJSON features deterministically, but the per-airport judgment text in FINDINGS '
            'is human-authored analysis of those responses (reading English free-text remarks is not a '
            'deterministic classification) -- re-running this script reproduces the same fetch/cross'
            '-check and the same table only because FINDINGS is fixed source, not because the judgment '
            'itself is re-derived from the raw JSON.',
            'historical_identity_confirmed / verification_tier in the existing pipeline output '
            '(mapping_table.csv, mapping_priority_investigation.csv) are NOT modified by this script; '
            'identity_determination here is a separate field for a human or a future coding round to '
            'act on.',
            'This is a cache-only reproduction of a manual judgment, not an automated proof of official '
            '-source completeness: an airport left unconfirmed here may mean this round found no '
            'linking record, not that no such record exists anywhere.',
            'PSG and WRG were downgraded to unconfirmed_program_linkage_insufficient: their fetched HOMR '
            'record is COOP-only, with no ASOS/AWOS platform entry, unlike IMT/KTN/SDF/SIT whose own HOMR '
            'record does carry an ASOS platform entry with ASOS-specific remarks. That ASOS-platform-linkage '
            'contrast is unaffected by the later current-vs-historical split (see the "Third round" bullet '
            'below): IMT/SDF/SIT no longer carry facility_continuity_confirmed themselves -- only KTN keeps '
            'a distinct confirmed-with-caveat determination.',
            'YUM was downgraded to identifier_mismatch_partially_resolved_source_conflict this round: '
            'this project\'s own cached IEM AZ_ASOS network file disagrees with NOAA HOMR about the '
            'closed "YUM" identifier\'s coordinates (see FINDINGS[\'YUM\'][\'inference_and_unresolved\']), '
            'and a 2021-12-16 HOMR metadata update adding an ASOS platform entry to KNYL is not treated '
            'as a verified ASOS installation date.',
            'SPN (Saipan) remains genuinely unconfirmed: NOAA HOMR shows an active ASOS (FAA id GSN) '
            'covering 2018-2019, but this project\'s cached IEM GU_ASOS network response does not list '
            'it under any tried id. Not resolved this round.',
            'The 8 identifier-mismatch findings (AZA/BKG/FCA/HHH/MQT/PBI/SCE/USA) are candidate bugs in '
            'candidate_network_and_sid() in notebooks/map_weather_stations.py; that script was not '
            'modified by this investigation.',
            'FAA facility master records (e.g. a dated Form 5010) were not queried; only NOAA/NCEI HOMR '
            'and the already-cached IEM network metadata.',
            'raw_evidence_fetch_date_estimate is a filesystem-mtime estimate of when each cache file was '
            'first written, not a verified per-request fetch log; evidence_reviewed_date is this run\'s '
            'own assembly date and is not backdated to look like an original fetch date.',
            'No new weather time series collection, no stratafix real collection, no model retraining, '
            'and no changes to any baseline_recovery_v2_weather_* production output were made.',
            'Third round: identity_determination for IMT/SDF/SIT was downgraded from '
            'facility_continuity_confirmed_tz_string_conflict_unchanged to '
            'current_facility_confirmed_historical_continuity_unconfirmed_tz_string_conflict_unchanged -- '
            'their fetched HOMR record confirms the CURRENT facility/ASOS-program linkage but carries no '
            'dated event, so 2018-2019 continuity specifically is left unconfirmed rather than folded into '
            '"confirmed". KTN keeps a separate determination because its record does carry a dated (1997) '
            'pre-period event, itself not proof of 2018-2019 either. All 20 rows now carry a '
            'historical_continuity_2018_2019 field distinguishing this from current-identifier/program '
            'confirmation; for the identifier-mismatch group, inference_and_unresolved no longer reads '
            '"none" where the only basis is an unbroken POR plus the absence of a relocation remark.',
            'Third round: EXPECTED_RECORD + verify_expected_record() connect the fetched HOMR JSON\'s '
            'actual stationCollection.stations to each judgment\'s cited ncdcStnId and required/forbidden '
            'platforms inside this run itself (record_verification above), not only in a separate test '
            'suite reading the cache independently -- a cache file existing is checked separately from the '
            'record it contains still matching what FINDINGS claims.',
            'Fourth round (this revision): PBI\'s remark text and continuity_basis previously attributed a '
            'dated "rename postdates 2018-2019" claim to the fetched HOMR record; the raw record (FAA_PBI. '
            'json) carries no such dated statement, only current identifiers (FAA=DJT, ICAO=KDJT, '
            'NWSLI=PBI) and an undated ad hoc metadata-name-update entry. PBI\'s historical_continuity_2018 '
            '_2019 was downgraded from explicit_statement_change_postdates_window to '
            'rename_confirmed_by_current_ids_undated_in_record; FCA\'s and KTN\'s actual dated remarks (2005 '
            'rename, 1997 station move) are unaffected. Separately, EXPECTED_IDENTIFIERS + the identifier '
            'check inside verify_expected_record() now require each judgment\'s cited FAA/ICAO/NWSLI/NEXRAD '
            'identifiers to actually be present on the selected ncdcStnId\'s record -- previously only '
            'ncdcStnId selection and platforms were checked, so a record that kept the same ncdcStnId and '
            'platforms but had its identifiers deleted or swapped would still have printed the fixed '
            'FINDINGS text unflagged. This check does not require every HOMR_QUERY lookup key (e.g. '
            'FAA:PBI) to itself appear among the record\'s current identifiers, since some lookups '
            '(PBI) are intentionally keyed by a historical id the record no longer carries as its own.',
        ],
    }
    manifest_path = out / f'{args.name}_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'limitations'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
