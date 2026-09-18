import csv
import json

import pytest

from notebooks import verify_priority_station_identity as vpsi

ROOT = vpsi.ROOT
MAPPING_TABLE_CSV = ROOT / 'output/baseline_recovery_v2_weather_scope_fix_20260918_mapping_table.csv'


def _prior_tiers():
    return vpsi.prior_tiers()


# ---- population: exactly these 20, disjoint from the 355 confirmed_period airports ----

def test_findings_and_query_tables_cover_exactly_the_same_20_airports():
    assert len(vpsi.HOMR_QUERY) == 20
    assert set(vpsi.FINDINGS) == set(vpsi.HOMR_QUERY) == set(vpsi.ISSUE_GROUP)


def test_20_priority_airports_match_prior_investigation_csv_exactly():
    tiers = _prior_tiers()
    assert set(tiers) == set(vpsi.HOMR_QUERY)
    # the three tiers this round covers, and no confirmed_period rows
    assert set(tiers.values()) == {
        'confirmed_current_only', 'tz_conflict_needs_resolution', 'unconfirmed'}


def test_20_priority_airports_are_disjoint_from_the_355_confirmed_period_airports():
    with open(MAPPING_TABLE_CSV, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    confirmed_period = {r['iata'] for r in rows if r['verification_tier'] == 'confirmed_period'}
    assert len(confirmed_period) == 355
    assert confirmed_period.isdisjoint(vpsi.HOMR_QUERY)


# ---- raw evidence: every cited file exists and hashes, extra evidence is wired up ----

def test_every_primary_raw_evidence_file_exists_in_the_default_cache():
    for iata, (id_type, id_value) in vpsi.HOMR_QUERY.items():
        cache = vpsi.DEFAULT_CACHE_DIR / f'{id_type}_{id_value}.json'
        assert cache.exists(), f'{iata}: missing cached raw evidence {cache}'
        json.loads(cache.read_text(encoding='utf-8'))  # must be valid JSON


def test_yum_links_both_the_faa_and_icao_raw_records():
    assert vpsi.HOMR_EXTRA_QUERY.get('YUM') == [('ICAO', 'KNYL')]
    cache = vpsi.DEFAULT_CACHE_DIR / 'ICAO_KNYL.json'
    assert cache.exists()
    # KNYL's own POR is the COOP thread (1960-), not a verified ASOS install date
    stn = json.loads(cache.read_text(encoding='utf-8'))['stationCollection']['stations'][0]
    assert stn['header']['por']['beginDate'].startswith('1960-07-01')
    platforms = {p['platform'] for p in stn.get('platforms', [])}
    assert {'COOP', 'ASOS'} <= platforms


def test_fetch_homr_cache_only_mode_refuses_network_fallback_when_missing(tmp_path):
    empty_cache = tmp_path / 'no_such_cache'
    with pytest.raises(FileNotFoundError):
        vpsi.fetch_homr('ICAO', 'KISN', empty_cache, allow_network=False)


def test_fetch_homr_cache_only_mode_succeeds_against_the_real_cache():
    result = vpsi.fetch_homr('ICAO', 'KISN', vpsi.DEFAULT_CACHE_DIR, allow_network=False)
    assert result['sha256'] == vpsi.digest(vpsi.DEFAULT_CACHE_DIR / 'ICAO_KISN.json')
    assert result['fetch_date_basis'] == 'filesystem_mtime_estimate_not_a_verified_fetch_log'


# ---- IEM cross-checks: exact file/feature used, not just a network name ----

def test_iem_feature_checks_reference_real_cached_geojson_files():
    for iata, checks in vpsi.IEM_FEATURE_CHECK.items():
        for network, sid in checks:
            result = vpsi.iem_feature_check(network, sid)
            assert result['file'].endswith(f'{network}.geojson')
            path = vpsi.NETWORKS_DIR / f'{network}.geojson'
            assert result['sha256'] == vpsi.digest(path)


def test_spn_is_not_found_in_any_tried_id_in_the_cached_gu_asos_network():
    results = {sid: vpsi.iem_feature_check('GU_ASOS', sid) for _, sid in vpsi.IEM_FEATURE_CHECK['SPN']}
    assert all(not r['found'] for r in results.values())


def test_yum_and_nyl_are_both_present_in_the_cached_az_asos_network():
    for sid in ('YUM', 'NYL'):
        result = vpsi.iem_feature_check('AZ_ASOS', sid)
        assert result['found'], f'{sid} unexpectedly missing from AZ_ASOS.geojson'


# ---- WRG/PSG downgrade: no ASOS platform in their fetched HOMR record ----

@pytest.mark.parametrize('iata,icao', [('WRG', 'PAWG'), ('PSG', 'PAPG')])
def test_wrg_and_psg_have_no_asos_or_awos_platform_in_their_raw_homr_record(iata, icao):
    id_type, id_value = vpsi.HOMR_QUERY[iata]
    assert (id_type, id_value) == ('ICAO', icao)
    cache = vpsi.DEFAULT_CACHE_DIR / f'{id_type}_{id_value}.json'
    stn = json.loads(cache.read_text(encoding='utf-8'))['stationCollection']['stations'][0]
    platforms = {p['platform'] for p in stn.get('platforms', [])}
    assert platforms == {'COOP'}
    assert vpsi.FINDINGS[iata]['identity_determination'] == 'unconfirmed_program_linkage_insufficient'


@pytest.mark.parametrize('iata,icao', [('IMT', 'KIMT'), ('KTN', 'PAKT'), ('SDF', 'KSDF'), ('SIT', 'PASI')])
def test_imt_ktn_sdf_sit_do_carry_an_asos_platform_in_their_raw_homr_record(iata, icao):
    id_type, id_value = vpsi.HOMR_QUERY[iata]
    assert (id_type, id_value) == ('ICAO', icao)
    cache = vpsi.DEFAULT_CACHE_DIR / f'{id_type}_{id_value}.json'
    stn = json.loads(cache.read_text(encoding='utf-8'))['stationCollection']['stations'][0]
    platforms = {p['platform'] for p in stn.get('platforms', [])}
    assert 'ASOS' in platforms
    assert vpsi.FINDINGS[iata]['identity_determination'] == 'facility_continuity_confirmed_tz_string_conflict_unchanged'


# ---- YUM: downgraded from a confident distinct-facility claim to a flagged conflict ----

def test_yum_determination_flags_the_homr_vs_iem_coordinate_conflict_instead_of_asserting_resolution():
    determination = vpsi.FINDINGS['YUM']['identity_determination']
    assert determination == 'identifier_mismatch_partially_resolved_source_conflict'
    assert 'conflict' in vpsi.FINDINGS['YUM']['inference_and_unresolved'].lower()


def test_aza_finding_names_the_specific_ncdc_stn_id_and_excludes_the_nexrad_record():
    linkage = vpsi.FINDINGS['AZA']['program_location_linkage']
    assert '10000826' in linkage
    assert 'NEXRAD' in linkage
    assert '30001870' in linkage


# ---- cache-only reproduction end to end: fresh name, no network, originals untouched ----

def test_main_cache_only_reproduction_leaves_existing_evidence_untouched(tmp_path):
    original_evidence = ROOT / 'output/baseline_recovery_v2_station_identity_investigation_20260918_evidence.csv'
    original_manifest = ROOT / 'output/baseline_recovery_v2_station_identity_investigation_20260918_manifest.json'
    before_evidence = original_evidence.read_bytes()
    before_manifest = original_manifest.read_bytes()

    fresh_name = 'baseline_recovery_v2_station_identity_pytest_tmp_20260918'
    new_evidence = ROOT / f'output/{fresh_name}_evidence.csv'
    new_manifest = ROOT / f'output/{fresh_name}_manifest.json'
    assert not new_evidence.exists() and not new_manifest.exists()
    try:
        vpsi.main(['--name', fresh_name])
        assert new_evidence.exists() and new_manifest.exists()
        manifest = json.loads(new_manifest.read_text(encoding='utf-8'))
        assert manifest['cache_mode'] == 'cache_only'
        assert manifest['row_count'] == 20

        with pytest.raises(FileExistsError):
            vpsi.main(['--name', fresh_name])
    finally:
        new_evidence.unlink(missing_ok=True)
        new_manifest.unlink(missing_ok=True)

    assert original_evidence.read_bytes() == before_evidence
    assert original_manifest.read_bytes() == before_manifest


def test_main_refuses_a_nonexistent_cache_dir_without_allow_network(tmp_path):
    fresh_name = 'baseline_recovery_v2_station_identity_pytest_tmp2_20260918'
    missing_cache = tmp_path / 'nope'
    with pytest.raises(FileNotFoundError):
        vpsi.main(['--name', fresh_name, '--cache-dir', str(missing_cache)])
    assert not (ROOT / f'output/{fresh_name}_evidence.csv').exists()
