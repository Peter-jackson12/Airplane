import csv
import json
import shutil

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


# ---- third round: current-identifier vs. historical-continuity separation ----

def test_every_finding_carries_a_historical_continuity_field_distinct_from_identity_determination():
    for iata, fnd in vpsi.FINDINGS.items():
        assert 'historical_continuity_2018_2019' in fnd, iata
        assert fnd['historical_continuity_2018_2019']


def test_ktn_keeps_its_own_determination_naming_the_dated_1997_event():
    assert vpsi.FINDINGS['KTN']['identity_determination'] == (
        'facility_continuity_supported_by_dated_1997_event_tz_string_conflict_unchanged')
    assert vpsi.FINDINGS['KTN']['historical_continuity_2018_2019'] == (
        'indirect_dated_pre_period_event_not_direct_2018_2019_confirmation')
    assert '1997' in vpsi.FINDINGS['KTN']['period_2018_2019_evidence']


@pytest.mark.parametrize('iata', ['IMT', 'SDF', 'SIT'])
def test_imt_sdf_sit_are_downgraded_to_current_facility_confirmed_not_historical_continuity(iata):
    fnd = vpsi.FINDINGS[iata]
    assert fnd['identity_determination'] == (
        'current_facility_confirmed_historical_continuity_unconfirmed_tz_string_conflict_unchanged')
    assert fnd['historical_continuity_2018_2019'] == 'unconfirmed_current_metadata_only'
    # no dated pre-period event exists for these three, unlike KTN
    assert '1997' not in fnd['period_2018_2019_evidence']


def test_imt_sdf_sit_no_longer_share_ktns_confirmed_determination():
    assert vpsi.FINDINGS['KTN']['identity_determination'] != vpsi.FINDINGS['IMT']['identity_determination']


def test_identifier_mismatch_inference_is_not_generically_none():
    for iata in vpsi._IDENTIFIER_MISMATCH:
        text = vpsi.FINDINGS[iata]['inference_and_unresolved']
        assert text.strip().lower() != 'none beyond remaining_gap below.', iata
        assert text.strip() != '', iata


def test_fca_has_a_dated_predating_remark_unlike_the_absence_only_group():
    assert vpsi.FINDINGS['FCA']['historical_continuity_2018_2019'] == 'dated_remark_predates_window'
    for iata in ('BKG', 'HHH', 'MQT', 'SCE', 'USA'):
        assert vpsi.FINDINGS[iata]['historical_continuity_2018_2019'] == 'absence_of_remark'
    assert vpsi.FINDINGS['AZA']['historical_continuity_2018_2019'] == 'undated_remark'


# ---- fourth round: PBI's raw-record boundary -- current ids vs. an undated rename ----

def test_pbi_continuity_reflects_current_ids_confirmed_but_no_dated_rename_remark():
    # Fourth round fix: the raw FAA_PBI.json record has no dated statement about when
    # the FAA/legal rename happened or that PBI was in use throughout 2018-2019 --
    # only current identifiers and an undated ad hoc metadata update. This must not be
    # confused with FCA's genuinely dated 2005 rename remark.
    fnd = vpsi.FINDINGS['PBI']
    assert fnd['historical_continuity_2018_2019'] == 'rename_confirmed_by_current_ids_undated_in_record'
    assert fnd['historical_continuity_2018_2019'] != vpsi.FINDINGS['FCA']['historical_continuity_2018_2019']
    assert 'background' in fnd['inference_and_unresolved'].lower() or 'outside' in fnd['inference_and_unresolved'].lower()


def test_pbi_raw_record_carries_no_dated_rename_or_2018_2019_statement():
    # Guards against re-attributing an outside/background-knowledge claim to the raw
    # HOMR remark: none of FAA_PBI.json's `remarks` mention DJT, TRUMP, or a rename,
    # and the only identifier-name-change entry anywhere on the record is the undated
    # "updates" block, not a dated remark.
    cache = vpsi.DEFAULT_CACHE_DIR / 'FAA_PBI.json'
    stn = json.loads(cache.read_text(encoding='utf-8'))['stationCollection']['stations'][0]
    remark_text = ' '.join(r.get('remark', '') for r in stn.get('remarks', [])).upper()
    for banned in ('DJT', 'TRUMP', 'RENAME', 'RENAMED', 'CHANGED FROM PBI'):
        assert banned not in remark_text, f'{banned} unexpectedly found in FAA_PBI.json remarks'
    updates = stn.get('updates', [])
    assert any('NAME' in u.get('description', '').upper() for u in updates)
    assert all('effectiveDate' in u or 'enteredDate' in u for u in updates)
    ids = {i['idType']: i['id'] for i in stn.get('identifiers', [])}
    assert ids['FAA'] == 'DJT' and ids['ICAO'] == 'KDJT' and ids['NWSLI'] == 'PBI'


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


# ---- third round: raw-record verification wired into the generation path ----

def test_load_station_collection_rejects_empty_object(tmp_path):
    cache = tmp_path / 'empty.json'
    cache.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError):
        vpsi.load_station_collection(cache)


def test_load_station_collection_rejects_empty_stations_list(tmp_path):
    cache = tmp_path / 'empty_stations.json'
    cache.write_text(json.dumps({'stationCollection': {'stations': []}}), encoding='utf-8')
    with pytest.raises(ValueError):
        vpsi.load_station_collection(cache)


def test_select_expected_station_returns_none_when_ncdc_stn_id_absent():
    stations = [{'ncdcStnId': '111', 'platforms': []}]
    assert vpsi.select_expected_station(stations, '222') is None
    assert vpsi.select_expected_station(stations, '111') is stations[0]


def test_verify_expected_record_flags_missing_required_platform(tmp_path):
    cache = tmp_path / 'ICAO_KIMT.json'
    cache.write_text(json.dumps(
        {'stationCollection': {'stations': [{'ncdcStnId': '20010418', 'platforms': [{'platform': 'COOP'}]}]}}),
        encoding='utf-8')
    checks = [dict(source='primary', ncdc_stn_id='20010418', requires={'ASOS'}, forbids=set())]
    failures = vpsi.verify_expected_record('IMT', {'primary': cache}, checks)
    assert failures and 'missing required platform' in failures[0]


def test_verify_expected_record_flags_forbidden_platform_present(tmp_path):
    cache = tmp_path / 'ICAO_PAWG.json'
    cache.write_text(json.dumps(
        {'stationCollection': {'stations': [{'ncdcStnId': '10000446', 'platforms': [{'platform': 'ASOS'}]}]}}),
        encoding='utf-8')
    checks = [dict(source='primary', ncdc_stn_id='10000446', requires=set(), forbids={'ASOS', 'AWOS'})]
    failures = vpsi.verify_expected_record('WRG', {'primary': cache}, checks)
    assert failures and 'forbidden platform' in failures[0]


def test_verify_expected_record_flags_missing_ncdc_stn_id(tmp_path):
    cache = tmp_path / 'ICAO_KIWA.json'
    cache.write_text(json.dumps(
        {'stationCollection': {'stations': [{'ncdcStnId': '99999999', 'platforms': [{'platform': 'AWOS'}]}]}}),
        encoding='utf-8')
    checks = [dict(source='primary', ncdc_stn_id='10000826', requires={'AWOS'}, forbids=set())]
    failures = vpsi.verify_expected_record('AZA', {'primary': cache}, checks)
    assert failures and 'not selectable' in failures[0]


def test_verify_expected_record_flags_missing_required_identifier(tmp_path):
    cache = tmp_path / 'FAA_PBI.json'
    cache.write_text(json.dumps({'stationCollection': {'stations': [{
        'ncdcStnId': '20004306',
        'platforms': [{'platform': 'COOP'}, {'platform': 'ASOS'}],
        'identifiers': [{'idType': 'FAA', 'id': 'DJT'}, {'idType': 'ICAO', 'id': 'KDJT'}],
        # NWSLI dropped entirely
    }]}}), encoding='utf-8')
    checks = [dict(source='primary', ncdc_stn_id='20004306', requires={'COOP', 'ASOS'}, forbids=set(),
                    expected_ids=[('FAA', 'DJT'), ('ICAO', 'KDJT'), ('NWSLI', 'PBI')])]
    failures = vpsi.verify_expected_record('PBI', {'primary': cache}, checks)
    assert failures and 'missing required identifier' in failures[0]
    assert "('NWSLI', 'PBI')" in failures[0]


def test_verify_expected_record_flags_changed_identifier_value(tmp_path):
    cache = tmp_path / 'FAA_PBI.json'
    cache.write_text(json.dumps({'stationCollection': {'stations': [{
        'ncdcStnId': '20004306',
        'platforms': [{'platform': 'COOP'}, {'platform': 'ASOS'}],
        'identifiers': [
            {'idType': 'FAA', 'id': 'ZZZ'},  # changed away from DJT
            {'idType': 'ICAO', 'id': 'KDJT'},
            {'idType': 'NWSLI', 'id': 'PBI'},
        ],
    }]}}), encoding='utf-8')
    checks = [dict(source='primary', ncdc_stn_id='20004306', requires={'COOP', 'ASOS'}, forbids=set(),
                    expected_ids=[('FAA', 'DJT'), ('ICAO', 'KDJT'), ('NWSLI', 'PBI')])]
    failures = vpsi.verify_expected_record('PBI', {'primary': cache}, checks)
    assert failures and 'missing required identifier' in failures[0]
    assert "('FAA', 'DJT')" in failures[0]


def test_expected_identifiers_covers_exactly_the_ncdc_stn_ids_in_expected_record():
    all_ids = {chk['ncdc_stn_id'] for checks in vpsi.EXPECTED_RECORD.values() for chk in checks}
    assert all_ids == set(vpsi.EXPECTED_IDENTIFIERS)


def test_verify_expected_record_passes_against_the_real_default_cache_for_every_airport():
    for iata, checks in vpsi.EXPECTED_RECORD.items():
        id_type, id_value = vpsi.HOMR_QUERY[iata]
        cache_files = {'primary': vpsi.DEFAULT_CACHE_DIR / f'{id_type}_{id_value}.json'}
        for i, (t, v) in enumerate(vpsi.HOMR_EXTRA_QUERY.get(iata, [])):
            cache_files[f'extra:{i}'] = vpsi.DEFAULT_CACHE_DIR / f'{t}_{v}.json'
        assert vpsi.verify_expected_record(iata, cache_files, checks) == [], iata


def _copy_default_cache(tmp_path):
    tmp_cache = tmp_path / 'homr_cache'
    shutil.copytree(vpsi.DEFAULT_CACHE_DIR, tmp_cache)
    return tmp_cache


def test_main_raises_and_writes_nothing_when_a_cached_record_loses_its_required_platform(tmp_path):
    tmp_cache = _copy_default_cache(tmp_path)
    imt_file = tmp_cache / 'ICAO_KIMT.json'
    obj = json.loads(imt_file.read_text(encoding='utf-8'))
    obj['stationCollection']['stations'][0]['platforms'] = [{'platform': 'COOP'}]
    imt_file.write_text(json.dumps(obj), encoding='utf-8')

    fresh_name = 'baseline_recovery_v2_station_identity_pytest_tmp3_20260918'
    with pytest.raises(RuntimeError, match='missing required platform'):
        vpsi.main(['--name', fresh_name, '--cache-dir', str(tmp_cache)])
    assert not (ROOT / f'output/{fresh_name}_evidence.csv').exists()
    assert not (ROOT / f'output/{fresh_name}_manifest.json').exists()


def test_main_raises_when_a_required_programs_platform_is_replaced_by_a_different_one(tmp_path):
    tmp_cache = _copy_default_cache(tmp_path)
    wrg_file = tmp_cache / 'ICAO_PAWG.json'
    obj = json.loads(wrg_file.read_text(encoding='utf-8'))
    obj['stationCollection']['stations'][0]['platforms'] = [{'platform': 'COOP'}, {'platform': 'ASOS'}]
    wrg_file.write_text(json.dumps(obj), encoding='utf-8')

    fresh_name = 'baseline_recovery_v2_station_identity_pytest_tmp4_20260918'
    with pytest.raises(RuntimeError, match='forbidden platform'):
        vpsi.main(['--name', fresh_name, '--cache-dir', str(tmp_cache)])
    assert not (ROOT / f'output/{fresh_name}_evidence.csv').exists()


def test_main_raises_and_writes_nothing_when_identifiers_are_deleted_but_platforms_are_untouched(tmp_path):
    # Fourth round regression: before EXPECTED_IDENTIFIERS existed, a cache file whose
    # `identifiers` were dropped entirely -- while ncdcStnId and platforms stayed the
    # same -- would still pass verify_expected_record() and print FINDINGS unflagged.
    tmp_cache = _copy_default_cache(tmp_path)
    pbi_file = tmp_cache / 'FAA_PBI.json'
    obj = json.loads(pbi_file.read_text(encoding='utf-8'))
    stn = obj['stationCollection']['stations'][0]
    assert stn['ncdcStnId'] == '20004306'
    stn['identifiers'] = []  # platforms/ncdcStnId left untouched
    pbi_file.write_text(json.dumps(obj), encoding='utf-8')

    fresh_name = 'baseline_recovery_v2_station_identity_pytest_tmp9_20260918'
    with pytest.raises(RuntimeError, match='missing required identifier'):
        vpsi.main(['--name', fresh_name, '--cache-dir', str(tmp_cache)])
    assert not (ROOT / f'output/{fresh_name}_evidence.csv').exists()
    assert not (ROOT / f'output/{fresh_name}_manifest.json').exists()


def test_main_raises_when_an_identifier_value_is_changed_but_platforms_are_untouched(tmp_path):
    tmp_cache = _copy_default_cache(tmp_path)
    pbi_file = tmp_cache / 'FAA_PBI.json'
    obj = json.loads(pbi_file.read_text(encoding='utf-8'))
    stn = obj['stationCollection']['stations'][0]
    for ident in stn['identifiers']:
        if ident['idType'] == 'FAA':
            ident['id'] = 'ZZZ'  # DJT -> a different FAA id, platforms untouched
    pbi_file.write_text(json.dumps(obj), encoding='utf-8')

    fresh_name = 'baseline_recovery_v2_station_identity_pytest_tmp10_20260918'
    with pytest.raises(RuntimeError, match='missing required identifier'):
        vpsi.main(['--name', fresh_name, '--cache-dir', str(tmp_cache)])
    assert not (ROOT / f'output/{fresh_name}_evidence.csv').exists()


def test_main_raises_when_a_cached_response_has_no_stations(tmp_path):
    tmp_cache = _copy_default_cache(tmp_path)
    isn_file = tmp_cache / 'ICAO_KISN.json'
    isn_file.write_text(json.dumps({'stationCollection': {'stations': []}}), encoding='utf-8')

    fresh_name = 'baseline_recovery_v2_station_identity_pytest_tmp5_20260918'
    with pytest.raises(RuntimeError, match='stationCollection.stations is missing or empty'):
        vpsi.main(['--name', fresh_name, '--cache-dir', str(tmp_cache)])
    assert not (ROOT / f'output/{fresh_name}_evidence.csv').exists()


def test_main_raises_when_a_required_non_spn_iem_feature_goes_missing(tmp_path, monkeypatch):
    real_check = vpsi.iem_feature_check

    def fake_check(network, sid):
        if network == 'AZ_ASOS' and sid == 'IWA':
            return {'network': network, 'sid': sid, 'file': 'fake', 'sha256': 'fake',
                     'found': False, 'summary': 'forced miss for test'}
        return real_check(network, sid)

    monkeypatch.setattr(vpsi, 'iem_feature_check', fake_check)
    fresh_name = 'baseline_recovery_v2_station_identity_pytest_tmp6_20260918'
    with pytest.raises(RuntimeError, match='required IEM feature'):
        vpsi.main(['--name', fresh_name])
    assert not (ROOT / f'output/{fresh_name}_evidence.csv').exists()


def test_main_still_allows_spns_expected_iem_miss(tmp_path):
    # SPN's GU_ASOS features are genuinely not found in the real cache (see
    # test_spn_is_not_found_in_any_tried_id_in_the_cached_gu_asos_network above);
    # main() must still succeed because SPN is excluded from
    # IEM_FEATURE_REQUIRED_FOUND, unlike the forced-miss case above for AZA.
    assert 'SPN' not in vpsi.IEM_FEATURE_REQUIRED_FOUND
    fresh_name = 'baseline_recovery_v2_station_identity_pytest_tmp7_20260918'
    try:
        vpsi.main(['--name', fresh_name])
        assert (ROOT / f'output/{fresh_name}_evidence.csv').exists()
    finally:
        (ROOT / f'output/{fresh_name}_evidence.csv').unlink(missing_ok=True)
        (ROOT / f'output/{fresh_name}_manifest.json').unlink(missing_ok=True)


def test_main_cache_only_reproduction_carries_the_new_columns_and_verification_summary(tmp_path):
    fresh_name = 'baseline_recovery_v2_station_identity_pytest_tmp8_20260918'
    try:
        vpsi.main(['--name', fresh_name])
        with open(ROOT / f'output/{fresh_name}_evidence.csv', newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert 'historical_continuity_2018_2019' in rows[0]
        assert 'record_verification_checks' in rows[0]
        assert all(row['historical_continuity_2018_2019'] for row in rows)
        manifest = json.loads((ROOT / f'output/{fresh_name}_manifest.json').read_text(encoding='utf-8'))
        assert manifest['record_verification']['passed'] is True
        assert manifest['record_verification']['checks_run'] > 20
    finally:
        (ROOT / f'output/{fresh_name}_evidence.csv').unlink(missing_ok=True)
        (ROOT / f'output/{fresh_name}_manifest.json').unlink(missing_ok=True)
