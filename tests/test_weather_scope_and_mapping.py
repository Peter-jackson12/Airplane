import hashlib

import pandas as pd
import pytest

from notebooks.map_weather_stations import candidate_network_and_sid, classify
from notebooks.scope_weather_collection import CENSUS_REGION, NON_CENSUS_TERRITORY, region_for
from notebooks.select_weather_sample_expanded import region_timezone_bucket, stable_rank


# ---- map_weather_stations: station-candidate rules and tiered classification ----

def test_conus_state_uses_iata_and_its_own_state_network():
    networks, sid = candidate_network_and_sid('KATL', 'ATL', 'Georgia', 'US')
    assert networks == ['GA_ASOS']
    assert sid == 'ATL'


def test_alaska_and_hawaii_use_icao_not_iata():
    networks, sid = candidate_network_and_sid('PANC', 'ANC', 'Alaska', 'US')
    assert networks == ['AK_ASOS'] and sid == 'PANC'
    networks, sid = candidate_network_and_sid('PHNL', 'HNL', 'Hawaii', 'US')
    assert networks == ['HI_ASOS'] and sid == 'PHNL'


def test_territories_use_icao_and_their_own_network():
    networks, sid = candidate_network_and_sid('TJSJ', 'SJU', 'Carolina', 'PR')
    assert networks == ['PR_ASOS'] and sid == 'TJSJ'
    # CNMI (Saipan/Rota) is grouped under Guam's network by IEM, not a separate MP_ASOS
    networks, sid = candidate_network_and_sid('PGSN', 'SPN', 'Saipan', 'MP')
    assert networks == ['GU_ASOS']


def test_unresolvable_state_country_yields_no_candidate_network():
    networks, sid = candidate_network_and_sid('ZZZZ', 'ZZZ', 'Nowhere', 'ZZ')
    assert networks == []


def test_classify_unconfirmed_when_station_not_found_in_network():
    result = classify({'mwgg_tz': 'America/New_York'}, None)
    assert result['verification_tier'] == 'unconfirmed'
    assert result['found_in_iem_network'] is False


def test_classify_confirmed_period_when_archive_brackets_2018_2019():
    props = {'tzname': 'America/New_York', 'archive_begin': '2000-01-01', 'archive_end': None}
    result = classify({'mwgg_tz': 'America/New_York'}, props)
    assert result['verification_tier'] == 'confirmed_period'
    assert result['period_covers_2018_2019'] is True


def test_classify_confirmed_current_only_when_archive_excludes_the_period():
    """Williston, ND: the old station (ISN) was replaced by XWA in Oct 2019,
    mid-way through the 2018-2019 window this project cares about."""
    props = {'tzname': 'America/Chicago', 'archive_begin': '1950-04-01', 'archive_end': '2019-10-18'}
    result = classify({'mwgg_tz': 'America/Chicago'}, props)
    assert result['verification_tier'] == 'confirmed_current_only'
    assert result['period_covers_2018_2019'] is False


def test_classify_tz_conflict_overrides_period_coverage():
    """A timezone disagreement between sources (e.g. IEM says a DST-observing
    zone, mwgg says a non-DST zone) is a correctness risk and must not be
    reported as confirmed just because the archive window looks fine."""
    props = {'tzname': 'Atlantic/Bermuda', 'archive_begin': '1950-01-01', 'archive_end': None}
    result = classify({'mwgg_tz': 'America/St_Thomas'}, props)
    assert result['verification_tier'] == 'tz_conflict_needs_resolution'
    assert result['tz_matches_mwgg'] is False


# ---- scope_weather_collection: region definition ----

def test_region_for_covers_all_four_census_regions():
    assert region_for('Georgia', 'US') == 'South'
    assert region_for('Illinois', 'US') == 'Midwest'
    assert region_for('New York', 'US') == 'Northeast'
    assert region_for('California', 'US') == 'West'


def test_region_for_keeps_alaska_hawaii_and_territories_separate():
    assert region_for('Alaska', 'US') == 'Alaska'
    assert region_for('Hawaii', 'US') == 'Hawaii'
    assert region_for('Carolina', 'PR') == NON_CENSUS_TERRITORY


def test_every_conus_state_in_census_region_has_a_region():
    assert set(CENSUS_REGION.values()) == {'Northeast', 'Midwest', 'South', 'West', 'Alaska', 'Hawaii'}


# ---- select_weather_sample_expanded: deterministic stratification helpers ----

def test_stable_rank_is_deterministic_and_matches_sha256():
    ids = pd.Series(['TRAIN_000001', 'TRAIN_000002'])
    ranks = stable_rank(ids)
    assert ranks.iloc[0] == hashlib.sha256(b'TRAIN_000001').hexdigest()
    # calling it again must give the identical order (no RNG involved)
    assert stable_rank(ids).tolist() == ranks.tolist()


def test_region_timezone_bucket_folds_timezone_into_mainland_regions_only():
    assert region_timezone_bucket('South', 'America/New_York') == 'South:New_York'
    assert region_timezone_bucket('West', 'America/Los_Angeles') == 'West:Los_Angeles'
    assert region_timezone_bucket('Alaska', 'America/Anchorage') == 'Alaska'
    assert region_timezone_bucket('Hawaii', 'Pacific/Honolulu') == 'Hawaii'
