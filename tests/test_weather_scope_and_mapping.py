import hashlib

import pandas as pd
import pytest

from notebooks.map_weather_stations import (candidate_network_and_sid, classify, haversine_km,
                                            subperiod_overlap, utc_offsets_match_over_period)
from notebooks.scope_weather_collection import CENSUS_REGION, NON_CENSUS_TERRITORY, region_for
from notebooks.select_weather_sample_expanded import (broad_inclusion_order, region_timezone_bucket,
                                                       round_robin_merge, stable_rank)


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
    # Atlantic/Bermuda observes DST, America/St_Thomas never does -- a genuine offset difference,
    # not just a naming difference, distinct from a case where the strings differ but offsets agree.
    assert result['utc_offset_equivalent_2018_2019'] is False


def test_classify_does_not_confirm_identity_or_location_by_itself():
    """confirmed_period must carry an explicit evidence-scope statement and a
    location signal, since the tier name alone invites over-reading it as
    "this is verifiably the same airport/location", which it is not."""
    props = {'tzname': 'America/New_York', 'archive_begin': '2000-01-01', 'archive_end': None,
            'iem_lat': 33.6367, 'iem_lon': -84.4281}
    result = classify({'mwgg_tz': 'America/New_York', 'lat': 33.6367, 'lon': -84.4281}, props)
    assert result['verification_tier'] == 'confirmed_period'
    assert 'evidence_basis' in result and 'location' in result['evidence_basis'].lower()
    assert result['location_distance_km'] == pytest.approx(0.0, abs=1e-6)


def test_classify_reports_valid_subperiod_for_partial_coverage():
    """A station whose archive window only partially overlaps 2018-2019 (e.g.
    Williston ISN->XWA mid-period) may still validly map individual rows
    whose own date falls inside the overlap -- confirmed_current_only is not
    "never usable", so the usable sub-range must be surfaced, not discarded."""
    props = {'tzname': 'America/Chicago', 'archive_begin': '1950-04-01', 'archive_end': '2019-10-18'}
    result = classify({'mwgg_tz': 'America/Chicago'}, props)
    assert result['verification_tier'] == 'confirmed_current_only'
    assert result['valid_subperiod_start'] == '2018-01-01'
    assert result['valid_subperiod_end'] == '2019-10-18'


# ---- haversine_km / utc_offsets_match_over_period / subperiod_overlap ----

def test_haversine_km_zero_for_identical_points():
    assert haversine_km(40.0, -80.0, 40.0, -80.0) == pytest.approx(0.0, abs=1e-9)


def test_haversine_km_none_when_a_coordinate_is_missing():
    assert haversine_km(None, -80.0, 40.0, -80.0) is None
    assert haversine_km(40.0, -80.0, 40.0, None) is None


def test_utc_offsets_match_over_period_true_for_identical_zone():
    start, end = pd.Timestamp('2018-01-01'), pd.Timestamp('2019-12-31')
    assert utc_offsets_match_over_period('America/New_York', 'America/New_York', start, end) is True


def test_utc_offsets_match_over_period_false_for_genuinely_different_offsets():
    """Atlantic/Bermuda observes DST; America/St_Thomas never does -- their
    offsets diverge for large parts of the year."""
    start, end = pd.Timestamp('2018-01-01'), pd.Timestamp('2019-12-31')
    assert utc_offsets_match_over_period('Atlantic/Bermuda', 'America/St_Thomas', start, end) is False


def test_utc_offsets_match_over_period_none_for_unresolvable_zone():
    start, end = pd.Timestamp('2018-01-01'), pd.Timestamp('2018-01-02')
    assert utc_offsets_match_over_period('Not/AZone', 'America/New_York', start, end) is None


def test_subperiod_overlap_no_overlap_returns_none_pair():
    start = pd.Timestamp('1950-01-01')
    end = pd.Timestamp('2007-12-31')  # entirely before the 2018-2019 period
    assert subperiod_overlap(start, end, pd.Timestamp('2018-01-01'), pd.Timestamp('2019-12-31')) == (None, None)


def test_subperiod_overlap_partial_overlap():
    start = pd.Timestamp('1950-01-01')
    end = pd.Timestamp('2019-10-18')
    result = subperiod_overlap(start, end, pd.Timestamp('2018-01-01'), pd.Timestamp('2019-12-31'))
    assert result == ('2018-01-01', '2019-10-18')


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


# ---- round_robin_merge / broad_inclusion_order: the 300-row sampling-bias fix ----

def test_round_robin_merge_interleaves_while_preserving_each_list_own_order():
    assert round_robin_merge([[1, 2, 3], [10, 20], [100]]) == [1, 10, 100, 2, 20, 3]


def test_round_robin_merge_skips_empty_sequences_without_erroring():
    assert round_robin_merge([[], [1, 2], []]) == [1, 2]


def test_broad_inclusion_order_visits_both_years_before_repeating_either():
    strata = [(2018, 'winter', 'R:a', 'small'), (2018, 'summer', 'R:a', 'small'),
             (2019, 'winter', 'R:a', 'small'), (2019, 'summer', 'R:a', 'small')]
    order = broad_inclusion_order(strata)
    assert {order[0][0], order[1][0]} == {2018, 2019}  # first two picks are NOT both the same year


def test_broad_inclusion_order_spreads_a_budget_shortfall_across_years():
    """The actual bug this fixes: sorting the plain stratum LABEL puts the
    year first alphabetically, so it exhausts one entire year before ever
    touching the other -- when a row cap lands mid-way through the second
    year, everything cut is concentrated in whichever season/region happened
    to sort last, not spread across the population. Here, of the first 4
    strata (a stand-in for "what a tight cap would keep"), both years must
    be represented, not 3-and-1 or 4-and-0."""
    strata = [(year, season, 'R:a', 'small') for year in (2018, 2019) for season in ('fall', 'spring', 'summer')]
    order = broad_inclusion_order(strata)
    first_four_years = [s[0] for s in order[:4]]
    assert first_four_years.count(2018) == 2
    assert first_four_years.count(2019) == 2


def test_broad_inclusion_order_is_deterministic_and_a_permutation_of_the_input():
    strata = [(2018, 'winter', 'South:New_York', 'small'), (2018, 'summer', 'West:Los_Angeles', 'large'),
             (2019, 'fall', 'Alaska', 'medium')]
    order1 = broad_inclusion_order(strata)
    order2 = broad_inclusion_order(list(reversed(strata)))  # input order must not matter, only content
    assert sorted(order1) == sorted(strata)
    assert order1 == order2
