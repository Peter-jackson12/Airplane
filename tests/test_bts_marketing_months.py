import calendar

import pandas as pd

from notebooks.summarize_bts_marketing_months import THANKSGIVING, YEARS


def fourth_thursday(year, month):
    thursdays = [d for d, wd in
                 ((d, calendar.weekday(year, month, d))
                  for d in range(1, calendar.monthrange(year, month)[1] + 1))
                 if wd == calendar.THURSDAY]
    return thursdays[3]


def test_thanksgiving_constants_are_the_actual_fourth_thursday():
    """The holiday check is only meaningful if these dates are right."""
    for year, (month, day) in THANKSGIVING.items():
        assert month == 11
        assert day == fourth_thursday(year, month), year


def test_the_two_candidate_years_have_different_thanksgiving_dates():
    days = {day for _, day in THANKSGIVING.values()}
    assert len(days) == len(THANKSGIVING) == len(YEARS)


def resolve(frame):
    from notebooks.summarize_bts_marketing_months import YEARS as ys
    for year in ys:
        frame[f'hit_{year}'] = frame[f'{year}_marketing'].gt(0)
    frame['candidate_years'] = frame[[f'hit_{y}' for y in ys]].sum(axis=1)
    frame['resolved_year'] = pd.NA
    for year in ys:
        frame.loc[frame[f'hit_{year}'] & frame.candidate_years.eq(1), 'resolved_year'] = year
    return frame


def test_a_row_matching_both_years_is_left_unresolved():
    frame = resolve(pd.DataFrame({'2018_marketing': [1, 0, 1], '2019_marketing': [0, 1, 1]}))
    assert frame.resolved_year.tolist()[:2] == [2018, 2019]
    assert pd.isna(frame.resolved_year.iloc[2])


def test_unresolved_rows_are_excluded_from_the_year_share_not_split():
    frame = resolve(pd.DataFrame({'2018_marketing': [1, 1, 1], '2019_marketing': [0, 0, 1]}))
    counts = {y: int(frame.resolved_year.eq(y).sum()) for y in YEARS}
    assert counts == {2018: 2, 2019: 0}
    assert sum(counts.values()) == 2, 'the dual-year row must not contribute to either side'


def test_a_row_matching_no_year_is_also_unresolved():
    frame = resolve(pd.DataFrame({'2018_marketing': [0], '2019_marketing': [0]}))
    assert pd.isna(frame.resolved_year.iloc[0])
