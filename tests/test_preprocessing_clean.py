"""Structural tests for conservative preprocessing; examples are test fixtures."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from src.features import impute_cross, build_speed_features, build_cyclic_features
import rerun_all_phases as runner


def test_unique_mapping_order_independent_and_observed_values_preserved():
    df = pd.DataFrame({'code':['A','A','A','B','B','C'], 'name':['one','two',None,'three',None,None]})
    clean = impute_cross(df, pairs=[('code','name')], conflict='unique')
    reverse = impute_cross(df.iloc[::-1], pairs=[('code','name')], conflict='unique').sort_index()
    pd.testing.assert_frame_equal(clean, reverse)
    assert pd.isna(clean.loc[2,'name']) and pd.isna(clean.loc[5,'name'])
    assert clean.loc[4,'name'] == 'three'
    pd.testing.assert_series_equal(clean.loc[df.name.notna(),'name'], df.loc[df.name.notna(),'name'])
    assert impute_cross(df, pairs=[('code','name')]).loc[2,'name'] == 'one'


def test_reverse_direction_uses_only_unique_observed_pairs():
    df = pd.DataFrame({'code':['A','B',None,'C',None], 'name':['shared','shared','shared','unique','unique']})
    out = impute_cross(df, pairs=[('code','name')], conflict='unique', bidirectional=True)
    assert pd.isna(out.loc[2,'code'])
    assert out.loc[4,'code'] == 'C'


def test_zero_negative_missing_nonfinite_denominator_and_input_unchanged():
    df = pd.DataFrame({'Estimated_Duration':[0., -1., np.nan, np.inf, 100.], 'Distance':[100.]*5})
    before = df.copy(deep=True)
    out = build_speed_features(df, safe=True)
    assert out.Distance_Per_Local_Minute.iloc[:4].isna().all()
    assert out.Distance_Per_Local_Minute.iloc[4] == 1.
    assert 'Air_Speed_Proxy' not in out and 'Estimated_Duration' not in out
    pd.testing.assert_frame_equal(df, before)


def test_unknown_hour_is_not_noon():
    out = build_cyclic_features(pd.DataFrame({'Dep_Hour':[-1., np.nan, 12., 23., 0.]}), add_missing_flag=True)
    assert out.Sin_Dep_Hour.iloc[:2].isna().all()
    assert out.Dep_Hour_Missing.tolist() == [1,1,0,0,0]
    assert out.Cos_Dep_Hour.iloc[2] == pytest.approx(-1.)


def test_clean_specs_have_distinct_cache_keys_and_noon_is_disabled():
    for base, key in [('P4','P4_clean'),('P6_fixed','P6_clean')]:
        old = next(s for s in runner.PHASES if s.key == base)
        new = next(s for s in runner.PHASES if s.key == key)
        assert old.feature_signature() != new.feature_signature()
        assert new.cyclic_missing == 'nan'
        assert not old.safe_preprocessing and new.safe_preprocessing


@pytest.mark.parametrize('base', ['P4','P6_fixed'])
@pytest.mark.parametrize('suffix,field', [('impute','unique_imputation'),
                                        ('ratio','safe_ratio'),
                                        ('missing','explicit_time_missing')])
def test_ablation_changes_only_one_control(base, suffix, field):
    from dataclasses import asdict
    old = next(s for s in runner.PHASES if s.key == base)
    new = next(s for s in runner.PHASES if s.key == f'{base}_{suffix}')
    before, after = asdict(old), asdict(new)
    changed = {k for k in before if before[k] != after[k]}
    assert changed == {'key','label','note',field}
    assert getattr(new, field) is True
    assert new.feature_signature() != old.feature_signature()


def test_clean_pipeline_preserves_labels_and_provenance_flags():
    raw = pd.DataFrame({
        'ID':['a','b','c','d'], 'Delay':['Delayed','Not_Delayed',None,None],
        'Month':[1]*4, 'Day_of_Month':[1]*4,
        'Estimated_Departure_Time':[1200.,np.nan,1200.,np.nan],
        'Estimated_Arrival_Time':[1300.,1300.,1200.,np.nan],
        'Origin_Airport':['AAA']*4, 'Destination_Airport':['BBB']*4,
        'Tail_Number':['T']*4, 'Airline':['A']*4, 'Distance':[100.]*4,
    })
    spec = next(s for s in runner.PHASES if s.key == 'P6_clean')
    X, y, U = runner.build_features(raw, spec)
    assert y.tolist() == [1,0] and len(U) == 2
    assert X.Dep_Hour_Originally_Missing.tolist() == [0,1]
    assert X.Dep_Hour_Missing.tolist() == [0,0]
    assert pd.isna(U.loc[0,'Distance_Per_Local_Minute'])
    assert pd.isna(U.loc[1,'Sin_Dep_Hour'])
    assert U.loc[1,'Dep_Hour_Missing'] == 1
