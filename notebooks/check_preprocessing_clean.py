"""Actual-data before/after checks; no model training. Run with python -u."""
from pathlib import Path
import sys
import json
import hashlib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import impute_cross, build_time_features, build_speed_features
from rerun_all_phases import build_features, PHASES

path = ROOT / 'data/train.csv'
if not path.is_file():
    raise FileNotFoundError('Actual train.csv required')
raw = pd.read_csv(path)
old = impute_cross(raw)
new = impute_cross(raw, conflict='unique')
reverse = impute_cross(raw.iloc[::-1], conflict='unique').sort_index()
pd.testing.assert_frame_equal(new, reverse)
assert new.loc[raw.Airline.notna(),'Airline'].equals(raw.loc[raw.Airline.notna(),'Airline'])
summary = {'rows':len(raw), 'airline_missing_raw':int(raw.Airline.isna().sum()),
           'airline_missing_legacy':int(old.Airline.isna().sum()),
           'airline_missing_clean':int(new.Airline.isna().sum()),
           'clean_reversal_changed_rows':0,
           'data_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
del old, new, reverse
time = build_time_features(raw)
old = build_speed_features(time)
new = build_speed_features(time, safe=True)
summary.update({'zero_gap_rows':int(time.Estimated_Duration.eq(0).sum()),
                'legacy_ratio_max':float(old.Air_Speed_Proxy.max()),
                'clean_ratio_max':float(new.Distance_Per_Local_Minute.max()),
                'clean_ratio_missing':int(new.Distance_Per_Local_Minute.isna().sum())})
del old, new, time
rows = []
for key in ['P4','P4_clean','P6_fixed','P6_clean']:
    spec = next(s for s in PHASES if s.key == key)
    X, y, U = build_features(raw, spec)
    combined = pd.concat([X,U], ignore_index=True)
    ratio = 'Distance_Per_Local_Minute' if spec.safe_preprocessing else 'Air_Speed_Proxy'
    rows.append({'phase':key, 'labeled':len(X), 'unlabeled':len(U), 'features':len(X.columns),
                 'sin_missing':int(combined.Sin_Dep_Hour.isna().sum()),
                 'ratio_missing':int(combined[ratio].isna().sum()),
                 'ratio_max':float(combined[ratio].max())})
    assert len(combined) == len(raw) and int(y.sum()) == 45000
    assert not np.isinf(combined.select_dtypes(include='number').to_numpy()).any()
    if spec.safe_preprocessing:
        assert 'Estimated_Duration' not in X and 'Air_Speed_Proxy' not in X
        assert (combined.loc[combined.Dep_Hour_Missing.eq(1),'Sin_Dep_Hour'].isna()).all()
    print(rows[-1], flush=True)
    del X,y,U,combined
out = ROOT / 'output'
pd.DataFrame(rows).to_csv(out / 'preprocessing_clean_comparison.csv', index=False)
(out / 'preprocessing_clean_checks.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2), flush=True)
