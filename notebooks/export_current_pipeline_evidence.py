"""Export actual clean input schemas and completed-run evidence; never fit models."""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rerun_all_phases as runner
from src.features import impute_cross


def main():
    source = ROOT / 'data/train.csv'
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = pd.read_csv(source)
    evidence = {'purpose': 'current raw-to-feature schema, not a new performance run',
                'data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'code_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                               for p in ['src/features.py','src/cv.py','rerun_all_phases.py']},
                'input_schemas': {}, 'completed_runs': []}
    for key in ['P4_clean','P6_clean']:
        spec = next(s for s in runner.PHASES if s.key == key)
        X,y,U = runner.build_features(raw,spec)
        te_cols = [c for c in spec.cat_cols if c in X]
        final = [c for c in X if not (spec.te_drop_original and c in te_cols)]
        final += ['TE_'+c for c in te_cols]
        assert len(X)==255001 and len(U)==744999 and int(y.sum())==45000
        assert X.columns.equals(U.columns)
        evidence['input_schemas'][key] = {
            'n_labeled':len(X),'n_unlabeled':len(U),
            'before_te':list(X.columns),'final_columns_from_te_contract':final,
            'dtypes_before_te':{c:str(t) for c,t in X.dtypes.items()}}
    for seed in [42,1,7]:
        runs=pd.read_csv(ROOT/f'output/baseline_recovery_v2_preprocessing_full_seed{seed}.csv')
        assert len(runs)==10
        for _, row in runs.iterrows():
            meta=json.loads(row.run_metadata)
            assert meta['data_sha256']==evidence['data_sha256']
            evidence['completed_runs'].append({
                'seed':seed, 'phase':row.phase_key, 'run_id':row.run_id,
                'versions':meta['versions'],'params':json.loads(row.lgbm_params),
                'elapsed_sec':float(row.elapsed_sec),
                'confusion_matrix':{k:int(row[k]) for k in ['tn','fp','fn','tp']},
                'thresholds':json.loads(row.per_fold_thresholds),
                'selected_n_estimators':json.loads(row.selected_n_estimators)})
    old=impute_cross(raw,conflict='first')
    clean=impute_cross(raw,conflict='unique')
    changed=raw.Airline.isna() & old.Airline.notna() & clean.Airline.isna()
    example=raw.loc[changed,['ID','Carrier_Code(IATA)','Airline']].head(8).copy()
    example['old_filled_airline']=old.loc[example.index,'Airline']
    example['clean_airline']=clean.loc[example.index,'Airline']
    example.to_csv(ROOT/'output/current_imputation_examples.csv',index=False)
    evidence['example_note']='Actual rows: clean leaves ambiguous Airline missing; not prediction errors.'
    (ROOT/'output/current_pipeline_evidence.json').write_text(
        json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Current schema, 30 completed runs, and actual imputation examples exported.')


if __name__=='__main__':
    main()
