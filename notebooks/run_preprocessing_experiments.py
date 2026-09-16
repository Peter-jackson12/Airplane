"""Approved full-data paired comparison, 10 conditions x 3 seeds. Resume-safe."""
from pathlib import Path
import subprocess
import sys
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'
SEEDS = [42, 1, 7]
KEYS = ['P4','P4_clean','P6_fixed','P6_clean',
        'P4_impute','P4_ratio','P4_missing',
        'P6_fixed_impute','P6_fixed_ratio','P6_fixed_missing']


def main():
    for seed in SEEDS:
        prefix = f'baseline_recovery_v2_preprocessing_full_seed{seed}'
        log = OUT / f'{prefix}.log'
        print(f'{datetime.now(timezone.utc).isoformat()} START seed={seed} log={log}', flush=True)
        with log.open('a', encoding='utf-8') as stream:
            completed = subprocess.run(
                [sys.executable, '-u', str(ROOT / 'rerun_all_phases.py'),
                 '--seed', str(seed), '--phases', ','.join(KEYS), '--output-prefix', prefix],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                encoding='utf-8', env={**__import__('os').environ, 'PYTHONIOENCODING':'utf-8'},
            )
        if completed.returncode:
            raise RuntimeError(f'seed={seed} failed; see {log}')
        print(f'DONE seed={seed}', flush=True)
    summarize()


def summarize():
    import pandas as pd
    import numpy as np
    frames = []
    for seed in SEEDS:
        df = pd.read_csv(OUT / f'baseline_recovery_v2_preprocessing_full_seed{seed}.csv')
        assert set(df.phase_key) == set(KEYS)
        assert df.n_rows.eq(255001).all()
        df['seed'] = seed
        frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)
    metrics = ['macro_f1_nested','log_loss','roc_auc']
    summary = all_rows.groupby('phase_key')[metrics].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(OUT / 'preprocessing_full_summary.csv')
    paired = []
    for base, variants in [('P4',['P4_clean','P4_impute','P4_ratio','P4_missing']),
                           ('P6_fixed',['P6_clean','P6_fixed_impute','P6_fixed_ratio','P6_fixed_missing'])]:
        ref = all_rows[all_rows.phase_key.eq(base)].set_index('seed')
        for key in variants:
            cur = all_rows[all_rows.phase_key.eq(key)].set_index('seed')
            row = {'reference':base, 'variant':key}
            for m in metrics:
                d = cur[m] - ref[m]
                row[m+'_delta_mean'] = d.mean()
                row[m+'_delta_std'] = d.std(ddof=1)
                row[m+'_deltas'] = json.dumps(d.reindex(SEEDS).tolist())
            paired.append(row)
    pd.DataFrame(paired).to_csv(OUT / 'preprocessing_full_paired_deltas.csv', index=False)
    all_rows[['phase_key','seed',*metrics,'selected_n_estimators','per_fold_thresholds','elapsed_sec']].to_csv(OUT / 'preprocessing_full_runs.csv', index=False)
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
