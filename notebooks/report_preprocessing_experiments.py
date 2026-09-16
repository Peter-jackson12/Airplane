"""Summarize completed paired runs without selecting from validation folds."""
from pathlib import Path
import json
import math
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'


def markdown(df):
    # Keep this report independent of optional tabulate installation.
    def fmt(v):
        return f'{v:.6f}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(df.columns)+' |',
                       '| '+' | '.join(['---']*len(df.columns))+' |'] +
                      ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False, name=None)])


def main():
    from run_preprocessing_experiments import summarize, KEYS, SEEDS
    summarize()
    normalized_metadata=[]
    for seed in SEEDS:
        original=pd.read_csv(OUT/f'baseline_recovery_v2_preprocessing_full_seed{seed}.csv')
        assert original.run_id.nunique()==1 and original.run_metadata.nunique()==1
        metadata=json.loads(original.run_metadata.iloc[0])
        assert metadata['sample'] is None
        assert metadata['cv']['seed']==seed and metadata['params']['random_state']==seed
        metadata['cv'].pop('seed')
        metadata['params'].pop('random_state')
        normalized_metadata.append(metadata)
    assert all(m==normalized_metadata[0] for m in normalized_metadata), 'Non-seed protocol drift'
    rows = pd.read_csv(OUT/'preprocessing_full_runs.csv')
    means = pd.read_csv(OUT/'preprocessing_full_summary.csv')
    delta = pd.read_csv(OUT/'preprocessing_full_paired_deltas.csv')
    means = means.sort_values('macro_f1_nested_mean', ascending=False)
    checks=[]
    for r in rows.itertuples():
        th=json.loads(r.per_fold_thresholds)
        trees=json.loads(r.selected_n_estimators)
        checks.append({'phase':r.phase_key,'seed':r.seed,'thresholds':str(th),'trees':str(trees),
                       'threshold_at_boundary':any(abs(t-.10)<1e-8 or abs(t-.70)<1e-8 for t in th)})
    diagnostics=pd.DataFrame(checks)
    diagnostics.to_csv(OUT/'preprocessing_full_selection_checks.csv', index=False)
    notes=[]
    for r in delta.itertuples():
        values=json.loads(r.macro_f1_nested_deltas)
        direction='3시드 모두 증가' if min(values)>0 else ('3시드 모두 감소' if max(values)<0 else '시드별 방향 혼재/동률')
        notes.append(f'- {r.reference} → {r.variant}: Macro F1 평균 차이 {r.macro_f1_nested_delta_mean:+.6f}, '
                     f'짝지은 차이의 표준편차 {r.macro_f1_nested_delta_std:.6f}; {direction}.')
    noisy=means.loc[means.macro_f1_nested_std.gt(.002),'phase_key'].tolist()
    parts=[
        '# 전처리 변경의 전체 데이터 성능 검증',
        '2026-09-16. 사용자의 전체 재학습 승인 후 실행. 기존 결과 파일은 보존했다.',
        '## 최종 판정',
        '**개선 전처리의 성능 향상은 입증되지 않았다.** P4 묶음 개선의 평균 Macro F1 차이는 '
        '-0.000375, P6 묶음 개선은 -0.000377이다. P4의 LogLoss/AUC도 평균상 악화됐고 '
        '(LogLoss 증가, AUC 감소), P6의 LogLoss/AUC 평균 차이는 거의 없다. '
        '차이가 작다는 사실이 통계적 동등성을 증명하지는 않는다.',
        'P6_clean은 시드 42/1에서는 F1이 +0.000390/+0.000678 개선됐지만 시드 7에서는 '
        '-0.002198 낮아졌다. 두 시드의 긍정 결과만으로 채택했다면 선택 편향이 생겼을 것이다. '
        '0.002보다 작은 시드 표준편차라는 조건만으로 개별 비교를 유의하다고 판정할 수도 없다.',
        '이번 비교에서 Macro F1 평균 최고는 기존 전처리의 **P6_fixed (0.576062 ± 0.001491 SD)**다. '
        '그러나 모호한 첫 관측값 대치나 0분 분모 처리의 타당성을 이 점수가 정당화하지는 않는다. '
        '**성능 기준선은 P6_fixed로 기록하고, P6_clean은 데이터 처리 개선 후보로 보존하되 '
        '성능 향상 모델로 승격하지 않는다.** 날씨 결합은 재개하지 않았다.',
        '세 가지 개별 변경 중 평균 F1이 기준보다 높아진 조건은 없었다. 특히 P6의 ratio-only는 '
        '세 시드 모두 F1이 낮아 평균 -0.001003이었다. 다만 이 조건에는 이름/분모/플래그 변경이 '
        '함께 있으므로 “0분 처리가 원인”이라고 세분해서 단정할 수 없다. '
        '다른 조건도 작은 차이를 확정적 인과·유의성 주장으로 확대하지 않는다.',
        '강사 발표에는 “전처리의 재현성과 의미를 개선했지만, 동일 프로토콜의 전체 데이터 '
        '3시드 비교에서는 예측 성능 향상으로 이어지지 않았다”라고 설명한다. '
        '기존 진단 시점 표는 과거 기록으로 남기고 새 전처리에는 이 보고서의 새 수치만 연결한다. '
        '과거 ES 붕괴 Phase 성능표 전체가 이번 일부 재측정으로 유효해지는 것은 아니다.',
        '## 검증 범위와 비교 기준',
        '실제 train.csv 100만 행에서 전처리하고 라벨 255,001행을 채점했다. '
        '10개 조건 × 시드 42/1/7 × 5 outer folds = 150개 채점 fold다. '
        '각 fold의 inner-train은 약 163,200행으로 동일하다. '
        'TE는 inner 경계에서 재계산하고, spw 없이 nested grid [10,25,50,75,100,150,300,600]를 '
        'inner-holdout LogLoss로 선택했다. 임계값도 해당 holdout에서만 선택했다. '
        '3개 시드는 CV 분할과 LightGBM random_state에 함께 적용했다.',
        '프로젝트 의사결정에서는 Macro F1을 먼저 보고 LogLoss/AUC를 함께 보고한다. '
        '원 대회의 공식 지표는 LogLoss이므로 강사 발표에서 둘을 혼동하지 않는다. '
        '과거 ES 붕괴 표는 이번 비교의 기준이 아니다. 기준은 이번에 다시 실행한 P4/P6_fixed다.',
        '조건: clean은 세 개선을 함께 적용, impute는 유일 대응 대치만, ratio는 분모 처리·명칭·관련 플래그만, '
        'missing은 시각/시간차 원래 결측 플래그와 NaN 순환 인코딩만 변경했다. '
        'P4는 원래도 NaN 인코딩이라 missing 조건의 추가 효과는 플래그다. '
        '각 구성 요소 내부의 플래그 효과까지 분리한 실험은 아니다.',
        '## 3시드 평균과 표준편차', markdown(means),
        '표준편차는 ddof=1이며 독립된 새 데이터셋에서의 불확실성이나 신뢰구간이 아니다. '
        '같은 데이터에서 시드만 바꾼 반복이라 일반화 유의성을 단정할 수 없다.',
        '## 동일 시드끼리 짝지은 차이 (개선/변경 − 기준)',
        markdown(delta.drop(columns=[c for c in delta if c.endswith('_deltas')])),
        'F1/AUC는 양수가 개선이고 LogLoss는 음수가 개선이다.',
        '\n'.join(notes),
        '## 0.002 기준과 시드 변동',
        f'Macro F1 시드 표준편차가 0.002를 넘는 조건: {noisy or "없음"}. '
        '0.002는 실무적 차이 판단의 참고선이며 통계적 유의수준이 아니다. '
        '3시드 평균 차이가 작으면 최고 평균이라는 이유만으로 우열을 확정하지 않는다. '
        '짝지은 차이의 부호와 분산을 함께 보고, 작은 차이를 주장해야 한다면 사전에 고정한 '
        '추가 시드와 별도 최종 holdout이 필요하다. 반복 수 증가는 데이터 누수나 방법 선택 편향을 해결하지 않는다.',
        '## fold별 트리 수·임계값', markdown(diagnostics),
        f'임계값 그리드 경계에 도달한 실행: {int(diagnostics.threshold_at_boundary.sum())}개. '
        '경계 도달은 후속 그리드 점검 대상이지 자동으로 잘못된 성능을 뜻하지 않는다.',
        '## 개별 실행 결과', markdown(rows),
        '## 재현 및 한계',
        '`uv run --offline python -u notebooks/run_preprocessing_experiments.py`로 실행한다. '
        '완료된 동일 지문 결과는 재사용하며 코드/설정/원본이 다르면 재개를 거부한다. '
        '세 시드별 baseline_recovery_v2_preprocessing_full_seed*.csv/md/log에 원본 결과, '
        '그리드 점수, 실행 시간, 코드/데이터 SHA256, 라이브러리 버전이 저장돼 있다.',
        '세 시드의 메타데이터에서 CV seed와 model random_state만 제외한 나머지가 동일함을 '
        '프로그램으로 검사했다. 실행 간 코드/데이터/설정 변경이 없음을 확인한 비교다.',
        '최종 전체 회귀 테스트: 158 passed. 개별 변경 옵션이 정확히 하나의 설정만 바꾸는 '
        '구조 테스트 6개도 포함한다. 새 실행 옵션은 --seed이며 CV/model에 함께 적용된다.',
        '이번 범위는 P4/P6와 전처리 ablation이다. P1~P3/P5의 새 전처리 적용 결과나 날씨 효과를 '
        '검증한 것이 아니다. 전체 데이터에서의 결측 복원/범주 사전이라는 기존 transductive '
        '계약은 유지했다. 배포 가용성·운항시각의 예정/실제 정의 문제는 남는다. '
        '이번 점수를 본 뒤 추가 선택한 조합은 확인 실험 전까지 탐색 결과로만 취급한다.',
    ]
    (OUT/'preprocessing_full_evaluation.md').write_text('\n\n'.join(parts)+'\n', encoding='utf-8')


if __name__=='__main__':
    main()
