"""Executed companion of completed full-data results; never trains models."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook(cells=[
md('''# 전처리 변경 후 실제 성능 비교

실제 라벨 255,001행, 10조건 × 3시드(42/1/7), 각 5-fold pooled OOF.
이 노트북은 실행이 완료된 실험 CSV만 읽는다. 전처리 규칙의 타당성과 성능 향상은 별개다.
**3시드 표준편차는 새 데이터에 대한 신뢰구간이 아니다.**

clean=세 변경 묶음, impute=대치만, ratio=시간차 비율·관련 플래그만,
missing=시각 결측 표현·원래 결측 플래그만. 이전 붕괴 프로토콜 점수는 비교하지 않는다.
'''),
code('''from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ROOT=next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/features.py').exists())
OUT=ROOT/'output'
runs=pd.read_csv(OUT/'preprocessing_full_runs.csv')
summary=pd.read_csv(OUT/'preprocessing_full_summary.csv')
paired=pd.read_csv(OUT/'preprocessing_full_paired_deltas.csv')
assert len(runs)==30 and runs.groupby('phase_key').size().eq(3).all()
assert not runs.duplicated(['phase_key','seed']).any()
plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False
display(summary.round(6))'''),
md('''## 같은 시드에서의 변경 효과
점은 변경 조건−기준 조건의 3시드 평균, 막대는 짝지은 차이의 표본 표준편차(±1 SD)다.
F1/AUC는 오른쪽, LogLoss는 왼쪽이 좋다. 0.002는 실무 참고선이며 통계적 유의수준이 아니다.
'''),
code('''metrics=[('macro_f1_nested','Macro F1 (높을수록 좋음)'),('log_loss','LogLoss (낮을수록 좋음)'),('roc_auc','AUC (높을수록 좋음)')]
fig, axes=plt.subplots(1,3,figsize=(15,5.5),sharey=True)
labels=(paired.reference+' → '+paired.variant).tolist()
for ax,(metric,title) in zip(axes,metrics):
    x=paired[metric+'_delta_mean']; e=paired[metric+'_delta_std']
    ax.errorbar(x,range(len(labels)),xerr=e,fmt='o',capsize=4,color='#267b91')
    ax.axvline(0,color='#555555',lw=1)
    ax.set_title(title); ax.set_xlabel('변경 - 기준 (평균 ± SD)')
    ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(range(len(labels)),labels); axes[0].invert_yaxis()
fig.suptitle('전처리 변경 효과: 전체 데이터 · 동일 시드끼리 비교')
fig.tight_layout(); fig.savefig(OUT/'preprocessing_full_effects.png',dpi=160); plt.show()
display(paired.drop(columns=[c for c in paired if c.endswith('_deltas')]).round(6))'''),
md('''## 시드별 Macro F1
조건별 색 선은 모델 복잡도 곡선이 아니라 시드 반복 결과다. 가로축 시드는 시간 순서가 아니다.
'''),
code('''groups=[['P4','P4_clean','P4_impute','P4_ratio','P4_missing'],['P6_fixed','P6_clean','P6_fixed_impute','P6_fixed_ratio','P6_fixed_missing']]
fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
colors=['#444444','#267b91','#bb633c','#65864b','#9671a5']
for ax,keys in zip(axes,groups):
    for key,color in zip(keys,colors):
        r=runs[runs.phase_key.eq(key)].set_index('seed').reindex([42,1,7])
        ax.plot(range(3),r.macro_f1_nested,marker='o',label=key,color=color)
    ax.set_xticks(range(3),['42','1','7']); ax.set_xlabel('반복 시드'); ax.set_title(keys[0]+' 계열')
    ax.grid(alpha=.2); ax.legend(frameon=False,fontsize=8)
axes[0].set_ylabel('pooled Macro F1'); fig.tight_layout()
fig.savefig(OUT/'preprocessing_full_seed_variation.png',dpi=160); plt.show()'''),
md('''## 개별 결과와 선택 기록
트리 수와 임계값은 outer-valid 라벨을 보지 않고 내부 holdout에서 선택했다.
'''),
code('''display(runs)
display(pd.read_csv(OUT/'preprocessing_full_selection_checks.csv'))'''),
md('''## 해석 범위
분할 시드를 반복해도 동일 데이터셋이다. 작은 차이의 통계적 유의성이나 배포 성능을 확정하지 않는다.
모든 조건의 학습 크기·nested TE·그리드·임계값 선택 절차는 같다.
전체 원본을 사용하는 비타깃 집계 계약은 유지했으므로 시간순 배포 검증은 별도다.
종합 판정은 `output/preprocessing_full_evaluation.md`, 원본 실행 로그/그리드 점수는
`output/baseline_recovery_v2_preprocessing_full_seed*.{csv,md,log}`에 있다.
'''),
])
nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
nbf.write(nb,ROOT/'notebooks/preprocessing_full_evaluation.ipynb')
html,_=HTMLExporter(exclude_input=True).from_notebook_node(nb)
(ROOT/'output/preprocessing_full_evaluation.html').write_text(html,encoding='utf-8')
print('Evaluation notebook executed and exported.')
