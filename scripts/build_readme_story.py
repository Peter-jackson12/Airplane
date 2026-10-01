"""Deterministic editorial story artwork; no network, raw data or training.

Counts are checked against the same tracked aggregates as the numerical figures.
Conceptual diagram geometry does not encode quantities.
"""
from pathlib import Path
import json
try:
    from .build_readme_assets import reviewed_data
    from .readme_editorial import SVG, PAPER, INK, MUTED, RULE, ACCENT, HEADING, GOLD, SKY
except ImportError:
    from build_readme_assets import reviewed_data
    from readme_editorial import SVG, PAPER, INK, MUTED, RULE, ACCENT, HEADING, GOLD, SKY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/readme'

def build():
    d = reviewed_data()
    assert d['attribution'][0]['rows'] == 706759
    weather = json.loads((ROOT / 'output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_summary.json').read_text(encoding='utf-8'))
    assert weather['evaluation_rows'] == 180332 and weather['weather_feature_count'] == 14
    OUT.mkdir(parents=True, exist_ok=True)
    # A short, typographic masthead. No decorative route or fictional operational data.
    a=SVG(570,'Airplane · 항공편 지연 예측과 날씨 정보 확장','전처리의 타당성과 새로운 정보의 예측력을 나누어 검증한 정적 교차검증 프로젝트. 문서 기준 2026-09-22.',header_band=False)
    a.rect(1160,88,440,426,SKY)
    a.running('Airplane', 'FLIGHT DELAY STUDY')
    a.text(80,208,'항공편 지연 예측과',76,weight=500,family=HEADING)
    a.text(80,314,'날씨 정보 확장',76,weight=500,family=HEADING)
    a.line(84,357,235,357,GOLD,5)
    a.text(84,420,'전처리의 타당성과 새로운 정보의 예측력을',30)
    a.text(84,468,'나누어 검증했습니다.',30)
    a.line(1160,144,1160,472,RULE,1)
    a.text(1210,177,'분석 방법',22,MUTED)
    a.text(1210,224,'LightGBM',29,weight=500)
    a.text(1210,270,'이진 분류',29,weight=500)
    a.text(1210,341,'검증 범위',22,MUTED)
    a.text(1210,387,'동일조건',29,weight=500)
    a.text(1210,431,'정적 교차검증',29,weight=500)
    a.line(84,514,1516,514,RULE,1)
    a.text(84,548,'연구 기록',19,MUTED)
    a.text(1516,548,'문서 기준 2026-09-22',19,MUTED,anchor='end')
    a.save(OUT / 'presentation_cover.svg')

    # Two separate population paths: the 180,332 row weather cohort is nested in the
    # accepted date-attributed population, not a score continuation of the 255,001 set.
    a=SVG(1070,'평가 집단은 두 개입니다','원본 1,000,000행. 전처리 평가에는 라벨 255,001행을 사용했습니다. 날짜 귀속 채택 706,759행 중 날짜와 라벨이 모두 있는 180,332행에서 날씨 유무를 비교했습니다. 도형의 크기와 선 길이는 수량을 나타내지 않습니다.')
    a.running('분석 범위','TWO EVALUATION POPULATIONS')
    a.text(80,175,'평가 집단은 두 개입니다',56,family=HEADING,weight=500)
    a.text(84,225,'전처리 평가와 날씨 비교의 점수를 직접 이어 읽지 않습니다.',28,MUTED)
    a.text(84,319,'원본 데이터',25,MUTED)
    a.text(355,324,'1,000,000',68,family=HEADING,weight=500)
    a.text(731,323,'행',27,MUTED)
    a.path('M84 356 H1516',INK,2)
    a.line(788,397,788,798,RULE,1)
    a.text(84,418,'A',22,ACCENT,weight=600)
    a.text(129,418,'전처리 평가',28,weight=500)
    a.text(80,520,'255,001',84,family=HEADING,weight=500)
    a.text(486,518,'행',27,MUTED)
    a.text(84,567,'라벨이 있는 행',28)
    a.text(84,622,'P4 / P6의 전처리 조건 비교',27,MUTED)
    a.line(84,672,695,672,RULE,1)
    a.text(84,718,'전처리는 더 타당해졌지만',31,family=HEADING,weight=500)
    a.text(84,766,'Macro F1 향상은 확인되지 않았습니다.',27,MUTED)
    a.text(844,418,'B',22,ACCENT,weight=600)
    a.text(889,418,'날짜 귀속 채택',28,weight=500)
    a.text(840,505,'706,759',68,family=HEADING,weight=500)
    a.text(1170,503,'행',27,MUTED)
    a.path('M866 539 V588 H908',MUTED,2)
    a.text(931,579,'그중 날짜와 라벨이 모두 있는',26,MUTED)
    a.text(925,674,'180,332',84,ACCENT,weight=500,family=HEADING)
    a.text(1330,672,'행',27,ACCENT)
    a.text(931,723,'날씨 유무를 비교하는 동일 집단',27)
    a.text(931,770,'P6_clean을 두 조건에서 각각 다시 학습',25,MUTED)
    a.line(84,830,1516,830,INK,2)
    a.text(84,879,'비교 원칙',24,ACCENT,weight=500)
    a.text(330,878,'같은 180,332행 · 같은 시드 · 같은 외부 폴드',30,weight=500)
    a.text(330,928,'날씨 14개 피처 추가 · 같은 내부 절차로 트리 수와 임계값 선택',26,MUTED)
    a.text(84,990,'인과 효과와 미래 운항 성능은 미검증',20,MUTED)
    a.text(1516,1030,'Airplane / 분석 범위',20,MUTED,anchor='end')
    a.text(84,1030,'도형 크기는 수량을 뜻하지 않습니다.',20,MUTED)
    a.save(OUT / 'analysis_journey.svg')

    a=SVG(920,'모델 선택과 최종 평가의 경계','층화 5분할 외부 학습 데이터 안에서 다시 80대20으로 분리합니다. 내부 학습 경계에서 타깃 인코딩을 계산하고 내부 검증으로 트리 수와 임계값을 선택한 뒤 외부 검증을 최종 채점에만 사용합니다.')
    a.running('평가 방법','MODEL SELECTION / FINAL SCORING')
    a.text(80,175,'고르는 데이터와 채점하는 데이터를 분리',52,family=HEADING,weight=500)
    a.text(84,229,'층화 5분할 × 시드 42 / 1 / 7 · 각 조건에서 같은 선택 절차',28,MUTED)
    a.text(84,318,'외부 학습 데이터 안에서만 선택',27,ACCENT,weight=500)
    a.line(84,343,1004,343,ACCENT,2)
    a.text(84,401,'내부 학습 / 80%',24,MUTED)
    a.text(84,479,'학습',59,family=HEADING,weight=500)
    a.text(84,538,'학습 경계에서 TE 계산',27)
    a.text(84,583,'트리 수 후보별 모델 학습',27)
    a.path('M448 466 H506 M494 456 L506 466 L494 476',INK,2)
    a.text(554,401,'내부 검증 / 20%',24,MUTED)
    a.text(554,479,'선택',59,family=HEADING,weight=500)
    a.text(554,538,'LogLoss → 트리 수',27)
    a.text(554,583,'Macro F1 → 임계값',27)
    a.line(1053,302,1053,682,INK,1)
    a.text(1110,318,'최종 채점 전용',27,ACCENT,weight=500)
    a.line(1110,343,1516,343,ACCENT,2)
    a.text(1110,401,'외부 검증',24,MUTED)
    a.text(1110,479,'채점',59,family=HEADING,weight=500)
    a.text(1110,538,'선택에 사용하지 않음',27)
    a.text(1110,583,'각 폴드의 예측 → OOF',27)
    a.line(84,642,1004,642,RULE,1)
    a.text(84,684,'선택한 모델 그대로 평가 · 외부 검증으로 선택하지 않음',27,MUTED)
    a.line(84,736,1516,736,INK,2)
    a.text(84,792,'OOF',27,ACCENT,weight=500)
    a.text(230,792,'각 행을 학습에 쓰지 않은 모델의 예측을 모아 평가',29)
    a.text(84,858,'라벨 누수 방지와 미래 시점의 입력 확보 가능성은 서로 다른 문제입니다.',26,MUTED)
    a.save(OUT / 'evaluation_boundary.svg')
    print('Built three editorial story assets; no network, raw data or model fitting.')


if __name__ == '__main__':
    build()
