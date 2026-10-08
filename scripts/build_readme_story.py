"""Reproducible flight-magazine C artwork using verified aggregate evidence.

Decorative flight paths and ticket geometry never encode observed quantities.
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

def barcode(a,x,y,h=50):
    """Decorative publishing mark, not a real boarding credential."""
    for i,w in enumerate([2,5,1,3,7,2,4,1,5,3,2,6,1,4,2,5,1,3,6,2]):
        a.rect(x+i*10,y,w,h,INK)

def perforation(a,x,y1,y2):
    for y in range(y1,y2,20): a.line(x,y,x,y+9,MUTED,1)

def build():
    d=reviewed_data()
    assert d['attribution'][0]['rows']==706759
    weather=json.loads((ROOT/'output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_summary.json').read_text())
    assert weather['evaluation_rows']==180332 and weather['weather_feature_count']==14
    OUT.mkdir(parents=True,exist_ok=True)
    a=SVG(1110,'Airplane · 항공편 지연 예측과 날씨 정보 확장','항공 연구 매거진 C안. 전처리의 타당성과 새로운 정보의 예측력을 나누어 검증한 정적 교차검증 프로젝트. 문서 기준 2026-09-22. 장식은 실제 비행 경로나 운항 데이터가 아닙니다.',False)
    a.rect(0,0,1600,460,SKY)
    a.text(58,59,'FLIGHT RESEARCH JOURNAL',21,weight=700,spacing=3)
    a.text(1540,59,'SPECIAL EDITION / C',21,anchor='end',spacing=2)
    a.text(44,256,'AIRPLANE',206,weight=900,spacing=-8)
    a.line(58,290,1540,290,INK,3)
    a.text(58,342,'항공편 지연 예측과 날씨 정보 확장', 40,weight=700)
    a.text(58,402,'전처리의 타당성과 새로운 정보의 예측력을 나누어 검증했습니다.',27)
    # Asymmetric editorial spread: short thesis, oversized chapter numeral, diagonal route.
    a.text(54,568,'THE',78,weight=900)
    a.text(48,673,'WEATHER',112,weight=900,spacing=-4)
    a.text(52,776,'QUESTION.',94,weight=900,spacing=-3)
    a.text(60,842,'날씨를 더하면,',35,weight=700)
    a.text(60,892,'예측은 달라지는가?',35,weight=700)
    a.rect(944,498,596,410,GOLD)
    a.text(980,552,'연구의 질문',23,weight=700)
    a.text(979,625,'더 타당한 전처리.',41,weight=700)
    a.text(979,683,'더 많은 정보.',41,weight=700)
    a.line(981,724,1500,724,INK,2)
    a.text(981,774,'같은 조건에서 따로 검증합니다.',25)
    a.text(981,824,'LightGBM · 이진 분류',25)
    a.text(981,868,'동일조건 정적 교차검증',25)
    a.path('M757 853 C867 836 794 595 932 555',ACCENT,3)
    a.path('M912 552 L932 555 L924 575',ACCENT,3)
    a.circle(757,853,7,INK,INK)
    a.rect(0,952,1600,158,INK)
    a.text(60,1008,'READING PASS',21,PAPER,700,spacing=3)
    a.text(60,1061,'본문 16분 + 전환·질문 3분',31,PAPER,700)
    a.text(1016,1010,'연구 기록 · Peter-jackson12 TF',23,PAPER)
    a.text(1016,1054,'문서 기준 2026-09-22',23,PAPER)
    a.text(1016,1087,'권장 배분이며 실제 낭독 측정이 아닙니다.',17,PAPER)
    a.save(OUT/'presentation_cover.svg')

    a=SVG(1310,'두 평가 집단을 잇지 않는 탑승권','원본 1,000,000행. 전처리 평가 255,001행. 날짜 귀속 채택 706,759행 중 날짜와 라벨이 모두 있는 180,332행에서 날씨 14개 피처를 비교합니다. 같은 외부 폴드를 사용합니다. 인과 효과와 미래 운항 성능은 미검증. 도형 크기는 수량을 뜻하지 않습니다.',False)
    a.text(60,66,'RESEARCH MANIFEST',23,weight=700,spacing=3)
    a.text(1540,66,'01 / POPULATIONS',22,anchor='end')
    a.text(54,181,'서로 다른 집단,',80,weight=900)
    a.text(54,279,'서로 다른 질문.',80,weight=900)
    a.rect(1010,108,530,209,SKY)
    a.text(1045,154,'출발점 / 원본 데이터',23,MUTED)
    a.text(1040,234,'1,000,000',70,weight=800)
    a.text(1045,284,'행',23)
    # Two deliberately staggered boarding-pass strips, no fake route code.
    a.rect(60,366,1380,318,SKY)
    a.rect(60,366,115,318,INK)
    a.text(85,435,'A', 60,PAPER,900)
    a.text(210,418,'전처리 평가',28,weight=700)
    a.text(204,522,'255,001',96,weight=900)
    a.text(212,579,'라벨이 있는 행',26)
    a.text(212,632,'P4 / P6의 전처리 조건 비교',25)
    perforation(a,853,391,660)
    a.text(897,423,'질문 01',22,ACCENT,700)
    a.text(897,480,'전처리는 더 타당해졌지만',28,weight=700)
    a.text(897,527,'Macro F1 향상은',28,weight=700)
    a.text(897,574,'확인되지 않았습니다.',28,weight=700)
    barcode(a,900,608,38)
    a.rect(159,730,1381,364,GOLD)
    a.rect(159,730,115,364,INK)
    a.text(184,799,'B',60,PAPER,900)
    a.text(308,782,'날짜 귀속 채택 706,759행',28,weight=700)
    a.text(308,830,'그중 날짜와 라벨이 모두 있는',24)
    a.text(302,938,'180,332',96,weight=900)
    a.text(310,990,'날씨 유무를 비교하는 동일 집단',25)
    a.text(310,1041,'P6_clean을 두 조건에서 각각 다시 학습',23)
    perforation(a,953,752,1070)
    a.text(996,788,'질문 02',22,ACCENT,700)
    a.text(996,847,'새 정보를 더하면',34,weight=700)
    a.text(996,902,'성능이 달라지나요?',34,weight=700)
    a.text(996,964,'날씨 14개 피처 추가',26)
    a.text(996,1022,'같은 내부 선택 절차',25)
    a.line(60,1141,1540,1141,INK,3)
    a.text(60,1196,'비교 원칙',26,ACCENT,700)
    a.text(325,1196,'같은 180,332행 · 같은 시드 · 같은 외부 폴드',29,weight=700)
    a.text(60,1254,'도형 크기는 수량을 뜻하지 않습니다.',21,MUTED)
    a.text(1540,1254,'인과 효과와 미래 운항 성능은 미검증',21,MUTED,anchor='end')
    a.save(OUT/'analysis_journey.svg')

    a=SVG(1250,'모델 선택과 최종 채점 사이의 경계','층화 5분할 외부 학습 데이터 안에서 80대20으로 분리합니다. 내부 학습 경계에서 TE를 계산하고 내부 검증으로 트리 수와 임계값을 선택합니다. 외부 검증은 최종 채점 전용이며 선택에 사용하지 않습니다.',False)
    a.text(60,63,'METHOD / CONTROLLED ACCESS',23,weight=700,spacing=2)
    a.text(1540,63,'02 / EVALUATION',22,anchor='end')
    a.text(52,176,'선택은 안쪽에서.',82,weight=900)
    a.text(52,274,'채점은 바깥에서.',82,weight=900)
    a.text(60,333,'층화 5분할 × 시드 42 / 1 / 7 · 각 조건에서 같은 선택 절차',27,MUTED)
    a.rect(60,390,973,574,SKY)
    a.text(95,440,'외부 학습 데이터 안에서만 선택',28,ACCENT,700)
    a.text(90,539,'01',76,weight=900)
    a.text(278,499,'내부 학습 / 80%',28,weight=700)
    a.text(278,548,'학습 경계에서 TE 계산',27)
    a.text(278,595,'트리 수 후보별 모델 학습',27)
    a.line(95,640,992,640,INK,2)
    a.text(90,740,'02',76,weight=900)
    a.text(278,702,'내부 검증 / 20%',28,weight=700)
    a.text(278,751,'LogLoss → 트리 수',27)
    a.text(278,798,'Macro F1 → 임계값',27)
    a.rect(95,852,897,71,INK)
    a.text(125,898,'선택한 모델 그대로 평가',30,PAPER,700)
    for y in range(406,969,37):a.line(1072,y,1100,y-20,GOLD,9)
    a.rect(1140,528,400,436,GOLD)
    a.text(1170,583,'최종 채점 전용',27,weight=700)
    a.text(1170,689,'03', 90,weight=900)
    a.text(1170,746,'외부 검증',34,weight=700)
    a.text(1170,807,'선택에 사용하지 않음',24)
    a.text(1170,857,'각 폴드의 예측 → OOF',23)
    a.text(1170,916,'OOF = 최종 평가',24,weight=700)
    a.line(60,1020,1540,1020,INK,3)
    a.text(60,1075,'외부 검증으로 선택하지 않음',31,ACCENT,700)
    a.text(60,1127,'각 행을 학습에 쓰지 않은 모델의 예측을 모아 평가합니다.',27)
    a.text(60,1190,'라벨 누수 방지와 미래 시점의 입력 확보 가능성은 서로 다른 문제입니다.',25,MUTED)
    a.save(OUT/'evaluation_boundary.svg')
    print('Built three flight-magazine C story assets; no network, raw data or training.')
if __name__=='__main__':build()
