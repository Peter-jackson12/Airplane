"""Reproducible evidence-first F artwork using verified aggregate evidence.

Semantic hierarchy is independent of color; cohort numbers are not area encodings.
"""
from pathlib import Path
import json
try:
    from .build_readme_assets import reviewed_data
    from .readme_editorial import SVG, PAPER, INK, MUTED, RULE, ACCENT, HEADING, CYAN, SKY
except ImportError:
    from build_readme_assets import reviewed_data
    from readme_editorial import SVG, PAPER, INK, MUTED, RULE, ACCENT, HEADING, CYAN, SKY
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/readme'

def build():
    d=reviewed_data()
    assert d['attribution'][0]['rows']==706759
    weather=json.loads((ROOT/'output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_summary.json').read_text())
    assert weather['evaluation_rows']==180332 and weather['weather_feature_count']==14
    OUT.mkdir(parents=True,exist_ok=True)
    first=d['weather_model'][0]
    a=SVG(760,'날씨 추가 후 Macro F1 +0.025035 · 정적 교차검증',
          'I안. 동일 라벨 180,332행, P6_clean, 시드 42/1/7의 평균. 날씨 14개 피처를 추가한 정적 교차검증에서 Macro F1 0.573910에서 0.598945. 10분은 공개 지연 가정이며 인과 효과와 미래 운항 성능은 미검증.',False)
    a.text(80,62,'AIRPLANE / RESEARCH FINDINGS',28,weight=700)
    a.text(1520,62,'I · 2026-09-22',28,MUTED,anchor='end')
    a.line(80,90,1520,90,INK,2)
    a.text(80,172,'날씨를 더한 동일조건 비교에서',64,weight=800)
    a.text(80,252,'지연 예측 세 지표가 개선됐습니다',64,weight=800)
    a.text(80,318,'동일 라벨 180,332행 · P6_clean · 날씨 14개 피처 · 3시드 평균',30,MUTED)
    a.rect(80,366,1440,222,SKY)
    a.rect(80,366,8,222,CYAN)
    a.text(120,420,'Macro F1 변화 · 사용−미사용',30,ACCENT,700)
    a.text(112,546,f"{first['delta_mean']:+.6f}",112,INK,800)
    a.text(990,430,'미사용 → 사용',30,MUTED)
    a.text(990,498,f"{first['off_mean']:.6f}",44,weight=500)
    a.text(990,553,f"→ {first['on_mean']:.6f}",44,weight=800)
    a.text(80,642,'범위  날짜가 확인된 선택 집단의 정적 교차검증',32,weight=700)
    a.text(80,696,'10분은 공개 지연 가정 · 인과 효과와 미래 운항 성능은 미검증',28,MUTED)
    a.save(OUT/'presentation_cover.svg')

    a=SVG(1060,'비교할 수 있는 집단부터 구분합니다',
          '원본 1,000,000행. 전처리 평가 255,001행. 날짜 귀속 채택 706,759행 중 날짜와 라벨이 모두 있는 180,332행에서 날씨 14개 피처를 비교합니다. 같은 외부 폴드. 인과 효과와 미래 운항 성능은 미검증. 도형 크기는 수량을 뜻하지 않습니다.',False)
    a.text(80,62,'01 / POPULATIONS',28,weight=700)
    a.text(1520,62,'AIRPLANE / I',28,MUTED,anchor='end')
    a.line(80,90,1520,90,INK,2)
    a.text(80,178,'두 질문은 서로 다른 집단에서 검증했습니다',58,weight=800)
    a.text(80,244,'출발점: 원본 1,000,000행 · 숫자를 이어 성능 변화로 읽지 않습니다',30,MUTED)
    for y,tag,title,count,scope,conclusion in [
        (306,'A','전처리 비교','255,001','라벨이 있는 행','Macro F1 향상은 확인되지 않았습니다'),
        (600,'B','날씨 유무 비교','180,332','날짜 귀속 채택 706,759행 중 라벨이 있는 행','날씨 14개 피처 추가 후 세 지표가 개선됐습니다')]:
        a.line(80,y,1520,y,INK,2)
        a.text(80,y+64,tag,44,ACCENT,800)
        a.text(164,y+64,title,36,weight=700)
        a.text(164,y+160,count,80,weight=800)
        a.text(164,y+222,scope,28,MUTED)
        a.text(840,y+102,'질문에 대한 답',28,ACCENT,700)
        # The weather conclusion uses two lines rather than squeezing its type.
        if tag=='A':
            a.text(840,y+156,'Macro F1 향상은',34,weight=700)
            a.text(840,y+208,'확인되지 않았습니다',34,weight=700)
        else:
            a.text(840,y+156,'날씨 14개 피처 추가 후',34,weight=700)
            a.text(840,y+208,'세 지표가 개선됐습니다',34,weight=700)
    a.rect(80,904,1440,80,SKY)
    a.text(104,955,'날씨 비교 원칙: 같은 180,332행 · 같은 시드 · 같은 외부 폴드',32,weight=700)
    a.text(80,1030,'도형 크기는 수량을 뜻하지 않습니다.',28,MUTED)
    a.save(OUT/'analysis_journey.svg')

    a=SVG(1120,'모델 선택은 내부에서, 외부 검증은 최종 채점만',
          '층화 5분할 외부 학습 데이터 안에서 80대20으로 분리. 내부 학습 경계에서 TE를 계산하고 내부 검증으로 트리 수와 임계값을 선택. 외부 검증은 최종 채점 전용.',False)
    a.text(80,62,'02 / EVALUATION BOUNDARY',28,weight=700)
    a.text(1520,62,'AIRPLANE / I',28,MUTED,anchor='end')
    a.line(80,90,1520,90,INK,2)
    a.text(80,178,'선택은 안쪽에서, 채점은 바깥에서',64,weight=800)
    a.text(80,246,'층화 5분할 × 시드 42 / 1 / 7 · 각 조건에서 같은 선택 절차',30,MUTED)
    a.rect(80,304,928,586,SKY)
    a.text(112,363,'외부 학습 데이터 안에서만 선택',32,ACCENT,700)
    for y,n,title,line1,line2 in [(438,'01','내부 학습 / 80%','학습 경계에서 TE 계산','트리 수 후보별 모델 학습'),(672,'02','내부 검증 / 20%','LogLoss → 트리 수','Macro F1 → 임계값')]:
        a.text(112,y,n,52,weight=800)
        a.text(240,y,title,38,weight=700)
        a.text(240,y+62,line1,32)
        a.text(240,y+114,line2,32)
    a.line(112,590,976,590,INK,2)
    a.line(1056,304,1056,890,INK,3)
    a.text(1100,360,'03 / 최종 채점 전용',28,ACCENT,700)
    a.text(1100,468,'외부 검증',48,weight=800)
    a.text(1100,540,'선택에 사용하지 않음',30,weight=700)
    a.text(1100,612,'각 폴드의 예측 → OOF',28)
    a.text(1100,676,'OOF = 최종 평가',30,weight=700)
    a.text(1100,798,'선택한 모델 그대로',30)
    a.text(1100,846,'최종 성능을 채점',30)
    a.text(80,962,'외부 검증으로 선택하지 않음',34,weight=700)
    a.text(80,1016,'각 행을 학습에 쓰지 않은 모델의 예측을 모아 평가합니다.',30,MUTED)
    a.text(80,1076,'라벨 누수 방지와 미래 시점의 입력 확보 가능성은 서로 다른 문제입니다.',28,MUTED)
    a.save(OUT/'evaluation_boundary.svg')
    print('Built three evidence-first F story assets; no network, raw data or training.')
if __name__=='__main__':build()
