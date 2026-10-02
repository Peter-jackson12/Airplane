"""Original H technical-announcement story figures; tracked evidence only."""
from pathlib import Path
import json
try:
    from .build_readme_assets import reviewed_data
    from .readme_editorial import SVG,PAPER,INK,MUTED,RULE,ACCENT,CYAN,SKY,HIGHLIGHT
except ImportError:
    from build_readme_assets import reviewed_data
    from readme_editorial import SVG,PAPER,INK,MUTED,RULE,ACCENT,CYAN,SKY,HIGHLIGHT
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/readme'
def build():
    d=reviewed_data(); w=d['weather_model'][0]
    OUT.mkdir(parents=True,exist_ok=True)
    a=SVG(1000,'항공편 지연 예측, 날씨 정보로 넓힌 검증',
      'H안. 동일 라벨 180,332행의 정적 교차검증. Macro F1 +0.025035. 10분은 공개 지연 가정. 인과 효과와 미래 운항 성능은 미검증.')
    a.rect(0,0,1600,648,SKY)
    a.text(120,100,'AIRPLANE  /  연구 업데이트',30,ACCENT,600)
    a.text(120,212,'항공편 지연 예측,',80,weight=500)
    a.text(120,318,'날씨 정보로 넓힌 검증',80,weight=500)
    a.text(120,418,'2026.09.22',28,MUTED)
    a.text(120,464,'Peter-jackson12 TF',28,MUTED)
    a.text(524,416,'날씨 14개 피처를 더한 동일조건 비교에서',34)
    a.text(524,472,'Macro F1 · LogLoss · ROC-AUC가 개선됐습니다.',34)
    a.line(120,528,1480,528,RULE,2)
    a.text(120,590,'동일 라벨 180,332행 · P6_clean · 시드 42 / 1 / 7',30,MUTED)
    # A benchmark-style hero, not an oversized decorative delta card.
    a.text(120,720,'Macro F1 ↑',38,weight=600)
    a.text(690,716,'날씨 미사용',28,MUTED,anchor='middle')
    a.text(1180,716,'날씨 사용',28,ACCENT,600,anchor='middle')
    a.rect(946,742,470,104,SKY,18)
    a.text(690,814,f"{w['off_mean']:.6f}",64,weight=400,anchor='middle')
    a.text(1180,814,f"{w['on_mean']:.6f}",64,ACCENT,600,anchor='middle')
    a.text(120,804,f"변화 {w['delta_mean']:+.6f}",36,ACCENT,600)
    a.line(120,884,1480,884,RULE,2)
    a.text(120,936,'정적 교차검증 · 10분 공개 지연 가정 · 인과 효과와 미래 운항 성능은 미검증',28,MUTED)
    a.save(OUT/'presentation_cover.svg')
    a=SVG(1050,'다른 질문에는 다른 평가 집단',
      '원본 1,000,000행. 전처리 평가 255,001행. 날짜 귀속 채택 706,759행 중 라벨 180,332행에서 날씨 14개 피처 비교. 같은 외부 폴드. 인과 효과와 미래 운항 성능은 미검증. 도형 크기는 수량을 뜻하지 않습니다.')
    a.text(800,110,'어떤 데이터에서 확인했나요?',64,weight=500,anchor='middle')
    a.text(800,174,'원본 1,000,000행에서 두 질문을 나누어 검증했습니다',30,MUTED,anchor='middle')
    a.rect(120,246,1360,166,HIGHLIGHT,24)
    a.text(170,308,'전체 입력',30,MUTED)
    a.text(170,375,'1,000,000행',58,weight=600)
    a.text(824,308,'날짜 귀속 채택',30,MUTED)
    a.text(824,375,'706,759행',58,weight=600)
    a.line(120,462,1480,462,RULE,2)
    a.rect(810,464,670,418,SKY,24)
    for x,title,count,line1,line2 in [(170,'전처리 비교','255,001행','라벨이 있는 전체 집단','Macro F1 향상은 확인되지 않음'),(860,'날씨 유무 비교','180,332행','날짜와 라벨이 모두 있는 집단','날씨 14개 피처 추가 후 세 지표 개선')]:
        a.text(x,538,title,38,weight=600)
        a.text(x,630,count,64,ACCENT if x>800 else INK,600)
        a.text(x,701,line1,28,MUTED)
        a.text(x,774,line2,28,weight=600)
    a.text(800,950,'날씨 비교: 같은 평가행 · 같은 시드 · 같은 외부 폴드',32,ACCENT,600,anchor='middle')
    a.text(800,1004,'도형 크기는 수량을 뜻하지 않습니다.',28,MUTED,anchor='middle')
    a.save(OUT/'analysis_journey.svg')
    a=SVG(1030,'모델 선택과 최종 채점은 분리합니다',
      '층화 5분할 외부 학습 데이터 내부를 80대20으로 분리. 내부 학습 경계에서 TE 계산. 내부 검증으로 트리 수와 임계값 선택. 외부 검증은 최종 채점 전용.')
    a.text(800,110,'선택은 안쪽에서, 채점은 바깥에서',64,weight=500,anchor='middle')
    a.text(800,174,'층화 5분할 × 시드 42 / 1 / 7 · 각 조건에서 같은 선택 절차',30,MUTED,anchor='middle')
    a.rect(120,252,1360,94,SKY,18)
    a.text(170,312,'단계',30,ACCENT,600);a.text(640,312,'사용 목적',30,ACCENT,600)
    for y,n,label,one,two in [(434,'01','내부 학습 / 80%','학습 경계에서 TE 계산','트리 수 후보별 모델 학습'),(598,'02','내부 검증 / 20%','LogLoss → 트리 수','Macro F1 → 임계값'),(762,'03','외부 검증','최종 채점 전용','각 폴드의 예측을 모아 OOF 평가')]:
        a.text(170,y,n,36,MUTED);a.text(258,y,label,38,weight=600)
        a.text(640,y,one,34);a.text(640,y+56,two,30,MUTED)
        a.line(120,y+96,1480,y+96,RULE,2)
    a.text(800,918,'외부 검증으로 선택하지 않음',36,ACCENT,600,anchor='middle')
    a.text(800,980,'라벨 누수 방지와 미래 시점의 입력 확보 가능성은 서로 다른 문제입니다.',28,MUTED,anchor='middle')
    a.save(OUT/'evaluation_boundary.svg')
    print('Built three original technical-announcement H story assets.')
if __name__=='__main__':build()
