"""Original E aviation evidence brief; report-grid references credited in README.
Only tracked aggregates are read. No copied artwork, network or model training.
"""
from pathlib import Path
try:
    from .build_readme_assets import reviewed_data
    from .readme_editorial import SVG, PAPER, INK, MUTED, RULE, ACCENT, CYAN, SKY
except ImportError:
    from build_readme_assets import reviewed_data
    from readme_editorial import SVG, PAPER, INK, MUTED, RULE, ACCENT, CYAN, SKY
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/readme'

def header(a,no,topic):
    a.rect(64,42,54,42,ACCENT);a.text(91,72,no,23,PAPER,700,anchor='middle')
    a.text(140,72,topic,23,weight=700)
    a.text(1536,72,'AIRPLANE / EVIDENCE BRIEF',20,MUTED,anchor='end',spacing=1)
    a.line(64,104,1536,104,INK,2)

def build():
    data=reviewed_data(); OUT.mkdir(parents=True,exist_ok=True)
    a=SVG(1010,'Airplane · 항공편 지연 예측과 날씨 정보 확장','항공 데이터 브리프 E안. 검증된 집계에 기반한 요약. 날씨 비교 180,332행과 전처리 평가 255,001행은 다른 집단. 10분은 가정. 미래 운항 성능은 미검증.',False)
    header(a,'E','항공 데이터 연구 / 2026.09.22')
    a.text(59,235,'AIRPLANE',120,weight=900,spacing=-4)
    a.text(65,302,'항공편 지연 예측과 날씨 정보 확장',43,weight=700)
    a.text(65,361,'전처리의 타당성과 새로운 정보의 예측력을 나누어 검증했습니다.',28,MUTED)
    a.rect(65,405,1470,168,INK)
    a.text(98,453,'RESEARCH FINDING',20,PAPER,700,spacing=2)
    a.text(98,512,'전처리 수정만으로는 성능 향상 미확인.',36,PAPER,700)
    a.text(98,553,'날씨를 추가한 동일조건 비교에서는 세 지표 모두 개선.',31,PAPER)
    for i,row in enumerate(data['weather_model']):
        x=65+i*500
        if i:a.line(x-15,617,x-15,824,RULE,2)
        a.text(x,641,row['metric']+(' ↑' if row['higher_is_better'] else ' ↓'),26,weight=700)
        a.text(x-3,723,f"{row['delta_mean']:+.6f}",63,ACCENT,800)
        a.text(x,769,f"{row['off_mean']:.6f} → {row['on_mean']:.6f}",27)
        a.text(x,808,'변화 / 사용−미사용 · 3시드 평균',21,MUTED)
    a.line(65,851,1535,851,INK,2)
    a.text(65,896,'동일 라벨 180,332행 · 시드 42 / 1 / 7 · 14개 날씨 피처',27,weight=700)
    a.text(65,941,'정적 교차검증 · 10분 공개 지연은 가정 · 인과 효과와 미래 운항 성능은 미검증',24,MUTED)
    a.text(65,984,'Peter-jackson12 TF',21,MUTED)
    a.text(1535,984,'읽기 16분 + 전환·질문 3분 / 권장 배분',21,MUTED,anchor='end')
    a.save(OUT/'presentation_cover.svg')

    a=SVG(980,'두 평가 집단을 분리한 데이터 명세','원본 1,000,000행. 전처리 평가 255,001행. 날짜 귀속 채택 706,759행 중 180,332행에서 날씨 14개 피처를 비교. 같은 외부 폴드. 인과 효과와 미래 운항 성능은 미검증. 도형 크기는 수량을 뜻하지 않습니다.',False)
    header(a,'01','DATA / 평가 집단 명세')
    a.text(65,191,'같은 원본, 두 개의 평가 질문',57,weight=800)
    a.text(65,242,'전처리 평가와 날씨 비교의 성능 수치를 서로 이어 붙이지 않습니다.',28,MUTED)
    a.rect(65,287,1470,105,INK)
    a.text(96,354,'원본 데이터',30,PAPER,600);a.text(1505,359,'1,000,000행',57,PAPER,800,anchor='end')
    for i,(title,num,sub,question) in enumerate([
        ('A / 전처리 평가','255,001','라벨이 있는 행','전처리가 더 타당하면 성능도 좋아질까?'),
        ('B / 날씨 비교','180,332','날짜와 라벨이 모두 있는 행','새로운 날씨 정보는 예측을 개선할까?')]):
        x=65+i*750
        a.rect(x,425,720,361,SKY)
        a.rect(x,425,720,7,ACCENT if i else INK)
        a.text(x+30,484,title,30,weight=700)
        a.text(x+25,581,num,79,ACCENT if i else INK,800)
        a.text(x+30,633,sub,26,MUTED)
        a.line(x+30,663,x+690,663,RULE,2)
        a.text(x+30,715,question,27,weight=700)
        a.text(x+30,759,'P4 / P6 전처리 조건 비교' if not i else '날짜 귀속 채택 706,759행 중 라벨 보유',24,MUTED)
    a.text(65,842,'날씨 비교의 통제',25,ACCENT,700)
    a.text(340,842,'같은 180,332행 · 같은 시드 · 같은 외부 폴드',29,weight=700)
    a.text(340,887,'P6_clean 재학습 · 같은 내부 선택 절차 · 날씨 14개 피처 추가',26)
    a.line(65,918,1535,918,INK,2)
    a.text(65,958,'도형 크기는 수량을 뜻하지 않습니다.',22,MUTED)
    a.text(1535,958,'인과 효과와 미래 운항 성능은 미검증',22,MUTED,anchor='end')
    a.save(OUT/'analysis_journey.svg')

    a=SVG(960,'선택과 채점의 분리를 보이는 평가 명세','층화 5분할. 외부 학습 안에서 내부 학습 80%, 내부 검증 20%. 내부 학습 경계에서 TE 계산, 내부 검증에서 트리 수·임계값 선택. 외부 검증은 최종 채점 전용.',False)
    header(a,'02','METHOD / 평가의 경계')
    a.text(65,191,'모델 선택과 최종 채점은 분리합니다',55,weight=800)
    a.text(65,246,'층화 5분할 × 시드 42 / 1 / 7 · 각 조건에 같은 선택 절차 적용',28,MUTED)
    a.rect(65,290,920,54,INK);a.text(92,328,'외부 학습 데이터 안에서만 선택',27,PAPER,700)
    for y,no,title,l1,l2 in [(378,'1','내부 학습 / 80%','학습 경계에서 TE 계산','트리 수 후보별 모델 학습'),(570,'2','내부 검증 / 20%','LogLoss → 트리 수 선택','Macro F1 → 임계값 선택')]:
        a.rect(65,y,920,163,SKY)
        a.text(95,y+79,no,65,ACCENT,800)
        a.text(230,y+49,title,30,weight=700)
        a.text(230,y+99,l1,28);a.text(620,y+99,l2,25)
    a.line(1040,290,1040,734,ACCENT,4)
    a.rect(1095,290,440,54,ACCENT);a.text(1124,328,'최종 채점 전용',27,PAPER,700)
    a.text(1124,433,'3',65,ACCENT,800)
    a.text(1124,494,'외부 검증',39,weight=800)
    a.text(1124,551,'선택에 사용하지 않음',27)
    a.text(1124,605,'각 폴드의 예측 → OOF',25)
    a.text(1124,660,'OOF = 최종 평가',28,weight=700)
    a.line(65,783,1535,783,INK,2)
    a.text(65,835,'외부 검증으로 선택하지 않음',31,ACCENT,700)
    a.text(65,886,'각 행을 학습에 쓰지 않은 모델의 예측을 모아 평가합니다.',27)
    a.text(65,935,'라벨 누수 방지와 미래 시점의 입력 확보 가능성은 서로 다른 문제입니다.',25,MUTED)
    a.save(OUT/'evaluation_boundary.svg')
    print('Built three original E evidence-brief assets; no network, raw data or training.')
if __name__=='__main__':build()
