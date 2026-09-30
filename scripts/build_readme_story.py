"""Deterministic presentation artwork; stdlib only, no network or raw data.

Counts are asserted against the same tracked aggregates as build_readme_assets.
Conceptual diagrams do not encode quantities through node sizes or distances.
"""
from pathlib import Path
from html import escape
import json
try:
    from .build_readme_assets import reviewed_data
except ImportError:  # Direct script execution from repository root.
    from build_readme_assets import reviewed_data
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/readme'
NAVY, INK, TEAL, MUTED, LINE = '#102D40', '#183B4E', '#007E80', '#526878', '#CCDAE1'

class Art:
    def __init__(self, height, title, desc, dark=False):
        self.h = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img" aria-labelledby="title desc">', f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>', '<style>text{font-family:"Noto Sans CJK KR","Malgun Gothic","Apple SD Gothic Neo",sans-serif;font-variant-numeric:tabular-nums}</style>']
        self.rect(0,0,1000,height,NAVY if dark else '#F6F9FA',20)
    def rect(self,x,y,w,h,fill,r=12): self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"/>')
    def text(self,x,y,t,size=23,fill=INK,weight=400): self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}">{escape(t)}</text>')
    def path(self,d,stroke=LINE,width=2): self.parts.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>')
    def arrow(self,x,y): self.path(f'M{x-6} {y-7} L{x} {y} L{x+6} {y-7}',TEAL)
    def node(self,x,y,w,h,num,title,line1,line2='',dark=False):
        self.rect(x,y,w,h,NAVY if dark else '#FFFFFF')
        c='#FFFFFF' if dark else INK
        self.text(x+22,y+31,num,17,'#8EDDD1' if dark else TEAL,700)
        self.text(x+22,y+66,title,26,c,700)
        self.text(x+22,y+98,line1,20,'#D4E4EC' if dark else MUTED)
        if line2:self.text(x+22,y+127,line2,20,'#D4E4EC' if dark else MUTED)
    def save(self,name): (OUT/name).write_text('\n'.join(self.parts+['</svg>'])+'\n',encoding='utf-8')

def build():
    d=reviewed_data()
    assert d['attribution'][0]['rows']==706759
    weather=json.loads((ROOT/'output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_summary.json').read_text())
    assert weather['evaluation_rows']==180332 and weather['weather_feature_count']==14
    a=Art(410,'Airplane · 항공편 지연 예측과 날씨 정보 확장','전처리의 타당성과 새로운 정보의 예측력을 나누어 검증한 항공편 지연 이진 분류 프로젝트. 장식용 항로는 실제 운항 노선을 나타내지 않습니다.',True)
    # Original decorative trajectory, deliberately not a geographic map.
    a.path('M660 350 C800 335 925 260 904 167 C890 98 829 104 802 141 C757 204 853 262 933 201','#294E61',2)
    a.path('M708 304 C784 300 880 255 904 184','#8EDDD1',3)
    a.parts.append('<g transform="translate(905 172) rotate(18)"><path d="M0 -25 L6 -3 L28 10 L28 17 L5 10 L4 27 L13 32 L13 37 L0 33 L-13 37 L-13 32 L-4 27 L-5 10 L-28 17 L-28 10 L-6 -3 Z" fill="#8EDDD1"/></g>')
    a.text(48,55,'FLIGHT DELAY  /  EVIDENCE-LED MACHINE LEARNING',16,'#8EDDD1',700)
    a.text(44,151,'Airplane',78,'#FFFFFF',700)
    a.text(48,205,'항공편 지연 예측과 날씨 정보 확장',31,'#FFFFFF',500)
    a.text(48,255,'전처리의 타당성과 새로운 정보의 예측력을',23,'#C8DBE4')
    a.text(48,290,'나누어 검증했습니다.',23,'#C8DBE4')
    a.path('M48 330 L628 330','#35566A',1)
    a.text(48,366,'이진 분류',19,'#FFFFFF',500)
    a.text(195,366,'LightGBM',19,'#FFFFFF',500)
    a.text(354,366,'동일조건 교차검증',19,'#FFFFFF',500)
    a.save('presentation_cover.svg')
    a=Art(756,'분석의 두 경로가 같은 조건의 날씨 비교로 만나는 과정','원본 100만 행에서 전처리 수정과 날짜 귀속을 진행합니다. 기존 전처리 평가는 라벨 255001행, 날씨 비교는 날짜와 라벨이 모두 있는 동일 180332행을 사용합니다. 도형의 크기와 선 길이는 수량을 뜻하지 않습니다.')
    a.text(36,48,'두 가지 질문, 하나의 비교 원칙',32,INK,700)
    a.text(36,82,'기존 입력의 의미를 바로잡기 → 새로운 정보의 예측력 확인하기',20,MUTED)
    a.node(260,108,480,113,'START','원본 1,000,000행','라벨 255,001행 · 미라벨 744,999행',dark=True)
    a.path('M500 221 V239 H265 V257 M500 239 H735 V257');a.arrow(265,257);a.arrow(735,257)
    a.node(36,260,452,158,'01 / 전처리','무엇을 확실하게 아는가?','모호한 대치 중단 · 시각의 의미 수정','라벨 255,001행에서 전처리 조건 비교')
    a.node(512,260,452,158,'02 / 정보 확장','어떤 정보를 더할 수 있는가?','BTS 대조 → 날짜 귀속 706,759행','출발·도착 공항 날씨를 시점에 맞춰 결합')
    a.text(58,452,'전처리 수정만으로 Macro F1 향상 미확인',20,MUTED)
    a.text(534,452,'날짜와 라벨이 모두 있는 180,332행 선택',20,MUTED)
    a.path('M265 468 V491 H500 M735 468 V491 H500 V512');a.arrow(500,512)
    a.node(170,518,660,149,'03 / 동일조건 비교','P6_clean  +  날씨 14개 피처','같은 180,332행 · 같은 시드 · 같은 외부 폴드','트리 수·임계값은 같은 내부 절차로 각각 선택',True)
    a.text(170,709,'결론: 세 지표 모두 개선 · 인과 효과와 미래 운항 성능은 미검증',21,TEAL,700)
    a.save('analysis_journey.svg')
    a=Art(543,'모델 선택과 최종 평가의 경계','층화 5분할 외부 학습 데이터 안에서 다시 80대20으로 분리합니다. 내부 학습 경계에서 타깃 인코딩을 계산하고 내부 검증으로 트리 수와 임계값을 선택한 뒤 외부 검증을 최종 채점에만 사용합니다.')
    a.text(36,48,'모델을 고르는 데이터와 채점하는 데이터를 분리',30,INK,700)
    a.text(36,84,'층화 5분할 × 시드 42 / 1 / 7 · 각 조건에서 같은 선택 절차',21,MUTED)
    a.rect(36,112,636,310,'#E8F1F3');a.text(58,148,'외부 학습 데이터 안에서만 선택',23,TEAL,700)
    a.node(56,173,287,164,'내부 학습 / 80%','학습','학습 경계에서 TE 계산','트리 수 후보별 모델 학습')
    a.node(365,173,287,164,'내부 검증 / 20%','선택','LogLoss → 트리 수','Macro F1 → 임계값')
    a.path('M345 246 H360',TEAL,2)
    a.text(58,389,'선택한 모델 그대로 평가 · 외부 검증으로 선택하지 않음',20,MUTED)
    a.node(701,112,263,310,'최종 채점 전용','외부 검증','선택에 사용하지 않음','각 폴드의 예측 → OOF',True)
    a.path('M672 267 H696',TEAL,2)
    a.text(36,469,'OOF: 각 행을 학습에 쓰지 않은 모델의 예측을 모아 평가',23,INK,500)
    a.text(36,510,'라벨 누수 방지와 미래 시점의 입력 확보 가능성은 서로 다른 문제입니다.',21,MUTED)
    a.save('evaluation_boundary.svg')
    print('Built three static presentation assets; no network, raw data or model fitting.')

if __name__=='__main__': build()
