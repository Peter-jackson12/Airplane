"""Render README figures from tracked aggregate evidence only.

No raw data, HTTP, model fitting or collection entrypoint is used. The JSON
receipt records source hashes, exact filters, plotted rows and figure hashes.
Use --check to validate the receipt without modifying any file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from html import escape
from math import isclose
from collections import Counter
from pathlib import Path
try:
    from . import readme_editorial as editorial
except ImportError:
    import readme_editorial as editorial

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets/readme'
SOURCES = {
    'models': 'output/preprocessing_full_summary.csv',
    'calibration': 'output/baseline_recovery_v2_calibration_20260917_level_summary.csv',
    'attribution': 'output/baseline_recovery_v2_row_date_attribution_20260918_status_summary.csv',
    'latency': 'output/baseline_recovery_v2_weather_expanded_stratafix_20260918_latency_sensitivity.csv',
    'weather_model': 'output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_summary.json',
}
PHASES = ('P4', 'P4_clean', 'P6_fixed', 'P6_clean')
CALIBRATORS = ('none', 'platt', 'isotonic')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: str) -> list[dict]:
    with (ROOT / path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def reviewed_data() -> dict:
    models = {r['phase_key']: r for r in load_rows(SOURCES['models'])}
    model_rows = [
        {'phase': phase, 'mean': float(models[phase]['macro_f1_nested_mean']),
         'sd': float(models[phase]['macro_f1_nested_std'])}
        for phase in PHASES
    ]
    calibration = [r for r in load_rows(SOURCES['calibration'])
                   if (r['phase_key'], r['arm'], r['group']) == ('P6_clean', 'shared', 'overall')]
    if len(calibration) != 3 or {r['calibrator'] for r in calibration} != set(CALIBRATORS):
        raise ValueError('Expected exactly three P6_clean/shared/overall calibration rows')
    by_calibrator = {r['calibrator']: r for r in calibration}
    calibration_rows = []
    for key in CALIBRATORS:
        row = by_calibrator[key]
        if int(row['ece_10_count']) != 3 or float(row['n_mean']) != 255001:
            raise ValueError('Calibration population or seed count changed; review the caption')
        calibration_rows.append({'calibrator': key, 'ece_percentage_points': float(row['ece_10_mean']) * 100,
                                 'log_loss': float(row['log_loss_mean']),
                                 'macro_f1': float(row['macro_f1_mean'])})
    counts = Counter()
    for row in load_rows(SOURCES['attribution']):
        counts[row['status']] += int(row['rows'])
    total = sum(counts.values())
    accepted = counts['complete_single_candidate_year']
    missing_single = counts['missing_key_single_candidate_year']
    if total != 1000000 or accepted != 706759:
        raise ValueError('Attribution evidence changed; review the README population contract')
    attribution_rows = [
        {'category': 'Accepted: complete key, one candidate year', 'rows': accepted},
        {'category': 'Held: missing key, one candidate year', 'rows': missing_single},
        {'category': 'Held: all other inspected cases', 'rows': total - accepted - missing_single},
    ]
    latency_rows = []
    for row in load_rows(SOURCES['latency']):
        eligible, matched = int(row['eligible_rows']), int(row['matched_rows'])
        if eligible != 268 or not 0 <= matched <= eligible:
            raise ValueError('Stratafix denominator changed; review the README caption')
        rate = matched / eligible
        if abs(rate - float(row['match_rate_of_eligible'])) > 1e-12:
            raise ValueError('Latency source numerator/rate mismatch')
        latency_rows.append({'latency_minutes': int(row['latency_minutes']), 'role': row['role'],
                             'eligible_rows': eligible, 'matched_rows': matched, 'match_rate_percent': rate * 100})
    expected = {(role, latency) for role in ('origin', 'destination') for latency in (0, 10, 30, 60)}
    if len(latency_rows) != 8 or {(r['role'], r['latency_minutes']) for r in latency_rows} != expected:
        raise ValueError('Latency source has missing or duplicate scenarios')

    weather = json.loads((ROOT / SOURCES['weather_model']).read_text(encoding='utf-8'))
    if (weather.get('base_phase') != 'P6_clean'
            or weather.get('headline_latency_minutes') != 10
            or weather.get('headline_latency_is_measured') is not False
            or weather.get('seeds') != [42, 1, 7]
            or weather.get('weather_feature_count') != 14
            or weather.get('evaluation_rows') != 180332
            or weather.get('positive_rows') != 31805):
        raise ValueError('Weather-model submission contract changed; review the README')
    weather_rows = []
    # Preserve the signed on-minus-off contrast from the evidence. Do not
    # subtract rounded display values or reverse LogLoss to a positive gain.
    for key, label, higher in (('macro_f1_nested', 'Macro F1', True),
                               ('log_loss', 'LogLoss', False),
                               ('roc_auc', 'ROC-AUC', True)):
        off = weather['conditions']['weather_off'][key]
        on = weather['conditions']['weather_on'][key]
        delta = weather['paired_deltas_on_minus_off'][key]
        seed_deltas = delta['values_by_seed']
        if set(seed_deltas) != {'42', '1', '7'}:
            raise ValueError('Weather-model paired seeds changed')
        for seed in seed_deltas:
            if not isclose(on['values_by_seed'][seed] - off['values_by_seed'][seed],
                           seed_deltas[seed], abs_tol=1e-12):
                raise ValueError('Weather-model paired delta does not match its levels')
        if not isclose(on['mean'] - off['mean'], delta['mean'], abs_tol=1e-12):
            raise ValueError('Weather-model mean delta does not match its levels')
        if not all(v > 0 if higher else v < 0 for v in seed_deltas.values()):
            raise ValueError('Weather-model per-seed improvement direction changed')
        weather_rows.append({'metric': label, 'higher_is_better': higher,
                             'off_mean': off['mean'], 'on_mean': on['mean'],
                             'off_sd': off['std'], 'on_sd': on['std'],
                             'delta_mean': delta['mean'], 'sd': delta['std'],
                             'definition': 'weather_on - weather_off',
                             'deltas_by_seed': seed_deltas})
    return {'models': model_rows, 'calibration': calibration_rows,
            'attribution': attribution_rows, 'latency': latency_rows,
            'weather_model': weather_rows}


def expected_filters() -> dict:
    return {'models': list(PHASES),
            'calibration': {'phase_key': 'P6_clean', 'arm': 'shared', 'group': 'overall'},
            'attribution': 'sum rows over all 12 months; three disjoint status buckets',
            'latency': 'stratafix only; recompute matched_rows / eligible_rows for each role/scenario',
            'weather_model': 'P6_clean; 10-minute assumed latency; 180,332 identical labeled rows; 3 paired seeds'}


FIGURES = {'models': 'model_comparison.svg', 'calibration': 'calibration_tradeoff.svg',
           'attribution': 'date_attribution.svg', 'latency': 'weather_latency.svg',
           'weather_model': 'weather_model_comparison.svg'}
INK, MUTED, BLUE = editorial.INK, editorial.MUTED, editorial.ACCENT
CYAN, SKY = editorial.CYAN, editorial.SKY
GRID, PAPER = editorial.RULE, editorial.PAPER


class SVG:
    """Deterministic 1600-unit magazine figures, with honest plotting geometry."""

    def __init__(self, height: int, title: str, description: str, chart_type: str):
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="{height}" '
            f'viewBox="0 0 1600 {height}" role="img" aria-labelledby="title desc" '
            f'data-design="technical-launch-h" data-chart-type="{chart_type}">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            '<style>text{font-family:"Noto Sans CJK KR","Malgun Gothic",'
            '"Apple SD Gothic Neo",sans-serif;font-variant-numeric:tabular-nums}</style>',
            f'<rect width="1600" height="{height}" fill="{PAPER}"/>']

    def text(self, x, y, text, size=28, color=INK, weight=400, anchor='start', heading=False):
        size = max(editorial.TYPE['caption'], size)
        family = f' style="font-family:{editorial.HEADING}"' if heading else ''
        self.parts.append(f'<text x="{x:.3f}" y="{y:.3f}" font-size="{size}" '
                          f'fill="{color}" font-weight="{weight}" text-anchor="{anchor}"{family}>'
                          f'{escape(str(text))}</text>')

    def line(self, x1, y1, x2, y2, color=GRID, width=1, dash=None):
        dashed = f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" '
                          f'y2="{y2:.3f}" stroke="{color}" stroke-width="{width}"{dashed}/>')

    def rect(self, x, y, w, h, color):
        self.parts.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" '
                          f'height="{h:.3f}" fill="{color}"/>')

    def dot(self, x, y, color=BLUE, hollow=False, radius=8):
        fill = PAPER if hollow else color
        self.parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius}" '
                          f'fill="{fill}" stroke="{color}" stroke-width="3"/>')

    def polygon(self, points, color):
        values = ' '.join(f'{x:.3f},{y:.3f}' for x, y in points)
        self.parts.append(f'<polygon points="{values}" fill="{color}"/>')

    def running(self, number, topic):
        self.rect(80, 38, 22, 22, CYAN)
        self.text(121, 58, topic, 24, weight=600)
        self.text(1516, 58, f'AIRPLANE  /  {number}', 23, MUTED, anchor='end')
        self.line(80, 82, 1516, 82, INK, 2)

    def footer(self, topic):
        self.text(80, self.height-28, '출처: README의 연결된 원본 집계', 28, MUTED)
        self.text(1516, self.height-28, topic, 21, MUTED, anchor='end')

    def marker(self, x, y, kind, color):
        if kind == 'none':
            self.dot(x, y, color, hollow=True, radius=9)
        elif kind == 'platt':
            self.rect(x-9, y-9, 18, 18, color)
        else:
            self.polygon([(x,y-12),(x-11,y+8),(x+11,y+8)], color)

    def paired_axis(self, row, x, y, width, lo, hi, ticks):
        # Each metric has its own explicitly labelled numerical domain. Only
        # reviewed means are marks; the connector is the signed on-minus-off move.
        xx = lambda value: x + (value-lo)/(hi-lo)*width
        self.line(x, y+34, x+width, y+34, MUTED, 1.5)
        for tick in ticks:
            tx = xx(tick)
            self.line(tx, y+27, tx, y+42, MUTED, 1.5)
            self.text(tx, y+70, f'{tick:.3f}', 23, MUTED, anchor='middle')
        off, on = xx(row['off_mean']), xx(row['on_mean'])
        self.line(off, y, on, y, BLUE, 4)
        middle, direction = (off+on)/2, 1 if on > off else -1
        self.polygon([(middle+direction*10,y), (middle-direction*6,y-7),
                      (middle-direction*6,y+7)], BLUE)
        self.dot(off, y, MUTED, hollow=True, radius=10)
        self.dot(on, y, BLUE, radius=10)

    def save(self, path):
        path.write_text('\n'.join(self.parts + ['</svg>']) + '\n', encoding='utf-8')


def check_receipt() -> dict:
    receipt = json.loads((ASSETS / 'sources.json').read_text(encoding='utf-8'))
    if receipt['schema_version'] != 1:
        raise ValueError('Unsupported figure receipt schema')
    if receipt['generator_sha256'] != sha256(Path(__file__)):
        raise ValueError('Figure builder changed; regenerate the README assets')
    if receipt.get('editorial_renderer_sha256') != sha256(Path(editorial.__file__)):
        raise ValueError('Editorial renderer changed; regenerate the README assets')
    if receipt['reviewed_data'] != reviewed_data():
        raise ValueError('Plotted rows differ from tracked aggregate evidence')
    if receipt['filters'] != expected_filters():
        raise ValueError('Figure source filters changed')
    if {key: row['path'] for key, row in receipt['sources'].items()} != SOURCES:
        raise ValueError('Figure source set changed')
    expected_paths = {key: f'assets/readme/{name}' for key, name in FIGURES.items()}
    if {key: row['path'] for key, row in receipt['figures'].items()} != expected_paths:
        raise ValueError('Figure set changed')
    for entry in receipt['sources'].values():
        if sha256(ROOT / entry['path']) != entry['sha256']:
            raise ValueError(f"Source hash drift: {entry['path']}")
    for entry in receipt['figures'].values():
        if sha256(ROOT / entry['path']) != entry['sha256']:
            raise ValueError(f"Figure hash drift: {entry['path']}")
    return receipt


def build() -> None:
    data=reviewed_data(); ASSETS.mkdir(parents=True,exist_ok=True); figures={}
    def finish(svg,key):
        path=ASSETS/FIGURES[key];svg.save(path)
        figures[key]={'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path)}
    def head(svg,title,subtitle):
        svg.text(800,105,title,58,weight=500,anchor='middle')
        svg.text(800,171,subtitle,30,MUTED,anchor='middle')
    # One complete 100% population; no enlarged sliver.
    a=SVG(900,'백만 행 중 날짜를 확인한 집단','서로 겹치지 않는 세 집단. 채택 706,759행, 결측 키 단일 후보 291,308행과 기타 1,933행은 보류.','horizontal-population-stack')
    head(a,'백만 행 중 날짜를 확인한 집단','원본 1,000,000행 전체를 대조 · 미검사 집단 없음')
    a.text(120,269,'날짜 귀속 채택',32,BLUE,600);a.text(1480,269,'706,759행 · 70.68%',42,BLUE,600,'end')
    x=120
    for row,color in zip(data['attribution'],(BLUE,'#DADCE0',INK)):
        width=1360*row['rows']/1000000;a.rect(x,310,width,90,color);x+=width
    for i,(row,label) in enumerate(zip(data['attribution'],('채택 · 완전한 키 + 단일 후보 연도','보류 · 결측 키 + 단일 후보 연도','보류 · 그 밖의 대조 결과'))):
        y=492+i*98;a.text(120,y,label,32);a.text(1480,y,f"{row['rows']:,}행",38,BLUE if i==0 else INK,600,'end');a.line(120,y+34,1480,y+34)
    a.text(120,815,'후보 연도가 하나여도 대조 키가 결측이면 채택하지 않습니다.',30,MUTED)
    a.footer('날짜 귀속');finish(a,'attribution')
    # Direct benchmark table with a blue highlighted focus column.
    a=SVG(1070,'전처리 타당성과 성능 변화는 다릅니다','동일 라벨 255,001행. 네 기준 조건의 Macro F1 평균과 SD. P6_clean 강조는 우승 표기가 아닙니다.','highlighted-benchmark-table')
    head(a,'전처리 타당성과 성능 변화는 다릅니다','동일 라벨 255,001행 · 전체 10조건 중 네 기준 조건 · 3시드 평균')
    a.parts.append('<rect x="1190" y="256" width="290" height="446" fill="#E8F0FE" data-focus-column="P6_clean"/>')
    for x,row in zip((570,825,1080,1335),data['models']):
        a.text(x,320,row['phase'],31,BLUE if row['phase']=='P6_clean' else INK,600,'middle')
        a.text(x,454,f"{row['mean']:.6f}",35,weight=600,anchor='middle')
        a.text(x,610,f"{row['sd']:.6f}",32,MUTED,anchor='middle')
    a.text(120,454,'Macro F1 ↑',34,weight=600);a.text(120,610,'표본 SD',32,MUTED)
    for y in (366,524,700):a.line(120,y,1480,y,GRID,2)
    a.text(120,798,'전처리는 더 타당해졌지만',42,weight=500)
    a.text(120,862,'Macro F1 향상은 확인되지 않았습니다.',42,weight=500)
    a.text(120,952,'SD는 신뢰구간이 아닙니다. 강조 열은 현재 기준 조건을 뜻합니다.',28,MUTED)
    a.footer('전처리 비교');finish(a,'models')
    # Three independent zero-start panels. Bars use actual means, never deltas.
    a=SVG(1710,'날씨 추가 후 세 지표 모두 개선','180,332행에서 동일 평가행·시드·외부 폴드. 지표별 0 기준 축. 각 지표의 척도가 달라 지표 간 막대 길이로 개선 크기를 비교하지 않습니다.','zero-baseline-weather-bars')
    head(a,'날씨 정보가 더해진 뒤, 세 지표 모두 개선','동일 라벨 180,332행 · P6_clean · 10분 공개 지연 가정 · 3시드 평균')
    for i,row in enumerate(data['weather_model']):
        top=252+i*436;bottom=top+252;axis_x=560;length=680;hi=1.0
        a.rect(80,top-30,1440,396,'#F8F9FA')
        a.text(120,top+44,row['metric'],42,weight=600)
        a.text(120,top+96,'높을수록 좋음' if row['higher_is_better'] else '낮을수록 좋음',28,MUTED)
        a.text(120,top+182,f"{row['delta_mean']:+.6f}",44,BLUE,600)
        a.text(120,top+230,'변화 · 사용−미사용',28,MUTED)
        a.text(120,top+298,f"차이 SD {row['sd']:.6f}",28,MUTED)
        for tick in (0,.25,.5,.75,1):
            x=axis_x+tick*length;a.line(x,top+22,x,bottom,GRID,1);a.text(x,bottom+42,f'{tick:g}',28,MUTED,anchor='middle')
        for j,(cond,key,label,color) in enumerate((('weather_off','off_mean','날씨 미사용','#DADCE0'),('weather_on','on_mean','날씨 사용',BLUE))):
            y=top+64+j*108;value=row[key];width=value/hi*length
            a.text(530,y+35,label,28,MUTED,anchor='end')
            a.parts.append(f'<rect x="{axis_x}" y="{y}" width="{width:.9f}" height="54" fill="{color}" data-metric="{row["metric"]}" data-condition="{cond}" data-value="{value}" data-axis-min="0" data-axis-max="{hi}" data-axis-length="{length}" data-baseline="{axis_x}" data-orientation="horizontal"/>')
            a.text(1280,y+39,f'{value:.6f}',32,BLUE if j else INK,600)
        a.text(1240,top+337,'0부터 시작하는 원래 값의 축',28,MUTED,anchor='end')
    a.text(120,1590,'지표별 척도가 달라 지표 간 막대 길이로 개선 크기를 비교하지 않습니다.',28,MUTED)
    a.text(120,1642,'SD는 신뢰구간이 아닙니다. 인과 효과와 미래 운항 성능은 미검증입니다.',28,MUTED)
    a.footer('날씨 유무 비교');finish(a,'weather_model')
    # Readable table instead of decorative/rank-confounding geometry.
    a=SVG(1000,'확률 보정은 지연 탐지 개선과 같지 않습니다','P6_clean 공유형 전체 라벨 255,001행. ECE는 비율을 100배 한 %p. 3시드 평균. ECE와 LogLoss는 낮을수록 좋습니다.','calibration-tradeoff-scorecard')
    head(a,'확률의 정확성과 지연 탐지 능력은 달랐습니다','P6_clean · 공유형 · 전체 라벨 255,001행 · 3시드 평균')
    a.rect(80,250,1440,100,SKY)
    for x,label in ((120,'보정 방법'),(760,'ECE (%p) ↓'),(1130,'LogLoss ↓'),(1480,'Macro F1 ↑')):
        a.text(x,310,label,32,BLUE,600,'start' if x==120 else 'end')
    for i,(row,label) in enumerate(zip(data['calibration'],('보정 없음','Platt','Isotonic'))):
        y=444+i*134;a.text(120,y,label,36,weight=600)
        a.text(760,y,f"{row['ece_percentage_points']:.4f}",38,anchor='end')
        a.text(1130,y,f"{row['log_loss']:.6f}",38,anchor='end')
        a.text(1480,y,f"{row['macro_f1']:.6f}",38,anchor='end');a.line(120,y+48,1480,y+48)
    a.text(120,844,'ECE와 LogLoss는 낮을수록 좋음 · Macro F1은 높을수록 좋음',30,MUTED)
    a.text(120,910,'보정이 일부 확률 지표를 바꿨지만 Macro F1 향상은 확인되지 않았습니다.',30,MUTED)
    a.footer('확률 보정');finish(a,'calibration')
    # True minute positions and no interpolation between unmeasured scenarios.
    a=SVG(1160,'날씨는 있어도 제때 공개되어야 합니다','층화 보정 표본 300행 중 수집 가능 268행. 0/10/30/60분은 공개 지연 가정. 결합률이지 모델 성능이 아닙니다.','discrete-latency-dot-panels')
    head(a,'날씨는 있어도, 제때 공개되어야 합니다','층화 보정 표본 300행 중 수집 가능 268행 · 관측 나이 상한 90분')
    for role,label,px,color in (('origin','출발 공항',190,BLUE),('destination','도착 공항',960,INK)):
        width=440;top=310;bottom=620
        a.text(px+width/2,254,label,38,weight=600,anchor='middle')
        for tick in (0,25,50,75,100):
            y=bottom-tick/100*(bottom-top);a.line(px,y,px+width,y);a.text(px-22,y+9,f'{tick}%',28,MUTED,anchor='end')
        for t in (0,10,30,60):
            row=next(r for r in data['latency'] if (r['role'],r['latency_minutes'])==(role,t));x=px+t/60*width;y=bottom-row['match_rate_percent']/100*(bottom-top)
            a.line(x,bottom,x,y,color,3);a.dot(x,y,color,hollow=role=='destination',radius=10);a.text(x,bottom+45,str(t),28,MUTED,anchor='middle')
        a.text(px+width/2,714,'공개 지연 가정 (분)',28,MUTED,anchor='middle')
    a.rect(80,760,1440,244,SKY)
    a.text(120,813,'결합 / 268행',28,weight=600)
    for t,x in zip((0,10,30,60),(500,800,1100,1400)):
        a.text(x,813,f'{t}분',30,weight=600,anchor='middle')
        for i,role in enumerate(('origin','destination')):
            row=next(r for r in data['latency'] if (r['role'],r['latency_minutes'])==(role,t));a.text(x,880+i*65,f"{row['matched_rows']}/268 · {row['match_rate_percent']:.2f}%",28,anchor='middle')
    a.text(120,880,'출발',30);a.text(120,945,'도착',30)
    a.text(120,1066,'실측 지연값이 아닌 가정입니다. 전체 집단 결합률이나 모델 성능이 아닙니다.',28,MUTED)
    a.footer('공개 시점');finish(a,'latency')
    receipt={'schema_version':1,'generator_sha256':sha256(Path(__file__)),
        'editorial_renderer_sha256':sha256(Path(editorial.__file__)),
        'sources':{k:{'path':v,'sha256':sha256(ROOT/v)} for k,v in SOURCES.items()},
        'filters':expected_filters(),'reviewed_data':data,'figures':figures,
        'limitations':['No raw data access, network calls or model retraining.',
        'No live collection progress is inferred.',
        'Korean text uses system fonts; no external assets.',
        'Means and signed paired deltas rounded independently.',
        'Three-seed SD is not a confidence interval; weather panels use zero-start axes.',
        'Latency scenarios are discrete marks on proportional minute axes; no interpolation.',
        'Attribution stack includes the full population without inflating small segments.']}
    (ASSETS/'sources.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Built five technical-announcement H figures and source receipt.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Read-only source, plotted-value and SVG hash validation')
    args = parser.parse_args()
    if args.check:
        check_receipt()
        print('README figure sources, filters, plotted values and file hashes: PASS')
    else:
        build()


if __name__ == '__main__':
    main()
