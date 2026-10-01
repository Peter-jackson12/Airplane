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
            f'data-design="bonobono-parody-g" data-chart-type="{chart_type}">',
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
        path.write_text(editorial.parody_frame('\n'.join(self.parts + ['</svg>']) + '\n', path.stem), encoding='utf-8')


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
    data = reviewed_data()
    ASSETS.mkdir(parents=True, exist_ok=True)
    figures = {}

    def finish(svg, key):
        path = ASSETS / FIGURES[key]
        svg.save(path)
        figures[key] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha256(path)}

    # A vertical 100% stack: segment heights use the complete 1,000,000-row
    # denominator, including the 1,933-row sliver without a minimum-height lie.
    svg = SVG(1060, '날짜를 확인한 706,759행만 채택',
              '원본 1,000,000행을 모두 대조했습니다. 세로 누적 막대의 세 집단은 서로 겹치지 않습니다. '
              '채택 706,759행, 결측 키의 단일 후보 연도 291,308행과 그 밖의 1,933행은 보류합니다.',
              'vertical-population-stack')
    svg.rect(1072, 0, 528, 1060, SKY)
    svg.rect(1056, 110, 16, 836, CYAN)
    svg.running('01', '날짜 귀속 · 원본 전체 대조')
    svg.text(80, 170, '확인한 날짜만,', 66, weight=700, heading=True)
    svg.text(80, 254, '다음 단계로', 66, weight=700, heading=True)
    svg.text(70, 420, f"{data['attribution'][0]['rows']:,}", 154, BLUE, 700, heading=True)
    svg.text(84, 466, '채택한 행 / 원본 1,000,000행', 29, MUTED)
    labels = (('채택', '완전한 대조 정보 + 단일 후보 연도'),
              ('보류', '대조 정보 결측 + 단일 후보 연도'),
              ('보류', '그 밖의 대조 결과'))
    for i, (row, labels_i, color) in enumerate(zip(data['attribution'], labels, (BLUE, INK, MUTED))):
        y = 550 + i*123
        svg.line(84, y-29, 1000, y-29, GRID, 1.5)
        svg.text(84, y+6, f'{i+1:02d}', 23, color, 700)
        svg.text(150, y+6, labels_i[0], 27, color, 700)
        svg.text(250, y+6, labels_i[1], 26)
        svg.text(1000, y+63, f"{row['rows']:,}행", 42, color, 600, 'end')
    svg.text(1130, 179, '70.68%', 78, BLUE, 700, heading=True)
    svg.text(1134, 225, '원본 중 날짜 귀속 채택', 25, MUTED)
    bar_x, bar_y, bar_w, bar_h = 1150, 326, 172, 530
    bottom = bar_y+bar_h
    for row, color in zip(data['attribution'], (BLUE, CYAN, INK)):
        height = bar_h*row['rows']/1000000
        bottom -= height
        svg.rect(bar_x, bottom, bar_w, height, color)
    for tick in (0,25,50,75,100):
        y = bar_y+bar_h*(1-tick/100)
        svg.line(1340, y, 1360, y, MUTED, 1.5)
        svg.text(1380, y+8, f'{tick}%', 24, MUTED)
    svg.text(1236, 659, '채택', 30, PAPER, 600, 'middle')
    svg.text(1236, 701, '70.68%', 27, PAPER, 500, 'middle')
    svg.text(1236, 394, '보류', 28, INK, 600, 'middle')
    svg.text(1236, 435, '29.13%', 25, INK, 500, 'middle')
    svg.line(1236, bar_y, 1236, 294, INK, 1.5)
    svg.text(1236, 281, '기타 0.19%', 23, INK, anchor='middle')
    svg.text(1134, 906, '세 집단 합계 100%', 26, weight=600)
    svg.text(84, 948, '보류 293,241행도 모두 대조했습니다. 미검사 집단이 아닙니다.', 27, MUTED)
    svg.text(84, 991, '대조 정보가 빠져 있다면 후보 연도가 하나여도 채택하지 않습니다.', 26, MUTED)
    svg.footer('날짜 귀속 / 01')
    finish(svg, 'attribution')

    # Change from the old horizontal forest plot to vertical mean/SD ranges.
    # The plotting domain is shared across all four conditions and labelled.
    svg = SVG(1110, '전처리는 타당해졌지만 Macro F1 향상은 확인되지 않음',
              '동일 라벨 255,001행. 전체 10조건 중 네 기준 조건의 3시드 평균과 ±1 표본 SD. '
              '세로축은 0.5725–0.5785의 확대 축이며 신뢰구간이 아닙니다.', 'vertical-mean-sd-ranges')
    svg.rect(0, 811, 1600, 199, SKY)
    svg.rect(1320, 112, 196, 174, CYAN)
    svg.running('02', '전처리 비교 · 의미와 성능의 분리')
    svg.text(80, 173, '전처리는 더 타당하게', 64, weight=700, heading=True)
    svg.text(80, 254, '성능 향상은 미확인', 64, weight=700, heading=True)
    svg.text(1418, 227, '04', 111, INK, 700, 'middle', heading=True)
    svg.text(1418, 267, '기준 조건', 23, INK, anchor='middle')
    svg.text(84, 310, '라벨 255,001행 · 동일 평가 절차 · 시드 42 / 1 / 7', 28, MUTED)
    svg.text(84, 352, '전체 10조건 중 네 기준 조건 · 점: 평균 / 오차막대: ±1 표본 SD', 25, MUTED)
    lo, hi = 0.5725, 0.5785
    yy = lambda v: 755 - (v-lo)/(hi-lo)*370
    xs = (465, 735, 1005, 1275)
    for x in (xs[1], xs[3]):
        svg.rect(x-91, 385, 182, 370, SKY)
    svg.text(84, 405, 'Macro F1', 29, weight=600)
    svg.text(84, 447, '↑ 높을수록 좋음', 23, MUTED)
    svg.text(84, 535, '확대 축', 28, BLUE, 600)
    svg.text(84, 578, '0.5725', 24, MUTED)
    svg.text(84, 614, '– 0.5785', 24, MUTED)
    for tick in (0.573,0.574,0.575,0.576,0.577,0.578):
        y = yy(tick)
        svg.line(341, y, 1492, y, GRID, 1)
        svg.text(323, y+8, f'{tick:.3f}', 23, MUTED, anchor='end')
    svg.line(341, 385, 341, 755, MUTED, 1.5)
    svg.line(341, 755, 1492, 755, MUTED, 1.5)
    for x, row in zip(xs, data['models']):
        clean = row['phase'].endswith('clean')
        color = BLUE if clean else INK
        top, bottom, center = yy(row['mean']+row['sd']), yy(row['mean']-row['sd']), yy(row['mean'])
        svg.line(x, top, x, bottom, color, 3)
        svg.line(x-14, top, x+14, top, color, 3)
        svg.line(x-14, bottom, x+14, bottom, color, 3)
        svg.dot(x, center, color, hollow=not clean, radius=10)
        svg.text(x, 798, row['phase'], 28, color, 600, 'middle')
        svg.text(x, 896, f"{row['mean']:.6f}", 35, color, 600, 'middle')
        svg.text(x, 944, f"± {row['sd']:.6f}", 26, MUTED, anchor='middle')
    svg.text(84, 863, '3시드 평균', 24, weight=600)
    svg.text(84, 912, '±1 표본 SD', 24, MUTED)
    svg.line(84, 1009, 1516, 1009, INK, 1.5)
    svg.text(84, 1056, 'SD는 신뢰구간이 아닙니다. 오차막대 겹침으로 동등성·유의성을 판정하지 않습니다.', 25, MUTED)
    svg.footer('전처리 비교 / 02')
    finish(svg, 'models')

    # One deliberately asymmetric lead result and two distinct metric strips.
    # The three independent quantitative axes may not be compared as magnitudes.
    svg = SVG(1380, '동일조건 비교에서 세 지표 모두 개선',
              'P6_clean, 동일 라벨 180,332행, 시드 42/1/7. Macro F1 0.573910에서 0.598945, '
              'LogLoss 0.448116에서 0.436689, ROC-AUC 0.640618에서 0.672936. '
              '변화량은 사용−미사용이며 원본 집계에서 따로 반올림했습니다. 각 지표는 독립 확대 축입니다. '
              '10분 공개 지연 가정. 지표마다 척도가 달라 직접 크기를 비교하지 않습니다. SD는 신뢰구간이 아닙니다. 인과 효과와 미래 운항 성능은 미검증입니다.', 'asymmetric-paired-metric-strips')
    svg.rect(0, 334, 1035, 435, SKY)
    svg.rect(1080, 334, 520, 435, INK)
    svg.rect(1059, 334, 12, 435, CYAN)
    svg.running('03', '날씨 정보 · 동일조건 비교')
    svg.text(80, 166, '같은 180,332행에서', 64, weight=700, heading=True)
    svg.text(80, 248, '날씨 추가 후 세 지표 개선', 64, weight=700, heading=True)
    svg.text(84, 303, 'P6_clean · 동일 라벨 180,332행 · 시드 42 / 1 / 7', 28, MUTED)
    first = data['weather_model'][0]
    svg.text(84, 395, 'Macro F1 ↑ 높을수록 좋음', 31, BLUE, 600)
    svg.text(73, 526, f"{first['delta_mean']:+.6f}", 129, BLUE, 700, heading=True)
    svg.text(86, 574, '변화 (사용−미사용) / 3시드 평균', 26, MUTED)
    svg.text(930, 605, '○ 미사용 · ● 사용', 22, BLUE, anchor='end')
    svg.paired_axis(first, 110, 635, 820, 0.565, 0.605, (0.565,0.575,0.585,0.595,0.605))
    svg.text(84, 747, 'Macro F1 · 확대 축', 23, MUTED)
    svg.text(1130, 394, '날씨 미사용', 27, SKY)
    svg.text(1130, 461, f"{first['off_mean']:.6f}", 58, PAPER, 500, heading=True)
    svg.line(1130, 497, 1516, 497, CYAN, 2)
    svg.text(1130, 549, '날씨 사용', 27, SKY)
    svg.text(1130, 616, f"{first['on_mean']:.6f}", 58, PAPER, 700, heading=True)
    svg.text(1130, 706, f"차이의 SD {first['sd']:.6f}", 25, SKY)
    svg.line(799, 813, 799, 1166, GRID, 1.5)
    for row, x, lo, hi, ticks in (
            (data['weather_model'][1], 84, 0.430, 0.455, (0.430,0.440,0.450,0.455)),
            (data['weather_model'][2], 865, 0.630, 0.680, (0.630,0.650,0.670,0.680))):
        guide = '높을수록 좋음 ↑' if row['higher_is_better'] else '낮을수록 좋음 ↓'
        svg.text(x, 834, row['metric'], 35, weight=600)
        svg.text(x+650, 833, guide, 24, MUTED, anchor='end')
        svg.text(x-4, 923, f"{row['delta_mean']:+.6f}", 76, BLUE, 700, heading=True)
        svg.text(x, 967, f"변화 (사용−미사용) · 차이의 SD {row['sd']:.6f}", 23, MUTED)
        svg.text(x, 1017, f"{row['off_mean']:.6f} → {row['on_mean']:.6f}", 32, weight=500)
        svg.text(x, 1055, '미사용 → 사용', 28, MUTED)
        svg.paired_axis(row, x+30, 1081, 580, lo, hi, ticks)
        svg.text(x, 1190, f"{row['metric']} · 독립 확대 축", 23, MUTED)
    svg.line(84, 1220, 1516, 1220, INK, 2)
    svg.text(84, 1264, '지표별 독립 확대 축 · 변화량 크기의 지표 간 비교 불가 · 각각 반올림', 28, MUTED)
    svg.text(84, 1305, '10분은 공개 지연 가정 · SD ≠ 신뢰구간 · 인과 효과·미래 운항 성능 미검증', 28, MUTED)
    svg.text(84, 1346, '전처리 255,001행과 다른 집단 · 세 시드 모두 각 지표의 개선 방향 일치', 28, MUTED)
    finish(svg, 'weather_model')

    # Two aligned point panels replace the old two-dimensional scatter. Every
    # marker remains keyed to a calibrator; metric domains/units are independent.
    svg = SVG(1210, '확률 보정은 지표마다 다른 방향',
              'P6_clean 공유형 보정기, 라벨 255,001행의 3시드 평균. '
              'ECE는 %p, LogLoss는 독립 확대 축이며 모두 낮을수록 좋습니다. '
              'Isotonic은 ECE가 낮아졌지만 LogLoss는 높아졌고 Macro F1 향상은 확인되지 않았습니다.',
              'aligned-independent-calibration-panels')
    svg.rect(0, 106, 585, 200, SKY)
    svg.rect(1129, 106, 387, 200, CYAN)
    svg.running('04', '확률 보정 · 지표의 상충 관계')
    svg.text(80, 174, '확률 보정,', 64, weight=700, heading=True)
    svg.text(80, 259, '지표마다 다른 방향', 64, weight=700, heading=True)
    svg.text(1323, 179, 'ECE ↓', 50, INK, 700, 'middle')
    svg.text(1323, 237, 'LogLoss ↑', 44, INK, 600, 'middle')
    svg.text(1323, 281, 'Isotonic / 보정 없음 대비', 21, INK, anchor='middle')
    svg.text(84, 346, 'P6_clean · 공유형 보정기 · 라벨 255,001행 · 3시드 평균', 28, MUTED)
    panels = ((395, 490, 'ECE (%p) ↓', 0.0, 0.5, (0,.1,.2,.3,.4,.5), 'ece_percentage_points'),
              (1070, 430, 'LogLoss ↓ / 확대 축', .4472, .4493, (.4475,.4480,.4485,.4490), 'log_loss'))
    for px, width, label, lo, hi, ticks, key in panels:
        svg.text(px, 414, label, 29, weight=600)
        xx = lambda value: px+(value-lo)/(hi-lo)*width
        for tick in ticks:
            x = xx(tick)
            svg.line(x, 448, x, 694, GRID, 1)
            svg.text(x, 738, f'{tick:.1f}' if key == 'ece_percentage_points' else f'{tick:.4f}', 23, MUTED, anchor='middle')
        svg.line(px, 694, px+width, 694, MUTED, 1.5)
        for i, (row, color) in enumerate(zip(data['calibration'], (INK, BLUE, INK))):
            y = 469+i*95
            svg.marker(xx(row[key]), y, row['calibrator'], color)
    svg.line(980, 394, 980, 751, CYAN, 3)
    for i, label in enumerate(('보정 없음', 'Platt', 'Isotonic')):
        y=469+i*95
        svg.text(84, y+8, label, 31, weight=600)
        svg.line(84, y+37, 290, y+37, GRID, 1)
    svg.text(84, 795, '두 지표 모두 낮을수록 좋음 · 각 축의 단위와 범위를 따로 읽습니다.', 25, MUTED)
    svg.rect(0, 837, 1600, 258, SKY)
    svg.text(84, 886, '3시드 평균', 24, weight=600)
    for x, label in ((770,'ECE (%p) ↓'),(1125,'LogLoss ↓'),(1516,'Macro F1 ↑')):
        svg.text(x, 886, label, 25, MUTED, anchor='end')
    svg.line(84, 908, 1516, 908, INK, 1.5)
    for i,(row,label) in enumerate(zip(data['calibration'],('보정 없음','Platt','Isotonic'))):
        y=954+i*58
        svg.text(84, y, label, 27, weight=500)
        svg.text(770, y, f"{row['ece_percentage_points']:.4f}", 30, BLUE, 600, 'end')
        svg.text(1125, y, f"{row['log_loss']:.6f}", 30, INK, 600, 'end')
        svg.text(1516, y, f"{row['macro_f1']:.6f}", 30, INK, 600, 'end')
    svg.text(84, 1138, 'ECE는 원본 비율을 100배 한 %p 단위입니다. Macro F1 향상은 확인되지 않았습니다.', 25, MUTED)
    svg.footer('확률 보정 / 04')
    finish(svg, 'calibration')

    # The scenarios are discrete lollipops on true, proportional minute axes.
    # No connecting line invents values or a continuous response between them.
    svg = SVG(1310, '공개 지연 가정별 날씨 결합률',
              '층화 보정 표본 300행 중 수집 가능 268행. 보류 32행은 분모에서 제외했습니다. '
              '출발과 도착의 두 패널은 0–100% 공통 세로축과 실제 간격의 0/10/30/60분 가로축입니다. '
              '모두 미검증 가정이며 결합률이지 모델 성능이 아닙니다.', 'latency-small-multiple-lollipops')
    svg.rect(0, 111, 546, 247, SKY)
    svg.rect(547, 111, 17, 247, CYAN)
    svg.running('05', '날씨 결합 · 공개 시점의 제약')
    svg.text(70, 282, '268', 170, BLUE, 700, heading=True)
    svg.text(84, 332, '수집 가능한 행 / 공통 분모', 29, MUTED)
    svg.text(637, 207, '있어도, 제때', 66, weight=700, heading=True)
    svg.text(637, 292, '공개되어야 한다', 66, weight=700, heading=True)
    svg.text(84, 408, '층화 보정 표본 300행 − 보류 32행 · 관측 나이 상한 90분', 28, MUTED)
    for role, label, px, color in (('origin','출발 공항',155,BLUE),('destination','도착 공항',932,INK)):
        width, top, bottom = 544, 511, 815
        svg.text(px-2, 475, label, 35, color, 700)
        xx = lambda value: px+value/60*width
        yy = lambda value: bottom-value/100*(bottom-top)
        for tick in (0,25,50,75,100):
            y=yy(tick)
            svg.line(px, y, px+width, y, GRID, 1)
            svg.text(px-22, y+8, f'{tick}%', 23, MUTED, anchor='end')
        svg.line(px, top, px, bottom, MUTED, 1.5)
        svg.line(px, bottom, px+width, bottom, MUTED, 1.5)
        for latency in (0,10,30,60):
            row=next(r for r in data['latency'] if (r['role'],r['latency_minutes']) == (role,latency))
            x,y=xx(latency),yy(row['match_rate_percent'])
            svg.line(x, bottom, x, y, color, 3)
            svg.dot(x, y, color, hollow=(role=='destination'), radius=9)
            svg.text(x, bottom+43, str(latency), 25, MUTED, anchor='middle')
        svg.text(px+width/2, bottom+89, '공개 지연 가정 (분)', 26, MUTED, anchor='middle')
    svg.line(797, 457, 797, 916, CYAN, 2)
    svg.rect(0, 948, 1600, 215, SKY)
    svg.text(84, 993, '결합 / 268행', 26, weight=600)
    for latency,x in zip((0,10,30,60),(524,817,1110,1403)):
        svg.text(x, 993, f'{latency}분 가정', 27, weight=600, anchor='middle')
        for i,(role,label) in enumerate((('origin','출발'),('destination','도착'))):
            row=next(r for r in data['latency'] if (r['role'],r['latency_minutes']) == (role,latency))
            svg.text(x, 1056+i*62, f"{row['matched_rows']}/268 · {row['match_rate_percent']:.2f}%", 26, BLUE if role=='origin' else INK, 500, 'middle')
    svg.line(84, 1012, 1516, 1012, INK, 1.5)
    svg.text(84, 1056, '출발 공항', 27, BLUE, 600)
    svg.text(84, 1118, '도착 공항', 27, INK, 600)
    svg.text(84, 1208, '0 / 10 / 30 / 60분은 실측 공개 지연 시간이 아니라 가정입니다.', 27, MUTED)
    svg.text(84, 1252, '300행 전체나 날짜 귀속 전체의 결합률이 아닙니다. 모델 성능과도 구별합니다.', 25, MUTED)
    svg.footer('날씨 결합 / 05')
    finish(svg, 'latency')

    receipt = {'schema_version': 1, 'generator_sha256': sha256(Path(__file__)),
               'editorial_renderer_sha256': sha256(Path(editorial.__file__)),
               'sources': {key: {'path': path, 'sha256': sha256(ROOT / path)} for key, path in SOURCES.items()},
               'filters': expected_filters(), 'reviewed_data': data, 'figures': figures,
               'limitations': ['No raw data access, network calls or model retraining.',
                               'No live collection progress is inferred.',
                               'Korean SVG text uses system font fallbacks; no font files or external assets are embedded.',
                               'Weather levels and signed paired deltas are rounded independently from tracked evidence.',
                               'Three-seed SD is not a confidence interval; each weather metric uses a labelled independent zoomed axis.',
                               'Latency scenarios are discrete marks on proportional minute axes; no interpolation is plotted.',
                               'The attribution stack uses the complete population without inflating small segments.']}
    (ASSETS / 'sources.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Built five README figures and their source receipt; no network or raw data used.')


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
