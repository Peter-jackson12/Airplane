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
INK, MUTED, BLUE, GOLD = '#172B4D', '#52647A', '#2459B8', '#9A651B'
GRID, SOFT = '#DEE5EE', '#F5F7FB'


class SVG:
    """Small deterministic SVG renderer; no external font, image or runtime."""

    def __init__(self, height: int, title: str, description: str):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height}" '
            f'viewBox="0 0 760 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            '<style>text{font-family:"Noto Sans CJK KR","Malgun Gothic",'
            '"Apple SD Gothic Neo",sans-serif;font-variant-numeric:tabular-nums}</style>',
            f'<rect width="760" height="{height}" fill="#FFFFFF"/>']

    def text(self, x, y, text, size=18, color=INK, weight=400, anchor='start'):
        self.parts.append(f'<text x="{x:.3f}" y="{y:.3f}" font-size="{size}" '
                          f'fill="{color}" font-weight="{weight}" text-anchor="{anchor}">'
                          f'{escape(str(text))}</text>')

    def line(self, x1, y1, x2, y2, color=GRID, width=1):
        self.parts.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" '
                          f'y2="{y2:.3f}" stroke="{color}" stroke-width="{width}"/>')

    def rect(self, x, y, w, h, color, radius=0):
        self.parts.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" '
                          f'height="{h:.3f}" rx="{radius}" fill="{color}"/>')

    def dot(self, x, y, color=BLUE, hollow=False, radius=6):
        fill = '#FFFFFF' if hollow else color
        self.parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius}" '
                          f'fill="{fill}" stroke="{color}" stroke-width="2"/>')

    def polygon(self, points, color):
        values = ' '.join(f'{x:.3f},{y:.3f}' for x, y in points)
        self.parts.append(f'<polygon points="{values}" fill="{color}"/>')

    def header(self, title, scope, detail=None):
        self.text(32, 43, title, 26, weight=700)
        self.text(32, 77, scope, 18, MUTED)
        if detail:
            self.text(32, 105, detail, 17, MUTED)

    def save(self, path):
        path.write_text('\n'.join(self.parts + ['</svg>']) + '\n', encoding='utf-8')


def check_receipt() -> dict:
    receipt = json.loads((ASSETS / 'sources.json').read_text(encoding='utf-8'))
    if receipt['schema_version'] != 1:
        raise ValueError('Unsupported figure receipt schema')
    if receipt['generator_sha256'] != sha256(Path(__file__)):
        raise ValueError('Figure builder changed; regenerate the README assets')
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

    svg = SVG(450, '날짜를 확인한 행만 채택', '원본 1,000,000행의 날짜 귀속: 채택 706,759행, 보류 293,241행.')
    svg.header('날짜를 확인한 706,759행만 채택', '원본 1,000,000행 전체를 대조 · 나머지 293,241행은 보류',
               '대조 정보가 빠져 있다면 후보 연도가 하나여도 채택하지 않음')
    x = 32
    for row, color in zip(data['attribution'], (BLUE, GOLD, MUTED)):
        width = 696 * row['rows'] / 1000000
        svg.rect(x, 135, width, 34, color)
        x += width
    labels = ('채택 · 완전한 대조 정보 + 단일 후보 연도',
              '보류 · 대조 정보 결측 + 단일 후보 연도', '보류 · 그 밖의 대조 결과')
    for i, (row, label, color) in enumerate(zip(data['attribution'], labels, (BLUE, GOLD, MUTED))):
        y = 216 + i * 60
        svg.rect(32, y-15, 10, 18, color)
        svg.text(54, y, label, 18)
        svg.text(728, y, f"{row['rows']:,}행", 23, color, 700, 'end')
    svg.line(32, 372, 728, 372)
    svg.text(32, 403, '세 집단은 서로 겹치지 않으며 합계는 원본 100만 행입니다.', 17, MUTED)
    svg.text(32, 429, '보류는 미검사가 아니라 날짜 채택 기준을 충족하지 못했다는 뜻입니다.', 17, MUTED)
    finish(svg, 'attribution')

    svg = SVG(488, '전처리 수정만으로 Macro F1 향상은 확인되지 않음',
              '동일 라벨 255,001행. 네 기준 조건의 3시드 평균과 ±1 표본 SD. 확대 축이며 신뢰구간이 아님.')
    svg.header('전처리는 타당해졌지만, 성능 향상은 미확인', '라벨 255,001행 · 동일 평가 절차 · 시드 42 / 1 / 7',
               '전체 10조건 중 네 기준 조건 · 점: 평균 / 오차막대: ±1 SD')
    lo, hi = 0.5725, 0.5785
    xx = lambda v: 186 + (v-lo)/(hi-lo) * 336
    for tick in (0.573, 0.575, 0.577):
        x = xx(tick)
        svg.line(x, 137, x, 369)
        svg.text(x, 399, f'{tick:.3f}', 17, MUTED, anchor='middle')
    svg.text(623, 139, '평균 ± SD', 16, MUTED, anchor='middle')
    for i, row in enumerate(data['models']):
        y = 172 + i * 60
        svg.text(32, y+6, row['phase'], 20, weight=600)
        left, right, center = xx(row['mean']-row['sd']), xx(row['mean']+row['sd']), xx(row['mean'])
        color = BLUE if row['phase'].endswith('clean') else MUTED
        svg.line(left, y, right, y, color, 2)
        svg.line(left, y-6, left, y+6, color, 2)
        svg.line(right, y-6, right, y+6, color, 2)
        svg.dot(center, y, color, hollow=not row['phase'].endswith('clean'))
        svg.text(575, y-1, f"{row['mean']:.6f}", 21, color, 600)
        svg.text(575, y+21, f"± {row['sd']:.6f}", 16, MUTED)
    svg.line(186, 369, 522, 369, MUTED)
    svg.text(186, 430, 'Macro F1 ↑ 높을수록 좋음 · 차이를 읽기 위한 확대 축', 17, MUTED)
    svg.text(32, 466, '3시드 SD는 신뢰구간이 아니며, 오차막대 겹침으로 유의성을 판정하지 않습니다.', 16, MUTED)
    finish(svg, 'models')

    svg = SVG(626, '날씨 추가 전후 세 지표의 직접 비교',
              'P6_clean, 동일 라벨 180,332행과 외부 폴드, 시드 42/1/7. 10분 공개 지연 가정. '
              '세 지표 모두 개선. 값과 차이는 원본 집계에서 각각 반올림. SD는 신뢰구간이 아님.')
    svg.header('날씨 정보 추가 후, 세 지표 모두 개선', 'P6_clean · 동일 라벨 180,332행 · 시드 42 / 1 / 7',
               '10분 공개 지연 가정 · 동일 외부 폴드와 내부 선택 절차')
    for i, row in enumerate(data['weather_model']):
        y = 128 + i * 142
        svg.rect(32, y, 696, 130, SOFT, 8)
        svg.text(50, y+35, row['metric'], 25, weight=700)
        svg.text(50, y+63, '높을수록 좋음 ↑' if row['higher_is_better'] else '낮을수록 좋음 ↓', 17, MUTED)
        svg.text(282, y+29, '날씨 미사용', 17, MUTED)
        svg.text(527, y+29, '날씨 사용', 17, BLUE, 600)
        svg.text(282, y+67, f"{row['off_mean']:.6f}", 30, MUTED, 600)
        svg.line(457, y+53, 498, y+53, MUTED, 2)
        svg.polygon([(498, y+53), (490, y+48), (490, y+58)], MUTED)
        svg.text(527, y+67, f"{row['on_mean']:.6f}", 30, BLUE, 700)
        svg.text(50, y+107, f"변화 {row['delta_mean']:+.6f}", 23, BLUE, 700)
        svg.text(355, y+106, f"시드별 차이의 SD {row['sd']:.6f}", 17, MUTED)
    svg.text(32, 583, '변화 = 사용 − 미사용 · 세 지표는 척도가 달라 변화량의 크기를 서로 비교하지 않습니다.', 16, MUTED)
    svg.text(32, 610, '평균과 차이는 원본 수치에서 각각 반올림했습니다. 3시드 SD는 신뢰구간이 아닙니다.', 16, MUTED)
    finish(svg, 'weather_model')

    svg = SVG(508, '확률 보정의 상충 관계: ECE와 LogLoss',
              'P6_clean 공유형 보정기, 라벨 255,001행의 3시드 평균. ECE는 %p, LogLoss와 모두 낮을수록 좋음.')
    svg.header('확률 보정: 지표에 따라 결과가 다름', 'P6_clean · 공유형 보정기 · 라벨 255,001행 · 3시드 평균',
               'Isotonic은 ECE가 낮아졌지만 LogLoss는 높아짐')
    xx = lambda v: 125 + (v-0.12)/0.40 * 565
    yy = lambda v: 380 - (v-0.4472)/0.0021 * 230
    for tick in (0.2, 0.3, 0.4, 0.5):
        x = xx(tick)
        svg.line(x, 144, x, 380)
        svg.text(x, 409, f'{tick:.1f}', 17, MUTED, anchor='middle')
    for tick in (0.4475, 0.4480, 0.4485, 0.4490):
        y = yy(tick)
        svg.line(125, y, 690, y)
        svg.text(112, y+6, f'{tick:.4f}', 17, MUTED, anchor='end')
    svg.text(32, 135, 'LogLoss ↓', 18, MUTED)
    for row, color in zip(data['calibration'], (MUTED, BLUE, GOLD)):
        x, y = xx(row['ece_percentage_points']), yy(row['log_loss'])
        key = row['calibrator']
        if key == 'none':
            svg.dot(x, y, color, hollow=True, radius=7)
            label, offset = '보정 없음', -16
        elif key == 'platt':
            svg.rect(x-6, y-6, 12, 12, color)
            label, offset = 'Platt', 31
        else:
            svg.polygon([(x,y-8),(x-8,y+6),(x+8,y+6)],color)
            label, offset = 'Isotonic', -16
        svg.text(x, y+offset, label, 20, color, 600, 'middle')
    svg.line(125, 144, 125, 380, MUTED)
    svg.line(125, 380, 690, 380, MUTED)
    svg.text(340, 443, 'ECE (%p) ↓ 낮을수록 좋음', 18, MUTED, anchor='middle')
    svg.text(32, 479, '두 축 모두 낮을수록 좋습니다. ECE는 원본 비율을 100배 한 %p 단위입니다.', 16, MUTED)
    svg.text(32, 501, 'Macro F1 향상은 확인되지 않았으며, 정확한 값은 본문의 보정 비교표에 있습니다.', 16, MUTED)
    finish(svg, 'calibration')

    svg = SVG(590, '공개 지연 가정별 날씨 결합률',
              '층화 보정 표본 300행 중 수집 가능 268행. 0/10/30/60분은 미검증 가정. '
              '60분에서 출발 150/268, 도착 157/268. 결합률이며 모델 성능이 아님.')
    svg.header('날씨가 있어도 예측 시점에 공개돼 있어야 사용', '층화 보정 표본 300행 중 수집 가능 268행 · 관측 나이 상한 90분',
               '공개 지연 가정별 결합률 · 분모에서 보류 32행은 제외한 표본 비교')
    svg.rect(455, 125, 13, 13, BLUE)
    svg.text(477, 137, '출발 공항', 17, MUTED)
    svg.rect(596, 125, 13, 13, MUTED)
    svg.text(618, 137, '도착 공항', 17, MUTED)
    xx = lambda v: 135 + v/100 * 350
    for tick in (0, 25, 50, 75, 100):
        x = xx(tick)
        svg.line(x, 164, x, 463)
        svg.text(x, 490, str(tick), 17, MUTED, anchor='middle')
    for i, latency in enumerate((0, 10, 30, 60)):
        y = 178 + i * 76
        svg.text(32, y+24, f'{latency}분 가정', 18, weight=600)
        for j, (role, color, label) in enumerate((('origin',BLUE,'출발'),('destination',MUTED,'도착'))):
            row = next(r for r in data['latency'] if (r['role'],r['latency_minutes']) == (role,latency))
            yy = y + j*26
            svg.rect(135, yy, xx(row['match_rate_percent'])-135, 17, color)
            svg.text(510, yy+15, f"{label} {row['matched_rows']}/268 · {row['match_rate_percent']:.2f}%", 17, color)
    svg.line(135, 463, 485, 463, MUTED)
    svg.text(205, 519, '수집 가능 행 중 결합률 (%)', 18, MUTED)
    svg.text(32, 558, '0 / 10 / 30 / 60분은 실측 공개 지연 시간이 아니라 가정입니다.', 17, MUTED)
    svg.text(32, 582, '전체 300행·전체 날짜 귀속 집단의 결합률과 구별하며, 모델 성능을 나타내지 않습니다.', 16, MUTED)
    finish(svg, 'latency')

    receipt = {'schema_version': 1, 'generator_sha256': sha256(Path(__file__)),
               'sources': {key: {'path': path, 'sha256': sha256(ROOT / path)} for key, path in SOURCES.items()},
               'filters': expected_filters(), 'reviewed_data': data, 'figures': figures,
               'limitations': ['No raw data access, network calls or model retraining.',
                               'No live collection progress is inferred.',
                               'Korean SVG text uses system font fallbacks; no font files or external assets are embedded.',
                               'Weather levels and signed paired deltas are rounded independently from tracked evidence.',
                               'Three-seed SD is not a confidence interval; the metrics do not share a quantitative axis.']}
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
