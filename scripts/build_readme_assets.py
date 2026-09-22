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
    # Every displayed level, spread and delta comes from the existing summary.
    # Keep the original signed delta; do not subtract rounded display values.
    weather_rows = []
    for metric, key, direction in (
        ('Macro F1', 'macro_f1_nested', 'higher'),
        ('LogLoss', 'log_loss', 'lower'),
        ('ROC-AUC', 'roc_auc', 'higher'),
    ):
        off = weather['conditions']['weather_off'][key]
        on = weather['conditions']['weather_on'][key]
        delta = weather['paired_deltas_on_minus_off'][key]
        sign = 1 if direction == 'higher' else -1
        if set(delta['values_by_seed']) != {'42', '1', '7'}:
            raise ValueError('Weather-model paired seeds changed')
        if not all(sign * value > 0 for value in delta['values_by_seed'].values()):
            raise ValueError('Weather-model seed improvement direction changed')
        weather_rows.append({
            'metric': metric, 'direction': direction,
            'off_mean': off['mean'], 'off_sd': off['std'],
            'on_mean': on['mean'], 'on_sd': on['std'],
            'delta_mean': delta['mean'], 'delta_sd': delta['std'],
            'delta_by_seed': delta['values_by_seed'],
            'definition': 'weather_on - weather_off',
        })
    return {'models': model_rows, 'calibration': calibration_rows,
            'attribution': attribution_rows, 'latency': latency_rows,
            'weather_model': weather_rows}


def check_receipt() -> dict:
    receipt = json.loads((ASSETS / 'sources.json').read_text(encoding='utf-8'))
    if receipt['schema_version'] != 1:
        raise ValueError('Unsupported figure receipt schema')
    if receipt['generator_sha256'] != sha256(Path(__file__)):
        raise ValueError('Figure builder changed; regenerate the README assets')
    if receipt['reviewed_data'] != reviewed_data():
        raise ValueError('Plotted rows differ from tracked aggregate evidence')
    for entry in receipt['sources'].values():
        if sha256(ROOT / entry['path']) != entry['sha256']:
            raise ValueError(f"Source hash drift: {entry['path']}")
    for entry in receipt['figures'].values():
        if sha256(ROOT / entry['path']) != entry['sha256']:
            raise ValueError(f"Figure hash drift: {entry['path']}")
    return receipt


# SVG text remains searchable and uses the reader's Korean system font.
# No external fonts/resources, timestamps or renderer-specific glyph paths are
# embedded. Generation is deterministic and uses only the Python standard library.
INK, MUTED, LINE = '#172b3a', '#526575', '#d8e1e8'
BLUE, TEAL, GOLD = '#215f9a', '#087a78', '#a65b11'


class Figure:
    def __init__(self, title: str, subtitle: str, height: int):
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
            f'viewBox="0 0 960 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(subtitle)}</desc>',
            '<style>text{font-family:"Noto Sans CJK KR","Noto Sans CJK SC",'
            '"Malgun Gothic","Apple SD Gothic Neo",sans-serif;'
            'font-variant-numeric:tabular-nums}</style>',
            f'<rect width="960" height="{height}" fill="white"/>',
        ]
        self.rect(32, 26, 5, 28, TEAL)
        self.text(51, 48, title, size=25, weight='700')
        self.text(32, 84, subtitle, size=17, fill=MUTED)

    def text(self, x, y, text, *, size=19, fill=INK, weight='400', anchor='start'):
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
                          f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">'
                          f'{escape(str(text))}</text>')

    def line(self, x1, y1, x2, y2, *, stroke=LINE, width=1.5, dash=''):
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                          f'stroke="{stroke}" stroke-width="{width}" stroke-dasharray="{dash}"/>')

    def rect(self, x, y, w, h, fill, *, stroke='none', radius=0):
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                          f'rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def dot(self, x, y, color, *, square=False, hollow=False):
        if square:
            self.rect(x - 6, y - 6, 12, 12, 'white' if hollow else color, stroke=color)
        else:
            self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" '
                              f'fill="{"white" if hollow else color}" stroke="{color}" stroke-width="2"/>')

    def note(self, lines):
        y = self.height - 24 - 26 * (len(lines) - 1)
        self.line(32, y - 26, 928, y - 26)
        for line in lines:
            self.text(32, y, line, size=17, fill=MUTED)
            y += 26

    def save(self, path: Path):
        path.write_text('\n'.join([*self.parts, '</svg>', '']), encoding='utf-8')


def build() -> None:
    data = reviewed_data()
    ASSETS.mkdir(parents=True, exist_ok=True)
    figures = {}

    def finish(fig, filename, key, notes):
        fig.note(notes)
        path = ASSETS / filename
        fig.save(path)
        figures[key] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}

    fig = Figure('날짜를 신뢰성 있게 귀속한 항공편은 706,759행',
                 '원본 1,000,000행 · BTS 2018/2019년 12개월 대조 · 서로 겹치지 않는 세 집단', 470)
    labels = [('채택', '완전한 대조 키 · 후보 연도 하나'),
              ('보류', '결측 대조 키 · 후보 연도 하나'), ('보류', '그 밖의 대조 결과')]
    left, right = 370, 775
    for tick in (0, 250000, 500000, 750000, 1000000):
        x = left + (right - left) * tick / 1000000
        fig.line(x, 125, x, 324)
        fig.text(x, 350, f'{tick // 10000}만' if tick else '0', size=16, anchor='middle', fill=MUTED)
    for i, (row, label) in enumerate(zip(data['attribution'], labels)):
        y = 149 + 72 * i
        color = TEAL if i == 0 else MUTED
        fig.text(32, y - 2, label[0], size=21, weight='700', fill=color)
        fig.text(32, y + 23, label[1], size=17, fill=MUTED)
        fig.rect(left, y - 16, (right - left) * row['rows'] / 1000000, 24, color)
        fig.text(919, y, f"{row['rows']:,}행", size=23, weight='700', anchor='end', fill=color)
        fig.text(919, y + 24, f"{row['rows'] / 10000:.2f}%", size=17, anchor='end', fill=MUTED)
    fig.line(left, 324, right, 324, stroke=MUTED)
    fig.text(572, 379, '행 수 · 0부터 시작하는 동일 축', size=17, anchor='middle', fill=MUTED)
    finish(fig, 'date_attribution.svg', 'attribution', [
        '보류 293,241행도 모두 대조했습니다. 키가 결측이면 후보 연도가 하나여도 채택하지 않습니다.',
        '날짜 귀속 집단과 라벨 집단은 다릅니다. 날씨 성능 비교는 귀속·라벨 보유 180,332행입니다.'])

    fig = Figure('전처리의 타당성 개선이 Macro F1 향상으로 이어지지는 않았습니다',
                 '동일 라벨 255,001행 · 전처리 10조건 중 네 기준 조건 · 3시드 평균과 표본 표준편차', 500)
    left, right, low, high = 240, 677, 0.5725, 0.5785
    scale = lambda value: left + (right - left) * (value - low) / (high - low)
    fig.text(925, 123, '평균 ± SD', size=17, anchor='end', fill=MUTED)
    for tick in (0.573, 0.574, 0.575, 0.576, 0.577, 0.578):
        x = scale(tick)
        fig.line(x, 137, x, 351)
        fig.text(x, 377, f'{tick:.3f}', size=16, anchor='middle', fill=MUTED)
    for i, row in enumerate(data['models']):
        y = 157 + i * 58
        clean = row['phase'].endswith('clean')
        color = TEAL if clean else BLUE
        fig.text(32, y + 6, row['phase'], size=21, weight='700' if clean else '400')
        x, x0, x1 = scale(row['mean']), scale(row['mean'] - row['sd']), scale(row['mean'] + row['sd'])
        fig.line(x0, y, x1, y, stroke=color, width=2)
        fig.line(x0, y - 6, x0, y + 6, stroke=color)
        fig.line(x1, y - 6, x1, y + 6, stroke=color)
        fig.dot(x, y, color, square=clean, hollow=not clean)
        fig.text(925, y + 6, f"{row['mean']:.6f} ± {row['sd']:.6f}", size=19, anchor='end')
    fig.line(left, 351, right, 351, stroke=MUTED)
    fig.text(460, 405, 'Macro F1 ↑ · 차이를 읽기 위한 확대 축', size=17, anchor='middle', fill=MUTED)
    finish(fig, 'model_comparison.svg', 'models', [
        '점: 3시드 평균 · 오차막대: ±1 표본 SD(신뢰구간 아님) · ○ 기존 조건 / ■ 수정 조건',
        '작은 차이나 오차막대의 겹침만으로 통계적 동등성·유의성·확정적 악화를 판정하지 않습니다.'])

    fig = Figure('날씨 추가 후 세 지표 모두 같은 방향으로 개선됐습니다',
                 'P6_clean · 동일 평가 180,332행 · 시드 42 / 1 / 7 · 동일 외부 폴드와 내부 선택 절차', 688)
    fig.text(32, 113, '출발·도착 날씨 14개 피처 추가 · 공개 지연 시간은 사전에 정한 10분 가정', size=17, fill=MUTED)
    for i, row in enumerate(data['weather_model']):
        top = 138 + i * 147
        fig.rect(32, top, 896, 130, '#f6f9fb', stroke=LINE, radius=8)
        fig.text(51, top + 37, row['metric'], size=25, weight='700')
        direction = '높을수록 좋음 ↑' if row['direction'] == 'higher' else '낮을수록 좋음 ↓'
        fig.text(51, top + 68, direction, size=17, fill=MUTED)
        for x, name, key, color in ((250, '날씨 미사용', 'off', MUTED), (495, '날씨 사용', 'on', TEAL)):
            fig.text(x, top + 27, name, size=17, fill=color)
            fig.text(x, top + 66, f"{row[key + '_mean']:.6f}", size=31, weight='700', fill=color)
            fig.text(x, top + 99, f"± {row[key + '_sd']:.6f}", size=18, fill=MUTED)
        fig.line(444, top + 59, 476, top + 59, stroke=MUTED, width=2)
        fig.line(468, top + 53, 476, top + 59, stroke=MUTED, width=2)
        fig.line(468, top + 65, 476, top + 59, stroke=MUTED, width=2)
        fig.line(708, top + 20, 708, top + 108)
        fig.text(728, top + 27, '변화량 · 사용−미사용', size=16, fill=MUTED)
        fig.text(728, top + 65, f"{row['delta_mean']:+.6f}".replace('-', '−'), size=28, fill=TEAL, weight='700')
        fig.text(728, top + 98, f"± {row['delta_sd']:.6f}", size=18, fill=MUTED)
    finish(fig, 'weather_model_comparison.svg', 'weather_model', [
        '모든 값: 3시드 평균 ± 표본 SD(신뢰구간 아님). 변화량은 반올림 전 원본 집계를 사용합니다.',
        '지표마다 척도와 해석이 다릅니다. 이 수치 카드는 변화량 크기를 공통 길이로 비교하지 않습니다.',
        '10분은 실측값이 아닙니다. 정적 교차검증의 개선이며 인과 효과·미래 운항 성능은 미검증입니다.'])

    fig = Figure('확률 보정 오차가 줄어도 LogLoss까지 좋아지지는 않았습니다',
                 'P6_clean · 공유형 보정기 · 라벨 255,001행 · 3시드 평균 · 두 축 모두 낮을수록 좋음', 500)
    left, right, top, bottom = 140, 575, 137, 347
    x = lambda v: left + (right - left) * (v - 0.15) / (0.50 - 0.15)
    y = lambda v: bottom - (bottom - top) * (v - 0.4473) / (0.4492 - 0.4473)
    for tick in (0.2, 0.3, 0.4, 0.5):
        fig.line(x(tick), top, x(tick), bottom)
        fig.text(x(tick), bottom + 27, f'{tick:.1f}', anchor='middle', size=16, fill=MUTED)
    for tick in (0.4475, 0.4480, 0.4485, 0.4490):
        fig.line(left, y(tick), right, y(tick))
        fig.text(left - 13, y(tick) + 5, f'{tick:.4f}', anchor='end', size=16, fill=MUTED)
    fig.line(left, top, left, bottom, stroke=MUTED)
    fig.line(left, bottom, right, bottom, stroke=MUTED)
    fig.text(36, 121, 'LogLoss ↓', size=18, fill=MUTED)
    fig.text(348, 409, 'ECE (%p) ↓ · 확대 축', size=18, anchor='middle', fill=MUTED)
    names = {'none': '보정 없음', 'platt': 'Platt', 'isotonic': 'Isotonic'}
    for i, row in enumerate(data['calibration']):
        color = (MUTED, BLUE, GOLD)[i]
        xp, yp = x(row['ece_percentage_points']), y(row['log_loss'])
        fig.dot(xp, yp, color, square=i == 2, hollow=i == 0)
        fig.text(xp, yp - 16, names[row['calibrator']], size=18, anchor='middle', fill=color)
        ty = 153 + i * 76
        fig.text(628, ty, names[row['calibrator']], size=21, weight='700', fill=color)
        fig.text(628, ty + 29, f"ECE {row['ece_percentage_points']:.4f}%p · LogLoss {row['log_loss']:.6f}", size=16)
    finish(fig, 'calibration_tradeoff.svg', 'calibration', [
        'ECE는 비율을 100배 한 %p입니다. 비교 조건의 정확한 Macro F1은 README 표에 함께 제시합니다.',
        'Isotonic은 ECE 감소와 LogLoss 악화가 함께 관찰됐습니다. 보정을 분류 성능 향상으로 해석하지 않습니다.'])

    fig = Figure('날씨 공개가 늦다고 가정할수록 사용할 수 있는 관측이 줄었습니다',
                 '층화 보정 표본(stratafix) 300행 중 수집 가능 268행 · 관측 나이 상한 90분', 534)
    left, right, top, bottom = 100, 865, 154, 344
    x = lambda v: left + (right - left) * v / 60
    y = lambda v: bottom - (bottom - top) * v / 100
    for tick in (0, 25, 50, 75, 100):
        fig.line(left, y(tick), right, y(tick))
        fig.text(left - 15, y(tick) + 6, f'{tick}%', size=17, anchor='end', fill=MUTED)
    for tick in (0, 10, 30, 60):
        fig.text(x(tick), bottom + 30, f'{tick}분', size=18, anchor='middle', fill=MUTED)
    fig.line(left, top, left, bottom, stroke=MUTED)
    fig.line(left, bottom, right, bottom, stroke=MUTED)
    fig.text(32, 120, '결합률 · 결합 행 / 268행', size=17, fill=MUTED)
    for i, role in enumerate(('origin', 'destination')):
        rows = sorted((r for r in data['latency'] if r['role'] == role), key=lambda r: r['latency_minutes'])
        color = BLUE if i == 0 else TEAL
        for a, b in zip(rows, rows[1:]):
            fig.line(x(a['latency_minutes']), y(a['match_rate_percent']),
                     x(b['latency_minutes']), y(b['match_rate_percent']), stroke=color, width=2, dash='6 4' if i else '')
        for row in rows:
            xp, yp = x(row['latency_minutes']), y(row['match_rate_percent'])
            fig.dot(xp, yp, color, square=bool(i), hollow=not i)
            fig.text(xp, yp + (30 if i == 0 else -19), f"{row['matched_rows']}/268", size=17, anchor='middle', fill=color)
        lx = 561 + 195 * i
        fig.dot(lx, 113, color, square=bool(i), hollow=not i)
        fig.text(lx + 15, 119, '출발 공항' if i == 0 else '도착 공항', size=18, fill=color)
    fig.text(470, 409, '날씨 공개 지연 시간 가정 · 실제 수신 지연의 측정값이 아님', size=18, anchor='middle', fill=MUTED)
    finish(fig, 'weather_latency.svg', 'latency', [
        '0 / 10 / 30 / 60분은 결합률 민감도를 살핀 가정입니다. 모델 성능 비교에는 10분만 사용했습니다.',
        '보류 32행을 포함한 전체 분모는 300행입니다. 이 그림은 날씨 모델의 성능 그래프가 아닙니다.'])

    receipt = {
        'schema_version': 1, 'generator_sha256': sha256(Path(__file__)),
        'sources': {key: {'path': path, 'sha256': sha256(ROOT / path)} for key, path in SOURCES.items()},
        'filters': {
            'models': list(PHASES),
            'calibration': {'phase_key': 'P6_clean', 'arm': 'shared', 'group': 'overall'},
            'attribution': 'sum rows over all 12 months; three disjoint status buckets',
            'latency': 'stratafix only; recompute matched_rows / eligible_rows for each role/scenario',
            'weather_model': 'P6_clean; 10-minute assumed latency; 180,332 identical labeled rows; 3 paired seeds',
        },
        'reviewed_data': data, 'figures': figures,
        'limitations': [
            'No raw data access, collection, joins or model retraining.',
            'Korean SVG text uses system font fallbacks; no external resources or font files are embedded.',
            'All weather levels, sample SDs and signed deltas come from the tracked summary before display rounding.',
            'Weather metric cards do not encode magnitude with a common axis or length.',
            'Sample SD is not a confidence interval; 10-minute publication latency is assumed, not measured.',
        ],
    }
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
