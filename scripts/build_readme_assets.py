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
    delta = weather['paired_deltas_on_minus_off']
    weather_rows = [
        {'metric': 'Macro F1', 'mean_improvement': float(delta['macro_f1_nested']['mean']),
         'sd': float(delta['macro_f1_nested']['std']), 'definition': 'weather_on - weather_off'},
        {'metric': 'ROC-AUC', 'mean_improvement': float(delta['roc_auc']['mean']),
         'sd': float(delta['roc_auc']['std']), 'definition': 'weather_on - weather_off'},
        {'metric': 'LogLoss reduction', 'mean_improvement': -float(delta['log_loss']['mean']),
         'sd': float(delta['log_loss']['std']), 'definition': 'weather_off - weather_on'},
    ]
    if not all(r['mean_improvement'] > 0 for r in weather_rows):
        raise ValueError('Weather-model plotted improvement direction changed')
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


def build() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    data = reviewed_data()
    ASSETS.mkdir(parents=True, exist_ok=True)
    # Stable SVG IDs and metadata; rely on Matplotlib's default chart palette.
    plt.rcParams.update({'svg.hashsalt': 'airplane-readme-v1', 'svg.fonttype': 'path',
                         'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.titlesize': 17,
                         'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 12})
    figures = {}

    def finish(fig, ax, filename: str, key: str, note: str) -> None:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.text(0.025, 0.025, note, fontsize=10)
        path = ASSETS / filename
        fig.savefig(path, format='svg', metadata={'Date': None, 'Creator': 'Airplane README evidence builder'})
        plt.close(fig)
        figures[key] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha256(path)}

    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.23, top=0.77)
    rows = data['models']
    ax.errorbar([r['mean'] for r in rows], range(4), xerr=[r['sd'] for r in rows],
                fmt='o', capsize=5, markersize=8, linewidth=1.8)
    ax.set_yticks(range(4), [r['phase'] for r in rows])
    ax.invert_yaxis()
    ax.set_ylim(3.7, -0.8)
    ax.set_xlim(0.5725, 0.5785)
    ax.set_xlabel('Macro F1 (higher is better; zoomed axis)')
    ax.grid(axis='x', alpha=0.2)
    for i, row in enumerate(rows):
        ax.annotate(f"{row['mean']:.6f}", (row['mean'], i), xytext=(0, 15),
                    textcoords='offset points', ha='center', fontsize=10)
    fig.suptitle('Preprocessing comparison: four reference conditions', x=0.025, ha='left', y=0.97)
    fig.text(0.025, 0.865, '255,001 labeled rows | same nested protocol | seeds 42, 1, 7', fontsize=11)
    finish(fig, ax, 'model_comparison.svg', 'models',
           'Points: 3-seed means. Error bars: +/- 1 sample SD, not confidence intervals.\nShown: 4 of the 10 evaluated conditions. No verified Macro F1 improvement from clean preprocessing.')

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.14, right=0.94, bottom=0.23, top=0.79)
    rows = data['calibration']
    for row, marker in zip(rows, ('o', 's', '^')):
        ax.scatter(row['ece_percentage_points'], row['log_loss'], s=100, marker=marker)
        label = {'none': 'No calibration', 'platt': 'Platt', 'isotonic': 'Isotonic'}[row['calibrator']]
        offset = (-12, 10) if row['calibrator'] == 'none' else (12, 10)
        align = 'right' if row['calibrator'] == 'none' else 'left'
        ax.annotate(label, (row['ece_percentage_points'], row['log_loss']),
                    xytext=offset, textcoords='offset points', ha=align, fontsize=11)
    ax.set_xlim(0.12, 0.52)
    ax.set_ylim(0.4472, 0.4493)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.set_xlabel('ECE (percentage points; lower is better)')
    ax.set_ylabel('LogLoss (lower is better)')
    ax.grid(alpha=0.2)
    fig.suptitle('Calibration: lower ECE is not the same as lower LogLoss', x=0.025, ha='left', y=0.97)
    fig.text(0.025, 0.87, 'P6_clean | shared arm | overall | 255,001 labeled rows | 3-seed means', fontsize=11)
    finish(fig, ax, 'calibration_tradeoff.svg', 'calibration',
           'Each point is one calibrator, not a fitted relationship. Both axes are zoomed.\nIsotonic lowers ECE while increasing LogLoss here; this does not establish a Macro F1 gain.')

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.subplots_adjust(left=0.40, right=0.96, bottom=0.23, top=0.78)
    rows = data['attribution']
    bars = ax.barh(range(3), [r['rows'] for r in rows], height=0.52)
    ax.set_yticks(range(3), ['Accepted: complete key\n+ one candidate year',
                            'Held: missing key\n+ one candidate year', 'Held: other inspected cases'])
    ax.invert_yaxis()
    ax.set_xlim(0, 1000000)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:.0f}k'))
    ax.set_xlabel('Rows (zero-based scale)')
    ax.grid(axis='x', alpha=0.2)
    for bar, row in zip(bars, rows):
        ax.text(row['rows'] + 16000, bar.get_y() + bar.get_height() / 2,
                f"{row['rows']:,}\n{row['rows'] / 10000:.2f}%", va='center', fontsize=11)
    fig.suptitle('Date attribution: accepted and held rows', x=0.025, ha='left', y=0.97)
    fig.text(0.025, 0.875, '1,000,000 source rows | BTS 2018/2019 comparison | 12 months inspected', fontsize=11)
    finish(fig, ax, 'date_attribution.svg', 'attribution',
           'Held rows were inspected, not forgotten. A unique candidate with incomplete keys is not accepted.\nThis selected subset is not the same population as the 255,001 labeled rows used for model evaluation.')

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.29, top=0.79)
    for role, marker in (('origin', 'o'), ('destination', 's')):
        rows = sorted((r for r in data['latency'] if r['role'] == role), key=lambda r: r['latency_minutes'])
        ax.plot([r['latency_minutes'] for r in rows], [r['match_rate_percent'] for r in rows],
                marker=marker, linewidth=2, markersize=7, label=role.title())
        for row in rows:
            offset = -16 if role == 'origin' else 9
            ax.annotate(f"{row['matched_rows']}/268", (row['latency_minutes'], row['match_rate_percent']),
                        xytext=(0, offset), textcoords='offset points', ha='center', fontsize=9)
    ax.set_xticks([0, 10, 30, 60])
    ax.set_xlim(-4, 64)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Assumed publication latency (minutes)')
    ax.set_ylabel('Matched / eligible rows (%)')
    ax.grid(alpha=0.2)
    ax.legend(loc='lower left', frameon=False)
    fig.suptitle('Stratafix: weather matching depends on availability assumptions', x=0.025, ha='left', y=0.97)
    fig.text(0.025, 0.875, 'Separate 300-row sample | 268 eligible rows | maximum observation age: 90 minutes', fontsize=11)
    finish(fig, ax, 'weather_latency.svg', 'latency',
           '0/10/30/60-minute latency values are scenarios, not measured historical publication delays.\n32 ineligible rows remain in the full 300-row denominator. This chart is not a model-performance result.')

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.24, top=0.78)
    rows = data['weather_model']
    means = [r['mean_improvement'] for r in rows]
    sds = [r['sd'] for r in rows]
    ax.errorbar(means, range(len(rows)), xerr=sds, fmt='o', capsize=5,
                markersize=8, linewidth=1.8)
    ax.axvline(0, linewidth=1, alpha=0.5)
    ax.set_yticks(range(len(rows)), [r['metric'] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.036)
    ax.set_xlabel('Mean paired improvement (positive is better)')
    ax.grid(axis='x', alpha=0.2)
    for i, row in enumerate(rows):
        ax.annotate(f"{row['mean_improvement']:+.6f}", (row['mean_improvement'], i),
                    xytext=(8, 0), textcoords='offset points', va='center', fontsize=10)
    fig.suptitle('Weather-on vs weather-off: paired improvement', x=0.025, ha='left', y=0.97)
    fig.text(0.025, 0.865,
             'P6_clean | 180,332 labeled adopted rows | 10-minute latency assumption | seeds 42, 1, 7',
             fontsize=11)
    finish(fig, ax, 'weather_model_comparison.svg', 'weather_model',
           'Points: 3-seed mean paired changes; error bars: +/- 1 sample SD, not confidence intervals.\n'
           'Macro F1 and ROC-AUC use weather-on minus weather-off; LogLoss is plotted as the reduction (off minus on).')

    receipt = {'schema_version': 1, 'generator_sha256': sha256(Path(__file__)),
               'sources': {key: {'path': path, 'sha256': sha256(ROOT / path)} for key, path in SOURCES.items()},
               'filters': {'models': list(PHASES), 'calibration': {'phase_key': 'P6_clean', 'arm': 'shared', 'group': 'overall'},
                           'attribution': 'sum rows over all 12 months; three disjoint status buckets',
                           'latency': 'stratafix only; recompute matched_rows / eligible_rows for each role/scenario',
                           'weather_model': 'P6_clean; 10-minute assumed latency; 180,332 identical labeled rows; 3 paired seeds'},
               'reviewed_data': data, 'figures': figures,
               'limitations': ['No raw data access or model retraining.', 'No live collection progress is inferred.',
                               'English plot labels use portable embedded glyphs; README provides Korean captions.',
                               'Weather-model scores come only from the tracked paired-comparison summary.']}
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
