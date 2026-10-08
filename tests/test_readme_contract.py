"""Git-only documentation checks; no network or ignored source data required."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / 'README.md'
# Section anchors of the portfolio README. `evidence-appendix` is also the
# inbound target of docs/README_EXPLAINED_KO.md and must keep resolving.
REQUIRED_ANCHORS = {
    'overview', 'results', 'engineering', 'limits', 'reproduce', 'docs',
    'evidence-appendix',
}


def prose_without_fences(text: str) -> str:
    lines = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith('```'):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    assert not in_fence, 'README has an unclosed fenced code block'
    return '\n'.join(lines)


def markdown_destinations(text: str) -> list[str]:
    # The README uses inline links with encoded spaces in filenames, no reference links.
    return re.findall(r'!?\[[^\]\n]*\]\(([^\s)]+)\)', prose_without_fences(text))


def readme_ids() -> list[str]:
    return re.findall(r'<a\s+id="([^"]+)"\s*>', README.read_text(encoding='utf-8'))


def test_readme_keeps_section_anchors_and_resolves_internal_links():
    text = README.read_text(encoding='utf-8')
    ids = readme_ids()
    assert len(ids) == len(set(ids)), 'Duplicate explicit README anchors'
    assert REQUIRED_ANCHORS <= set(ids)
    for destination in markdown_destinations(text):
        if destination.startswith('#'):
            assert unquote(destination[1:]) in ids, destination


def test_repository_docs_links_into_readme_resolve():
    ids = set(readme_ids())
    checked = 0
    for doc in sorted((ROOT / 'docs').glob('*.md')):
        for destination in markdown_destinations(doc.read_text(encoding='utf-8')):
            parts = urlsplit(destination)
            if parts.scheme or parts.netloc or not parts.path.endswith('README.md'):
                continue
            if (doc.parent / unquote(parts.path)).resolve() != README.resolve():
                continue
            if parts.fragment:
                assert unquote(parts.fragment) in ids, f'{doc.name}: {destination}'
                checked += 1
    assert checked, 'Expected at least one docs link into a README section'


def test_readme_relative_evidence_links_exist_in_git_checkout():
    text = README.read_text(encoding='utf-8')
    checked = []
    for destination in markdown_destinations(text):
        parts = urlsplit(destination)
        if parts.scheme or parts.netloc or not parts.path:
            continue
        path = (ROOT / unquote(parts.path)).resolve()
        assert path.is_relative_to(ROOT), destination
        assert path.exists(), f'Broken README link: {destination}'
        checked.append(destination)
    assert checked, 'No repository-relative evidence links were checked'
    # The documentation guide links the deep-dive, tutor feedback and wiki hub.
    for required in ('docs/README_EXPLAINED_KO.md', 'docs/TUTOR_FEEDBACK_HANDOFF_KO.md', 'AGENTS.md'):
        assert required in checked, required


def test_readme_details_and_fences_are_balanced():
    prose = prose_without_fences(README.read_text(encoding='utf-8'))
    depth = 0
    summaries = 0
    openings = 0
    for tag in re.findall(r'</?details\b[^>]*>|<summary\b[^>]*>', prose):
        if tag.startswith('</details'):
            depth -= 1
            assert depth >= 0, 'Closing details without opening details'
        elif tag.startswith('<details'):
            depth += 1
            openings += 1
        else:
            assert depth > 0, 'Summary outside details'
            summaries += 1
    assert depth == 0, 'Unclosed details element'
    assert openings == summaries and openings > 0
    # Supplementary analyses may fold, but the visible reading path must
    # retain the problem, headline results and interpretation limits.
    reader_body = prose.split('<a id="reproduce"></a>', 1)[0]
    visible_body = re.sub(r'<details>.*?</details>', '', reader_body, flags=re.S)
    assert '0.598945' in visible_body and '180,332' in visible_body
    assert '인과 효과' in visible_body and '실측 공개 지연 시간이 아닙니다' in visible_body
    for phrase in ('지연 여부', '255,001', '706,759', '신뢰구간이 아닙니다',
                   'Macro F1 향상을 확인하지 못했', '10분은 실측 공개 지연 시간이 아닙니다',
                   '지연 경보의 정확성이 개선됐다', 'Logistic Regression'):
        assert phrase in visible_body, phrase


def test_readme_current_model_table_matches_tracked_summary():
    text = README.read_text(encoding='utf-8')
    with (ROOT / 'output/preprocessing_full_summary.csv').open(encoding='utf-8', newline='') as handle:
        summary = {row['phase_key']: row for row in csv.DictReader(handle)}
    for phase in ('P4', 'P4_clean', 'P6_fixed', 'P6_clean'):
        row = summary[phase]
        expected = (
            f"| {phase} | {float(row['macro_f1_nested_mean']):.6f} ± "
            f"{float(row['macro_f1_nested_std']):.6f} | {float(row['log_loss_mean']):.6f} | "
            f"{float(row['roc_auc_mean']):.6f} |"
        )
        assert expected in text, f'Current model summary drift: {phase}'


def test_readme_stratafix_denominators_match_selection_manifest():
    text = README.read_text(encoding='utf-8')
    manifest = json.loads((ROOT / (
        'output/baseline_recovery_v2_weather_expanded_stratafix_20260918_selection_manifest.json'
    )).read_text(encoding='utf-8'))
    expected = f"{manifest['selected_rows']}행(수집 가능 {manifest['collectible_rows']}행)"
    assert expected in text
    assert manifest['collectible_rows'] + manifest['not_collectible_rows'] == manifest['selected_rows']


def test_readme_weather_population_matches_tracked_summary():
    text = README.read_text(encoding='utf-8')
    summary = json.loads((ROOT / (
        'output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_summary.json'
    )).read_text(encoding='utf-8'))
    assert f"**{summary['evaluation_rows']:,}행**(지연 {summary['positive_rows']:,}행)" in text
    assert "날씨 14개 피처" in text and summary['weather_feature_count'] == 14
    assert f"{summary['full_join_10min_coverage']['both_matched']:,}행" in text
    assert f"미결합 {summary['full_join_10min_coverage']['none_matched']:,}행" in text
    assert summary['headline_latency_minutes'] == 10 and summary['headline_latency_is_measured'] is False
    assert sorted(summary['seeds']) == [1, 7, 42] and '42/1/7' in text


def test_readme_notebook_module_commands_have_entry_files():
    text = README.read_text(encoding='utf-8')
    modules = re.findall(r'\bpython(?:\.exe)?\s+(?:-u\s+)?-m\s+(notebooks\.[A-Za-z0-9_]+)', text)
    assert modules
    for module in modules:
        path = ROOT.joinpath(*module.split('.')).with_suffix('.py')
        assert path.is_file(), f'Missing module in README command: {module}'


def test_readme_classifier_table_matches_tracked_summary():
    text = README.read_text(encoding='utf-8')
    summary = json.loads((ROOT / (
        'output/baseline_recovery_v2_classifier_compare_20261008_summary.json'
    )).read_text(encoding='utf-8'))
    assert summary['smoke_only_not_evidence'] is False
    assert summary['population']['rows'] == 180332
    assert sorted(summary['seeds']) == [1, 7, 42]
    assert summary['lightgbm_reproduction_vs_20260922_weather_on']['all_exact'] is True
    names = {'lightgbm': 'LightGBM', 'logistic_regression': 'Logistic Regression',
             'random_forest': 'Random Forest'}
    for key, label in names.items():
        metrics = summary['models'][key]
        expected = (
            f"| {label} | {metrics['macro_f1_nested']['mean']:.6f} ± "
            f"{metrics['macro_f1_nested']['std']:.6f} | {metrics['log_loss']['mean']:.6f} | "
            f"{metrics['roc_auc']['mean']:.6f} |"
        )
        assert expected in text, f'Classifier table drift: {key}'
    deltas = summary['paired_deltas_model_minus_lightgbm']
    for key in ('logistic_regression', 'random_forest'):
        for metric, label in (('macro_f1_nested', 'Macro F1'), ('log_loss', 'LogLoss'),
                              ('roc_auc', 'ROC-AUC')):
            delta = deltas[key][metric]
            assert delta['sign_consistent_across_seeds'] is True, (key, metric)
            value = f"{delta['mean']:+.6f}".replace('-', '−')
            assert re.search(re.escape(label) + r'(?:는)? ' + re.escape(value), text), (key, metric)
    # Random Forest leads LightGBM on all three metrics in every seed.
    rf = deltas['random_forest']
    assert all(v > 0 for v in rf['macro_f1_nested']['values_by_seed'].values())
    assert all(v < 0 for v in rf['log_loss']['values_by_seed'].values())
    assert all(v > 0 for v in rf['roc_auc']['values_by_seed'].values())
    # The grid-edge caveat in the README is backed by the per-run selections.
    with (ROOT / 'output/baseline_recovery_v2_classifier_compare_20261008_runs.csv').open(
            encoding='utf-8', newline='') as handle:
        rf_params = [p for row in csv.DictReader(handle) if row['model'] == 'random_forest'
                     for p in json.loads(row['selected_params'])]
    assert len(rf_params) == 15
    assert max(summary['grids']['random_forest']['max_features_grid'],
               key=lambda v: -1 if v == 'sqrt' else v) == 0.5
    assert all(p['max_features'] == 0.5 for p in rf_params)
    assert '15개 폴드 모두 후보의 끝값(`max_features=0.5`)' in text
    # min_samples_leaf was also identical in every fold (a mid-grid value).
    assert all(p['min_samples_leaf'] == 25 for p in rf_params)
    assert '`min_samples_leaf=25`' in text


def _signed(value: float) -> str:
    return f'{value:+.6f}'.replace('-', '−')


def test_readme_classifier_tuning_numbers_match_tracked_summary():
    text = README.read_text(encoding='utf-8')
    section = text.split('<a id="classifier-tuning"></a>', 1)[1].split('\n### ', 1)[0]
    summary = json.loads((ROOT / (
        'output/baseline_recovery_v2_classifier_tuning_20261008_summary.json'
    )).read_text(encoding='utf-8'))
    assert summary['smoke_only_not_evidence'] is False
    assert summary['population']['rows'] == 180332
    assert sorted(summary['seeds']) == [1, 7, 42]
    assert summary['fold_fingerprints'] == summary['reference_fold_fingerprints_20261008']
    assert len(summary['grids']['lightgbm_tuned']['configurations']) == 24
    assert len(summary['grids']['random_forest_tuned']['configurations']) == 9
    assert '24개 설정' in section and '9개 설정' in section
    models = summary['models']
    for key, label in (('weather_on/lightgbm_tuned', 'LightGBM (24개 설정 탐색)'),
                       ('weather_on/random_forest_tuned', 'Random Forest (9개 설정 탐색)')):
        m = models[key]
        row = (f"| {label} | {m['macro_f1_nested']['mean']:.6f} | "
               f"{m['log_loss']['mean']:.6f} | {m['roc_auc']['mean']:.6f} |")
        assert row in section, key
    comps = {c['comparison']: c['metrics'] for c in summary['paired_comparisons']}
    metrics = ('macro_f1_nested', 'log_loss', 'roc_auc')
    rf_lgbm = comps['weather_on: random_forest_tuned - lightgbm_tuned']
    row = '| 차이 (Random Forest − LightGBM) | ' + ' | '.join(
        _signed(rf_lgbm[m]['mean']) for m in metrics) + ' |'
    assert row in section
    # "세 시드 모두 세 지표에서 Random Forest가 앞섰습니다" is backed per seed.
    assert all(rf_lgbm[m]['sign_consistent_across_seeds'] for m in metrics)
    assert all(v > 0 for v in rf_lgbm['macro_f1_nested']['values_by_seed'].values())
    assert all(v < 0 for v in rf_lgbm['log_loss']['values_by_seed'].values())
    assert all(v > 0 for v in rf_lgbm['roc_auc']['values_by_seed'].values())
    seed_f1 = rf_lgbm['macro_f1_nested']['values_by_seed'].values()
    assert f'+{min(seed_f1):.4f}~+{max(seed_f1):.4f}' in section
    # Tuned LightGBM vs its fixed 20261008 config: small, same direction in every seed.
    lgbm = comps['weather_on: lightgbm_tuned - lightgbm(20261008 fixed config)']
    assert all(lgbm[m]['sign_consistent_across_seeds'] for m in metrics)
    assert (f"Macro F1 {_signed(lgbm['macro_f1_nested']['mean'])}, "
            f"LogLoss {_signed(lgbm['log_loss']['mean'])}, "
            f"ROC-AUC {_signed(lgbm['roc_auc']['mean'])}") in section
    rf_old = comps['weather_on: random_forest_tuned - random_forest(20261008 6-config grid)']
    assert rf_old['macro_f1_nested']['sign_consistent_across_seeds'] is False
    assert f"Macro F1 {_signed(rf_old['macro_f1_nested']['mean'])}, 시드별 방향 혼재" in section
    # Random Forest weather on/off vs the existing LightGBM weather on/off delta. The
    # headline RF delta now comes from the identical-grid extension (checked in
    # test_readme_classifier_grid_extension_numbers_match_tracked_summary); the
    # 9-config delta stays quoted as the earlier, edge-bound reference.
    rf_w = comps['random_forest_tuned: weather_on - weather_off']
    lgbm_w = comps['lightgbm(20260922 fixed config): weather_on - weather_off']
    assert all(rf_w[m]['sign_consistent_across_seeds'] for m in metrics)
    assert ' / '.join(_signed(rf_w[m]['mean']) for m in metrics) in section
    assert '(' + ' / '.join(_signed(lgbm_w[m]['mean']) for m in metrics) + ')' in section
    # Grid-edge counts quoted in the README.
    sel = summary['selection']
    lg_edges = sel['weather_on/lightgbm_tuned']['edge_counts_by_axis']
    assert sel['weather_on/lightgbm_tuned']['n_folds'] == 15
    assert "14개에서 학습률 최솟값 0.03" in section and lg_edges['learning_rate']['low'] == 14
    assert "9개에서 잎 수 최댓값 127" in section and lg_edges['num_leaves']['high'] == 9
    rf_on = sel['weather_on/random_forest_tuned']['edge_counts_by_axis']
    assert rf_on['max_features'] == {'low': 11, 'interior': 4}
    assert '11개 폴드에서 `max_features` 최솟값 0.5(나머지 4개는 0.7)' in section
    rf_off = sel['weather_off/random_forest_tuned']['edge_counts_by_axis']
    assert rf_off['min_samples_leaf'] == {'high': 15}
    assert '15개 폴드 모두 `min_samples_leaf` 최댓값 50' in section


def _load(name: str) -> dict:
    return json.loads((ROOT / 'output' / name).read_text(encoding='utf-8'))


def _config_counts(selection: dict) -> list[tuple[dict, int]]:
    return [(json.loads(key), count) for key, count in selection['selected_config_counts'].items()]


def test_readme_classifier_grid_extension_numbers_match_tracked_summary():
    text = README.read_text(encoding='utf-8')
    section = text.split('<a id="classifier-grid-extension"></a>', 1)[1].split(
        '<a id="classifier-calibration"></a>', 1)[0]
    lgbm = _load('baseline_recovery_v2_classifier_grid_ext_20261008_lgbm_summary.json')
    rf = _load('baseline_recovery_v2_classifier_grid_ext_20261008_rf_summary.json')
    combined = _load('baseline_recovery_v2_classifier_grid_ext_20261008_combined_summary.json')
    tuning = _load('baseline_recovery_v2_classifier_tuning_20261008_summary.json')
    for part in (lgbm, rf):
        assert part['smoke_only_not_evidence'] is False
        assert part['population']['rows'] == 180332
        assert sorted(part['seeds']) == [1, 7, 42]
        assert part['fold_fingerprints'] == part['reference_fold_fingerprints_20261008']
    assert combined['smoke_only_not_evidence'] is False
    lg_grid = lgbm['grids']['lightgbm_ext']
    rf_grid = rf['grids']['random_forest_ext']
    assert lg_grid['n_configurations'] == 18 and rf_grid['n_configurations'] == 15
    assert '18개 설정' in section and '15개 설정' in section
    assert lg_grid['axes'] == {'learning_rate': [0.01, 0.02, 0.03], 'num_leaves': [127, 255, 511],
                               'min_child_samples': [20, 100]}
    assert lg_grid['max_trees_by_learning_rate'] == {'0.01': 1500, '0.02': 1000, '0.03': 600}
    assert rf_grid['axes'] == {'min_samples_leaf': [25, 50, 100, 200, 400],
                               'max_features': [0.5, 0.7, 1.0]}
    metrics = ('macro_f1_nested', 'log_loss', 'roc_auc')
    for label, m in (('LightGBM (18개 설정 탐색)', lgbm['models']['weather_on/lightgbm_ext']),
                     ('Random Forest (15개 설정 탐색)', rf['models']['weather_on/random_forest_ext'])):
        assert f"| {label} | " + ' | '.join(f"{m[k]['mean']:.6f}" for k in metrics) + ' |' in section
    # Random Forest still leads the extended LightGBM in every seed on all three metrics.
    gap = combined['paired_comparisons'][0]
    assert gap['comparison'] == 'weather_on: random_forest_ext - lightgbm_ext'
    gap = gap['metrics']
    assert '| 차이 (Random Forest − LightGBM) | ' + ' | '.join(
        _signed(gap[k]['mean']) for k in metrics) + ' |' in section
    assert all(gap[k]['sign_consistent_across_seeds'] for k in metrics)
    assert all(v > 0 for v in gap['macro_f1_nested']['values_by_seed'].values())
    assert all(v < 0 for v in gap['log_loss']['values_by_seed'].values())
    assert all(v > 0 for v in gap['roc_auc']['values_by_seed'].values())
    seed_f1 = gap['macro_f1_nested']['values_by_seed'].values()
    assert f'+{min(seed_f1):.4f}~+{max(seed_f1):.4f}' in section
    # Extended LightGBM vs the 24-config search: no consistent gain.
    lcomps = {c['comparison']: c['metrics'] for c in lgbm['paired_comparisons']}
    ext_vs_tuned = lcomps['weather_on: lightgbm_ext - lightgbm_tuned(20261008, 24 configs)']
    assert not any(ext_vs_tuned[k]['sign_consistent_across_seeds'] for k in metrics)
    assert (f"Macro F1 {_signed(ext_vs_tuned['macro_f1_nested']['mean'])}, "
            f"LogLoss {_signed(ext_vs_tuned['log_loss']['mean'])}, "
            f"ROC-AUC {_signed(ext_vs_tuned['roc_auc']['mean'])}") in section
    assert '시드별 방향이 섞였습니다' in section
    # Two learning-rate regimes, interior tree counts, and leaves 511 never chosen.
    sel = lgbm['selection']['weather_on/lightgbm_ext']
    counts = _config_counts(sel)
    assert sum(c for _, c in counts) == sel['n_folds'] == 15
    assert sum(c for p, c in counts if p['learning_rate'] == 0.03 and p['n_estimators'] == 150) == 8
    assert sum(c for p, c in counts if p['learning_rate'] == 0.01
               and 300 <= p['n_estimators'] <= 600) == 7
    assert '학습률 0.03·트리 150개(8개 폴드)와 학습률 0.01·트리 300~600개(7개 폴드)' in section
    assert sel['edge_counts_by_axis']['n_estimators'] == {'interior': 15}
    leaves = {n: sum(c for p, c in counts if p['num_leaves'] == n) for n in (127, 255, 511)}
    assert leaves == {127: 8, 255: 7, 511: 0}
    assert '127(8개 폴드)과 255(7개 폴드)' in section
    # max_depth=8 caps a tree at 256 leaves, so 255 and 511 score identically.
    assert lg_grid['base_params']['max_depth'] == 8 and '`max_depth=8`' in section
    with (ROOT / 'output/baseline_recovery_v2_classifier_grid_ext_20261008_lgbm_folds.csv').open(
            encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            scores = {(g['config']['learning_rate'], g['config']['min_child_samples'],
                       g['config']['num_leaves']): g['n_estimators_scores']
                      for g in json.loads(row['grid_scores'])}
            for (lr, mcs, leaves_n), s in scores.items():
                if leaves_n == 255:
                    assert s == scores[(lr, mcs, 511)]
    # Random Forest selections under the identical 15-config grid.
    rsel = rf['selection']
    on = _config_counts(rsel['weather_on/random_forest_ext'])
    assert all(p['min_samples_leaf'] == 25 for p, _ in on)
    assert rsel['weather_on/random_forest_ext']['edge_counts_by_axis']['min_samples_leaf'] == {'low': 15}
    tuned_on = _config_counts(tuning['selection']['weather_on/random_forest_tuned'])
    assert all(p['min_samples_leaf'] == 25 for p, _ in tuned_on)  # 10 was in that grid, never chosen
    assert '10을 함께 두었을 때도 15개 폴드 모두 25' in section
    off = _config_counts(rsel['weather_off/random_forest_ext'])
    assert sum(c for p, c in off if p['min_samples_leaf'] == 100) == 13
    assert sum(c for p, c in off if p['min_samples_leaf'] == 50) == 2
    assert rsel['weather_off/random_forest_ext']['edge_counts_by_axis']['min_samples_leaf'] == {'interior': 15}
    assert '13개에서 `min_samples_leaf=100`, 2개에서 50' in section
    rcomps = {c['comparison']: c['metrics'] for c in rf['paired_comparisons']}
    off_gain = rcomps['weather_off: random_forest_ext - random_forest_tuned(20261008, leaf 10/25/50)']
    assert off_gain['macro_f1_nested']['sign_consistent_across_seeds'] is True
    assert f"Macro F1 {_signed(off_gain['macro_f1_nested']['mean'])}로 세 시드 같은 방향" in section
    # RF weather on/off under the identical grid is the headline RF weather delta.
    weather = rcomps['random_forest_ext (identical 15-config grid): weather_on - weather_off']
    assert all(weather[k]['sign_consistent_across_seeds'] for k in metrics)
    tuning_section = text.split('<a id="classifier-tuning"></a>', 1)[1].split(
        '<a id="classifier-grid-extension"></a>', 1)[0]
    assert '똑같은 15개 설정 후보' in tuning_section
    assert (f"Macro F1 {_signed(weather['macro_f1_nested']['mean'])}, "
            f"LogLoss {_signed(weather['log_loss']['mean'])}, "
            f"ROC-AUC {_signed(weather['roc_auc']['mean'])}") in tuning_section
    # Still a bounded-budget comparison.
    assert '사전 선언한 유한 예산 안의 비교' in section
    assert '일반적으로 더 나은 알고리즘이라는 뜻으로 확장하지 않습니다' in section


CALIBRATION_ARMS_KO = {'none': '없음', 'platt_crossfit': 'Platt 교차적합',
                       'isotonic_crossfit': 'Isotonic 교차적합'}
CALIBRATION_MODELS = {'lightgbm_tuned': 'LightGBM', 'logistic_regression': 'Logistic Regression',
                      'random_forest_tuned': 'Random Forest'}


def test_readme_classifier_calibration_numbers_match_tracked_summary():
    text = README.read_text(encoding='utf-8')
    section = text.split('<a id="classifier-calibration"></a>', 1)[1].split('\n**기록 참고.**', 1)[0]
    summary = _load('baseline_recovery_v2_classifier_calibration_20261008_summary.json')
    assert summary['smoke_only_not_evidence'] is False
    assert summary['population']['rows'] == 180332
    assert sorted(summary['seeds']) == [1, 7, 42]
    assert summary['fold_fingerprints_equal_reference'] is True
    cal = summary['protocol']['probability_calibration']
    assert cal['outer_valid_labels_used_for_calibration_or_threshold'] is False
    assert summary['protocol']['outer_valid_labels_used_for_selection'] is False
    assert summary['protocol']['ece']['primary'].startswith('15 equal-frequency bins')
    levels, deltas = summary['levels'], summary['paired_deltas_arm_minus_none']
    for model, label in CALIBRATION_MODELS.items():
        for arm, arm_ko in CALIBRATION_ARMS_KO.items():
            m = levels[model][arm]
            row = (f"| {label} | {arm_ko} | {m['log_loss']['mean']:.6f} | "
                   f"{m['ece_ef15']['mean']:.6f} | {m['macro_f1_nested']['mean']:.6f} |")
            assert row in section, (model, arm)
    # LightGBM under-predicts most; Platt cross-fit improves LogLoss and ECE in every seed.
    biases = {m: levels[m]['none']['calibration_bias']['mean'] for m in CALIBRATION_MODELS}
    assert biases['lightgbm_tuned'] < 0
    assert max(abs(v) for v in biases.values()) == abs(biases['lightgbm_tuned'])
    assert f"약 {-biases['lightgbm_tuned']:.4f} 낮았고" in section
    assert biases['random_forest_tuned'] > 0
    assert f"과대 예측({biases['random_forest_tuned']:+.4f})" in section
    assert abs(biases['logistic_regression']) < 1e-4
    platt = deltas['lightgbm_tuned']['platt_crossfit']
    assert platt['log_loss']['sign_consistent_across_seeds'] and platt['log_loss']['mean'] < 0
    assert platt['ece_ef15']['sign_consistent_across_seeds'] and platt['ece_ef15']['mean'] < 0
    assert (f"LogLoss {_signed(platt['log_loss']['mean'])}, "
            f"ECE15 {_signed(platt['ece_ef15']['mean'])}") in section
    lr_platt = deltas['logistic_regression']['platt_crossfit']['ece_ef15']
    assert lr_platt['mean'] > 0 and lr_platt['sign_consistent_across_seeds']
    assert f"({_signed(lr_platt['mean'])})" in section
    # LogLoss and Macro F1 barely move.
    cross_ll = [deltas[m][a]['log_loss']['mean'] for m in CALIBRATION_MODELS
                for a in ('platt_crossfit', 'isotonic_crossfit')]
    assert f"{_signed(min(cross_ll))}~{_signed(max(cross_ll))}" in section
    all_f1 = [abs(d['macro_f1_nested']['mean']) for m in CALIBRATION_MODELS
              for d in deltas[m].values()]
    assert len(all_f1) == 12 and f"절댓값 {max(all_f1):.6f} 이하" in section
    # Ranking holds under every arm (sign-consistent across seeds).
    by_arm = summary['model_minus_lightgbm_by_arm']
    assert set(by_arm) == {'none', 'platt_inner_holdout', 'isotonic_inner_holdout',
                           'platt_crossfit', 'isotonic_crossfit'}
    for arm, models in by_arm.items():
        rf_gap, lr_gap = models['random_forest_tuned'], models['logistic_regression']
        for metric, sign in (('macro_f1_nested', 1), ('log_loss', -1), ('roc_auc', 1)):
            assert rf_gap[metric]['sign_consistent_across_seeds'], (arm, metric)
            assert all(v * sign > 0 for v in rf_gap[metric]['values_by_seed'].values()), (arm, metric)
        for metric, sign in (('macro_f1_nested', -1), ('log_loss', 1)):
            assert lr_gap[metric]['sign_consistent_across_seeds'], (arm, metric)
            assert all(v * sign > 0 for v in lr_gap[metric]['values_by_seed'].values()), (arm, metric)
    assert '분류기 순서는 어떤 보정에서도 같았습니다' in section
    # Isotonic fitted on the selection rows (inner holdout) worsens LogLoss for all three.
    iso = [deltas[m]['isotonic_inner_holdout']['log_loss'] for m in CALIBRATION_MODELS]
    assert all(d['mean'] > 0 and d['sign_consistent_across_seeds'] for d in iso)
    means = [d['mean'] for d in iso]
    assert f"{_signed(min(means))}~{_signed(max(means))}" in section
    # Reproduction of the recorded uncalibrated runs.
    units = summary['reproduction_of_recorded_uncalibrated_runs']['run_level']['by_unit']
    for unit, rec in units.items():
        assert rec['per_fold_thresholds_equal'] is True
        if not unit.startswith('logistic_regression/'):
            assert max(rec['abs_diff'].values()) < 1e-15, unit
    lr_max = max(rec['abs_diff']['macro_f1_nested'] for unit, rec in units.items()
                 if unit.startswith('logistic_regression/'))
    assert f"최대 {lr_max:.5f}" in section and '0.596816' in section
