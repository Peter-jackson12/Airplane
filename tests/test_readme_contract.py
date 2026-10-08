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
