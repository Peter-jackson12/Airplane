"""README presentation checks: tracked aggregate evidence only, no network."""
import ast
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import pytest
from pathlib import Path

from scripts import build_readme_assets as builder

ROOT = Path(__file__).resolve().parents[1]


def test_readme_figure_receipt_matches_sources_filters_and_hashes():
    receipt = builder.check_receipt()
    assert set(receipt['figures']) == {'models', 'calibration', 'attribution', 'latency', 'weather_model'}
    assert len(receipt['sources']) == 5
    for source in receipt['sources'].values():
        assert source['path'].startswith('output/')
        assert len(source['sha256']) == 64


def test_readme_check_is_read_only():
    paths = [ROOT / 'README.md', *sorted((ROOT / 'assets/readme').glob('*'))]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    builder.check_receipt()
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


def test_readme_figures_are_local_svg_and_have_descriptive_alt_text():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    receipt = builder.check_receipt()
    for figure in receipt['figures'].values():
        path = figure['path']
        matches = re.findall(r'!\[([^\]]+)\]\(' + re.escape(path) + r'\)', text)
        assert len(matches) == 1 and len(matches[0]) > 20, path
        svg = ET.fromstring((ROOT / path).read_text(encoding='utf-8'))
        assert svg.tag.endswith('svg') and 'viewBox' in svg.attrib
        assert not any(node.tag.endswith('script') for node in svg.iter())
        for node in svg.iter():
            for key, value in node.attrib.items():
                if key.endswith('href'):
                    assert value.startswith('#'), f'External SVG resource: {value}'


def test_date_attribution_plot_has_disjoint_complete_population():
    rows = builder.reviewed_data()['attribution']
    assert [r['rows'] for r in rows] == [706759, 291308, 1933]
    assert sum(r['rows'] for r in rows) == 1000000


def test_latency_plot_preserves_scenario_denominator_and_numerators():
    rows = builder.reviewed_data()['latency']
    assert len(rows) == 8
    for row in rows:
        assert row['eligible_rows'] == 268
        assert abs(row['match_rate_percent'] - row['matched_rows'] / 268 * 100) < 1e-10
    by_key = {(r['role'], r['latency_minutes']): r for r in rows}
    assert by_key['origin', 60]['matched_rows'] == 150
    assert by_key['destination', 60]['matched_rows'] == 157

def test_weather_model_plot_uses_frozen_paired_submission_result():
    rows = builder.reviewed_data()['weather_model']
    assert [r['metric'] for r in rows] == ['Macro F1', 'LogLoss', 'ROC-AUC']
    expected = [
        (0.5739098053816244, 0.5989451594849345, 0.025035354103310186, True),
        (0.4481163223494364, 0.4366887714640056, -0.011427550885430774, False),
        (0.6406182693684818, 0.6729355261382809, 0.032317256769799164, True),
    ]
    for row, (off, on, delta, higher) in zip(rows, expected):
        assert row['off_mean'] == pytest.approx(off)
        assert row['on_mean'] == pytest.approx(on)
        assert row['delta_mean'] == pytest.approx(delta)
        assert row['higher_is_better'] is higher
        assert row['definition'] == 'weather_on - weather_off'
        assert row['sd'] > 0 and row['off_sd'] > 0 and row['on_sd'] > 0
        assert set(row['deltas_by_seed']) == {'42', '1', '7'}
        assert all(v > 0 if higher else v < 0 for v in row['deltas_by_seed'].values())


PORTFOLIO_ANCHORS = ('overview', 'results', 'engineering', 'limits', 'reproduce', 'docs',
                     'full-weather-row-join', 'full-weather-transport', 'join-reproduction',
                     'evidence-appendix')


def test_portfolio_readme_keeps_sections_scope_and_figure_sources_explicit():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    for anchor in PORTFOLIO_ANCHORS:
        assert text.count(f'<a id="{anchor}"></a>') == 1, anchor
    for heading in ('## 1. 프로젝트 한눈에', '## 2. 핵심 결과', '## 3. 문제 해결 과정과 기술적 의사결정',
                    '## 4. 한계와 다음 단계', '## 5. 재현 방법', '## 6. 문서·근거 안내'):
        assert heading in text, heading
    for boundary in ('신뢰구간이 아닙니다', '날씨 모델의 성능이 아닙니다',
                     '10분은 실측 공개 지연 시간이 아닙니다', '인과 효과의 증명이 아닙니다',
                     '서로 다른 평가 집단'):
        assert boundary in text, boundary
    assert '(assets/readme/sources.json)' in text
    assert 'scripts/build_readme_assets.py --check' in text


def test_readme_records_completed_weather_transport_join_and_model_comparison():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    for evidence in (
        'output/baseline_recovery_v2_weather_full_bulk_20260921_full_weather_plan_manifest.json',
        'output/baseline_recovery_v2_weather_full_bulk_20260921_full_weather_fetch_manifest.json',
        'output/baseline_recovery_v2_weather_full_bulk_20260921_full_weather_fetch_audit.json',
        'output/baseline_recovery_v2_weather_full_join_20260922_full_weather_join_summary.json',
        'output/baseline_recovery_v2_weather_full_join_20260922_full_weather_join_manifest.json',
        'output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_summary.json',
        'output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_manifest.json',
        'output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_runs.csv',
    ):
        assert f'({evidence})' in text
    for value in ('1,317개 요청 묶음, 27개 분할', '7,299,100', '1,093,058,768 bytes',
                  'HTTP 503 25건 + 중단 상태 불명 1건', '689,457', '180,332', '0.598945', '+0.025035'):
        assert value in text
    assert '정적 교차검증' in text
    assert '미래 운항 성능으로 일반화하지 않습니다' in text


def test_figure_builder_does_not_import_collection_or_network_code():
    tree = ast.parse((ROOT / 'scripts/build_readme_assets.py').read_text(encoding='utf-8'))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split('.')[0])
    assert not imports.intersection({'requests', 'httpx', 'urllib', 'socket', 'subprocess', 'notebooks', 'src'})


def test_headline_results_match_all_frozen_weather_levels_and_signed_deltas():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    overview = text.split('<a id="results"></a>', 1)[0]
    for count in ('1,000,000행', '255,001행', '706,759행', '180,332행'):
        assert count in overview
    assert '`Delay`' in overview
    # The headline result and its evaluation scope are visible before any fold.
    headline = text.split('<a id="results"></a>', 1)[1].split('<details>', 1)[0]
    assert '동일 평가행·동일 시드(42/1/7)·동일 외부 5폴드' in headline
    assert 'Macro F1 향상을 확인하지 못했' in headline
    for row in builder.reviewed_data()['weather_model']:
        direction = '↑' if row['higher_is_better'] else '↓'
        delta = f"{row['delta_mean']:+.6f}".replace('-', '−')
        expected = (f"| {row['metric']} {direction} | {row['off_mean']:.6f} | "
                    f"**{row['on_mean']:.6f}** | **{delta}** |")
        assert expected in headline
    assert '10분은 실측 공개 지연 시간이 아닙니다' in headline
    assert '인과 효과나 미래 운항 성능으로 일반화하지 않습니다' in headline
    assert '신뢰구간이 아닙니다' in headline


def _weather_runs():
    path = ROOT / 'output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_runs.csv'
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row['condition'], []).append(row)
    assert set(by_condition) == {'weather_off', 'weather_on'}
    assert all(len(v) == 3 and {r['seed'] for r in v} == {'42', '1', '7'} for v in by_condition.values())
    return by_condition


def test_class_metrics_are_derived_from_tracked_confusion_matrices():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    runs = _weather_runs()

    def mean_metric(rows, metric):
        values = []
        for r in rows:
            tn, fp, fn, tp = (int(r[k]) for k in ('tn', 'fp', 'fn', 'tp'))
            assert tn + fp + fn + tp == int(r['n_rows']) == 180332
            values.append(metric(tn, fp, fn, tp))
        return sum(values) / len(values)

    metrics = {
        'Delayed precision': lambda tn, fp, fn, tp: tp / (tp + fp),
        'Delayed recall': lambda tn, fp, fn, tp: tp / (tp + fn),
        'Delayed F1': lambda tn, fp, fn, tp: 2 * tp / (2 * tp + fp + fn),
        'Not_Delayed F1': lambda tn, fp, fn, tp: 2 * tn / (2 * tn + fn + fp),
    }
    for name, metric in metrics.items():
        off = mean_metric(runs['weather_off'], metric)
        on = mean_metric(runs['weather_on'], metric)
        assert f'| {name} | {off:.6f} | {on:.6f} |' in text, name
    precision = [mean_metric(runs[c], metrics['Delayed precision']) for c in ('weather_off', 'weather_on')]
    recall = [mean_metric(runs[c], metrics['Delayed recall']) for c in ('weather_off', 'weather_on')]
    assert f'약 {precision[0]:.2%}→{precision[1]:.2%}' in text
    assert f'약 {recall[0]:.2%}→{recall[1]:.2%}(+{(recall[1] - recall[0]) * 100:.2f}%p)' in text
    # Interpretation agreed with tutor feedback: precision moved, recall barely did.
    assert '지연 경보의 정확성이 개선됐다' in text
    assert '지연을 훨씬 많이 잡아낸 것이 아닙니다' in text


def test_selected_thresholds_and_tree_counts_match_tracked_runs():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    summary = {}
    for condition, rows in _weather_runs().items():
        thresholds = [t for r in rows for t in json.loads(r['per_fold_thresholds'])]
        trees = [t for r in rows for t in json.loads(r['selected_n_estimators'])]
        assert len(thresholds) == len(trees) == 15
        summary[condition] = (min(thresholds), max(thresholds), min(trees), max(trees))
    off, on = summary['weather_off'], summary['weather_on']
    assert off[2] == off[3]
    assert (f'임계값은 미사용 {off[0]:.2f}~{off[1]:.2f}, 사용 {on[0]:.2f}~{on[1]:.2f}, '
            f'트리 수는 미사용 {off[2]}, 사용 {on[2]}~{on[3]}') in text
    assert '3시드 × 5폴드 = 15개 선택값' in text


def test_current_prose_uses_korean_work_terms_without_translating_identifiers():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    # Code, paths, anchors and explicit identifiers are not reader-facing prose.
    body = re.sub(r'```.*?```', '', text, flags=re.S)
    body = re.sub(r'`[^`]+`', '', body)
    body = re.sub(r'\]\([^)]+\)', ']', body)
    body = re.sub(r'<a id="[^"]+"></a>', '', body)
    for term in ('shard', 'request group', 'weather-off', 'weather-on', 'paired',
                 'transport', 'checkpoint', 'manifest', 'latency', 'finalize',
                 'outer', 'inner', 'resume', 'cache-only', 'fail-closed'):
        assert not re.search(r'\b' + re.escape(term) + r'\b', body, re.I), term
    for identifier in ('P4_clean', 'P6_fixed', 'P6_clean', 'LightGBM', 'Macro F1',
                       'LogLoss', 'ROC-AUC', 'OOF', 'TE'):
        assert identifier in text


def test_weather_svg_shows_original_levels_signed_changes_and_distinct_scales():
    svg = ET.fromstring((ROOT / 'assets/readme/weather_model_comparison.svg').read_text(encoding='utf-8'))
    text = ' '.join(svg.itertext())
    for row in builder.reviewed_data()['weather_model']:
        assert row['metric'] in text
        assert f"{row['off_mean']:.6f}" in text
        assert f"{row['on_mean']:.6f}" in text
        assert f"{row['delta_mean']:+.6f}" in text
        assert f"{row['sd']:.6f}" in text
    for phrase in ('180,332행', '10분 공개 지연 가정', '날씨 미사용', '날씨 사용',
                   '낮을수록 좋음', '척도가 달라', '신뢰구간이 아닙니다'):
        assert phrase in text


def test_all_five_svg_have_korean_accessible_titles_and_no_external_font_dependency():
    for filename in builder.FIGURES.values():
        svg = ET.fromstring((ROOT / 'assets/readme' / filename).read_text(encoding='utf-8'))
        assert svg.attrib['role'] == 'img'
        assert svg.attrib['aria-labelledby'] == 'title desc'
        title = svg.find('{http://www.w3.org/2000/svg}title')
        desc = svg.find('{http://www.w3.org/2000/svg}desc')
        assert title is not None and re.search('[가-힣]', title.text)
        assert desc is not None and len(desc.text) > 20
        raw = ET.tostring(svg, encoding='unicode')
        assert '@import' not in raw and '@font-face' not in raw and 'url(' not in raw


@pytest.mark.parametrize('field', ['filters', 'sources', 'figures'])
def test_figure_receipt_rejects_changed_filter_or_asset_contract(tmp_path, monkeypatch, field):
    receipt = builder.check_receipt()
    if field == 'filters':
        receipt[field]['calibration']['group'] = 'different_population'
    else:
        receipt[field].pop(next(iter(receipt[field])))
    (tmp_path / 'sources.json').write_text(json.dumps(receipt), encoding='utf-8')
    monkeypatch.setattr(builder, 'ASSETS', tmp_path)
    with pytest.raises(ValueError, match='changed'):
        builder.check_receipt()


def test_all_five_figures_and_receipt_rebuild_byte_for_byte(tmp_path, monkeypatch):
    # Only the five tracked aggregate inputs are copied. No raw data,
    # model or network entrypoint is available in this build sandbox.
    for relative in builder.SOURCES.values():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    output = tmp_path / 'assets/readme'
    monkeypatch.setattr(builder, 'ROOT', tmp_path)
    monkeypatch.setattr(builder, 'ASSETS', output)
    builder.build()
    builder.check_receipt()
    for name in [*builder.FIGURES.values(), 'sources.json']:
        assert (output / name).read_bytes() == (ROOT / 'assets/readme' / name).read_bytes(), name


def test_presentation_story_artwork_is_accessible_and_local():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    for name in ('presentation_cover.svg', 'analysis_journey.svg', 'evaluation_boundary.svg'):
        path = f'assets/readme/{name}'
        matches = re.findall(r'!\[([^\]]+)\]\(' + re.escape(path) + r'\)', text)
        assert len(matches) == 1 and len(matches[0]) > 20
        raw = (ROOT / path).read_text(encoding='utf-8')
        svg = ET.fromstring(raw)
        assert svg.attrib['role'] == 'img'
        assert svg.attrib['aria-labelledby'] == 'title desc'
        assert svg.find('{http://www.w3.org/2000/svg}title') is not None
        assert svg.find('{http://www.w3.org/2000/svg}desc') is not None
        assert not any(n.tag.endswith('script') or n.tag.endswith('image') for n in svg.iter())
        assert '@import' not in raw and '@font-face' not in raw


def test_presentation_story_artwork_rebuilds_byte_for_byte(tmp_path, monkeypatch):
    from scripts import build_readme_story as story
    monkeypatch.setattr(story, 'OUT', tmp_path)
    story.build()
    for name in ('presentation_cover.svg', 'analysis_journey.svg', 'evaluation_boundary.svg'):
        assert (tmp_path / name).read_bytes() == (ROOT / 'assets/readme' / name).read_bytes()


def test_presentation_story_retains_population_and_evaluation_boundaries():
    journey = (ROOT / 'assets/readme/analysis_journey.svg').read_text(encoding='utf-8')
    for phrase in ('1,000,000행', '255,001행', '706,759행', '180,332행', '14개 피처',
                   '같은 외부 폴드', '인과 효과와 미래 운항 성능은 미검증'):
        assert phrase in journey
    evaluation = (ROOT / 'assets/readme/evaluation_boundary.svg').read_text(encoding='utf-8')
    for phrase in ('내부 학습 / 80%', '내부 검증 / 20%', '최종 채점 전용',
                   '외부 검증으로 선택하지 않음', '라벨 누수 방지와 미래 시점'):
        assert phrase in evaluation


def test_all_readme_artwork_retains_flat_sky_gold_palette():
    filenames = [*builder.FIGURES.values(), 'presentation_cover.svg',
                 'analysis_journey.svg', 'evaluation_boundary.svg']
    for filename in filenames:
        raw = (ROOT / 'assets/readme' / filename).read_text(encoding='utf-8')
        svg = ET.fromstring(raw)
        rectangles = svg.findall('{http://www.w3.org/2000/svg}rect')
        assert rectangles and rectangles[0].attrib['fill'] == '#FBF8F0'
        assert all(float(rect.attrib.get('rx', 0)) == 0 for rect in rectangles)
        assert not any(node.tag.endswith(('linearGradient', 'radialGradient')) for node in svg.iter())
        assert '#304750' in raw and 'Noto Sans CJK KR' in raw
        if filename != 'presentation_cover.svg':
            assert '#356A86' in raw
        assert 'Noto Serif CJK KR' not in raw
        for previous_color in ('#102D40', '#183B4E', '#007E80', '#EFF6F6', '#F5F2EB', '#A5442B'):
            assert previous_color not in raw


def test_figure_receipt_rejects_changed_editorial_renderer(tmp_path, monkeypatch):
    changed_renderer = tmp_path / 'readme_editorial.py'
    changed_renderer.write_text('# Changed drawing primitives\n', encoding='utf-8')
    monkeypatch.setattr(builder.editorial, '__file__', str(changed_renderer))
    with pytest.raises(ValueError, match='Editorial renderer changed'):
        builder.check_receipt()


def test_magazine_c_uses_oversized_type_and_asymmetric_boarding_passes():
    cover = ET.fromstring((ROOT / 'assets/readme/presentation_cover.svg').read_text())
    assert max(float(n.attrib.get('font-size', 0)) for n in cover.iter()) >= 200
    assert 'THE' in ''.join(cover.itertext()) and 'WEATHER' in ''.join(cover.itertext())
    journey = ET.fromstring((ROOT / 'assets/readme/analysis_journey.svg').read_text())
    rects = journey.findall('{http://www.w3.org/2000/svg}rect')
    assert any(r.attrib.get('x') == '60' and r.attrib.get('fill') == '#E4F0F5' for r in rects)
    assert any(r.attrib.get('x') == '159' and r.attrib.get('fill') == '#D5BD8F' for r in rects)
    assert '도형 크기는 수량을 뜻하지 않습니다.' in ''.join(journey.itertext())


def test_portfolio_readme_drops_presentation_only_material():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert 'badge.svg?branch=master' in text
    assert 'actions/workflows/ci.yml' in text
    for presentation_only in ('본문 설명 약 16분', '낭독', '발표를 마무리하는', '발표할 때는',
                              '메인 디자인 · C안', '읽기 패스', '질문 대비'):
        assert presentation_only not in text, presentation_only
    for design_branch in ('style/a-editorial', 'design/gigi-sky-gold',
                          'design/korean-air-palette-d', 'design/refined-evidence-f'):
        assert design_branch not in text, design_branch
    # The README stays skimmable: substantially shorter than the presentation version.
    assert len(text.splitlines()) <= 400


def test_readme_code_blocks_reference_existing_entrypoints_and_locked_environment():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    blocks = re.findall(r'```[^\n]*\n(.*?)```', text, re.S)
    assert blocks
    commands = [line.strip() for block in blocks for line in block.splitlines()
                if line.strip() and not line.strip().startswith('#')]
    assert 'uv sync --locked --python 3.14 --dev' in commands
    assert 'uv run --locked --offline python -m pytest -q' in commands
    scripts = []
    for command in commands:
        assert command.startswith(('uv sync --locked', 'uv run --locked --offline ')), command
        scripts += re.findall(r'\bpython(?:\s+-u)?\s+([\w/]+\.py)\b', command)
    assert scripts
    for script in scripts:
        assert (ROOT / script).is_file(), script
    # The documented default test selection is the one CI runs.
    ci = (ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8')
    pyproject = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    assert '-m "not local_data"' in ci and "-m 'not local_data'" in pyproject
    assert 'requires-python = ">=3.14"' in pyproject and 'python-version: "3.14"' in ci


def test_magazine_c_quantitative_figures_use_distinct_chart_geometries():
    chart_types = set()
    for filename in builder.FIGURES.values():
        svg = ET.fromstring((ROOT / 'assets/readme' / filename).read_text())
        assert svg.attrib['data-design'] == 'flight-magazine-c'
        chart_types.add(svg.attrib['data-chart-type'])
    assert len(chart_types) == 5
