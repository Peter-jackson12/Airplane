"""README presentation checks: tracked aggregate evidence only, no network."""
import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import pytest
from pathlib import Path

from scripts import build_readme_assets as builder

ROOT = Path(__file__).resolve().parents[1]
SVG_NS = '{http://www.w3.org/2000/svg}'
STORY_FIGURES = ('presentation_cover.svg', 'analysis_journey.svg', 'evaluation_boundary.svg')
DESIGN_REFERENCE = 'https://blog.google/intl/ko-kr/products/gemini-4-argon-kr/'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':')
    ).encode()).hexdigest()


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


def test_submission_readme_keeps_scope_and_completed_results_explicit():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    for anchor in ('submission-overview', 'weather-glossary', 'submission-completion',
                   'reliability-design', 'readme-figures', 'presentation-route'):
        assert f'<a id="{anchor}"></a>' in text
    for boundary in ('실시간 다운로드 모니터가 아닙니다', '신뢰구간이 아닙니다',
                     '날씨 모델의 성능 그래프가 아닙니다', '10분은 실측 공개 지연 시간이 아닙니다',
                     '원인 프로세스를 특정하지 못한', '중간에 pull하지 않습니다'):
        assert boundary in text
    assert text.count('```mermaid') >= 3
    assert '[그림 생성기](scripts/build_readme_assets.py)' in text


def test_readme_records_completed_weather_transport_join_and_model_comparison():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert '<a id="full-weather-transport"></a>' in text
    assert '<a id="full-weather-row-join"></a>' in text
    for evidence in (
        'output/baseline_recovery_v2_weather_full_bulk_20260921_full_weather_plan_manifest.json',
        'output/baseline_recovery_v2_weather_full_bulk_20260921_full_weather_fetch_manifest.json',
        'output/baseline_recovery_v2_weather_full_bulk_20260921_full_weather_fetch_audit.json',
        'output/baseline_recovery_v2_weather_full_join_20260922_full_weather_join_summary.json',
        'output/baseline_recovery_v2_weather_full_join_20260922_full_weather_join_manifest.json',
        'output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_summary.json',
        'output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_manifest.json',
    ):
        assert f'({evidence})' in text
    for value in ('1,317 / 27', '1,317 / 7,299,100', '1,093,058,768 bytes',
                  'HTTP 503 25건 + 중단 상태 불명(`interrupted_unknown`) 1건',
                  '689,457', '180,332', '0.598945', '+0.025035'):
        assert value in text
    assert '날씨 유무 동일조건 비교까지 완료' in text
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


def test_opening_summary_matches_all_frozen_weather_levels_and_signed_deltas():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    opening = text.split('<a id="1-목적과문제정의"></a>', 1)[0]
    for count in ('1,000,000행', '255,001행', '706,759행', '180,332행'):
        assert count in opening
    assert '`Delay`' in opening and '14개 피처' in opening
    assert '동일 평가행·동일 시드·동일 외부 폴드' in opening
    assert 'Macro F1 향상을 확인하지 못했습니다' in opening
    for row in builder.reviewed_data()['weather_model']:
        direction = '↑' if row['higher_is_better'] else '↓'
        delta = f"{row['delta_mean']:+.6f}".replace('-', '−')
        expected = (f"| {row['metric']} {direction} | {row['off_mean']:.6f} | "
                    f"**{row['on_mean']:.6f}** | **{delta}** |")
        assert expected in opening
    assert '실측 공개 지연 시간이 아닙니다' in opening
    assert '인과 효과나 미래 운항 성능으로 일반화하지 않습니다' in opening
    assert '신뢰구간이 아닙니다' in opening
    assert '<details>' not in opening


def test_current_prose_uses_korean_work_terms_without_translating_identifiers():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    body = text.split('<a id="weather-glossary"></a>', 1)[0]
    # Code, paths and explicit identifiers are not reader-facing prose.
    body = re.sub(r'```.*?```', '', body, flags=re.S)
    body = re.sub(r'`[^`]+`', '', body)
    body = re.sub(r'\]\([^)]+\)', ']', body)
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


def test_all_readme_artwork_uses_the_h_technical_launch_palette():
    from scripts import readme_editorial as style
    assert style.PAPER == '#FFFFFF'
    assert style.INK == '#202124'
    assert style.ACCENT == '#185ABC'
    assert style.RULE == '#DADCE0'
    filenames = [*builder.FIGURES.values(), *STORY_FIGURES]
    for filename in filenames:
        raw = (ROOT / 'assets/readme' / filename).read_text(encoding='utf-8')
        svg = ET.fromstring(raw)
        rectangles = svg.findall(SVG_NS + 'rect')
        assert rectangles and rectangles[0].attrib['fill'] == '#FFFFFF'
        assert svg.attrib['data-design'] == 'technical-launch-h'
        assert not any(node.tag.endswith(('linearGradient', 'radialGradient')) for node in svg.iter())
        assert '#202124' in raw and '#185ABC' in raw
        assert 'Noto Sans CJK KR' in raw and 'Noto Serif CJK KR' not in raw
        # Gray controls are shape encodings, never unreadable text labels.
        assert all(node.attrib.get('fill') != '#DADCE0' for node in svg.iter(SVG_NS + 'text'))
        for previous_color in ('#051766', '#004766', '#57BBEB', '#DDF1FB',
                               '#102D40', '#183B4E', '#007E80', '#EFF6F6', '#F5F2EB', '#A5442B'):
            assert previous_color not in raw


def test_figure_receipt_rejects_changed_editorial_renderer(tmp_path, monkeypatch):
    changed_renderer = tmp_path / 'readme_editorial.py'
    changed_renderer.write_text('# Changed drawing primitives\n', encoding='utf-8')
    monkeypatch.setattr(builder.editorial, '__file__', str(changed_renderer))
    with pytest.raises(ValueError, match='Editorial renderer changed'):
        builder.check_receipt()


def test_h_leads_with_bounded_result_and_readable_type():
    cover = ET.fromstring((ROOT / 'assets/readme/presentation_cover.svg').read_text())
    assert int(cover.attrib['width']) == 1600
    assert int(cover.attrib['height']) == 1000
    content = ''.join(cover.itertext())
    for value in ('+0.025035', '180,332행', '정적 교차검증', '미래 운항 성능은 미검증'):
        assert value in content
    for filename in [*builder.FIGURES.values(), 'presentation_cover.svg',
                     'analysis_journey.svg', 'evaluation_boundary.svg']:
        svg = ET.fromstring((ROOT / 'assets/readme' / filename).read_text())
        labels = svg.findall('{http://www.w3.org/2000/svg}text')
        assert all(float(n.attrib['font-size']) >= 28 for n in labels), filename
    journey = (ROOT / 'assets/readme/analysis_journey.svg').read_text()
    assert '도형 크기는 수량을 뜻하지 않습니다.' in journey


def test_h_readme_credits_reference_preserves_styles_and_uses_its_branch_ci():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    opening = text.split('<a id="1-목적과문제정의"></a>', 1)[0]
    assert 'H안' in opening
    assert DESIGN_REFERENCE in text
    assert '[A안](https://github.com/Peter-jackson12/Airplane/tree/style/a-editorial)' in text
    for branch in ('design/gigi-sky-gold', 'design/gigi-flight-magazine-c',
                   'design/korean-air-palette-d', 'design/reference-report-e',
                   'design/refined-evidence-f'):
        assert f'https://github.com/Peter-jackson12/Airplane/tree/{branch}' in text
    assert 'badge.svg?branch=design%2Ftechnical-launch-h' in opening
    assert 'ci.yml?query=branch%3Adesign%2Ftechnical-launch-h' in opening
    assert 'badge.svg?branch=master' not in opening
    assert '본문 설명 약 16분 + 전환·질문 여유 약 3분' in text
    design_doc = (ROOT / 'assets/readme/DESIGN.md').read_text(encoding='utf-8')
    assert DESIGN_REFERENCE in design_doc
    for token in ('technical-launch-h', '#202124', '#185ABC', '#DADCE0'):
        assert token in design_doc


def test_h_preserves_all_19_readme_code_blocks_byte_for_byte():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    blocks = re.findall(r'```[^\n]*\n.*?```', text, re.S)
    assert len(blocks) == 19
    assert hashlib.sha256(json.dumps(blocks, ensure_ascii=False).encode()).hexdigest() == 'c8e177b591f64156f5275a938294c77773d5e98125ff7b855fee20a56c58267c'


def test_h_quantitative_figures_use_five_distinct_chart_geometries():
    chart_types = set()
    for filename in builder.FIGURES.values():
        svg = ET.fromstring((ROOT / 'assets/readme' / filename).read_text())
        assert svg.attrib['data-design'] == 'technical-launch-h'
        chart_types.add(svg.attrib['data-chart-type'])
    assert len(chart_types) == 5


def test_h_technical_launch_palette_text_contrast_is_at_least_aa():
    from scripts import readme_editorial as style
    def luminance(color):
        channels = [int(color[i:i+2], 16) / 255 for i in (1, 3, 5)]
        channels = [v / 12.92 if v <= 0.04045 else ((v + .055) / 1.055) ** 2.4 for v in channels]
        return sum(v * w for v, w in zip(channels, (.2126, .7152, .0722)))
    pairs = [(style.INK, bg) for bg in (style.PAPER, style.SKY, style.CYAN)]
    pairs += [(style.ACCENT, bg) for bg in (style.PAPER, style.SKY, style.CYAN)]
    pairs += [(style.MUTED, bg) for bg in (style.PAPER, style.SKY)]
    pairs += [(style.PAPER, style.INK), (style.PAPER, style.ACCENT), (style.SKY, style.INK)]
    for foreground, background in pairs:
        a, b = sorted((luminance(foreground), luminance(background)))
        assert (b + .05) / (a + .05) >= 4.5, (foreground, background)


def test_h_preserves_all_eight_local_images_exactly_once():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    images = re.findall(r'!\[[^\]]+\]\((assets/readme/[^)]+)\)', text)
    expected = [f'assets/readme/{name}' for name in [*builder.FIGURES.values(), *STORY_FIGURES]]
    assert len(images) == 8
    assert sorted(images) == sorted(expected)


def test_h_keeps_the_frozen_evidence_sources_filters_and_exact_values():
    # Captured from the reviewed pre-H aggregate receipt, before visual changes.
    # These contracts are independent of the generator's own implementation:
    # changing a renderer and its receipt together cannot change the evidence.
    receipt = builder.check_receipt()
    expected_hashes = {
        'sources': 'c8f4c8450e4e67501cbd895f9986d29fb45ffd7c8789ebf5f904a5a2f43a8925',
        'filters': '407bbb75126489247734380f69604b95dcc6b852395be76b38f8b97c9be500f7',
        'reviewed_data': 'a71d41a0d130de95f698c69d51ceb8ce680cc5b1a23cf7ec5a543fa350749020',
    }
    for field, expected in expected_hashes.items():
        assert canonical_hash(receipt[field]) == expected, f'Frozen H evidence drift: {field}'
    assert canonical_hash(builder.reviewed_data()) == expected_hashes['reviewed_data']


def test_h_benchmark_table_retains_all_four_conditions_and_highlights_focus_column():
    svg = ET.fromstring((ROOT / 'assets/readme/model_comparison.svg').read_text(encoding='utf-8'))
    text = ' '.join(svg.itertext())
    for row in builder.reviewed_data()['models']:
        assert row['phase'] in text
        assert f"{row['mean']:.6f}" in text
        assert f"{row['sd']:.6f}" in text
    assert '255,001행' in text and '신뢰구간이 아닙니다' in text
    highlights = [node for node in svg.iter() if node.attrib.get('data-focus-column') == 'P6_clean']
    assert len(highlights) == 1
    focus = highlights[0]
    assert focus.tag == SVG_NS + 'rect' and focus.attrib['fill'] == '#E8F0FE'
    headings = [node for node in svg.iter(SVG_NS + 'text') if node.text == 'P6_clean']
    assert len(headings) == 1 and headings[0].attrib['fill'] == '#185ABC'
    x, y = float(headings[0].attrib['x']), float(headings[0].attrib['y'])
    assert float(focus.attrib['x']) < x < float(focus.attrib['x']) + float(focus.attrib['width'])
    assert float(focus.attrib['y']) < y < float(focus.attrib['y']) + float(focus.attrib['height'])


def test_h_weather_bars_encode_original_levels_from_true_zero_baselines():
    svg = ET.fromstring((ROOT / 'assets/readme/weather_model_comparison.svg').read_text(encoding='utf-8'))
    bars = [node for node in svg.iter(SVG_NS + 'rect') if 'data-condition' in node.attrib]
    assert len(bars) == 6
    expected = {(row['metric'], condition): row[field]
                for row in builder.reviewed_data()['weather_model']
                for condition, field in (('weather_off', 'off_mean'), ('weather_on', 'on_mean'))}
    actual = {}
    for bar in bars:
        attrs = bar.attrib
        key = (attrs['data-metric'], attrs['data-condition'])
        assert key in expected and key not in actual
        actual[key] = attrs
        value = float(attrs['data-value'])
        assert value == pytest.approx(expected[key], rel=0, abs=1e-12)
        assert float(attrs['data-axis-min']) == 0
        maximum = float(attrs['data-axis-max'])
        length = float(attrs['data-axis-length'])
        baseline = float(attrs['data-baseline'])
        assert maximum >= value > 0 and length > 0
        assert attrs['fill'] == ('#DADCE0' if key[1] == 'weather_off' else '#185ABC')
        if attrs['data-orientation'] == 'vertical':
            extent = float(attrs['height'])
            assert float(attrs['y']) + extent == pytest.approx(baseline, abs=.002)
        else:
            assert attrs['data-orientation'] == 'horizontal'
            extent = float(attrs['width'])
            assert float(attrs['x']) == pytest.approx(baseline, abs=.002)
        assert extent == pytest.approx(length * value / maximum, abs=.002)
    assert set(actual) == set(expected)
    for metric, _ in expected:
        off, on = actual[metric, 'weather_off'], actual[metric, 'weather_on']
        for field in ('data-axis-min', 'data-axis-max', 'data-axis-length',
                      'data-baseline', 'data-orientation'):
            assert off[field] == on[field], (metric, field)


def test_h_preserves_all_137_existing_aggregate_evidence_destinations():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    destinations = re.findall(r'!?\[[^\]\n]*\]\((output/[^\s)]+)\)', text)
    unique = sorted(set(destinations))
    assert len(unique) == 137
    assert canonical_hash(unique) == '6345edc5f985d27e886e78b9ed73484251626dc8ae75a975f1d9ccd036ba286a'
