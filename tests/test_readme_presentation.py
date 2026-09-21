"""README presentation checks: tracked aggregate evidence only, no network."""
import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts import build_readme_assets as builder

ROOT = Path(__file__).resolve().parents[1]


def test_readme_figure_receipt_matches_sources_filters_and_hashes():
    receipt = builder.check_receipt()
    assert set(receipt['figures']) == {'models', 'calibration', 'attribution', 'latency'}
    assert len(receipt['sources']) == 4
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


def test_submission_readme_keeps_scope_and_future_results_explicit():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    for anchor in ('submission-overview', 'weather-glossary', 'submission-completion',
                   'reliability-design', 'readme-figures', 'presentation-route'):
        assert f'<a id="{anchor}"></a>' in text
    for boundary in ('실시간 다운로드 모니터가 아닙니다', '결과 미반영',
                     '신뢰구간이 아닙니다', '날씨 모델의 성능 그래프가 아닙니다',
                     '원인 프로세스를 특정하지 못한', '중간에 pull하지 않습니다'):
        assert boundary in text
    assert text.count('```mermaid') >= 3
    assert '[그림 생성기](scripts/build_readme_assets.py)' in text


def test_figure_builder_does_not_import_collection_or_network_code():
    tree = ast.parse((ROOT / 'scripts/build_readme_assets.py').read_text(encoding='utf-8'))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split('.')[0])
    assert not imports.intersection({'requests', 'httpx', 'urllib', 'socket', 'subprocess', 'notebooks', 'src'})
