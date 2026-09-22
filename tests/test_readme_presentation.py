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
    # Protect the actual comparison, not the old chart's positive-only delta format.
    expected = [
        (0.5739098053816244, 0.5989451594849345, 0.025035354103310186, 'higher'),
        (0.4481163223494364, 0.4366887714640056, -0.011427550885430774, 'lower'),
        (0.6406182693684818, 0.6729355261382809, 0.032317256769799164, 'higher'),
    ]
    for row, (off, on, delta, direction) in zip(rows, expected):
        assert row['off_mean'] == pytest.approx(off, abs=1e-14)
        assert row['on_mean'] == pytest.approx(on, abs=1e-14)
        assert row['delta_mean'] == pytest.approx(delta, abs=1e-14)
        assert row['definition'] == 'weather_on - weather_off'
        assert row['direction'] == direction
        assert all(row[key] >= 0 for key in ('off_sd', 'on_sd', 'delta_sd'))
        assert set(row['delta_by_seed']) == {'42', '1', '7'}
        sign = 1 if direction == 'higher' else -1
        assert all(sign * value > 0 for value in row['delta_by_seed'].values())


def test_submission_readme_keeps_scope_and_completed_results_explicit():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    for anchor in ('submission-overview', 'weather-glossary', 'submission-completion',
                   'reliability-design', 'readme-figures', 'presentation-route'):
        assert f'<a id="{anchor}"></a>' in text
    for boundary in ('실시간 다운로드 모니터가 아닙니다', '신뢰구간이 아닙니다',
                     '날씨 모델의 성능 그래프가 아닙니다', '10분은 실측 날씨 공개 지연 시간이 아닙니다',
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


def test_weather_svg_displays_source_levels_spreads_and_signed_deltas():
    svg = ET.parse(ROOT / 'assets/readme/weather_model_comparison.svg')
    text = '\n'.join(svg.getroot().itertext())
    for label in ('날씨 미사용', '날씨 사용', '사용−미사용', '신뢰구간 아님',
                  '공통 길이'):
        assert label in text
    for row in builder.reviewed_data()['weather_model']:
        assert row['metric'] in text
        for key in ('off_mean', 'on_mean'):
            assert f"{row[key]:.6f}" in text
        for key in ('off_sd', 'on_sd', 'delta_sd'):
            assert f"± {row[key]:.6f}" in text
        assert f"{row['delta_mean']:+.6f}".replace('-', '−') in text


def test_asset_rebuild_is_deterministic_and_preserves_source_bytes(tmp_path, monkeypatch):
    import shutil

    target = tmp_path / 'repo'
    target.mkdir()
    before = {}
    for relative in builder.SOURCES.values():
        source = ROOT / relative
        copied = target / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copied)
        before[relative] = source.read_bytes()
    expected = {p.name: p.read_bytes() for p in (ROOT / 'assets/readme').iterdir() if p.is_file()}
    monkeypatch.setattr(builder, 'ROOT', target)
    monkeypatch.setattr(builder, 'ASSETS', target / 'assets/readme')
    builder.build()
    builder.check_receipt()
    for name, content in expected.items():
        assert (target / 'assets/readme' / name).read_bytes() == content
    for relative, content in before.items():
        assert (target / relative).read_bytes() == content
