"""Git-only documentation checks; no network or ignored source data required."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / 'README.md'
LEGACY_ANCHORS = {
    '1-목적과문제정의', '2-데이터와분석범위', '3-전처리결정과근거',
    '4-현재파이프라인', '5-최신검증결과', '6-한계와다음단계',
    '7-코드구조와재현', '8-문서안내',
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


def test_readme_keeps_legacy_entry_anchors_and_resolves_internal_links():
    text = README.read_text(encoding='utf-8')
    ids = re.findall(r'<a\s+id="([^"]+)"\s*>', text)
    assert len(ids) == len(set(ids)), 'Duplicate explicit README anchors'
    assert LEGACY_ANCHORS <= set(ids)
    assert {'project-status', 'next-local-run', 'evidence-appendix'} <= set(ids)
    for destination in markdown_destinations(text):
        if destination.startswith('#'):
            assert unquote(destination[1:]) in ids, destination


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
    assert prose.index('id="evidence-appendix"') < prose.index('<details>')


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
    expected = (
        f"| stratafix 표본 | {manifest['selected_rows']} | {manifest['collectible_rows']} | "
        f"{manifest['not_collectible_rows']} |"
    )
    assert expected in text
    assert manifest['collectible_rows'] + manifest['not_collectible_rows'] == manifest['selected_rows']


def test_readme_notebook_module_commands_have_entry_files():
    text = README.read_text(encoding='utf-8')
    modules = re.findall(r'\bpython(?:\.exe)?\s+(?:-u\s+)?-m\s+(notebooks\.[A-Za-z0-9_]+)', text)
    assert modules
    for module in modules:
        path = ROOT.joinpath(*module.split('.')).with_suffix('.py')
        assert path.is_file(), f'Missing module in README command: {module}'
