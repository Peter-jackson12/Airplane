"""Keep completed work, open follow-ups and historical evidence distinct in the README."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_completed_work_is_stated_without_todo_language():
    readme = _readme()
    assert "아직 완료되지 않았습니다" not in readme
    assert "1,317개 요청 묶음, 27개 분할" in readme
    assert "실제 706,759행 전체 행 날씨 결합과 독립 gzip 감사까지 완료했습니다" in readme
    assert "항공편 한 건의 지연 여부(`Delay`)" in readme
    # Dated per-round history moved out of the portfolio README but stays
    # reachable at an immutable commit rather than being silently dropped.
    archived = re.search(
        r"https://github\.com/Peter-jackson12/Airplane/blob/([0-9a-f]{40})/README\.md#evidence-appendix",
        readme)
    assert archived, "Archived presentation README link must pin a full commit SHA"


def test_limits_distinguish_paired_result_from_causality_and_open_followups():
    readme = _readme()
    limits = readme.split('<a id="limits"></a>', 1)[1].split('<a id="reproduce"></a>', 1)[0]
    assert "인과 효과의 증명이 아닙니다" in limits
    assert "선택 집단의 정적 교차검증" in limits
    assert "docs/TUTOR_FEEDBACK_HANDOFF_KO.md" in limits
    # An unexecuted algorithm comparison must never read as done.
    rows = [line for line in limits.splitlines() if "Random Forest" in line]
    assert rows and all("미실시" in line for line in rows)
    assert "기존 CSV에서 도출 가능(재학습 불필요)" in limits
    for overclaim in ("Random Forest보다", "Logistic Regression보다", "알고리즘 비교 결과"):
        assert overclaim not in readme


def test_figure_reproduction_documents_tracked_sources():
    readme = _readme()
    section = readme.split('<a id="reproduce"></a>', 1)[1].split("### 코드 지도", 1)[0]
    assert "추적된 집계 파일만 사용" in section
    assert "원본 데이터가 필요하지 않습니다" in section
    assert "scripts/build_readme_assets.py --check" in section
