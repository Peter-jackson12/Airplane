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
    # The executed classifier comparison is recorded as done, with evidence,
    # and nothing in the README still claims it was not run.
    assert "미실시" not in readme
    rows = [line for line in limits.splitlines()
            if line.startswith("| 1)") and "Random Forest" in line]
    assert len(rows) == 1
    assert "완료" in rows[0]
    assert "(output/baseline_recovery_v2_classifier_compare_20261008_summary.json)" in rows[0]
    assert "(#classifier-comparison)" in rows[0]
    # Every tutor item 1-11 has a row, and items resolved by the feedback
    # documents link to them.
    for item in ("1)", "2)", "3)", "4)", "5)", "6)", "7)", "8)", "9)", "10)", "11)"):
        assert f"| {item}" in limits or f"·{item}" in limits, item
    assert "docs/FEEDBACK_PREPROCESSING_KO.md" in limits
    assert "docs/FEEDBACK_MODEL_WEATHER_KO.md" in limits
    assert "기존 CSV에서 도출 가능(재학습 불필요)" in limits
    # Open next steps stay framed as candidates, not as completed work.
    nxt = limits.split("### 다음 단계 후보", 1)[1]
    for candidate in ("더 넓은 탐색 예산", "`max_features` 후보 확장",
                      "Random Forest에서 날씨 사용/미사용 비교", "확률 보정"):
        assert candidate in nxt, candidate


def test_classifier_comparison_is_interpreted_within_its_budget():
    readme = _readme()
    results = readme.split('<a id="results"></a>', 1)[1].split('<a id="engineering"></a>', 1)[0]
    section = results.split('<a id="classifier-comparison"></a>', 1)[1].split("\n### ", 2)[1]
    # Honest reading: the heading and interpretation are scoped to the small
    # predeclared candidate range, the unequal search budget is stated, whether
    # a LightGBM search would close the gap is left unverified, it is not a
    # general algorithm ranking, and the paired weather conclusion is untouched.
    heading = section.splitlines()[0]
    assert "작은 사전 선언 후보 범위에서" in heading
    assert "현재 LightGBM 고정 설정은 이 후보 범위에서 최선이 아니었습니다" in section
    assert "차이가 줄어드는지는 **미검증**" in section
    assert "LightGBM은 트리 수 1축" in section and "Random Forest는 2축 6개 후보" in section
    assert "`min_samples_leaf=25`" in section
    assert "Random Forest가 일반적으로 더 나은 알고리즘이라는 증거는 아닙니다" in section
    assert "날씨 사용/미사용 결론은 바뀌지 않습니다" in section
    assert "전역 하이퍼파라미터 탐색이 아니므로" in section
    assert "`max_features=0.5`" in section and "확률 보정은 하지 않았고" in section
    for overclaim in ("Random Forest보다", "Logistic Regression보다", "최적 알고리즘",
                      "Random Forest가 더 우수", "알고리즘 비교 결과",
                      "최적이 아닐 가능성이 큽니다"):
        assert overclaim not in readme, overclaim


def test_figure_reproduction_documents_tracked_sources():
    readme = _readme()
    section = readme.split('<a id="reproduce"></a>', 1)[1].split("### 코드 지도", 1)[0]
    assert "추적된 집계 파일만 사용" in section
    assert "원본 데이터가 필요하지 않습니다" in section
    assert "scripts/build_readme_assets.py --check" in section
