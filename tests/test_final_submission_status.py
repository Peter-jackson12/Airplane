"""Keep current submission status distinct from preserved historical evidence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_current_transport_status_preserves_history_without_declaring_incomplete():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    current = readme.split('<a id="evidence-appendix"></a>', 1)[0]
    assert "전체 27-shard 네트워크 실행은 아직 완료되지 않았습니다." not in current
    history = readme.split('<a id="evidence-appendix"></a>', 1)[1]
    # Preserve the dated initial state, but do not make a completed project
    # read like a to-do list by requiring that history in the current body.
    assert "초기 실행 기록(2026-09-21, 전체 완료 전)" in history
    assert "현재 제출을 위해 다시 실행할 작업이 아닙니다." in history
    assert "27/27 분할" in current and "1,317 요청 묶음 완료" in current
    assert "날씨 유무 동일조건 비교까지 완료" in current
    assert "전체 수집·전체 행 결합·날씨 모델 비교·제출 정리" in current
    assert "2026-09-22 완료" in current
    assert "**한 줄 목적:** 항공편 한 건의 `Delay`" in current


def test_presentation_route_distinguishes_paired_result_from_causality():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    route = readme.split('<a id="presentation-route"></a>', 1)[1].split('<a id="evidence-appendix"></a>', 1)[0]
    assert "“날씨 효과는?” → 아직 미검증" not in route
    assert "동일조건 교차검증 개선" in route
    assert "인과 효과·미래 운항 성능은 미검증" in route


def test_figure_reproduction_documents_csv_and_json_sources():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split('<a id="readme-figures"></a>', 1)[1].split("### 코드 지도", 1)[0]
    assert "Git에 추적된 집계 CSV·JSON만 읽습니다." in section
    assert "공개 집계 CSV만 읽습니다." not in section
