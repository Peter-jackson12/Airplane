"""Keep current submission status distinct from preserved historical evidence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_current_transport_status_preserves_history_without_declaring_incomplete():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    current = readme.split('<a id="evidence-appendix"></a>', 1)[0]
    assert "전체 27-shard 네트워크 실행은 아직 완료되지 않았습니다." not in current
    assert "초기 실행 기록(2026-09-21, 전체 완료 전)" in current
    assert "현재는 27/27 shard 수집·finalize와 full join·weather-off/on paired 비교까지 완료" in current
    assert "현재 제출을 위해 다시 실행할 작업이 아닙니다." in current


def test_presentation_route_distinguishes_paired_result_from_causality():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    route = readme.split('<a id="presentation-route"></a>', 1)[1].split('<a id="evidence-appendix"></a>', 1)[0]
    assert "“날씨 효과는?” → 아직 미검증" not in route
    assert "동일조건 paired CV 개선" in route
    assert "인과 효과·미래 운항 성능은 미검증" in route


def test_figure_reproduction_documents_csv_and_json_sources():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split('<a id="readme-figures"></a>', 1)[1].split("### 코드 지도", 1)[0]
    assert "Git에 추적된 집계 CSV·JSON만 읽습니다." in section
    assert "공개 집계 CSV만 읽습니다." not in section
