"""Keep current submission status distinct from preserved historical evidence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_current_transport_status_preserves_history_without_declaring_incomplete():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    current, history = readme.split('<a id="evidence-appendix"></a>', 1)
    assert "전체 27-shard 네트워크 실행은 아직 완료되지 않았습니다." not in current
    assert "초기 실행 기록(2026-09-21, 전체 완료 전)" not in current
    assert "초기 실행 기록(2026-09-21, 전체 완료 전)" in history
    assert 'id="appendix-execution-receipts"' in history
    assert "현재는 27/27 분할 수집·최종 검증과 전체 행 날씨 결합·날씨 미사용/사용 동일조건 비교까지 완료" in current
    assert "현재 제출을 위해 다시 실행할 작업이 아닙니다." in current
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


def test_first_read_explains_problem_population_result_and_limits():
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    overview = readme.split('<a id="1-목적과문제정의"></a>', 1)[0]
    for required in ('지연 여부', '1,000,000행', '255,001행', '706,759행', '180,332행',
                     '전처리 수정만으로 Macro F1 향상은 확인하지 못했습니다',
                     '세 시드 모두', '정적 교차검증', '실측 날씨 공개 지연 시간이 아닙니다',
                     '인과 효과나 미래 운항 성능으로 일반화하지 않습니다', '신뢰구간이 아닙니다'):
        assert required in overview
    assert '<details>' not in overview
    results = readme.split('## 5. 최신 검증 결과', 1)[1].split('<a id="6-한계와다음단계"></a>', 1)[0]
    assert results.index('### 5.1 날씨 정보 추가') < results.index('### 5.2 전처리 수정')
