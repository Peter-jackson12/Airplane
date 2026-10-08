<a id="home"></a>
# Airplane 위키 · 에이전트 공통 작업 지침

항공편 지연 예측(LightGBM)과 날씨 정보 확장 프로젝트의 **길찾기 허브이자 에이전트 공통 규칙 파일**이다.
이 저장소에서 작업하는 모든 에이전트(도구·모델 무관)는 이 파일에서 시작한다.
CLAUDE.md는 이 파일로 연결하는 진입점이며 별도 규칙을 중복 관리하지 않는다.
이 파일은 "무엇이 어디에 있는지"와 "어떻게 작업하는지"만 기록하며, 실험 수치·완료 테스트 수·현재 브랜치·임시 세션 ID는 복제하지 않는다. 수치와 결론은 [README](README.md)와 실행 근거에서 확인한다.
강의 발표는 끝났고 README는 포트폴리오 형태의 설명 문서로 운영된다.

## 목차

| 절 | 내용 |
|---|---|
| [빠른 길찾기](#quick-nav) | "하고 싶은 일 → 읽거나 실행할 곳" |
| [문서 지도](#doc-map) | 현행 참조 문서와 과거 기록 문서 |
| [코드 지도](#code-map) | `src/`, `scripts/`, `notebooks/`, 루트 스크립트, `tests/` |
| [실행 근거 지도](#evidence-map) | `output/` 근거 파일을 실험별로 분류 |
| [작업 현황·백로그](#backlog) | 튜터 피드백 반영 현황, 진행 중 실험, 검토 후속, 마무리 점검 |
| [작업 규칙](#rules) | 시작 절차, 역할, Git, 문서 관리, 데이터·평가 경계, 실행·보존, 보고 |

<a id="quick-nav"></a>
## 빠른 길찾기

| 하고 싶은 일 | 읽거나 실행할 곳 |
|---|---|
| 현재 결론·파이프라인·한계 확인 | [README.md](README.md) |
| README를 쉬운 말로 깊게 읽기 | [docs/README_EXPLAINED_KO.md](docs/README_EXPLAINED_KO.md) |
| 튜터 피드백 11개와 반영 현황 확인 | [docs/TUTOR_FEEDBACK_HANDOFF_KO.md](docs/TUTOR_FEEDBACK_HANDOFF_KO.md) (0절이 현재 반영 현황) |
| 결측 복원의 집단·분모, 시각 복원 규칙, Traffic 집계 경계, 양쪽 시각 결측 집단 | [docs/FEEDBACK_PREPROCESSING_KO.md](docs/FEEDBACK_PREPROCESSING_KO.md), 재집계 [scripts/feedback_missing_audit.py](scripts/feedback_missing_audit.py) |
| 날씨 피처 명세, LightGBM 고정/탐색 설정·정오표, Macro F1 정의 | [docs/FEEDBACK_MODEL_WEATHER_KO.md](docs/FEEDBACK_MODEL_WEATHER_KO.md) |
| 분류기 3종(Logistic Regression / Random Forest / LightGBM) 동일조건 비교 | [src/classifier_compare.py](src/classifier_compare.py), [notebooks/run_classifier_comparison.py](notebooks/run_classifier_comparison.py), 결과는 README 2절 |
| 분류기 탐색 예산 확장·Random Forest 날씨 사용/미사용 | [notebooks/run_classifier_tuning.py](notebooks/run_classifier_tuning.py), 결과는 [README 2절 후속 확인](README.md#classifier-tuning) |
| Tableau Public 대시보드 제작 | [docs/TABLEAU_HANDOFF_KO.md](docs/TABLEAU_HANDOFF_KO.md), 집계 데이터 [output/tableau/README_DATA_KO.md](output/tableau/README_DATA_KO.md), 생성기 [scripts/build_tableau_extracts.py](scripts/build_tableau_extracts.py) |
| 평가 경로(nested grid, inner 경계 TE) | [src/cv.py](src/cv.py) |
| 전처리·피처 엔지니어링 | [src/features.py](src/features.py) |
| 날씨 피처 계약(예측 시점·가정 지연) | [src/weather_model.py](src/weather_model.py), [src/weather.py](src/weather.py) |
| 날씨 전체 결합 코드 | [src/weather_full.py](src/weather_full.py) |
| 확률 보정(Platt/Isotonic) | [src/calibration.py](src/calibration.py) |
| 행별 OOF 계약·진단 | [src/oof.py](src/oof.py) |
| 실행 체크포인트·지문 검사 | [src/run_store.py](src/run_store.py) |
| Phase 1~6 통일 프로토콜 러너 | [rerun_all_phases.py](rerun_all_phases.py) |
| 전처리 10조건 × 3시드 실험 | [notebooks/run_preprocessing_experiments.py](notebooks/run_preprocessing_experiments.py) |
| 날씨 사용/미사용 쌍비교 | [notebooks/run_weather_model_comparison.py](notebooks/run_weather_model_comparison.py) |
| README 그림 재생성·출처 검증 | [scripts/build_readme_assets.py](scripts/build_readme_assets.py) (`--check`로 읽기 전용 검사), [scripts/build_readme_story.py](scripts/build_readme_story.py) |
| 테스트(Git만으로 가능한 기본 검사) | `uv run --locked --offline python -m pytest -q` (기본으로 `local_data` marker 제외) |
| 테스트(로컬 원자료 필요 검사) | `uv run --locked --offline python -m pytest -q -m local_data` |
| CI가 무엇을 도는지 | [.github/workflows/ci.yml](.github/workflows/ci.yml) |
| 과거 감사·계획 확인 | [AUDIT.md](AUDIT.md), [PLAN.md](PLAN.md) (과거 기록, 아래 문서 지도 참고) |
| 전처리 결정 이유 | [output/preprocessing_decisions.md](output/preprocessing_decisions.md) |
| 데이터 사용 계약 | [output/pipeline_data_contract.md](output/pipeline_data_contract.md) |
| 새 실험을 시작 | [작업 규칙](#rules)의 데이터·평가 경계와 실행·결과 보존을 먼저 읽는다 |

<a id="doc-map"></a>
## 문서 지도

상태 표기: **현행** = 현재 설명·작업의 기준 또는 참조, **보존** = 근거로 유지하되 현재 사양이 아님, **과거 기록** = 당시 시점의 진단·계획(수치·미해결 상태를 현재 사양으로 되살리지 않음).

### 현행 참조 문서

| 문서 | 역할 | 상태 |
|---|---|---|
| [README.md](README.md) | 현재 결론·유효한 결과·남은 한계·재현 안내의 기준 문서(포트폴리오 README) | 현행 |
| [docs/README_EXPLAINED_KO.md](docs/README_EXPLAINED_KO.md) | README의 한국어 심화 해설(분모 구분, 날씨 피처, 가용 시점 등) | 현행 |
| [docs/TUTOR_FEEDBACK_HANDOFF_KO.md](docs/TUTOR_FEEDBACK_HANDOFF_KO.md) | 튜터 피드백 11개의 보존본. 0절은 현재 반영 현황, 1~4절은 기준 커밋 시점의 상태 대조 기록 | 현행(0절) / 과거 기록(1~4절) |
| [docs/FEEDBACK_PREPROCESSING_KO.md](docs/FEEDBACK_PREPROCESSING_KO.md) | 피드백 2·3·4·10: 결측 복원의 집단·분모, 시각 복원 규칙, Traffic 집계 경계, 양쪽 시각 결측 집단 | 현행 참조 |
| [docs/FEEDBACK_MODEL_WEATHER_KO.md](docs/FEEDBACK_MODEL_WEATHER_KO.md) | 피드백 5·6·7: 날씨 피처 명세, LightGBM 고정/탐색 설정과 protocol 라벨 정오표, 지표 정의, 분류기 비교 기록의 코드 해시 정오표(3.6절) | 현행 참조 |
| [docs/TABLEAU_HANDOFF_KO.md](docs/TABLEAU_HANDOFF_KO.md) | Tableau Public 대시보드 3종 제작 인계서(연결할 CSV, 화면 구성, 게시 전 점검) | 현행 |
| [output/tableau/README_DATA_KO.md](output/tableau/README_DATA_KO.md) | Tableau용 집계 CSV의 열·집단·분모 설명과 근거 대조 결과 | 현행 참조 |
| [output/pipeline_data_contract.md](output/pipeline_data_contract.md) | 현재 파이프라인의 데이터 사용 계약 | 현행 참조 |
| [output/preprocessing_decisions.md](output/preprocessing_decisions.md) | 전처리 결정표 | 현행 참조 |
| [output/preprocessing_clean_implementation.md](output/preprocessing_clean_implementation.md) | 전처리 개선 구현 및 전수 검증 | 현행 참조 |
| [output/preprocessing_full_evaluation.md](output/preprocessing_full_evaluation.md) | 전처리 변경의 전체 데이터 성능 검증 | 현행 참조 |
| [output/preprocessing_current_report.md](output/preprocessing_current_report.md) | 전처리 중심 모델링 보고서 | 현행 참조 |
| [output/nested_grid_implementation.md](output/nested_grid_implementation.md) | nested `n_estimators` 선택 구현 | 현행 참조 |
| [output/pipeline_hardening_implementation.md](output/pipeline_hardening_implementation.md) | 파이프라인 감사 후 구현 결과 | 현행 참조 |
| [output/label_coverage_review.md](output/label_coverage_review.md) | 라벨 유무 분포 진단(전체 실제 데이터) | 현행 참조 |
| [output/weather_recovery_review.md](output/weather_recovery_review.md) | 날씨 활용 재검토 및 재개 준비 | 보존(날씨 조사 시점) |

### 과거 기록·보존 문서 (별도 수정 요청 없이 보존)

| 문서 | 역할 | 상태 |
|---|---|---|
| [PLAN.md](PLAN.md) | 프로젝트 마스터 플랜·로드맵 | 과거 기록 |
| [AUDIT.md](AUDIT.md) | 파이프라인 감사 보고서 | 과거 기록 |
| [output/audit_addendum_te_leak.md](output/audit_addendum_te_leak.md) | AUDIT 부록: ES-inner-holdout 경계의 TE 누수 | 과거 기록 |
| [output/es_diagnosis.md](output/es_diagnosis.md) | Early Stopping `best_iteration` 붕괴 원인 진단 | 과거 기록 |
| [output/es_protocol_final.md](output/es_protocol_final.md) | Early Stopping 선택 분산 문제의 최종 프로토콜 | 과거 기록 |
| [output/pipeline_architecture_review.md](output/pipeline_architecture_review.md) | 파이프라인 구조 감사(P5 라벨 흐름·임계값·재개 로직) | 과거 기록 |
| [output/baseline_recovery.md](output/baseline_recovery.md), [output/baseline_recovery.csv](output/baseline_recovery.csv) | Phase 1~6 통일 프로토콜 재측정(기준선 복구) | 과거 기록(보존 필수) |

그 밖에 `output/*.html`(시각화 보고서)과 추적되는 한글 파일명의 `.docx`(보고서 템플릿)가 있다. 템플릿은 작성 시점의 초안 서식이므로 독립적인 최신 사양으로 취급하지 않는다.

<a id="code-map"></a>
## 코드 지도

### `src/` — 재사용 모듈

| 파일 | 역할 |
|---|---|
| [src/cv.py](src/cv.py) | 교차검증 프로토콜(`CVConfig`), nested grid, 지표·임계값 선택. 평가 프로토콜을 스크립트가 아닌 모듈이 소유 |
| [src/features.py](src/features.py) | 공통 전처리·피처 엔지니어링. target-free(fold 밖)와 target-dependent(fold 안) 단계 분리 |
| [src/calibration.py](src/calibration.py) | outer-train 내부 inner holdout에서만 적합하는 확률 보정기 |
| [src/oof.py](src/oof.py) | 행별 OOF 계약과 기술 진단. 모델 적합 없음 |
| [src/run_store.py](src/run_store.py) | 스키마·실험 지문 검사가 있는 원자적 Phase 체크포인트(단일 작성자) |
| [src/weather.py](src/weather.py) | 날씨 정렬 원시 연산. 연도·도착일을 추측하지 않고 `available_at`을 명시적으로 받음 |
| [src/weather_full.py](src/weather_full.py) | 캐시 전용·메모리 제한 전체 날씨 결합. 타깃·모델 코드 없음 |
| [src/weather_model.py](src/weather_model.py) | 사전 선언된 날씨 모델 계약(10분 공개 지연 가정이 대표 시나리오). 모델 적합 없음 |
| [src/classifier_compare.py](src/classifier_compare.py) | 분류기 3종 동일조건 비교 하네스. 모델별 전처리·후보를 내부 경계 안에서만 적합. 데이터 로드·파일 쓰기 없음 |

### `scripts/` — README 그림 생성·문서 근거 재집계·Tableau 집계

| 파일 | 역할 |
|---|---|
| [scripts/build_readme_assets.py](scripts/build_readme_assets.py) | 추적된 집계 근거만으로 README 그림 생성, 출처·해시 영수증 기록, `--check` 검증 |
| [scripts/build_readme_story.py](scripts/build_readme_story.py) | 검증된 집계 근거 기반의 표지·흐름·경계 그림 |
| [scripts/readme_editorial.py](scripts/readme_editorial.py) | 위 두 생성기가 공유하는 표준 라이브러리 전용 SVG 구성 요소 |
| [scripts/feedback_missing_audit.py](scripts/feedback_missing_audit.py) | 피드백 2·4·10 수치를 원자료에서 읽기 전용으로 재집계(로컬 원자료 필요, 학습 없음) |
| [scripts/build_tableau_extracts.py](scripts/build_tableau_extracts.py) | Tableau Public용 집계 CSV(`output/tableau/`)와 대조 manifest 생성(학습 없음) |

### `notebooks/` — 실행·분석 스크립트(대부분 `.py`)

| 분류 | 파일 |
|---|---|
| 전처리 실험 | [run_preprocessing_experiments.py](notebooks/run_preprocessing_experiments.py)(10조건 × 3시드, 재개 가능), [report_preprocessing_experiments.py](notebooks/report_preprocessing_experiments.py), [check_preprocessing_clean.py](notebooks/check_preprocessing_clean.py)(실제 데이터 전/후 검사), [build_preprocessing_evaluation.py](notebooks/build_preprocessing_evaluation.py), [build_preprocessing_walkthrough.py](notebooks/build_preprocessing_walkthrough.py), [preprocessing_full_evaluation.ipynb](notebooks/preprocessing_full_evaluation.ipynb), [preprocessing_walkthrough.ipynb](notebooks/preprocessing_walkthrough.ipynb), [export_current_pipeline_evidence.py](notebooks/export_current_pipeline_evidence.py) |
| OOF·오류 진단 | [run_oof_diagnostics.py](notebooks/run_oof_diagnostics.py), [analyze_oof_error_profile.py](notebooks/analyze_oof_error_profile.py), [plot_oof_error_profile.py](notebooks/plot_oof_error_profile.py) |
| 확률 보정 실험 | [run_calibration_experiment.py](notebooks/run_calibration_experiment.py), [report_calibration_experiment.py](notebooks/report_calibration_experiment.py), [plot_calibration_experiment.py](notebooks/plot_calibration_experiment.py), [compare_calibration_runs.py](notebooks/compare_calibration_runs.py) |
| 날짜 복원·달력 진단 | [analyze_calendar_signature.py](notebooks/analyze_calendar_signature.py), [analyze_calendar_mixture.py](notebooks/analyze_calendar_mixture.py), [analyze_record_integrity.py](notebooks/analyze_record_integrity.py), [assign_row_dates.py](notebooks/assign_row_dates.py)(행별 날짜 귀속 규칙) |
| BTS 원본 대조 | [verify_bts_november.py](notebooks/verify_bts_november.py), [diagnose_bts_november_mismatch.py](notebooks/diagnose_bts_november_mismatch.py), [assess_bts_november_recovery.py](notebooks/assess_bts_november_recovery.py), [verify_bts_marketing.py](notebooks/verify_bts_marketing.py), [summarize_bts_marketing_months.py](notebooks/summarize_bts_marketing_months.py) |
| 라벨 커버리지 | [build_label_coverage_review.py](notebooks/build_label_coverage_review.py), [label_coverage_review.ipynb](notebooks/label_coverage_review.ipynb) |
| 날씨 조사·표본 | [weather_feasibility.py](notebooks/weather_feasibility.py), [build_weather_review.py](notebooks/build_weather_review.py), [weather_recovery_review.ipynb](notebooks/weather_recovery_review.ipynb), [map_weather_stations.py](notebooks/map_weather_stations.py), [verify_priority_station_identity.py](notebooks/verify_priority_station_identity.py), [scope_weather_collection.py](notebooks/scope_weather_collection.py), [scope_weather_collection_refined.py](notebooks/scope_weather_collection_refined.py), [select_weather_sample.py](notebooks/select_weather_sample.py), [select_weather_sample_expanded.py](notebooks/select_weather_sample_expanded.py), [fetch_weather_sample.py](notebooks/fetch_weather_sample.py), [fetch_weather_sample_expanded.py](notebooks/fetch_weather_sample_expanded.py), [join_weather_sample.py](notebooks/join_weather_sample.py), [join_weather_sample_expanded.py](notebooks/join_weather_sample_expanded.py), [diagnose_weather_expanded_cache.py](notebooks/diagnose_weather_expanded_cache.py), [reconcile_weather_cache_recombination.py](notebooks/reconcile_weather_cache_recombination.py) |
| 날씨 전체 수집·결합·비교 | [fetch_weather_full_sharded.py](notebooks/fetch_weather_full_sharded.py), [run_weather_full_collection.py](notebooks/run_weather_full_collection.py)(실패 시 중단하는 순차 오케스트레이션), [join_weather_full.py](notebooks/join_weather_full.py), [run_weather_model_comparison.py](notebooks/run_weather_model_comparison.py) |
| 분류기 비교 | [run_classifier_comparison.py](notebooks/run_classifier_comparison.py)(날씨 사용 집단에서 Logistic Regression / Random Forest / LightGBM, 기존 출력 덮어쓰기 거부), [run_classifier_tuning.py](notebooks/run_classifier_tuning.py)(LightGBM 24개·Random Forest 9개 설정 탐색 확장과 Random Forest 날씨 사용/미사용, 폴드 단위 체크포인트) |

### 루트 스크립트

| 파일 | 역할 | 상태 |
|---|---|---|
| [rerun_all_phases.py](rerun_all_phases.py) | Phase 1~6을 단일 `CVConfig`로 통일 재측정하는 현행 러너 | 현행 |
| [run_baseline.py](run_baseline.py), [run_tuned.py](run_tuned.py), [run_target_encoded.py](run_target_encoded.py), [run_hybrid.py](run_hybrid.py), [run_pseudo_labeling.py](run_pseudo_labeling.py), [run_advanced_features.py](run_advanced_features.py) | Phase 1~6 단계별 초기 실행 스크립트(통일 프로토콜 이전) | 과거 기록 |
| [run_grand_slam_v0_leaky.py](run_grand_slam_v0_leaky.py) | 누수가 있던 v0 시제품(파일명 그대로 누수 포함) | 과거 기록, 성능 근거로 인용 금지 |
| [run_phase7_weather_model.py](run_phase7_weather_model.py) | Phase 7 외부 날씨 피처 LightGBM 초기 실험 | 과거 기록 |
| [merge_weather_pipeline.py](merge_weather_pipeline.py), [find_flight_year.py](find_flight_year.py) | 초기 날씨 병합·연도 추정 스크립트 | 과거 기록 |
| [main.py](main.py) | `uv init`이 만든 자리표시 파일 | 사용 안 함 |

### `tests/`

`pyproject.toml`의 기본 pytest 설정은 `local_data` marker를 제외한다. CI([.github/workflows/ci.yml](.github/workflows/ci.yml))도 같은 Git-only 검사만 실행한다. 대응 범위별로 분류한다.

| 범위 | 파일 |
|---|---|
| 핵심 파이프라인 | [test_features.py](tests/test_features.py), [test_oof.py](tests/test_oof.py), [test_calibration.py](tests/test_calibration.py), [test_pipeline_regressions.py](tests/test_pipeline_regressions.py), [test_preprocessing_clean.py](tests/test_preprocessing_clean.py) |
| 분류기 비교 | [test_classifier_comparison.py](tests/test_classifier_comparison.py), [test_classifier_tuning.py](tests/test_classifier_tuning.py) |
| Tableau 집계 | [test_tableau_extracts.py](tests/test_tableau_extracts.py) |
| 문서 근거 재집계 | [test_feedback_missing_audit.py](tests/test_feedback_missing_audit.py)(순수 로직, 원자료 불필요) |
| README·문서 계약 | [test_readme_contract.py](tests/test_readme_contract.py), [test_readme_presentation.py](tests/test_readme_presentation.py), [test_final_submission_status.py](tests/test_final_submission_status.py) |
| 날짜·BTS 대조 | [test_assign_row_dates.py](tests/test_assign_row_dates.py), [test_bts_november.py](tests/test_bts_november.py), [test_bts_november_diagnosis.py](tests/test_bts_november_diagnosis.py), [test_bts_marketing.py](tests/test_bts_marketing.py), [test_bts_marketing_months.py](tests/test_bts_marketing_months.py) |
| 날씨 | [test_weather.py](tests/test_weather.py), [test_weather_model.py](tests/test_weather_model.py), [test_weather_protocol_label.py](tests/test_weather_protocol_label.py), [test_weather_full.py](tests/test_weather_full.py), [test_weather_full_artifact_boundaries.py](tests/test_weather_full_artifact_boundaries.py), [test_weather_full_collection_runner.py](tests/test_weather_full_collection_runner.py), [test_weather_full_finalize.py](tests/test_weather_full_finalize.py), [test_weather_full_serialization.py](tests/test_weather_full_serialization.py), [test_weather_full_sharded.py](tests/test_weather_full_sharded.py), [test_join_weather_full_driver.py](tests/test_join_weather_full_driver.py), [test_weather_sample.py](tests/test_weather_sample.py), [test_weather_expanded_diagnostics.py](tests/test_weather_expanded_diagnostics.py), [test_weather_expanded_pipeline.py](tests/test_weather_expanded_pipeline.py), [test_weather_reconciliation.py](tests/test_weather_reconciliation.py), [test_weather_scope_and_mapping.py](tests/test_weather_scope_and_mapping.py), [test_verify_priority_station_identity.py](tests/test_verify_priority_station_identity.py) |

<a id="evidence-map"></a>
## 실행 근거 지도

`output/`은 실행 근거 보관소다. 파일명은 `baseline_recovery_v2_<실험>_<날짜>_<산출물>` 형식이 많다. 각 실행의 `*_manifest.json`이 입력 지문·설정을 기록한다. 파일명의 날짜·final/current만으로 최신성을 판단하지 않고 README의 근거 링크와 manifest로 확인한다. 아래는 접두사(glob)로 묶은 지도이며, 개별 파일은 `git ls-files output`으로 열거한다.

| 실험 | 대표 근거 | 상태 |
|---|---|---|
| 날씨 사용/미사용 쌍비교 | `output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_{runs.csv, paired_deltas.csv, summary.json, manifest.json}` — 예: [runs.csv](output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_runs.csv) | 현행 |
| 분류기 3종 동일조건 비교 | `output/baseline_recovery_v2_classifier_compare_20261008_{runs.csv, folds.csv, paired_deltas.csv, summary.json, manifest.json}`, [실행 로그](output/baseline_recovery_v2_classifier_compare_20261008.log) — 예: [summary](output/baseline_recovery_v2_classifier_compare_20261008_summary.json) | 현행 |
| 분류기 탐색 예산 확장·Random Forest 날씨 사용/미사용 | `output/baseline_recovery_v2_classifier_tuning_20261008_{runs.csv, folds.csv, paired_deltas.csv, summary.json, manifest.json}`, [실행 로그](output/baseline_recovery_v2_classifier_tuning_20261008.log) — 예: [summary](output/baseline_recovery_v2_classifier_tuning_20261008_summary.json) | 현행 |
| Tableau Public용 집계 | `output/tableau/*.csv`, [manifest](output/tableau/tableau_extracts_manifest.json), [데이터 설명](output/tableau/README_DATA_KO.md) | 현행(기존 근거의 재집계, 학습 없음) |
| 피드백 문서용 결측·Traffic·취약 집단 재집계 | [output/feedback_missing_audit_20261008.json](output/feedback_missing_audit_20261008.json) | 현행 근거(읽기 전용 재집계, 학습 없음) |
| 날씨 전체 행 결합 | [join manifest](output/baseline_recovery_v2_weather_full_join_20260922_full_weather_join_manifest.json), [join summary](output/baseline_recovery_v2_weather_full_join_20260922_full_weather_join_summary.json) | 현행 |
| 날씨 전체 수집(bulk) | [plan manifest](output/baseline_recovery_v2_weather_full_bulk_20260921_full_weather_plan_manifest.json), [fetch manifest](output/baseline_recovery_v2_weather_full_bulk_20260921_full_weather_fetch_manifest.json), [fetch audit](output/baseline_recovery_v2_weather_full_bulk_20260921_full_weather_fetch_audit.json) | 현행 |
| 날씨 사전 표본·확장 표본·범위·매핑 | `output/baseline_recovery_v2_weather_{sample, expanded, expanded_stratafix, expanded_diagnostic*, scope, scope_fix, scope_refined, recombination*}_*`, 관측소 확인 `output/baseline_recovery_v2_station_identity_*` | 보존(전체 수집 이전의 검증 단계) |
| 날씨 조사 초기 | [output/weather_recovery_review.md](output/weather_recovery_review.md), `output/weather_review/`, [output/phase7_weather_feature_importance.png](output/phase7_weather_feature_importance.png) | 과거 기록 |
| 전처리 전체 비교(10조건 × 3시드) | [summary](output/preprocessing_full_summary.csv), [paired deltas](output/preprocessing_full_paired_deltas.csv), [runs](output/preprocessing_full_runs.csv), [selection checks](output/preprocessing_full_selection_checks.csv), [clean 비교](output/preprocessing_clean_comparison.csv), [전/후 검사](output/preprocessing_clean_checks.json), 시드별 `output/baseline_recovery_v2_preprocessing_full_seed{1,7,42}.{csv,md,log}` | 현행 |
| 전처리 결정·예시·시각화 | [output/current_imputation_examples.csv](output/current_imputation_examples.csv), [output/current_pipeline_evidence.json](output/current_pipeline_evidence.json), `output/preprocessing_review/`, [walkthrough html](output/preprocessing_walkthrough.html), [report html](output/preprocessing_current_report.html), [evaluation html](output/preprocessing_full_evaluation.html) | 현행 참조 |
| OOF 3시드 재측정과 오류 진단 | `output/baseline_recovery_v2_oof_20260916*` (그룹은 [groups](output/baseline_recovery_v2_oof_20260916_v2_groups.csv)), [오분류 분석 보고서](output/baseline_recovery_v2_error_profile_20260917_report.md) 및 같은 접두사 파일 | 현행 근거 |
| 확률 보정 실험 | [보고서](output/baseline_recovery_v2_calibration_20260917_report.md), [summary](output/baseline_recovery_v2_calibration_20260917_summary.csv), [paired delta](output/baseline_recovery_v2_calibration_20260917_paired_delta_summary.csv) 및 `*_calibration_equivalence_*`·`*_calibration_local_*`·`*_local_verify_*` | 현행 근거(smoke 파일은 성능 근거 아님) |
| 날짜 복원·달력 진단 | `output/baseline_recovery_v2_calendar_signature_*`, `*_calendar_mixture_*`, `*_record_integrity_*`, `*_row_date_attribution_20260918_*` | 보존 근거 |
| BTS 원본 대조 | `output/baseline_recovery_v2_bts_november_*`, `*_bts_marketing_m01..m12_*`, `*_bts_marketing_summary_*` | 보존 근거 |
| 기준선 재측정 | [output/baseline_recovery.md](output/baseline_recovery.md), [output/baseline_recovery_v2_verified_8000.md](output/baseline_recovery_v2_verified_8000.md), `*_cloud_parity_*` | 과거 기록(smoke는 성능 근거 아님) |
| Early Stopping·임계값·트리 수 진단 그림 | `output/es_curve_*.png`, [output/n_estimators_curve.png](output/n_estimators_curve.png), [output/threshold_optimization.png](output/threshold_optimization.png), [output/protocol_fold_optimum_curves.png](output/protocol_fold_optimum_curves.png), [output/te_leak_check_fold2.png](output/te_leak_check_fold2.png), `output/_deprecated/` | 과거 기록 |
| 라벨 커버리지 | [output/label_coverage_review.md](output/label_coverage_review.md), [output/label_coverage_review.html](output/label_coverage_review.html), `output/label_coverage/` | 현행 참조 |
| README 그림 출처 | [assets/readme/sources.json](assets/readme/sources.json) (그림 SVG는 `assets/readme/`) | 현행 |

<a id="backlog"></a>
## 작업 현황·백로그

README는 포트폴리오 README로 운영되며 강의 발표는 끝났다.

### 튜터 피드백 11개: 반영 완료

항목별 반영 위치·근거·커밋은 [인계 문서 0절](docs/TUTOR_FEEDBACK_HANDOFF_KO.md#0-2026-10-08-반영-현황)이 기준이다. 수치는 여기에 복제하지 않는다.

| 항목 | 상태 | 반영 위치 |
|---|---|---|
| 1) Logistic Regression / Random Forest / LightGBM 비교 | 완료(새 실험) | [README 2절](README.md#classifier-comparison), 근거는 [실행 근거 지도](#evidence-map)의 분류기 비교 |
| 2) 결측 대치의 분모와 전후 | 완료 | [docs/FEEDBACK_PREPROCESSING_KO.md](docs/FEEDBACK_PREPROCESSING_KO.md) 1절 |
| 3) 시각 복원 규칙 | 완료 | 같은 문서 2절 |
| 4) Traffic의 의미와 집계 경계 | 완료 | 같은 문서 3절, `src/features.py`의 `build_traffic_features` 설명 |
| 5) 날씨 피처의 의미·단위·결측 계약 | 완료(미확인 부분은 미확인으로 표시) | [docs/FEEDBACK_MODEL_WEATHER_KO.md](docs/FEEDBACK_MODEL_WEATHER_KO.md) 2절 |
| 6) LightGBM 고정/탐색 설정 | 완료 | 같은 문서 3절 |
| 7) Macro F1과 Delayed F1 표기 | 완료 | 같은 문서 1절·4절 |
| 8) 날씨 비교 클래스별 지표·혼동행렬 | 완료(기존 CSV) | [README 2절](README.md#results), 인계 문서 3절 |
| 9) 날씨 조건별 선택 임계값·트리 수 | 완료(기존 CSV) | [README 2절](README.md#results), 인계 문서 3절 |
| 10) 양쪽 시각 결측 집단과 날씨 평가 집단의 구분 | 완료 | [docs/FEEDBACK_PREPROCESSING_KO.md](docs/FEEDBACK_PREPROCESSING_KO.md) 4절 |
| 11) 가정 가용성과 실제 수신 이력의 구분 | 유지(이미 반영) | [docs/README_EXPLAINED_KO.md 6절](docs/README_EXPLAINED_KO.md#availability) |

### 열린 후속 작업

새 결과는 새 출력 이름으로 분리한다. 제안 항목의 실행은 그때의 사용자 요청과 범위를 확인한 뒤에만 한다. 근거와 동기는 [README 4절](README.md#limits)을 따른다.

**실험**

| 항목 | 성격 | 상태 | 완료 시 함께 할 일 |
|---|---|---|---|
| `classifier_tuning_20261008` — LightGBM·Random Forest 탐색 예산 확장, `max_features` 후보 확장, Random Forest 날씨 사용/미사용 동일조건 비교 | 새 실험(같은 행·폴드·시드·내부 선택 경계 유지, 예산 사전 선언). 러너 [notebooks/run_classifier_tuning.py](notebooks/run_classifier_tuning.py) | 완료(`03fe25c`) | [README 2절 후속 확인](README.md#classifier-tuning), 근거는 [실행 근거 지도](#evidence-map)의 분류기 탐색 예산 확장 |
| LightGBM 후보를 끝값 너머로 확장(학습률 0.03 미만, 잎 수 127 초과) | 새 실험(탐색 확장에서 선택값이 후보 끝에 몰림) | 제안/미실행 | – |
| 날씨 미사용 Random Forest의 `min_samples_leaf` 50 초과 후보 | 새 실험(같은 이유) | 제안/미실행 | – |
| 분류기별 확률 보정 | 새 실험(보정기는 외부 학습 데이터 내부에서만 적합) | 제안/미실행 | – |
| Tableau Public 대시보드 3개 제작·게시 (Codex 담당, [docs/TABLEAU_HANDOFF_KO.md](docs/TABLEAU_HANDOFF_KO.md)) | 시각화 제작(새 실험 아님, `output/tableau/` 집계만 사용) | 대기 | – |

**검토 후속(코드·테스트)** — 2026-10-08 독립 검토 지적. 탐색 확장 실험 커밋 뒤 처리했다. 기존 실행 근거 파일은 수정하지 않았다.

| 항목 | 위치 | 상태 |
|---|---|---|
| (a) 분류기 비교 실행 이후 `src/features.py`는 문서 문자열만 바뀌었음(db117ed)을 기록(`code_sha256_lf` 불일치 설명) | [README 2절 기록 참고](README.md#classifier-tuning), [코드 해시 정오표](docs/FEEDBACK_MODEL_WEATHER_KO.md#code-hash-errata) | 완료(근거 파일 대신 문서에 기록) |
| (b) 체크포인트 identity에 `code_sha256_lf` 포함 | [notebooks/run_classifier_comparison.py](notebooks/run_classifier_comparison.py)(탐색 확장 러너는 이미 포함) | 완료 |
| (c) Logistic Regression 후보별 `n_iter`·`converged`를 `grid_scores`에 기록 | [src/classifier_compare.py](src/classifier_compare.py) | 완료(이후 실행부터 기록, 선택·지표 불변) |
| (d) 누수 테스트 보강: Random Forest·LightGBM 경로의 outer-valid 라벨 반전 불변, inner-holdout 라벨 변경 시 inner-train TE·전처리기 불변, OneHot 범주·infrequent 처리가 inner-train에서만 학습, fingerprint 불일치 실패 경로, Random Forest `n_jobs>1` 결정성 | [tests/test_classifier_comparison.py](tests/test_classifier_comparison.py) | 완료(`n_jobs>1`은 나무 동일, 확률 합은 수 ulp 차이 허용 — 정오표 3.6절) |
| (e) fold fingerprint에 타깃 해시·행 키 해시 추가(선택) | [src/classifier_compare.py](src/classifier_compare.py), 두 러너 | 완료(기존 지문 정의는 유지하고 `target_sha256`·`row_key_sha256` 별도 필드로 추가) |

**마무리 점검**

| 항목 | 상태 |
|---|---|
| 작업 트리 정리: 줄바꿈 표시만 다른 output CSV 4개와 `scripts/build_readme_story.py`의 미커밋 표시를 정리하고 실험 커밋에 섞지 않기 | 완료 |
| 마지막 커밋 기준 Linux CI 통과 확인 | 완료 |
| Windows 로컬 전체 pytest 통과 확인 | 완료 |
| GitHub에서 README 그림 렌더링 확인 | 완료 |

<a id="rules"></a>
## 작업 규칙

### 1. 시작할 때 읽을 것

1. 이 파일을 읽고 사용자 요청의 범위를 확인한다.
2. README.md에서 현재 결론·파이프라인·남은 한계·문서 안내를 확인한다.
3. git branch/status로 현재 브랜치와 기존 변경을 확인한다. 완료됐다는 이전 대화 요약만으로 코드 상태를 단정하지 않는다.
4. 해당 작업의 코드와 README에 연결된 실행 근거만 추가로 읽는다. 과거 보고서 전체를 매번 읽지 않는다. 위의 [빠른 길찾기](#quick-nav)와 지도를 이용한다.

README는 현재 상태의 요약이며, 구체적인 사실은 코드·원본 데이터·실행 로그로 확인한다.
둘이 다르면 불일치를 보고하고 필요한 검증으로 해결한다. 파일명의 final/current만으로 최신성을 판단하지 않는다.

### 2. 역할과 작업 범위

- 현재 에이전트가 설계·검토·구현·파일 수정·터미널 실행·테스트를 직접 수행할 수 있다.
- 사용자 변경을 보존하고 같은 파일을 다른 에이전트와 동시에 수정하지 않는다.
- 기본 작업 위치는 `master`다. 일반적인 구현·분석·문서 수정은 `master`에서 검증 후 커밋·푸시한다. 장기간 분리할 실험이나 병렬 작업 등 구체적인 필요가 있을 때만 임시 브랜치 또는 별도 worktree를 사용한다.
- 요청에 포함된 읽기·구현·단위 테스트·축소 스모크는 진행한다. 읽기 전용 감사 요청을 임의의 코드 수정으로 확대하지 않는다.
- 사용자 목표에 필요한 설계·구현·검증·분석·전체 데이터 Phase 재학습은 에이전트가 범위를 판단해 별도 사전 승인 없이 진행한다. 실행할 조건·시드·출력 경로를 알리고, 축소 검증 후 필요한 전체 실행을 수행한다. 사용자가 명시한 읽기 전용·실행 금지·자원 제한은 우선한다.

#### 조율 에이전트와 실행 에이전트

- 작업이 크거나 독립적으로 나눌 수 있으면, 사용자와 대화하는 에이전트가 조율 에이전트(컨트롤 타워)를 맡고 서브에이전트 또는 새 세션의 실행 에이전트에게 작업을 위임할 수 있다. 특정 도구·모델에 역할을 고정하지 않는다. 사용자가 모델·추론 수준을 지정하면 그 지정을 따른다.
- 조율 에이전트는 현재 저장소와 실행 근거를 확인해 작업을 나누고, 실행 에이전트끼리 같은 파일을 동시에 수정하지 않도록 담당 파일을 분리한다. 실행 에이전트의 결과는 조율 에이전트가 검토·검증한 뒤 커밋한다.
- 실행 에이전트는 받은 범위 안에서 직접 구현·검증하며, 범위 밖 변경이나 오래돼 보이는 규칙은 임의로 고치지 않고 보고한다. 사용자가 특정 에이전트에 구현·분석을 직접 요청하면 그 요청을 우선한다.
- 위임 지시(프롬프트)에는 작업 목적과 범위, 확인할 파일, 완료된 결과와 근거, 우선순위가 있는 다음 작업, 금지 사항과 정보 경계, 검증 방법, 산출물과 종료 조건을 포함한다. 작업 규모에 맞게 작성하고 이전 대화 전체를 반복하지 않는다.
- 직접 확인한 사실·이전 에이전트의 보고·추론을 구분한다. 완료된 작업의 불필요한 반복을 피하고, 미검증 결론이나 오래된 상태를 확정된 사실로 전달하지 않는다.
- 필요한 데이터가 없으면 정확한 확보 대상과 필요한 이유를 명시하고, 데이터 확보와 독립적으로 진행할 수 있는 작업도 지시한다.
- 사용자가 파일 저장을 요청하지 않으면 별도 인계 파일이나 상태 요약 문서를 계속 만들지 않는다. 사용자가 다른 도구로 넘길 인계 프롬프트를 요청하면 대화에 제공한다.

#### Git 정리의 상시 승인

- 사용자는 검증된 변경의 커밋·푸시·fast-forward 병합을 별도 확인 없이 처리하도록 승인했다. 작업을 마무리하는 에이전트(위임한 경우 조율 에이전트)가 필요한 Git 정리를 수행하며, 다른 세션에 작업을 넘길 때는 이 승인도 함께 전달한다. 이후 사용자의 커밋·푸시 금지 등 명시적 제한이 있으면 그 제한을 우선한다.
- 변경 범위와 검증 결과를 확인하고 원격 최신 상태를 가져온 뒤 `master`에 커밋·푸시한다. 임시 브랜치를 사용한 경우에만 fast-forward 가능 여부를 확인해 병합하며, 완료 후에는 `master`에 머문다. 로컬·원격 상태를 확인하고 강제 푸시·강제 초기화는 하지 않는다. 기존 브랜치는 불필요하게 동기화하거나 별도 요청 없이 삭제하지 않는다.
- 다른 에이전트가 작업 중이거나 출처가 불명확한 변경·기존 스테이징이 있으면 임의로 포함·해제·폐기하지 않는다. 동시 작업에 영향을 줄 수 있는 커밋·브랜치 전환·병합은 보류하고 이유를 보고한다. `Claude outputs/`는 별도 요청 없이 커밋하지 않는다.
- Git 잠금 파일은 관련 작업이 종료됐는지 확인하기 전에는 삭제하지 않는다. 분기 충돌이나 검증 실패가 있으면 상태를 보고하고 기존 작업을 보존한다.

### 3. 문서 관리

- README.md가 사람이 읽는 전체 설명·현재 결론·유효한 결과·남은 한계의 기준 문서다.
- 사용자는 README 중심 통합을 승인했다. 승인된 작업으로 상태가 바뀌면 README의 관련 설명과 근거 링크를 먼저 갱신한다.
- 보고서·노트북은 재현 가능한 실행 증거 또는 지정 제출 형식이 필요할 때만 추가한다. 매 작업마다 새로운 상태 요약 문서를 만들지 않는다.
- 제출용 보고서는 작성 시점의 초안이다. 제출 전에 README와 맞추며 독립적인 최신 사양으로 관리하지 않는다. (발표는 끝났으므로 이후에는 보존된 작성 시점 기록으로 취급한다.)
- PLAN.md, AUDIT.md, output/baseline_recovery.csv/.md 및 진단 시점 문서는 별도 수정 요청 없이 보존한다. 과거 기록의 수치·미해결 상태를 현재 사양으로 되살리지 않는다.
- 문서·코드·근거 파일을 추가하거나 옮기면 이 파일의 문서·코드·실행 근거 지도를 함께 갱신한다. 이 파일에는 지속적인 작업 규칙과 위치 안내만 둔다. 실험 수치·완료 테스트 수·현재 브랜치·임시 세션 ID는 복제하지 않는다.

### 4. 데이터·평가 경계

- 데이터가 필요한 작업은 실제 data/train.csv의 존재를 먼저 확인한다. 없으면 해당 작업을 중단하고 알린다. 합성 데이터를 실제 실행 결과로 대체하지 않는다.
- 타깃 결측은 미라벨이며 음성 정답이 아니다.
- 현행 평가 경로는 src/cv.py의 nested grid와 inner 경계 TE다. 상세 설정은 README와 실행 메타데이터를 확인한다.
- outer-valid는 트리 수·임계값·teacher/pseudo 선택에 사용하지 않는다.
- 전체 입력 묶음 기반 비타깃 집계와 미래 예측 시점의 정보 가용성은 구별한다. 미래 예측 성능을 주장하려면 별도 계약·검증이 필요하다.
- 스모크 결과는 성능 근거로 인용하지 않는다. 일부 Phase 재검증을 전체 Phase 검증으로 확대하지 않는다.
- 전처리 의미 개선과 성능 향상, 시드 표준편차와 통계적 유의성은 구별한다.

### 5. 실행·결과 보존

- 재현 명령·코드 구조는 README의 재현 절을 따른다. 장시간 Python 실행은 python -u로 로그를 남긴다.
- 새 학습 결과는 baseline_recovery_v2 계열 신규 경로로 분리한다. 데이터·코드·설정이 바뀌면 새 출력 이름을 사용한다.
- 스키마·실험 지문 검사를 유지한다. --force는 검증 우회 옵션이 아니다.
- 같은 출력 경로에 여러 프로세스를 동시에 쓰지 않는다.
- 변경에 맞는 검증을 수행하고 실행 여부를 정확하게 보고한다. 문서만 고쳤을 때 과거 테스트를 이번 실행 결과로 표현하지 않는다.

### 6. 보고와 새 대화 인계

- 직접 확인한 사실, 실제 실행 결과, 추론·권고를 구별한다.
- 수치에는 평가 대상·분할·시드·프로토콜과 근거 파일을 연결한다.
- 완료 보고에는 변경 내용, 검증, 남은 한계를 적는다. 미실행 작업을 완료로 쓰지 않는다.
- 새 대화에서는 이 파일 → README → 현재 작업에 필요한 코드/근거 순서로 재개한다.
- 새 대화에 전달할 내용은 구체적인 다음 작업, 미완료 실행 여부, 작업 범위와 사용자 제약이다. 긴 대화 전체를 다시 붙이거나 별도 상태 문서를 늘리지 않는다.
