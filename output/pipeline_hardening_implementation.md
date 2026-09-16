# 파이프라인 감사 후 구현 결과 — 2026-09-16

## 범위와 상태

- 사용자 지시에 따라 대화 모델의 직접 구현·터미널 실행 제한을 해제했다.
- CLAUDE.md의 역할 분담을 갱신하고 AGENTS.md에 공통 규칙을 추가했다.
- 작업 브랜치: feat/nested-n-estimators. 이번 작업에서 commit/push하지 않았다.
- data/train.csv 실존 확인: 116,707,688 bytes. 전체 Phase 재학습은 실행하지 않았다.
- PLAN.md / README.md / AUDIT.md / baseline_recovery.csv·md 및 기존 진단·감사 보고서는 보존했다.

## 변경 사항 [코드로 확인]

| 감사 항목 | 수정 | 확인 범위 |
| :--- | :--- | :--- |
| F1/F2 | v2 출력 분리, 엄격한 CSV 스키마/실험 ID 검증, 원자적 Phase upsert | 구 스키마·다른 실험 거부, 강제 재실행에서도 검증 우회 금지 |
| F3 | student inner 분할 이후 extra_fit_factory 호출 | honest teacher는 student inner-train 피처/라벨만 받음 |
| F4 | teacher도 run_fold_nested_grid로 전환 | teacher 자체 holdout 경계에서 TE를 재계산 |
| F5/F7 | 선택 모델의 inner-holdout 확률로 임계값 선택 | outer-valid는 k와 임계값 선택에 관여하지 않음 |
| F6 | 현재 코드 설명과 새 보고서의 서사를 정정 | 역사적 문서는 수정하지 않음 |
| F8 | 실제 run_phase → teacher → student 경로의 행 ID 검사 추가 | teacher 입력과 student holdout/outer-valid의 교집합이 공집합 |

주요 구현 위치:
- src/cv.py: run_fold_nested_grid에 extra_fit_factory / selection_metadata 선택 인자 추가.
  기존 FoldFitGrid 반환 형태와 run_fold/oof_target_encode 시그니처는 보존한다.
- src/cv.py: evaluate_oof에 사전 선택 per_fold_thresholds 인자 추가.
  이 경로는 Macro F1뿐 아니라 혼동행렬/recall에도 동일한 fold별 임계값을 적용한다.
- rerun_all_phases.py: honest pseudo 생성 시점을 student 내부 분할 이후로 이동.
  teacher 자체 그리드는 student inner-train 안에서만 선택한다.
- src/run_store.py: 헤더/행 폭/중복 Phase/실험 ID를 검사하고 임시 파일→원자적 교체로 저장.
  단일 writer를 전제로 하며 동시에 여러 학습 프로세스가 같은 결과 파일에 쓰면 안 된다.

### 출력 및 재개 계약

기본 전체 실행 경로는 output/baseline_recovery_v2.csv·md다.
sample 실행 기본 경로에는 sample 행 수를 붙인다. --output-prefix는
baseline_recovery_v2로 시작하는 파일명만 허용해 기존 진단 파일을 보호한다.

각 행은 run_id/run_metadata를 기록한다. 식별 정보에는 데이터 SHA256,
핵심 소스 SHA256, CV/TE/모델/그리드/Phase 설정, sample 크기, 라이브러리 버전,
teacher·임계값 선택 방식을 포함한다. 실제 경로는 inner_early_stopping=False로 기록한다.
--force는 동일 실험의 선택 Phase만 교체하고 다른 Phase는 보존한다.
코드나 설정이 달라지면 --force로도 이전 파일에 섞어 쓰지 못하며 새 prefix가 필요하다.
주석 수정도 소스 SHA256을 바꾸므로 재개가 보수적으로 차단될 수 있다.

### 임계값 선택의 의미

각 outer fold의 inner-holdout에서 k와 임계값을 모두 선택하고 outer-valid에 적용한다.
튜닝용 holdout 점수는 성능 추정치로 보고하지 않는다. 동일 holdout의 다중 선택으로
선택 분산은 발생할 수 있으나 outer-valid 라벨의 교차 유입 경로는 제거했다.
기존 tune_threshold_nested는 호환용으로 남기고 cross-fold OOF 의존성 한계를 명시했다.
naive F1과 새 보고 F1의 차이는 진단값이지 낙관 편향의 인과 추정치가 아니다.

### 의도적 대조군과 서사 정정

- P5_leaky의 전체 teacher 앙상블은 의도적 outer 라벨 누수 대조군으로 보존했다.
  honest 경로의 안전성 주장을 이 조건에 적용하지 않는다. 성능 순위에서 제외한다.
- P4_nospw는 현재 P4와 같은 호환 별칭이다. spw ablation/순위에 재사용하지 않는다.
- 54는 과거 fold 2의 선택된 best_iteration이고, 499는 ES 없이 강제 학습한
  오염된 holdout 곡선의 argmin이다. 이를 ES가 499까지 실제 진행했다는 뜻으로 쓰지 않는다.
- train 라벨→holdout TE는 정상이다. holdout 라벨→학습/검증 피처의 유입이 문제다.
- 고정 트리 수 자체가 누수를 제거하지 않는다. 특징 생성과 선택의 라벨 경계가 핵심이다.
- 자동 보고서의 단일 시드 인과 추론과 0.002 기반 유의성 판정을 제거했다.
  Macro F1을 우선 정렬하고 LogLoss/AUC를 병기하는 기술 통계만 출력한다.

## 검증 [실행 결과]

| 검증 | 결과 |
| :--- | :--- |
| 기존 tests/test_features.py | 124 passed |
| 추가 회귀 테스트 포함 tests 전체 | 140 passed, 6.35초 |
| 실제 train.csv 앞 8,000행, P4/P5_honest 5-fold, 실제 LightGBM | 두 Phase 정상 완료 |
| 동일 CLI로 재개 | 두 Phase skip, CSV 바이트 불변 |
| 같은 스모크에서 --force --phases P4 | P4 교체, P5 행 동일, 총 2행, 열 밀림 없음 |
| git diff --check | 통과 |

신규 테스트는 합성 행을 단위 검증에만 사용한다. teacher 입력 범위, holdout 라벨을
뒤집어도 학습용 TE와 pseudo 생성자 입력이 불변인 점, outer-valid 피처가 변해도
k/임계값이 불변인 점, 고정 임계값과 혼동행렬의 일치, 외부 eval_set/사전 TE 입력
거부, 구 CSV 스키마와 실험 지문 변경 거부를 확인했다.

스모크 파일: output/baseline_recovery_v2_verified_8000.csv·md.
라벨 2,058행 / 미라벨 5,942행이며 이 수치를 전체 데이터 성능으로 인용하지 않는다.
먼저 생성한 baseline_recovery_v2_smoke_8000 파일은 메타데이터 설명 보완 전 중간
스모크 기록이다. 각 파일의 소스 지문은 해당 실행 시점의 것이다. 검증 이후
레거시 run_fold 독스트링의 잘못 삽입된 인자 설명만 제거했다(동작 변화 없음).

## 남은 실험과 승인 지점 [판단]

- 전체 데이터 성능, 변경된 임계값 선택의 F1 영향, P5 teacher 변경 영향은 미측정이다.
  과거 Phase 4 F1 회귀 목표를 새 임계값 프로토콜에 그대로 요구할 수 없다.
- Teacher도 8개 후보를 학습하므로 P5 비용이 증가한다. 성능 개선을 단정하지 않는다.
- 후속 Phase 비교는 같은 시드·분할의 차이를 짝지어 보고하고 3시드 표준편차만으로
  유의/동률을 확정하지 않는다. 배포용 전체 데이터 재학습·임계값 검증은 별도다.
- 구조 수정/단위 검증/축소 실행은 완료했다. 전체 데이터 Phase 재학습은 사용자
  승인 전 실행하지 않는다. 결과는 baseline_recovery_v2 계열 새 파일에 저장한다.

추후 문서 갱신 검토: AUDIT의 inner/outer 경계와 threshold 독립성 범위,
PLAN의 0.002 해석 및 최신 프로토콜, README의 v2 실행/재개 사용법.
원문들은 이번 작업에서 변경하지 않았다.
