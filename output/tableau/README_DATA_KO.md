# Tableau 추출 데이터 사전

Tableau Public 대시보드 3종(지연 패턴 탐색 / 날씨와 지연 / 모델이 틀리는 곳)에 쓰는 집계 CSV 15개의 설명서입니다. 대시보드 제작 절차는 [docs/TABLEAU_HANDOFF_KO.md](../../docs/TABLEAU_HANDOFF_KO.md)를 봅니다.

- 생성기: [scripts/build_tableau_extracts.py](../../scripts/build_tableau_extracts.py) — **모델 학습 없음.** 로컬 원자료와 추적된 실행 근거를 읽기만 합니다.
- 재생성: `PYTHONUTF8=1 uv run --locked --offline python -u scripts/build_tableau_extracts.py` (로컬 `data/train.csv`, 10분 가정 날씨 결합 gzip, P6_clean 행별 OOF가 필요)
- 입력 SHA-256·출력 행수·대조 결과: [tableau_extracts_manifest.json](tableau_extracts_manifest.json)
- 모든 CSV는 UTF-8(BOM 없음), LF 줄바꿈입니다. 무한대 구간 끝값은 빈칸으로 둡니다(Tableau가 숫자 열로 읽도록).
- 테스트: [tests/test_tableau_extracts.py](../../tests/test_tableau_extracts.py)(합성 데이터)

## 1. 집단(population)과 분모

모든 표에는 `population` 열이 있고, 대부분 `population_rows`(분모)·`n_rows`·`n_delayed`·`delay_rate = n_delayed / n_rows`가 있습니다. **서로 다른 집단의 지연율이나 점수를 한 축에 섞어 비교하지 않습니다.**

| `population` | 정의 | 행 수 | 지연 행 | 쓰는 대시보드 |
|---|---|---:|---:|---|
| `labeled_all` | `data/train.csv` 1,000,000행 중 `Delay`가 비결측인 행. 타깃 결측(미라벨)은 음성이 아니므로 제외 | 255,001 | 45,000 | 1 |
| `weather_eval` | 날짜가 귀속된 706,759행 중 라벨 행. 10분 공개 지연 **가정** 날씨 결합. 날씨 사용/미사용·분류기 비교의 평가 집단 | 180,332 | 31,805 | 2, 3(모델 비교) |
| `both_time_missing` | 원본 예정 출발·도착 시각이 모두 결측인 라벨 행. `weather_eval`에 포함된 행 0 | 3,031 | 519 | 3 |
| `oof_p6_clean` | `labeled_all`의 P6_clean 행별 OOF 예측 × 분할 시드 3개(42/1/7). 시드별 분모 255,001, 세 시드 합산 시 행-예측 765,003건 | 255,001 × 3 | 45,000 × 3 | 3 |

## 2. 파일별 설명

| 파일 | 행 | 크기(bytes) | 단위(grain) | 집단 | 출처 |
|---|---:|---:|---|---|---|
| `overview_populations.csv` | 3 | 459 | 집단 1개 | 위 3개 | 원자료 + 날씨 결합 |
| `d1_delay_by_airline.csv` | 29 | 2,639 | 항공사명(원본, 결측은 `(결측)`) | `labeled_all` | `data/train.csv` |
| `d1_delay_by_month.csv` | 12 | 963 | 월(1~12, **연도 없음**) | `labeled_all` | 같음 |
| `d1_delay_by_hour.csv` | 51 | 5,449 | `time_role`(departure/arrival) × 원본 예정 시각의 시(hour) | `labeled_all` | 같음 |
| `d1_delay_by_airport.csv` | 747 | 108,241 | `airport_role`(origin/destination) × 공항 | `labeled_all` | 같음 + 공항 좌표 |
| `d1_delay_by_route.csv` | 6,374 | 820,202 | 노선(출발-도착), 전 노선 | `labeled_all` | 같음 + 공항 좌표 |
| `d1_delay_by_airline_month.csv` | 331 | 28,958 | 항공사 × 월 | `labeled_all` | 같음 |
| `d2_weather_bins_long.csv` | 72 | 10,726 | `role` × `feature` × 구간 | `weather_eval` | 10분 가정 결합 + `src/weather_model.py` 피처 |
| `d2_delay_by_year_month.csv` | 24 | 1,935 | 귀속 연-월(2018-01~2019-12) | `weather_eval` | 결합 파일의 `attributed_date` |
| `model_confusion_long.csv` | 60 | 6,821 | 실험 × 시드 × 조건/모델 × 혼동행렬 칸(TN/FP/FN/TP) | `weather_eval` | 추적된 runs CSV 2개 |
| `model_metrics_long.csv` | 210 | 26,151 | 실험 × 시드 × 조건/모델 × 지표 | `weather_eval` | 같음(+ 칸 수에서 계산한 클래스별 지표) |
| `model_comparison_all.csv` | 729 | 194,114 | 실험 × 조건 × 모델 × 보정 조건 × 시드 × 지표 (분류기 실행 전체) | `weather_eval`(날씨 미사용 행은 같은 행에서 날씨 피처만 뺀 조건) | 추적된 분류기 runs CSV 5개(비교·탐색 확장·확대 2부분·보정) |
| `model_calibration_reliability.csv` | 1,125 | 339,564 | 시드 × 모델 × 보정 조건 × 구간 방식 × 구간 | `weather_eval` | 추적된 보정 실험의 `reliability_bins.csv` |
| `d3_oof_errors_by_group.csv` | 3,081 | 351,033 | `dimension` × `group` × 시드, TN/FP/FN/TP | `oof_p6_clean` | 로컬 행별 OOF(P6_clean, v2) |
| `d3_oof_error_stability.csv` | 1,027 | 71,630 | `dimension` × `group`, 세 시드 중 틀린 횟수(0~3) 분포 | `oof_p6_clean`(행 단위 255,001) | 같음 |
| `tableau_extracts_manifest.json` | – | – | 입력 해시·출력 해시·대조 결과 | – | 생성기 |

### 열 설명 (공통)

| 열 | 의미 |
|---|---|
| `population`, `population_rows`, `population_delayed` | 집단 코드와 그 집단 전체의 행·지연 행(분모) |
| `n_rows`, `n_delayed`, `n_not_delayed` | 해당 칸의 행 수와 실제 지연/정상 행 수 |
| `delay_rate` | `n_delayed / n_rows` (실제 라벨 기준 지연율, 예측 아님) |
| `share_of_population` | `n_rows / population_rows` |
| `rank_by_rows` | 행 수 기준 순위(동률은 먼저 나온 순) — Tableau Top N 필터용 |

### 파일별 주의

- **`d1_delay_by_hour.csv`**: 원본 `Estimated_Departure_Time`/`Estimated_Arrival_Time`(HHMM)의 `floor(값/100)`입니다. **모델 전처리의 시각 복원값이 아닙니다.** 원본 시각이 없는 행은 `결측(원본 시각 없음)`(정렬값 −1)으로 남깁니다(출발 27,841행, 도착 27,684행). 도착 `2400` 표기 5행은 `24 (2400 표기)`로 따로 둡니다. 시각은 각 공항의 현지 예정 시각입니다.
- **`d1_delay_by_airline.csv`**: 원본 `Airline` 문자열입니다(결측 27,540행은 `(결측)`). 같은 운항사 코드에 여러 이름이 대응하는 원자료 특성이 있어(README 3절 ③) 이름 단위 비교는 원본 표기 기준입니다.
- **`d1_delay_by_airport.csv` / `d1_delay_by_route.csv` 좌표**: 추적된 [`output/baseline_recovery_v2_weather_scope_fix_20260918_mapping_table.csv`](../baseline_recovery_v2_weather_scope_fix_20260918_mapping_table.csv)의 `lat`/`lon`(관측소 매핑 때 캐시한 공항 목록의 공항 기준점)입니다. 라벨 행의 공항 374개가 모두 좌표를 가집니다(누락 0). 인터넷에서 새로 가져오지 않았습니다. `weather_station_tier`는 날씨 관측소 매핑 등급이며 좌표 정확도 등급이 아닙니다.
- **`d1_delay_by_route.csv`**: 6,374개 노선 전체입니다. 행 수가 작은 노선의 지연율은 불안정하므로 Tableau에서 `n_rows` 최소값 필터(권장 100 이상, 해당 582개 노선)를 둡니다.
- **`d2_weather_bins_long.csv`**: 각 `role`(origin/destination) × `feature`의 구간 합계가 180,332행이 되도록 모든 행에 구간을 하나씩 줍니다.
  - `bin_kind`: `value`(수치 구간), `special`(풍속 정확히 0, 강수 0, 미량), `field_missing`(관측 보고는 결합됐지만 해당 필드가 `M`), `unmatched`(관측 보고 미결합), `partial`(결합 상태: 한쪽만 결합).
  - 강수 `p01i`: `0 inch(강수 없음 보고)`, `미량(trace, 0.0001 표기)`, `(0.0001, 0.05) inch`, `[0.05, 0.1) inch`, `≥ 0.1 inch`. **0.0001은 IEM 요청 `trace=0.0001`로 표기된 미량 강수이며 측정된 0.0001 inch가 아닙니다.**
  - 풍속 `sknt`: `0 knot(무풍 보고)`과 `(0, 6) knot`을 구분합니다. 기온은 °F, 시정은 mile, 관측 나이는 분(10분 가정상 최솟값 10, 상한 90)입니다.
  - `feature = matched_status`, `role = both`: 양쪽 결합 / 출발만 / 도착만 / 양쪽 미결합.
  - 구간 경계는 결과를 보고 고른 것이 아니라 단위의 관례적 구간입니다. 구간별 지연율은 **연관(association)** 이며 날씨가 지연을 일으켰다는 근거가 아닙니다.
- **`model_confusion_long.csv`**: `experiment = weather_on_off_lightgbm`(variant `weather_off`/`weather_on`, 3시드)과 `classifier_compare_weather_on`(variant `lightgbm`/`logistic_regression`/`random_forest`, 3시드). 칸 값은 추적된 runs CSV의 `tn/fp/fn/tp`를 그대로 옮긴 것이며, 각 폴드에서 내부 선택한 임계값 기준의 pooled OOF입니다. `actual_class_rows`는 그 칸의 실제 클래스 행 수(행 기준 비율의 분모)입니다.
- **`model_metrics_long.csv`**: `source = tracked runs CSV`는 원본 값(`macro_f1_nested`, `log_loss`, `roc_auc`, `f1_at_050`, `deployment_threshold`), `derived from tracked tn/fp/fn/tp`는 칸 수에서 계산한 클래스별 지표입니다. `f1_at_050`은 임계값 0.5의 **Macro F1**입니다(Delayed F1 아님). 3시드 평균은 Tableau에서 `AVG`로 계산하며, 시드는 같은 행의 재분할이라 SD는 신뢰구간이 아닙니다.
- **`model_comparison_all.csv`**: 같은 180,332행·외부 폴드·시드에서 실행한 분류기 결과를 한 표로 모았습니다(새 학습 없음, 추적된 runs CSV 값을 그대로 옮김). 행은 (`experiment`, `condition`, `model`, `calibration_arm`, `seed`, `metric`) 하나입니다.
  - `experiment`: `classifier_compare_20261008`(작은 후보 비교), `classifier_tuning_20261008`(LightGBM 24개·Random Forest 9개 설정), `classifier_grid_ext_20261008_lgbm`·`_rf`(LightGBM 18개·Random Forest 15개 설정 확대), `classifier_calibration_20261008`(보정 조건 5개).
  - `model`은 실행 기록의 원래 키(`lightgbm`, `lightgbm_tuned`, `lightgbm_ext`, `random_forest*`, `logistic_regression`), `model_family`는 알고리즘 묶음, `model_label`은 표시명입니다. `search_budget`·`n_configurations`는 각 실행의 사전 선언 탐색 예산이며 실험마다 다릅니다(비교 표에서 예산을 함께 표시).
  - `condition`: `weather_on`/`weather_off`(날씨 미사용은 탐색 확장·확대의 Random Forest만). `calibration_arm`: `none`, `platt_crossfit`, `isotonic_crossfit`, `platt_inner_holdout`, `isotonic_inner_holdout`(보정 실험만, 나머지는 `none`). `inner_holdout` 조건은 설정·임계값을 고른 행을 보정에 다시 쓴 비교용 조건입니다.
  - `metric`: 모든 실험에 `macro_f1_nested`, `log_loss`, `roc_auc`, `precision_delayed`, `recall_delayed`, `f1_delayed`, `f1_not_delayed`. 보정 실험에는 `brier`, `ece_ef15`(동일 빈도 15구간, 주 지표), `ece_ew10`(동일 폭 10구간), `calibration_bias`(평균 예측 확률 − 실제 지연율), `mean_probability`도 있습니다. 3시드 평균은 Tableau에서 `AVG(value)`로 계산합니다(시드별 값이라 평균이 정의대로 맞음).
  - **같은 실행이 두 번 나오는 경우가 있습니다.** 확대 실험의 날씨 사용 Random Forest는 9개 설정 결과와 같은 설정을 골라 값이 같고, 보정 실험의 `none`은 기록된 설정을 다시 적합한 값입니다(LightGBM·Random Forest는 기존 실행과 같고, Logistic Regression은 Macro F1이 시드별 최대 0.00022 다름). 한 화면에서는 `experiment`를 하나 고르거나 예산별로 나눠 보이고, 서로 다른 실험의 행을 합산하지 않습니다.
- **`model_calibration_reliability.csv`**: 보정 실험의 신뢰도 구간(reliability bin)을 옮긴 표입니다. 구간은 외부 검증 예측을 기준으로 시드·모델·보정 조건마다 따로 정했습니다.
  - `binning`: `equal_frequency_15`(주 지표, 구간마다 행 수가 거의 같음), `equal_width_10`(보조, 빈 구간은 `n = 0`이고 확률·비율 열이 빈칸).
  - `mean_probability`(구간 평균 예측 확률), `observed_rate`(구간 실제 지연율), `abs_gap = |mean_probability − observed_rate|`, `gap_signed = mean_probability − observed_rate`(양수 = 과대 예측).
  - 시드를 합칠 때 쓰는 개수 열: `n_delayed = observed_rate × n`(정수로 확인), `sum_probability = mean_probability × n`. 합친 구간의 실제 지연율은 `SUM(n_delayed)/SUM(n)`, 평균 예측 확률은 `SUM(sum_probability)/SUM(n)`입니다. 다만 동일 빈도 구간의 경계는 시드마다 달라 같은 `bin` 번호를 합치면 근사입니다. 정확한 곡선은 시드 하나씩 봅니다.
  - ECE(구간 가중 평균 `SUM(n × abs_gap)/SUM(n)`)는 runs CSV의 `ece_ef15`/`ece_ew10`과 일치함을 생성기가 확인합니다.
- **`d3_oof_errors_by_group.csv`**: `dimension` ∈ `airline`, `dep_hour`(원본 예정 출발 시), `month`, `origin_airport`, `route`, `raw_time_pattern`. `route`는 라벨 100행 이상 582개 노선(93,975행)만 개별로 두고 나머지는 `(기타: 라벨 100행 미만 노선)`으로 묶어 시드별 합계가 255,001이 되게 했습니다. `group_sort`는 월·시 정렬용입니다. 지표: `recall_delayed = tp/(tp+fn)`, `precision_delayed = tp/(tp+fp)`, `false_positive_rate = fp/(fp+tn)`, `error_rate = (fp+fn)/n_rows`. 세 시드를 합칠 때는 칸 수를 먼저 `SUM`한 뒤 비율을 계산합니다.
- **`d3_oof_error_stability.csv`**: 행마다 세 시드 중 틀린 횟수(0~3)를 세어 그룹별로 분포를 냅니다. 분모는 행 255,001(시드 곱하지 않음).

## 3. 공통 주의사항(대시보드 캡션에 반영)

- **`Traffic`은 실제 공항 혼잡도가 아니라 표본 레코드 수**이고 집계 키에 연도가 없습니다. 이 추출에는 `Traffic`을 넣지 않았으며, 대시보드에서 "혼잡도"로 표현하지 않습니다.
- **원본에는 연도가 없습니다.** `month`는 2018·2019년이 섞인 월입니다. 연-월은 `weather_eval`(날짜 귀속 행)에서만 `attributed_date`로 표시할 수 있습니다.
- **날씨는 10분 공개 지연 가정**으로 결합했습니다. 10분은 실측 공개 지연 시간이 아닙니다. 예측 시점은 예정 출발 60분 전이고, 관측 나이 90분 이하만 사용합니다.
- 모델 결과는 **선택 집단의 정적 교차검증**입니다. 미래 운항 성능이나 날씨의 인과 효과로 일반화하지 않습니다.
- **시드는 같은 행의 재분할**입니다. 독립 데이터셋이 아니며 시드 SD는 신뢰구간이 아닙니다.
- `d3_*`는 **P6_clean(날씨 미사용) LightGBM의 라벨 255,001행 OOF**입니다. 날씨 사용 모델·분류기 비교의 행별 예측은 로컬에도 없으므로(체크포인트에는 집계값만 있음) 날씨 모델의 그룹별 오류는 만들지 않았습니다.
- 3,031행 집단은 날짜 귀속 키가 불완전해 `weather_eval`에 들어 있지 않습니다(교집합 0행). 날씨 개선을 이 집단의 개선 근거로 쓰지 않습니다.

## 4. 대조 결과 (2026-10-08 실행, 분류기 표 2개 추가 후 재실행)

생성기가 실행 중 확인하고 manifest의 `reconciliation.checks`에 기록합니다. 하나라도 어긋나면 생성기가 실패합니다. 이번 실행은 **전 항목 통과**(`all_checks_passed: true`, 334개)입니다. 분류기 표 2개를 추가하며 다시 실행했을 때 기존 CSV 13개는 바이트 단위로 같았습니다(manifest에는 입력·출력 항목만 추가).

| 대조 항목 | 기대값 | 결과 |
|---|---|---|
| D1 표(항공사, 월, 출발 시, 도착 시, 출발 공항, 도착 공항, 전체 노선, 항공사×월)의 `n_rows`/`n_delayed` 합 | 255,001 / 45,000 | 8개 표 모두 일치 |
| D2 구간 표: role × feature 11개 조합(출발·도착 각 5개 피처 + 결합 상태) 각각의 합 | 180,332 / 31,805 | 모두 일치 |
| D2 연-월 표 합 | 180,332 / 31,805 | 일치 |
| 날씨 사용/미사용 혼동행렬(3시드 × 2조건) | runs CSV의 `tn/fp/fn/tp` | 6개 모두 칸별 일치, 칸 합 180,332, FN+TP 31,805 |
| 분류기 비교 혼동행렬(3시드 × 3모델) | runs CSV의 `tn/fp/fn/tp` | 9개 모두 일치, 칸 합 180,332 |
| `model_comparison_all`: 실험 5개의 (조건 × 모델 × 보정 조건) 24개 묶음 | 시드 42/1/7, 분모 180,332 / 31,805, Macro F1·LogLoss·ROC-AUC 3시드 평균이 각 summary JSON과 같음(1e-12 이내), 클래스별 precision·recall·F1이 `tn/fp/fn/tp`와 같음 | 모두 일치, summary JSON의 모델 목록과 빠짐없이 대응 |
| `model_calibration_reliability`: 시드 × 모델 × 보정 조건 × 구간 방식 90개 묶음 | 구간 `n` 합 180,332, `n_delayed` 합 31,805, 구간 가중 ECE = runs CSV의 `ece_ef15`/`ece_ew10`(1e-9 이내) | 90개 모두 일치 |
| D3 OOF 시드별 전체 TN/FP/FN/TP | 추적된 `baseline_recovery_v2_oof_20260916_v2_groups.csv`의 P6_clean `overall` | 3시드 모두 일치 |
| D3 `raw_time_pattern` 4개 그룹 × 3시드 | 같은 groups CSV | 12개 모두 일치 |
| D3 각 dimension × 시드 합 / 안정성 표 합 | 255,001 / 45,000 | 모두 일치 |
| 양쪽 시각 결측 집단 | 3,031행 / 지연 519행, `weather_eval` 교집합 0 | 일치 |
| 양쪽 시각 결측 집단의 지연 recall(시드 42/1/7) | README 3시드 평균 10.02% | 10.597% / 9.441% / 10.019%, 평균 10.019% |
| 날씨 미결합 행(origin / destination) | 해설 문서 4,185 / 4,154 | 4,185 / 4,154 |
| 미량 강수 0.0001 행(origin / destination) | 해설 문서 10,952 / 11,099 | 10,952 / 11,099 |
| 좌표 없는 공항 | – | 0개(374개 모두 좌표 있음) |

## 5. 행 단위 파일 (로컬 전용, 커밋 금지)

- 위치: `data/tableau/flights_rowlevel.csv` (255,001행, 약 59 MB, `data/` 전체가 `.gitignore` 대상)
- **데이터셋 재배포 조건이 확인되지 않았으므로 커밋·업로드·Tableau Public 게시에 쓰지 않습니다.** Tableau Desktop에서 로컬 탐색용으로만 씁니다. Tableau Public에 게시하는 통합 문서는 `output/tableau/*.csv` 집계 파일만 연결합니다.
- 열: `ID`, `month`, `day_of_month`, 원본 시각 기반 `dep_hour_*`/`arr_hour_*`, `raw_time_pattern`, 공항·노선·항공사·`carrier_code`·`distance_miles`, `delayed`, `in_weather_eval`, 공항 좌표, `attributed_date`와 날씨 피처 14개(`weather_eval` 행만 값이 있음), P6_clean OOF 확률·예측(시드 42/1/7), `oof_p6clean_error_seed_count`.

## 6. Git 추적 메모

`.gitignore`에 `!output/tableau/*.csv` 예외가 있어 이 폴더의 집계 CSV는 추적됩니다. 행 단위 파일(`data/tableau/`)은 `data/` 규칙으로 계속 무시됩니다.
