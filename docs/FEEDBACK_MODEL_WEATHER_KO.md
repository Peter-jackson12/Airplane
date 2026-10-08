# 튜터 피드백 5·6·7 대응: 날씨 피처 계약, LightGBM 설정, F1 표기

> 이 문서는 [튜터 피드백 인계 문서](TUTOR_FEEDBACK_HANDOFF_KO.md)의 5)·6)·7)을 닫기 위한 **대조 기록**입니다. 현재 결론의 기준은 [README](../README.md)이며, 이 문서는 새로운 실험 결과가 아닙니다.
>
> - 대조 기준: `master`의 `54f011f` 시점 코드와 추적된 실행 근거, 그리고 로컬에만 있는 결합 산출물·IEM 캐시.
> - **모델 학습, 외부 수집, 기존 `output/` 파일 수정은 하지 않았습니다.** 2.3~2.5절의 범위·개수는 로컬 결합 파일과 캐시를 읽기 전용으로 집계한 값입니다(재현 방법은 2.7절).
> - 확인하지 못한 부분은 **미확인**으로 표시했습니다.
> - LightGBM과 다른 분류기(Logistic Regression, Random Forest)의 공정 비교는 2026-10-08에 별도로 수행했습니다. 결과는 [README 분류기 3종 비교](../README.md#classifier-comparison)를 참고하세요.

## 목차

| 절 | 내용 | 피드백 |
|---|---|---|
| [1. 지표 정의](#metrics) | Delayed=1 / Not_Delayed=0, Macro F1과 Delayed F1, 필드 이름의 실제 계산 | 7) |
| [2. 날씨 피처 명세](#weather) | 14개 피처의 의미·원천 필드·단위·변환·관측 범위, `M`·trace·결합 실패 표기 | 5) |
| [3. LightGBM 고정/탐색 설정](#lightgbm) | 선택 이유, 설정표, 버전, protocol 라벨 오기 | 6) |
| [4. 표기 정리 내역](#wording) | 해설 문서 수정 목록과 README 점검 결과 | 7) |
| [5. 남은 한계](#limits) | 미확인 항목 | 5)·6)·7) |

<a id="metrics"></a>
## 1. 지표 정의 (피드백 7)

### 1.1 클래스와 계산 위치

- 정답 매핑: `Not_Delayed → 0`, `Delayed → 1` ([`TARGET_MAP`](../src/features.py)). 양성(positive)은 **Delayed=1**입니다.
- 지표 계산: [`src/cv.py`](../src/cv.py)의 `evaluate_oof`, 임계값 선택은 같은 파일의 `run_fold_nested_grid`.
- 모든 F1 계산은 scikit-learn `f1_score(..., average="macro")`이며 `pos_label`을 지정하지 않습니다. `average="macro"`에서는 `pos_label`이 쓰이지 않습니다. 이진 라벨에 두 클래스가 모두 있으므로 대상 클래스는 {0, 1}입니다. ([`src/oof.py`](../src/oof.py)의 그룹 진단은 `labels=[0, 1]`, `zero_division=0`을 명시합니다.)

### 1.2 정의

| 이름 | 정의 | 이 저장소에서 쓰는 곳 |
|---|---|---|
| **Delayed F1** (= F1(Delayed), 지연 클래스 F1) | `2TP / (2TP + FP + FN)` (양성=Delayed) | 인계 문서 3절, README 결과표의 클래스별 행 |
| **Not_Delayed F1** (정상 클래스 F1) | `2TN / (2TN + FN + FP)` (Not_Delayed를 양성으로 본 F1) | 같은 곳 |
| **Macro F1** | `(Delayed F1 + Not_Delayed F1) / 2`, 두 클래스의 단순 평균(가중 평균 아님) | 주 지표. 모든 임계값 선택과 표지 수치 |

대조: 날씨 비교의 3시드 평균 Delayed F1 0.313735 / Not_Delayed F1 0.834084의 평균은 0.573910이고, 사용 조건은 (0.344182 + 0.853708) / 2 = 0.598945입니다. 두 값은 [요약 JSON](../output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_summary.json)의 `macro_f1_nested` 평균과 일치합니다. 시드별로 먼저 계산한 뒤 평균한 값이며, 클래스별 수치의 출처는 [인계 문서 3절](TUTOR_FEEDBACK_HANDOFF_KO.md)입니다.

### 1.3 저장된 필드 이름과 실제 계산

| 필드 (`runs.csv`, `summary.json`) | 실제 계산 | 주의 |
|---|---|---|
| `macro_f1_nested` (`evaluate_oof`의 `macro_f1`) | **Macro F1.** 각 외부 fold의 내부 holdout에서 선택한 임계값으로 그 fold의 OOF 행을 판정하고, 5개 fold를 합친(pooled) 예측으로 한 번 계산 | README의 주 지표 |
| `f1_at_050` | **Macro F1**, 임계값 0.5 고정, pooled OOF | 이름에 "macro"가 없지만 Delayed F1이 아닙니다. 3시드 평균 미사용 0.453731 / 사용 0.478858 |
| `naive_macro_f1` | Macro F1, 전체 OOF에서 사후에 최댓값 임계값을 고른 값 | 낙관 편향 진단용. 성능 수치로 인용하지 않습니다 |
| `recall` | **Delayed recall** = `TP / (TP + FN)`, fold별 선택 임계값 기준 | Macro 평균이 아닙니다 |
| `tn, fp, fn, tp` | 행=실제 [0, 1], 열=예측 [0, 1]인 `confusion_matrix(labels=[0, 1])` | fold별 선택 임계값 기준 |

기존 실행 파일의 필드 이름은 바꾸지 않았습니다. 문서에서 `f1_at_050`을 인용할 때는 "임계값 0.5의 Macro F1"이라고 씁니다.

<a id="weather"></a>
## 2. 날씨 피처 명세 (피드백 5)

### 2.1 원천과 수집 계약

| 항목 | 값 | 근거 |
|---|---|---|
| 원천 | Iowa Environmental Mesonet(IEM) ASOS/AWOS 관측 다운로드 | [수집 코드](../notebooks/fetch_weather_full_sharded.py) `build_bulk_request_url` |
| 요청 필드 | `tmpf, dwpf, relh, sknt, gust, vsby, p01i, skyc1, wxcodes, snowdepth, metar` | 같은 코드, [결합 코드](../src/weather_full.py) `WEATHER_FIELDS` |
| 시간대 | `tz=UTC` | 같은 코드 |
| 보고 유형 | `report_type=[3, 4]` | 같은 코드, [수집 manifest](../output/baseline_recovery_v2_weather_full_bulk_20260921_full_weather_fetch_manifest.json) |
| 결측 표기 요청 | `missing=M` | 같은 코드, manifest `missing_representation: "M"` |
| 미량 강수 표기 요청 | **`trace=0.0001`** | 같은 코드, manifest `trace_representation: "0.0001"` |
| 캐시 규모 | 1,317개 파일, 7,299,100행 | manifest `cache_file_count`, `total_cache_rows` |

단위는 [해설 5.2절](README_EXPLAINED_KO.md)이 [IEM 변수 설명](https://mesonet.agron.iastate.edu/request/download.phtml)을 기준으로 정리한 것을 따랐습니다. 이번 작업에서는 외부 문서를 다시 조회하지 않았습니다(2.6절).

### 2.2 14개 모델 피처

[`src/weather_model.py`](../src/weather_model.py)의 `WEATHER_MODEL_FEATURES`입니다. `{role}`은 `origin`(출발 공항)과 `destination`(도착 공항)이며 두 공항 모두 **같은 `prediction_at`**(출발 예정 UTC − 60분)을 기준으로 결합합니다.

| 피처 (role별 1개씩) | 의미 | 원천 | 단위 | 변환 |
|---|---|---|---|---|
| `weather_{role}_tmpf` | 기온 | IEM `tmpf` | °F | 없음. 문자열을 실수로 변환만 함 |
| `weather_{role}_dwpf` | 이슬점 | IEM `dwpf` | °F | 없음 |
| `weather_{role}_sknt` | 풍속 | IEM `sknt` | knot | 없음 |
| `weather_{role}_vsby` | 시정 | IEM `vsby` | mile | 없음 |
| `weather_{role}_p01i` | 1시간 강수 관측. 누적 구간은 관측소의 직전 시간 강수 초기화 시각에 따라 다를 수 있음 | IEM `p01i` | inch | 없음. 단 미량 강수는 원천 요청 단계에서 `0.0001`로 표기(2.4절) |
| `weather_{role}_age_minutes` | 예측 시점 − 관측 시각(관측 나이) | 결합 단계 계산값 `{role}_weather_age_minutes` | 분 | 계산 피처. 결합 조건상 0~90 |
| `weather_{role}_matched` | 관측 보고가 결합됐는지 | `{role}_observed_at`의 비결측 여부 | 0/1 정수(`int8`) | 파생 피처. 결합 실패 시 0 |

- 피처 수: 5개 기상값 + 관측 나이 + 결합 플래그 = role당 7개, 2개 role로 **14개**. 실행 기록의 `n_features`는 미사용 30 / 사용 44로 정확히 14 차이입니다([`runs.csv`](../output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_runs.csv)).
- 수치 처리: `extract_weather_features`는 `pd.to_numeric(errors="raise")`로 읽고 유한값인지 검사합니다. 단위 변환, 대치, 클리핑, 행 제외는 없습니다. LightGBM이 결측(NaN)을 직접 처리합니다.
- 제외 필드(`relh, gust, snowdepth, wxcodes, skyc1, metar`)와 사유는 [요약 JSON](../output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_summary.json)의 `excluded_weather_fields`에 있습니다. 이 피처 목록은 모델 점수로 고르지 않았습니다(`feature_selection_policy`).

### 2.3 관측 범위 (결합된 행의 실제 값)

대상은 모델이 읽은 10분 가정 결합 파일 `joined_latency10min.csv.gz`입니다. 이 파일은 로컬 전용이며, SHA-256 `28f015b5…` 값이 [모델 manifest](../output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_manifest.json)의 기록과 일치함을 확인했습니다. 아래 값은 **평가 집단 180,332행 중 해당 role이 결합된 행**에서 결측을 뺀 값입니다. 괄호 안은 날짜 채택 전체 706,759행에서 다른 경우의 값입니다.

| 피처 | 최솟값 | 1% 분위 | 중앙값 | 99% 분위 | 최댓값 | 값이 정확히 0인 행 |
|---|---:|---:|---:|---:|---:|---:|
| origin `tmpf` | −33 (−40) | 14 | 63 | 96 (97) | 115 (116) | 51 |
| destination `tmpf` | −35 (−40) | 14 | 63 | 96 (97) | 118 (125.6) | 43 |
| origin `dwpf` | −33 (−35) | 1 | 49 | 76 | 82.4 (83) | 203 |
| destination `dwpf` | −35 (−36) | 1 | 49 | 76 | 82 (83) | 198 |
| origin `sknt` | 0 | 0 | 7 | 21 | **97.19** | 17,293 |
| destination `sknt` | 0 | 0 | 7 | 21 | 39 (44) | 17,097 |
| origin `vsby` | 0 | 0.75 | 10 | 10 | 20 | 22 |
| destination `vsby` | 0 | 0.75 | 10 | 10 | 20 | 14 |
| origin `p01i` | 0 | 0 | 0 | 0.09 (0.1) | 1.92 (3.26) | 155,847 |
| destination `p01i` | 0 | 0 | 0 | 0.09 (0.1) | 2.33 (2.34) | 155,519 |
| origin/destination `age_minutes` | 10 | 11 | 39 | 69 | 90 | 0 |

읽는 법:

- `tmpf`와 `dwpf`의 0은 0°F이며 결측이 아닙니다. `sknt`의 0은 무풍 보고로, `vsby`의 0은 시정 0 mile 보고로 저장된 값입니다. 결측은 NaN으로 따로 표시됩니다.
- origin `sknt` 최댓값 97.19 knot와 `vsby` 최댓값 20 mile은 원천 값을 그대로 둔 것입니다. 이상값 여부는 검증하지 않았고 필터링하지 않았습니다(미확인). 대부분의 `vsby`가 10에 몰려 있지만, 10이 보고 상한인지는 외부 계약으로 확인하지 않았습니다(미확인).
- `age_minutes`의 최솟값 10은 10분 공개 지연 가정과 일치합니다. 관측 시각에 10분을 더한 가용 시각이 예측 시점 이하여야 하기 때문입니다. 최댓값 90은 관측 나이 상한입니다.

### 2.4 결측·미량 강수·실제 0의 표기

#### 원시 캐시 → 결합 → 모델 피처

| 단계 | 결측 | 미량 강수(trace) | 실제 0 |
|---|---|---|---|
| IEM 원시 캐시 CSV | 문자열 `M` | 문자열 `0.0001`(요청 `trace=0.0001`) | `0.00` |
| 결합 코드 `read_cache` | `na_values=['M']`과 빈 문자열을 결측으로 바꾼 뒤 `pd.to_numeric(errors="raise")` 적용 | 실수 0.0001 | 실수 0.0 |
| 결합 산출물 `joined_latency*.csv.gz` | 예약 토큰 **`<NA>`**(`CSV_NULL`). 빈 문자열은 결측이 아님(`empty_string_is_null: false`) | `0.0001` | `0` |
| 모델 피처(`read_joined_csv` → `extract_weather_features`) | NaN | 0.0001 | 0.0 |

원시 캐시 1,317개 파일, 7,299,100행 전체에서 5개 모델 필드의 문자열 토큰을 직접 세었습니다.

| 필드 | `M` | 숫자 형식 | 그 밖의 토큰(`T`, 빈 문자열 등) |
|---|---:|---:|---:|
| `tmpf` | 13,441 | 7,285,659 | 0 |
| `dwpf` | 26,309 | 7,272,791 | 0 |
| `sknt` | 29,240 | 7,269,860 | 0 |
| `vsby` | 17,206 | 7,281,894 | 0 |
| `p01i` | 36,079 | 7,263,021 | 0 |

`p01i`의 숫자 값 중 `0.00`은 5,935,599행, `0.0001`은 670,033행입니다. **캐시에 리터럴 `T`는 하나도 없습니다.** 따라서 이 프로젝트에서 미량 강수는 결측이 아닌 **수치 0.0001**로 들어갑니다. 0과 구별되며 측정된 0.0001 inch를 뜻하지도 않습니다. 결합된 평가 집단에서는 origin `p01i` = 0.0001이 10,952행, destination이 11,099행입니다. 0과 0.0001 사이의 값은 0행입니다. 모델이 0과 0.0001을 별도 분할로 실제 구분해 사용했는지는 확인하지 않았습니다(미확인).

#### 행 상태 세 가지 (평가 집단 180,332행, 10분 가정)

| 상태 | 정의 | `matched` | `age_minutes` | 기상값 5개 | origin 행 수 | destination 행 수 |
|---|---|---:|---|---|---:|---:|
| **미결합** | 결합된 보고 없음. `{role}_observed_at`이 `<NA>` | 0 | NaN | 모두 NaN | 4,185 | 4,154 |
| **결합됐지만 개별 필드 결측** | 보고는 결합됐으나 원천이 해당 필드를 `M`으로 보고 | 1 | 값 있음 | 해당 필드만 NaN | `tmpf` 51 · `dwpf` 68 · `sknt` 328 · `vsby` 59 · `p01i` 217 | `tmpf` 50 · `dwpf` 82 · `sknt` 358 · `vsby` 48 · `p01i` 221 |
| **실제 0** | 결합됐고 원천이 0을 보고 | 1 | 값 있음 | 0.0 | 위 2.3절 마지막 열 | 같음 |

미결합 사유(결합 산출물의 `{role}_reason`, 평가 집단 기준):

| 사유 | origin | destination | 의미 |
|---|---:|---:|---|
| `not_mapping_eligible` | 3,923 | 3,923 | 출발·도착 중 한 공항이라도 `confirmed_period` 매핑 등급이 아니어서 결합을 시도하지 않음. 이 행은 두 role 모두 미결합 |
| `stale_beyond_90_minutes` | 214 | 198 | 예측 시점 이전의 마지막 보고가 90분보다 오래됨 |
| `not_yet_available_under_latency_assumption` | 48 | 33 | 90분 안에 보고가 있으나 10분 가정상 예측 시점까지 가용하지 않음 |
| 합계 | 4,185 | 4,154 | [요약 JSON](../output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_summary.json) `labeled_population_coverage`의 180,332 − 결합 수와 일치 |

전체 706,759행 기준 사유와 필드 결측 수는 [결합 요약](../output/baseline_recovery_v2_weather_full_join_20260922_full_weather_join_summary.json)의 `scenarios[latency_minutes=10]`에 있습니다(`feature_missing_matched_rows` 등). 이번 집계의 전체 행 기준 값(예: origin 결합 후 `tmpf` 결측 201, `p01i` 774)은 그 기록과 일치했습니다. 결합 요약의 `feature_blank_matched_rows`는 모든 필드에서 0입니다.

### 2.5 `matched=1`이 보장하는 것과 보장하지 않는 것

보장하는 것은 다음과 같습니다([`src/weather.py`](../src/weather.py) `join_weather_asof`, [`src/weather_full.py`](../src/weather_full.py) `join_block`).

- 매핑된 같은 관측소의 보고가 하나 결합됐습니다.
- 관측 시각 ≤ 예측 시점이고, 가정 가용 시각(관측 시각 + 10분) ≤ 예측 시점입니다.
- 관측 나이가 0~90분이며 `age_minutes`가 비결측입니다. `extract_weather_features`가 `matched`와 `age.notna()`의 일치를 검사합니다.
- 그 보고는 가정 가용 시각 기준으로 예측 시점까지 가용한 **가장 최근 보고 하나**입니다.

보장하지 않는 것은 다음과 같습니다.

- **5개 기상값이 모두 있다는 보장은 없습니다.** 결합은 보고 단위로 이루어지며, 최근 보고의 특정 필드가 `M`이면 더 이른 보고의 값으로 채우지 않습니다. 그 필드만 NaN이 됩니다(2.4절 둘째 상태).
- 실제 과거 시점의 수신 이력은 보장하지 않습니다. 10분은 실측이 아닌 가정입니다(`measured_publication_latency: false`).
- 관측소의 2018~2019년 물리적 연속성은 보장하지 않습니다(`confirmed_period`의 의미는 [해설 6.3절](README_EXPLAINED_KO.md) 참고).
- 값의 물리적 타당성은 보장하지 않습니다(2.3절의 이상값 후보).

`matched=0`은 "날씨가 맑았다"나 "강수 0"이 아니라 **정보 없음**입니다. 기상값은 NaN이고 0으로 채우지 않습니다.

### 2.6 미확인으로 남긴 것

- IEM의 각 필드 정의·보고 규칙(`p01i` 누적 구간, `vsby` 보고 상한, ASOS 보고 유형 3/4의 정확한 의미)을 외부 원문과 다시 대조하지는 않았습니다. 해설 5.2절의 기존 확인을 인용했습니다.
- IEM이 trace를 `0.0001`로 바꾸는 범위(모든 trace 보고가 빠짐없이 변환되는지)는 요청 계약과 "캐시에 `T`가 0건"이라는 관측으로만 확인했습니다. 원천 내부의 변환 규칙은 미확인입니다.
- origin `sknt` 97.19 knot 같은 극단값이 실제 관측인지 원천 오류인지는 미확인입니다.

### 2.7 재현 방법 (읽기 전용)

저장소에 스크립트를 추가하지 않았습니다. 같은 수치는 다음 순서로 다시 얻을 수 있습니다.

1. `src.weather_full.read_joined_csv(joined_latency10min.csv.gz, usecols=src.weather_model.joined_usecols())`로 읽습니다.
2. `extract_weather_features`를 적용합니다.
3. `data/train.csv`의 `ID, Delay`를 `align_raw_to_join`으로 정렬한 뒤 `labeled_mask`로 평가 집단(180,332행)을 고릅니다.
4. role별 `matched == 1` 행에서 결측, 0, 0.0001, 분위수를 셉니다.
5. 원시 토큰은 수집 manifest의 `requests[].cache_file`을 `dtype=str, keep_default_na=False`로 읽어 정규식 `^-?\d+(\.\d+)?$`에 맞지 않는 토큰을 셉니다.

<a id="lightgbm"></a>
## 3. LightGBM 고정/탐색 설정 (피드백 6)

### 3.1 LightGBM을 쓴 이유와 그 범위

- 입력은 범주형(공항·항공사·노선·기체번호), 수치형, 결측이 섞인 표 형식이며 약 18만~25만 행입니다. LightGBM은 pandas `category` 범주형과 결측 수치를 별도의 대치나 원-핫 인코딩 없이 받습니다. 이 저장소의 날씨 계약도 이 성질에 기대어 "날씨 결측을 대치하지 않는다"고 정했습니다([`src/weather_model.py`](../src/weather_model.py) 독스트링).
- Phase 1~6의 초기 스크립트(`run_baseline.py` 등)부터 LightGBM을 사용했습니다. 이후 통일 프로토콜 재측정도 같은 모델 계열을 유지해, **모델을 고정하고 전처리·정보 추가의 효과만 비교**하는 실험 설계를 택했습니다.
- 정직한 한계: 이 선택은 다른 알고리즘보다 우수하다는 비교를 근거로 하지 않습니다. 고정 하이퍼파라미터도 체계적 탐색의 결과가 아닙니다. 과거 Phase 2 스크립트가 `num_leaves 47→63`, `max_depth 7→8`로 수동으로 늘린 값을 Phase 4(`run_hybrid.py`) 기준으로 이어받았습니다([`run_tuned.py`](../run_tuned.py) 독스트링, [`rerun_all_phases.py`](../rerun_all_phases.py) `LGBM_PARAMS` 주석).
- Logistic Regression / Random Forest / LightGBM의 공정 비교 결과는 [README 분류기 3종 비교](../README.md#classifier-comparison)에 있습니다. 같은 조건에서 Random Forest가 더 높은 점수를 냈으므로 "LightGBM이 최적"이라고 쓰지 않습니다.

### 3.2 고정 설정과 탐색 설정

**전역 하이퍼파라미터 최적화는 수행하지 않았다; 탐색 대상은 트리 수와 임계값뿐이다.** 두 값도 외부 검증 fold가 아닌 각 외부 fold의 내부 holdout에서만 선택했습니다.

대상 실행은 날씨 비교 `baseline_recovery_v2_weather_model_compare_20260922`(P6_clean, 180,332행, 시드 42/1/7)입니다. 근거는 [`rerun_all_phases.py`](../rerun_all_phases.py)(`CFG`, `LGBM_PARAMS`, `N_ESTIMATORS_GRID`, `TE_SMOOTHING_M`, `PhaseSpec`), [`src/cv.py`](../src/cv.py), [모델 manifest](../output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_manifest.json)의 `experiment_identity`입니다.

| 구분 | 설정 | 값 | 비고 |
|---|---|---|---|
| 고정 | `objective` / `metric` / `boosting_type` | `binary` / `binary_logloss` / `gbdt` | manifest `lgbm_params` |
| 고정 | `learning_rate` | 0.05 | |
| 고정 | `num_leaves` | 63 | |
| 고정 | `max_depth` | 8 | |
| 고정 | `colsample_bytree` | 0.8 | 트리마다 피처 80% 사용 |
| 고정(효과 없음) | `subsample` | 0.8 | **`subsample_freq`가 기본값 0이라 행 bagging이 꺼져 있습니다.** 설치된 LightGBM 4.7.0 docstring: "subsample_freq … <=0 means no enable". 설정표의 0.8을 "행 80% 샘플링"으로 읽으면 안 됩니다 |
| 고정(기본값) | `min_child_samples` | 20 | 지정하지 않음. LightGBM 기본값 |
| 고정(기본값) | `reg_alpha` / `reg_lambda` / `min_split_gain` / `min_child_weight` | 0 / 0 / 0 / 0.001 | 지정하지 않음. 명시적 규제 없음 |
| 고정(미사용) | `scale_pos_weight` / `class_weight` | 미지정(1.0 / None) | es_protocol_final 작업 5 결정. `rerun_all_phases.py` 주석 |
| 고정 | `random_state`(LightGBM) | **42, 세 시드 모두** | 3.5절 참고 |
| 고정 | 범주형 처리 | `Tail_Number, Route, Origin_Airport, Destination_Airport, Airline` 5개를 정수 코드의 pandas `category` dtype으로 넣고, LightGBM의 기본 `categorical_feature="auto"`로 범주형 처리. 같은 5개 열의 TE 수치 열도 함께 사용(`te_drop_original=False`) | `STANDARD_CATS`, `encode_categoricals`, P6_clean = P6_fixed + clean |
| 고정 | TE | 평활 m=20, inner-train 내부 5-fold 교차 인코딩. inner-holdout과 outer-valid는 inner-train 라벨만으로 인코딩 | `TE_SMOOTHING_M`, `TE_INNER_SPLITS`, `run_fold_nested_grid` |
| 고정 | 외부 분할 | `StratifiedKFold(n_splits=5, shuffle=True, random_state=시드)` | `make_folds` |
| 고정 | 내부 holdout 비율 | outer-train의 **20%**, 층화, `random_state=시드+fold` | `inner_holdout_frac=0.2`, `_make_inner_split` |
| 고정 | 시드 | 42, 1, 7. 각 시드가 외부 분할, 내부 분할, TE 내부 분할을 바꿈 | `SEEDS`, `replace(CFG, seed=...)` |
| 고정 | 집계 | 5개 fold의 OOF를 합친 pooled 지표 | `metric_aggregation="pooled_oof"` |
| **탐색** | `n_estimators` | 후보 `[10, 25, 50, 75, 100, 150, 300, 600]`. 각 후보를 조기 종료 없이 끝까지 학습하고 **inner-holdout LogLoss 최소**인 값 선택. 동률이면 작은 값 | `N_ESTIMATORS_GRID`, `run_fold_nested_grid`. 선택 결과: 미사용 모두 50, 사용 75~150(인계 문서 3절) |
| **탐색** | 분류 임계값 | 0.10~0.70, 0.01 간격(61개). 선택된 모델의 inner-holdout 확률에서 **Macro F1 최대**인 값 선택. 동률이면 작은 값 | `threshold_grid`, `selection_metadata`. 선택 결과: 0.22~0.24 |
| 미탐색 | 그 밖의 모든 하이퍼파라미터, 피처 부분집합, 날씨 지연 가정 | — | manifest `weather_feature_selection_frozen: true`, `latency_selected_by_model_score: false` |

### 3.3 패키지 버전: 잠금파일과 실행 당시 기록의 구분

| 패키지 | 현재 [`uv.lock`](../uv.lock) | 날씨 비교 실행 당시 manifest 기록 | 현재 로컬 환경(`uv run --locked`) |
|---|---|---|---|
| Python | `requires-python >=3.14` (`pyproject.toml`) | **3.14.6** (`python_version`) | 3.14.6 |
| pandas | 3.0.5 | **3.0.5** (`pandas_version`) | 3.0.5 |
| LightGBM | 4.7.0 | **기록 없음** | 4.7.0 |
| scikit-learn | 1.9.0 | **기록 없음** | 1.9.0 |
| NumPy | 2.5.3 | **기록 없음** | 2.5.3 |

- 날씨 비교 manifest는 `uv_lock_sha256 = d73d3485…dc22`를 기록했습니다. 현재 `uv.lock`의 SHA-256도 같은 값이므로 **잠금파일은 실행 이후 바뀌지 않았습니다.**
- 그러나 실행 당시 실제로 import된 LightGBM, scikit-learn, NumPy 버전은 이 실행의 manifest에 기록되지 않았습니다. 잠금파일과 일치했을 가능성이 높지만 기록으로는 **미확인**입니다.
- 대조로, Phase 러너 [`rerun_all_phases.py`](../rerun_all_phases.py)의 `RUN_METADATA`는 `versions`(lightgbm, pandas, numpy, scikit-learn)를 기록합니다. 날씨 비교 러너에는 이 필드가 없습니다.

### 3.4 protocol 라벨 오기 (숫자에는 영향 없음)

- 증상: 20260922 날씨 비교의 [`runs.csv`](../output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_runs.csv) `protocol` 열과 [manifest](../output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_manifest.json) `experiment_identity.cfg`에 `"inner_early_stopping": true`, `"stopping_rounds": 40`이 기록되어 있습니다.
- 실제 경로: 이 실행은 `run_fold_nested_grid`로 트리 수를 골랐습니다. 이 함수는 `callbacks`를 제거하고 각 후보 트리 수를 조기 종료 없이 끝까지 학습합니다([`src/cv.py`](../src/cv.py)). `stopping_rounds`는 이 경로에서 쓰이지 않습니다. `runs.csv`의 `selected_n_estimators`와 `grid_scores`(8개 후보 각각의 inner LogLoss)가 이 경로의 산출물입니다.
- 원인: `CVConfig.describe()`가 `inner_early_stopping=True`를 하드코딩합니다. Phase 러너는 `protocol_description()`에서 이를 `False`로 덮어쓰지만, 20260922 시점의 날씨 비교 러너는 `describe()` 값을 그대로 저장했습니다.
- 판정: **라벨 기록 오류이며, 저장된 지표·선택값은 nested grid 경로로 계산된 값이므로 수치에는 영향이 없습니다.** 기존 출력 파일은 덮어쓰지 않습니다. 러너의 라벨은 2026-10-08 커밋 15a0b3c에서 수정되어 이후 실행부터 정확한 프로토콜을 기록합니다.

### 3.5 추가로 확인한 사실

1. **LightGBM `random_state`는 세 시드 모두 42였습니다.** Phase 러너는 `--seed`로 실행할 때 `LGBM_PARAMS["random_state"]`를 시드로 바꿉니다(`main()`). 날씨 비교 러너는 `rerun_all_phases`를 모듈로 import하므로 `main()`이 실행되지 않습니다. 이 러너는 `CVConfig`의 시드만 바꿉니다. manifest의 `lgbm_params.random_state: 42`도 이와 일치합니다. 따라서 날씨 비교의 "시드 반복"은 **분할(외부·내부·TE 내부)의 반복**이며, LightGBM 내부 난수(`colsample_bytree` 피처 샘플링)는 세 시드에서 같은 시드를 썼습니다. 날씨 미사용/사용은 같은 시드에서 같은 설정이므로 쌍비교의 공정성에는 영향이 없습니다. 다만 "시드 3개"를 모델 난수까지 바꾼 반복으로 설명하면 안 됩니다.
2. `subsample=0.8`은 위 표대로 실제로 적용되지 않았습니다. 설정표에서 행 샘플링이 있었다고 쓰지 않습니다.
3. `n_jobs`는 지정하지 않았습니다(LightGBM 기본 스레드 사용). 다중 스레드에서 동일 입력을 재실행했을 때 비트 단위로 같은지는 이 문서에서 확인하지 않았습니다(미확인).

<a id="wording"></a>
## 4. 표기 정리 내역 (피드백 7)

### 4.1 [해설 문서](README_EXPLAINED_KO.md)에서 고친 곳

단독 "F1"이 실제로 Macro F1(또는 클래스 F1)을 뜻하던 곳만 최소한으로 바꿨습니다. 앵커와 링크는 바꾸지 않았습니다.

| 위치 | 이전 | 이후 |
|---|---|---|
| 0절 "짧게 말하면" | 전처리 수정만으로 F1 향상은 | … Macro F1 향상은 |
| 1.2 정의 목록 | `F1 = 2 × Precision × Recall / …` | `클래스 F1(예: Delayed F1) = …` 및 양성 클래스 기준이라는 설명 |
| 1.3 "짧게 말하면" | F1은 두 클래스를 함께 분류하는 능력 | Macro F1은 … |
| 4.2 pooled 설명 | F1은 비선형이므로 "fold별 F1의 단순 평균"과 … F1 | Macro F1 |
| 5.3 재현율 문단 | F1이 개선됐다고 | Macro F1이 개선됐다고 |
| 5.4 보정 문단·"짧게 말하면" | F1 향상도 없었습니다 / F1 개선을 확인하지 못했고 | Macro F1 |
| 6.4 latency 문단 | "60분 모델의 F1이 떨어졌다" | "60분 모델의 Macro F1이 떨어졌다" |
| 부록 D-1 | 시각 결측 그룹의 F1·재현율 | 그룹의 Macro F1·지연 재현율(`src/oof.py`의 그룹 진단은 `average="macro"`) |
| Q2, Q8, Q9 | F1 향상 여부 / F1 0.599 / F1·LogLoss·순위, F1 향상도 | Macro F1 |

### 4.2 README.md 점검 결과 (수정하지 않음)

[README](../README.md)의 "F1"은 모두 `Macro F1`, `Delayed F1`, `Not_Delayed F1`처럼 한정어가 붙어 있습니다(19, 21, 37, 43, 51, 52, 54, 58, 62, 69, 71, 76, 78, 94행). **같은 이유로 고칠 단독 "F1"은 없습니다.** 다만 README 3절 ① "설정" 줄(95행)은 `learning_rate`·`num_leaves`·`max_depth`만 나열합니다. 3.2절의 `subsample` 무효, LightGBM `random_state` 고정, protocol 라벨 오기를 README에 반영할지는 README 담당자가 판단할 사항으로 남깁니다.

<a id="limits"></a>
## 5. 남은 한계

- IEM 필드 정의·보고 규칙의 외부 원문은 이번에 다시 대조하지 않았습니다(2.6절).
- 날씨 비교 실행 당시의 LightGBM·scikit-learn·NumPy 버전은 기록이 없어 잠금파일 해시 일치로만 간접 추정합니다(3.3절).
- 0과 trace(0.0001)를 모델이 실제로 구분해 썼는지, 다중 스레드 재실행의 비트 단위 재현성은 확인하지 않았습니다.
- 다른 분류기와의 공정 비교 결과는 이 문서에 없습니다. 이 문서는 LightGBM이 최적이라는 근거가 아닙니다.
- 2.3~2.4절의 집계는 로컬 원자료(`data/`)가 있어야 재현됩니다. Git만으로는 다시 계산할 수 없습니다.
