# Tableau Public 대시보드 제작 인계서

> 대상: Tableau Desktop을 조작해 대시보드 3종을 만들고 Tableau Public에 게시할 실행 에이전트(Codex 등).
> 데이터 설명·대조 결과는 [output/tableau/README_DATA_KO.md](../output/tableau/README_DATA_KO.md), 생성기는 [scripts/build_tableau_extracts.py](../scripts/build_tableau_extracts.py)입니다.
> 이 문서는 새 실험 결과가 아닙니다. 수치의 기준은 [README](../README.md)와 추적된 `output/` 근거입니다.

## 0. 시작 전 확인

1. `output/tableau/`에 CSV 15개와 `tableau_extracts_manifest.json`이 있는지, manifest의 `reconciliation.all_checks_passed`가 `true`인지 확인합니다. 없거나 `false`이면 제작을 멈추고 생성기를 다시 실행합니다.
2. **게시용 통합 문서에는 `output/tableau/*.csv`만 연결합니다.** `data/tableau/flights_rowlevel.csv`(행 단위)는 데이터셋 재배포 조건이 확인되지 않았으므로 Tableau Public 통합 문서에 넣지 않습니다(추출에 포함되면 원자료 재배포가 됨).
3. Tableau Desktop 또는 Tableau Public 앱에서 **텍스트 파일** 커넥터로 각 CSV를 엽니다. 필드 구분자 쉼표, 문자 집합 **UTF-8**(한글이 깨지면 연결 설정에서 UTF-8을 지정), 첫 행 헤더를 사용합니다.
4. 데이터 원본은 파일별로 따로 만들고 관계(join)를 걸지 않습니다. 파일마다 집단·단위가 달라 결합하면 분모가 중복됩니다.
5. 데이터 형식 지정: `n_*`, `tn/fp/fn/tp`, `n`, `*_rows`, `seed`, `month`, `hour_order`, `bin_order`, `bin`, `group_sort`, `rank_by_rows`는 정수, `*_rate`, `value`, `lat`/`lon`, `bin_lower`/`bin_upper`, `lower`/`upper`, `mean_probability`, `sum_probability`, `abs_gap`, `gap_signed`는 실수. `seed`, `month`, `bin`은 **차원(불연속)** 으로 바꿉니다. `lat`/`lon`은 지리적 역할 위도/경도로 지정합니다.

## 1. 공통 디자인 규칙

### 1.1 색상 (README C안 토큰, `scripts/readme_editorial.py`)

| 토큰 | 값 | 용도 |
|---|---|---|
| PAPER | `#FBF8F0` | 대시보드·워크시트 배경 |
| INK | `#304750` | 제목·본문 텍스트, 주요 축 |
| MUTED | `#61727A` | 캡션·주석, 보조 계열(예: 날씨 미사용, Logistic Regression) |
| RULE | `#D7DCD9` | 격자선·구분선·테두리 |
| ACCENT | `#356A86` | 주 계열(막대·점), 날씨 사용, LightGBM |
| GOLD | `#D5BD8F` | 강조(선택 항목, 전체 기준선 주석), Random Forest, FP 칸 |
| SKY | `#E4F0F5` | 연속 색상의 낮은 끝, 패널 배경 |
| HIGHLIGHT | `#EAF2F4` | 툴팁·필터 패널 배경 |

- 지연율 연속 색상: **SKY `#E4F0F5` → ACCENT `#356A86`** 2단계 사용자 지정 순차 팔레트. 빨강/초록 사용 금지.
- 혼동행렬: TN·TP는 SKY→ACCENT 순차 음영, FP는 GOLD, FN은 MUTED 테두리로 구분(색만으로 구분하지 않도록 칸 라벨에 TN/FP/FN/TP를 항상 표시).
- 글꼴: 제목 Tableau Book/Bold 18pt INK, 본문 11pt, 캡션 9~10pt MUTED. 한글 표시가 깨지면 'Malgun Gothic'을 지정.
- 격자선은 RULE 1px, 0 기준선만 표시. 테두리·그림자 없음.

### 1.2 공통 매개변수와 계산 필드

각 데이터 원본에 필요한 것만 만듭니다(Tableau 계산식 문법).

```text
// 매개변수
[최소 행 수]      정수, 기본값 100, 범위 1~5000         (D1·D3 작은 칸 숨김용, D2 표본 주의 표시용)
[노선 Top N]      정수, 기본값 20, 범위 5~100
[구간 피처]       문자열 목록: tmpf, sknt, vsby, p01i, age_minutes, matched_status (표시 별칭: 기온, 풍속, 시정, 1시간 강수, 관측 나이, 결합 상태)
[공항 역할]       문자열 목록: origin, destination (별칭: 출발 공항, 도착 공항)

// 지연율 (항상 합계로 다시 계산 — AVG([delay_rate]) 금지)
[지연율] = SUM([n_delayed]) / SUM([n_rows])

// 집단 전체 지연율(기준선). population_* 열은 모든 행에 같은 값
[집단 지연율] = MIN([population_delayed]) / MIN([population_rows])

// 기준선 대비 차이(%p)
[기준선 대비 차이] = [지연율] - [집단 지연율]

// 작은 칸 필터 (필터 선반에 놓고 '참'만)
[표본 충분] = SUM([n_rows]) >= [최소 행 수]

// 참고용 95% Wilson 구간 (행 독립 가정의 근사, 오차막대·툴팁 전용)
[Wilson 하한] =
  ( [지연율] + 1.96^2/(2*SUM([n_rows]))
    - 1.96*SQRT( [지연율]*(1-[지연율])/SUM([n_rows]) + 1.96^2/(4*SUM([n_rows])^2) ) )
  / (1 + 1.96^2/SUM([n_rows]))
[Wilson 상한] =
  ( [지연율] + 1.96^2/(2*SUM([n_rows]))
    + 1.96*SQRT( [지연율]*(1-[지연율])/SUM([n_rows]) + 1.96^2/(4*SUM([n_rows])^2) ) )
  / (1 + 1.96^2/SUM([n_rows]))
```

숫자 서식: 지연율·recall·precision은 백분율 소수 1자리(툴팁은 2자리), 행 수는 천 단위 구분 정수.

### 1.3 공통 문구 규칙

- 모든 대시보드 상단 부제에 **집단과 분모**를 씁니다(예: "라벨 255,001행 · 지연 45,000행").
- 지연율은 "실제 라벨 기준 지연 비율"입니다. 예측값이나 확률로 부르지 않습니다.
- "~ 때문에", "~가 지연을 일으킨다", "~하면 지연이 줄어든다", "미래에도" 같은 인과·미래 표현을 쓰지 않습니다. "~에서 지연 비율이 높게 관측됨", "연관"을 씁니다.
- 시드 3개는 "같은 행의 재분할 3회"라고 씁니다. SD를 신뢰구간이라 부르지 않습니다.
- Wilson 구간을 보이면 "행 독립 가정의 참고 구간"이라고 캡션에 씁니다.

### 1.4 대시보드 공통 크기·배치

- 크기: **고정 1200 × 800 px**(Tableau Public 임베드와 README 스크린샷에 맞춤).
- 상단 90px: 제목(18pt) + 부제(집단·분모, 11pt MUTED).
- 하단 60px: 캡션 박스(9~10pt MUTED, 주의 문구·출처 파일명).
- 필터·매개변수 컨트롤은 오른쪽 세로 패널(폭 200px, HIGHLIGHT 배경) 또는 상단 가로 띠에 둡니다.

---

## 2. 대시보드 1 — 지연 패턴 탐색

**이야기(1~2문장).** 라벨이 있는 255,001편에서 실제 지연 비율이 항공사·공항·시간대·월·노선별로 어떻게 다른지 살펴봅니다. 이는 관측된 분포이며 원인 분석이 아닙니다.

**데이터 원본:** `d1_delay_by_airline.csv`, `d1_delay_by_airport.csv`, `d1_delay_by_hour.csv`, `d1_delay_by_month.csv`, `d1_delay_by_route.csv`, `d1_delay_by_airline_month.csv`, `overview_populations.csv`

### 시트

| 시트 | 원본 | 열/행 | 마크 | 색상/크기/레이블 | 필터 |
|---|---|---|---|---|---|
| D1-KPI | `overview_populations` | 텍스트 표: `population_ko` / `SUM(n_rows)`, `SUM(n_delayed)`, `[지연율]` | 텍스트 | INK | `population = labeled_all` |
| D1-항공사 | `d1_delay_by_airline` | 행: `airline`(정렬: `[지연율]` 내림차순), 열: `[지연율]` | 막대 | 색 ACCENT 단색, 레이블 `[지연율]`; 참조선 `[집단 지연율]`(GOLD 점선, 라벨 "라벨 전체 지연율") | `[표본 충분]` = 참 |
| D1-지도 | `d1_delay_by_airport` | 열: `lon`(경도), 행: `lat`(위도) | 원 | 색 `[지연율]`(SKY→ACCENT), 크기 `SUM(n_rows)`, 세부 `airport` | `airport_role` = `[공항 역할]`, `[표본 충분]` = 참 |
| D1-시간대 | `d1_delay_by_hour` | 열: `hour_label`(정렬: `hour_order` 오름차순), 행: `[지연율]` | 막대(결측 칸은 MUTED) | 색 계산식 `IF MIN([hour_order]) < 0 THEN "원본 시각 결측" ELSE "관측 시각" END` (관측 ACCENT, 결측 MUTED) | `time_role`(단일 값 목록, 기본 departure) |
| D1-월 | `d1_delay_by_month` | 열: `month`(불연속), 행: `[지연율]` | 선 + 원 | ACCENT, 레이블 끝점만 | 없음 |
| D1-노선 | `d1_delay_by_route` | 행: `route`, 열: `[지연율]` | 막대 | 색 ACCENT, 레이블 `SUM(n_rows)` | `rank_by_rows` ≤ `[노선 Top N]` (행 수 기준 상위 N), `[표본 충분]` |
| D1-항공사×월(선택) | `d1_delay_by_airline_month` | 행: `airline`, 열: `month` | 사각형(히트맵) | 색 `[지연율]`(SKY→ACCENT), 레이블 없음 | `[표본 충분]` = 참(작은 칸은 빈칸) |

참고: `[노선 Top N]` 필터는 `rank_by_rows`(행 수 순위)를 쓰므로 "가장 운항 기록이 많은 노선 N개"입니다. 지연율 상위 N개가 아닙니다(작은 노선이 상위를 차지하는 왜곡 방지).

### 계산 필드(추가)

```text
[시각 상태] = IF MIN([hour_order]) < 0 THEN "원본 시각 결측" ELSE "관측 시각" END
```

### 제목·캡션

- 제목: **항공편 지연 패턴 탐색**
- 부제: `라벨 전체 255,001행 · 실제 지연 45,000행(17.6%) · 원본 data/train.csv`
- 시트 제목: "항공사별 지연 비율", "공항별 지연 비율(원 크기 = 운항 기록 수)", "예정 시각(현지)별 지연 비율", "월별 지연 비율(2018·2019년 혼합)", "운항 기록 상위 노선의 지연 비율"
- 캡션:
  > 지연 비율 = 실제 지연 행 ÷ 해당 칸의 라벨 행. 라벨이 없는 행(744,999행)은 제외했고 음성으로 보지 않았습니다. 시각은 원본 예정 시각의 시(hour)이며 결측은 복원하지 않고 따로 표시했습니다. 원본에는 연도가 없어 월은 2018·2019년이 섞여 있습니다. 칸별 비율은 관측된 분포이며 원인을 뜻하지 않습니다. 행 수 [최소 행 수] 미만 칸은 숨겼습니다. 출처: output/tableau/d1_*.csv

### 배치 스케치

```text
+--------------------------------------------------------------+--------------+
| 제목 / 부제                                       [KPI 3칸]   |  필터 패널   |
+-------------------------------+------------------------------+  공항 역할   |
| D1-지도 (560x330)             | D1-항공사 (380x330)          |  time_role   |
+-------------------------------+------------------------------+  최소 행 수  |
| D1-시간대 (400x250)  | D1-월 (240x250) | D1-노선 (300x250)  |  노선 Top N  |
+--------------------------------------------------------------+--------------+
| 캡션                                                                        |
+-----------------------------------------------------------------------------+
```

- 동작: D1-항공사 막대 클릭 → 강조 동작(D1-항공사×월이 있으면 필터). 지도 원 클릭 → 툴팁만.

### 툴팁

```text
<airport> (<airport_state>) · <airport_role_ko>
지연 비율: <[지연율]> (참고 95% 구간 <[Wilson 하한]>–<[Wilson 상한]>)
라벨 행: <SUM(n_rows)> · 실제 지연: <SUM(n_delayed)>
라벨 전체 지연율: <[집단 지연율]>
```

(항공사·시간대·월·노선 시트도 같은 형식으로, 첫 줄만 해당 차원으로 바꿉니다.)

---

## 3. 대시보드 2 — 날씨와 지연

**이야기.** 날짜를 귀속해 날씨를 결합한 180,332편에서 출발·도착 공항의 날씨 구간별 실제 지연 비율을 보여 주고, 같은 행에서 날씨 피처를 넣고 뺀 LightGBM 교차검증의 혼동행렬을 시드별로 비교합니다. 구간별 비율은 연관이며, 모델 비교는 정적 교차검증입니다.

**데이터 원본:** `d2_weather_bins_long.csv`, `d2_delay_by_year_month.csv`, `model_confusion_long.csv`, `model_metrics_long.csv`, `overview_populations.csv`

### 시트

| 시트 | 원본 | 열/행 | 마크 | 색상/레이블 | 필터 |
|---|---|---|---|---|---|
| D2-구간 | `d2_weather_bins_long` | 행: `[구간 표시명]`(정렬: `bin_order` 오름차순), 열: `[지연율]`; 열 분할 `role_ko` | 막대 | 색 `bin_kind`: value=ACCENT, special=ACCENT 70% 투명도, field_missing=MUTED, unmatched=MUTED 빗금 없음·연한 회색 `#B9C3C7`, partial=GOLD; 참조선 `[집단 지연율]` GOLD 점선 | `feature` = `[구간 피처]`; `role` ≠ `both`(결합 상태 선택 시에는 `role = both`) |
| D2-구간 행 수 | 같음 | D2-구간과 같은 행, 열: `SUM(n_rows)` | 막대(가늘게) | MUTED, 레이블 천 단위 + `[표본 주의]` | 같음(최소 행 수로 숨기지 않음) |
| D2-연월 | `d2_delay_by_year_month` | 열: `year_month`(불연속), 행: `[지연율]` | 선 | ACCENT | 없음 |
| D2-혼동행렬 | `model_confusion_long` | 행: `actual`, 열: `predicted`, 열 분할: `variant` | 사각형 + 텍스트 | 색 `[칸 유형]`, 레이블 `cell` + `SUM(n)` + `[실제 클래스 대비 비율]` | `experiment = weather_on_off_lightgbm`, `seed`(단일 값 목록: 42/1/7, 기본 42) |
| D2-FP·FN 시드별 | `model_confusion_long` | 열: `seed`, 행: `SUM(n)`; 열 분할 `cell`(FP, FN만) | 막대(나란히) | 색 `variant`: weather_off=MUTED, weather_on=ACCENT | `experiment = weather_on_off_lightgbm`, `cell` ∈ {FP, FN} |
| D2-지표 | `model_metrics_long` | 행: `metric`(macro_f1_nested, log_loss, roc_auc, precision_delayed, recall_delayed), 열: `AVG(value)` + `MIN([3시드 평균])`(동기화 이중 축) | 시드점 마크에만 `seed` 세부(투명도 40%), 평균 막대는 별도 마크(`seed` 없음) | 색 `variant` 동일 | `experiment = weather_on_off_lightgbm` |

`variant` 별칭: weather_off → "날씨 미사용", weather_on → "날씨 사용". `role_ko`는 이미 한글입니다.

### 계산 필드

```text
[칸 유형] = CASE ATTR([cell]) WHEN "TN" THEN "정상→정상" WHEN "TP" THEN "지연→지연"
            WHEN "FP" THEN "오경보(FP)" WHEN "FN" THEN "놓침(FN)" END
[실제 클래스 대비 비율] = SUM([n]) / SUM([actual_class_rows])
       // 실제 지연 행 중 TP·FN 비율, 실제 정상 행 중 TN·FP 비율. 여러 시드를 합쳐도 분모가 함께 늘어남
[구간 표시명] = IF [bin_kind] = "special" AND [feature] = "p01i" AND [bin_order] = 1
              THEN "미량(trace) — 0.0001 표기, 측정량 아님" ELSE [bin_label] END
[표본 주의] = IF SUM([n_rows]) < [최소 행 수] THEN "표본 주의: 최소 행 수 미만" ELSE "" END
// model_metrics_long: variant가 날씨 조건을 구분하며 모델은 LightGBM, 보정 없음
[3시드 평균] = {FIXED [experiment], [variant], [metric] : AVG([value])}
```

D2 구간·행 수 시트에는 `[표본 충분]` 필터를 걸지 않습니다. 작은 구간과 미결합·필드 결측 구간도 모두 남겨 역할별 합계 180,332행을 유지하고, `[표본 주의]`를 레이블·툴팁에 표시합니다. 강수 축과 툴팁에는 `[구간 표시명]`을 사용합니다. D2-지표의 평균 마크에는 `seed`를 넣지 않으며, 시드 선택 필터는 혼동행렬에만 적용합니다(지표 시트는 42/1/7을 모두 유지).

### 제목·캡션

- 제목: **날씨 구간과 지연 · 날씨 피처 유무 비교**
- 부제: `날씨 평가 집단 180,332행(날짜 귀속 + 라벨) · 실제 지연 31,805행 · 날씨 공개 지연 10분 가정`
- 시트 제목: "날씨 구간별 실제 지연 비율", "구간별 행 수", "귀속 연-월별 지연 비율", "혼동행렬: 날씨 미사용 vs 사용 (시드 <seed>)", "시드별 오경보(FP)·놓침(FN) 수", "3시드 지표(점 = 시드, 막대 = 평균)"
- 캡션:
  > 구간별 지연 비율은 같은 시각대·공항·계절 등과 겹쳐 있는 연관이며 날씨가 지연을 일으켰다는 근거가 아닙니다. 날씨는 예정 출발 60분 전까지 이용 가능했다고 가정한 관측(공개 지연 10분 가정, 실측 아님, 관측 나이 90분 이하)입니다. 미결합·필드 결측 행도 분모에 남겼고, 최소 행 수 미만 구간은 숨기지 않고 표본 주의를 표시했습니다. 강수 0.0001은 미량 강수 표기입니다. 모델 비교는 같은 180,332행·같은 외부 5폴드·같은 시드(같은 행의 재분할 3회)의 정적 교차검증이며, 혼동행렬은 폴드별 내부 선택 임계값 기준입니다. 미래 운항 성능을 뜻하지 않습니다. 출처: output/tableau/d2_*.csv, model_*.csv (원본 runs CSV와 칸별 일치 확인)

README와 같은 결론 문장만 씁니다: "날씨 사용 조건에서 Macro F1·LogLoss·ROC-AUC가 세 시드 모두 개선, 주된 구성은 오경보(FP) 감소, 지연 recall 증가는 작고 시드별로 고르지 않음."

### 배치 스케치

```text
+--------------------------------------------------------------+--------------+
| 제목 / 부제                                                  | 구간 피처    |
+--------------------------------+-----------------------------+ seed         |
| D2-구간 (출발|도착, 520x330)   | D2-혼동행렬 (400x330)       | 최소 행 수   |
| D2-구간 행 수 (520x120)        | D2-FP·FN 시드별 (400x170)   |              |
+--------------------------------+-----------------------------+              |
| D2-연월 (520x150)              | D2-지표 (400x150)           |              |
+--------------------------------------------------------------+--------------+
| 캡션                                                                        |
+-----------------------------------------------------------------------------+
```

### 툴팁

```text
D2-구간: <role_ko> · <feature_label_ko> · <[구간 표시명]>
         지연 비율 <[지연율]> (참고 95% 구간 <[Wilson 하한]>–<[Wilson 상한]>)
         행 <SUM(n_rows)> (집단의 <SUM(share_of_population)>) · 실제 지연 <SUM(n_delayed)>
         구간 종류: <bin_kind> · <[표본 주의]>
D2-혼동행렬: <variant> · 시드 <seed> · <[칸 유형]>
         행 <SUM(n)> / 실제 <actual> <SUM(actual_class_rows)>행 중 <[실제 클래스 대비 비율]>
         평가 행 180,332 · 임계값은 각 외부 폴드의 내부 검증에서 선택
```

---

## 4. 대시보드 3 — 모델이 틀리는 곳

**이야기.** 날씨 미사용 P6_clean LightGBM의 라벨 255,001행 교차검증(OOF) 예측에서 놓침(FN)과 오경보(FP)가 어느 항공사·시간대·노선·원본 시각 결측 집단에 몰리는지 보고, 같은 날씨 평가 집단에서 분류기 3종을 탐색 예산별로 비교하고 보정 전후의 신뢰도 곡선(예측 확률 대 실제 지연율)을 보여 줍니다. 왼쪽 OOF와 오른쪽 분류기 영역은 서로 다른 집단·모델이므로 같은 축에 놓지 않습니다.

**데이터 원본:** `d3_oof_errors_by_group.csv`, `d3_oof_error_stability.csv`, `model_confusion_long.csv`, `model_comparison_all.csv`, `model_calibration_reliability.csv`

### 시트

| 시트 | 원본 | 열/행 | 마크 | 색상/레이블 | 필터 |
|---|---|---|---|---|---|
| D3-그룹 recall·FPR | `d3_oof_errors_by_group` | 행: `group`(정렬: `group_sort` 오름차순, 없으면 `[시드합산 놓침률]` 내림차순), 열: `[시드합산 recall]`, `[시드합산 FPR]`(이중 축 아님, 나란히) | 막대 | recall=ACCENT, FPR=GOLD; 참조선 각 지표의 `[dimension 전체 recall]`/`[dimension 전체 FPR]` | `dimension`(단일 값 목록, 기본 `raw_time_pattern`), `[표본 충분(OOF)]` = 참 |
| D3-3,031행 강조 | `d3_oof_errors_by_group` | 열: `seed`, 행: `[시드합산 recall]` | 막대 | GOLD | `dimension = raw_time_pattern`, `group = both_missing`(이 시트는 세 시드 모두 유지) |
| D3-오류 안정성 | `d3_oof_error_stability` | 행: `group`, 열: `SUM(n_wrong_0_of_3)` … `SUM(n_wrong_3_of_3)`를 측정값 이름/값으로 누적 | 누적 막대(100% 비율: 표 계산 '구간 합계 비율') | 0회 SKY, 1회 `#B9C3C7`, 2회 MUTED, 3회 INK | `dimension` 같은 매개변수, `SUM(n_rows) >= [최소 행 수]` |
| D3-분류기 혼동행렬 | `model_confusion_long` | 행: `actual`, 열: `predicted`, 열 분할: `variant` | 사각형 + 텍스트 | D2와 같은 `[칸 유형]` 색 | `experiment = classifier_compare_weather_on`, `seed` |
| D3-예산별 분류기 비교 | `model_comparison_all` | 행: `[예산 표시]`(정렬: `[예산 순서]`), 열: `AVG(value)` + `MIN([3시드 평균])`(동기화 이중 축); 열 분할 `metric`(macro_f1_nested, log_loss, roc_auc만, 지표별 축 독립) | 평균 원은 별도 마크(`seed` 없음), 시드점 마크에만 `seed` 세부(투명도 40%) | 색 `model_family`: lightgbm=ACCENT, logistic_regression=MUTED, random_forest=GOLD; 평균 레이블 `MIN([3시드 평균])` 소수 4자리 | `experiment` ≠ `classifier_calibration_20261008`, `condition = weather_on`, `metric` ∈ {macro_f1_nested, log_loss, roc_auc} |
| D3-보정 신뢰도 곡선 | `model_calibration_reliability` | 열: `[구간 평균 예측 확률]`, 행: `[구간 실제 지연율]`; `bin`을 세부(Detail)·경로(Path)에 배치하고 오름차순 정렬; 대각선 참조(아래 계산 필드 `[완전 보정선]`을 이중 축 선으로) | 선 + 원(원 크기 `SUM(n)`) | 색 `model_label`(D3-예산별과 같은 색), 모양 없음 | `binning = equal_frequency_15`, `seed`(단일 값, 기본 42), `[보정 조건]`(단일 값 목록: none / platt_crossfit / isotonic_crossfit, 기본 none) |

`variant` 별칭: lightgbm → "LightGBM", logistic_regression → "Logistic Regression", random_forest → "Random Forest". `model_comparison_all`과 `model_calibration_reliability`의 `model_label`·`calibration_arm_ko`·`binning_ko`는 이미 한글·표시명입니다. `metric` 별칭: macro_f1_nested → "Macro F1 ↑", log_loss → "LogLoss ↓", roc_auc → "ROC-AUC ↑". `dimension` 별칭: airline → 항공사, dep_hour → 예정 출발 시(원본), month → 월, origin_airport → 출발 공항, route → 노선(라벨 100행 이상, 나머지 기타), raw_time_pattern → 원본 시각 결측 유형. `raw_time_pattern` 별칭: both_observed → 출발·도착 시각 모두 있음, departure_missing → 출발 시각 결측, arrival_missing → 도착 시각 결측, both_missing → 양쪽 결측(3,031행).

### 계산 필드

```text
// 세 시드를 합칠 때는 칸 수를 먼저 합한 뒤 비율 계산 (AVG([recall_delayed]) 금지)
[시드합산 recall] = SUM([tp]) / (SUM([tp]) + SUM([fn]))
[시드합산 놓침률] = SUM([fn]) / (SUM([tp]) + SUM([fn]))
[시드합산 FPR]    = SUM([fp]) / (SUM([fp]) + SUM([tn]))
[시드합산 precision] = SUM([tp]) / (SUM([tp]) + SUM([fp]))
[시드합산 오류율] = (SUM([fp]) + SUM([fn])) / SUM([n_rows])
[dimension 전체 recall] = {FIXED [dimension] : SUM([tp])} / {FIXED [dimension] : SUM([tp]) + SUM([fn])}
[dimension 전체 FPR]    = {FIXED [dimension] : SUM([fp])} / {FIXED [dimension] : SUM([fp]) + SUM([tn])}
[행 수(시드당)] = SUM([n_rows]) / COUNTD([seed])
[표본 충분(OOF)] = [행 수(시드당)] >= [최소 행 수]
[3회 모두 틀림 비율] = SUM([n_wrong_3_of_3]) / SUM([n_rows])

// model_comparison_all (D3-예산별 분류기 비교)
[3시드 평균] = {FIXED [experiment], [condition], [model], [calibration_arm], [metric] : AVG([value])}
[예산 표시] = [model_label] + " · " + [search_budget]
[예산 순서] = CASE [experiment]
                WHEN "classifier_compare_20261008" THEN 1
                WHEN "classifier_tuning_20261008" THEN 2
                WHEN "classifier_grid_ext_20261008_lgbm" THEN 3
                WHEN "classifier_grid_ext_20261008_rf" THEN 3 END
// 보정 실험(classifier_calibration_20261008)은 이 시트의 experiment 필터에서 제외
// (그 실험의 none 행은 위 실험들의 재적합 결과라 같은 점이 두 번 찍힘)
// 평균 마크의 레이블은 MIN([3시드 평균]), 시드점의 툴팁은 AVG([value])

// model_calibration_reliability (D3-보정 신뢰도 곡선) — 개수로 다시 계산, AVG 금지
[구간 실제 지연율]     = SUM([n_delayed]) / SUM([n])
[구간 평균 예측 확률] = SUM([sum_probability]) / SUM([n])
[완전 보정선]         = [구간 평균 예측 확률]      // 이중 축 선(INK 점선 1px), 축 동기화
[보정 조건]           매개변수, 문자열 목록: none, platt_crossfit, isotonic_crossfit
                      (별칭: 보정 없음, Platt 교차적합, Isotonic 교차적합)
[보정 조건 필터]      = [calibration_arm] = [보정 조건]   // 필터 선반에 놓고 '참'
```

D3-예산별 시트는 `experiment`·`condition`·`model`·`calibration_arm`·`metric`별로 평균을 고정합니다. 평균 마크에는 `seed`를 넣지 않고, `seed`는 시드점 마크에만 놓습니다. 시드 선택 필터는 분류기 혼동행렬·신뢰도 곡선에만 적용하고 예산 비교와 3,031행 강조에는 적용하지 않습니다. 신뢰도 곡선은 모델별 15개 구간이 별도 마크가 되도록 `bin`을 Detail/Path에 넣습니다.

- 신뢰도 곡선은 `seed`를 하나만 고르는 것이 기본입니다. 동일 빈도 구간의 경계는 시드마다 달라 여러 시드를 합치면 근사가 됩니다(툴팁에 "시드 <seed>"를 항상 표시).
- 동일 빈도 구간의 평균 예측 확률이 약 0.04~0.46 범위라 축을 0~0.6으로 고정하고, 축 제목은 "구간 평균 예측 확률", "구간 실제 지연율"로 씁니다.

(`dimension` 필터는 FIXED보다 먼저 적용되도록 **컨텍스트에 추가**합니다.)

### 제목·캡션

- 제목: **모델이 틀리는 곳 · 오류 분포와 분류기 비교**
- 부제(왼쪽 OOF 영역): `P6_clean LightGBM(날씨 미사용) · 라벨 255,001행 × 분할 시드 3개 OOF · 시드별 지연 45,000행`
- 부제(오른쪽 분류기 영역): `날씨 사용 피처 · 날씨 평가 집단 180,332행 · 분류기 3종 동일 행·폴드·시드 · 탐색 예산은 모델·실험마다 다름`
- 시트 제목: "<dimension>별 지연 recall과 오경보율(FPR)", "원본 출발·도착 시각이 모두 없는 3,031행의 시드별 지연 recall", "세 시드 중 틀린 횟수 분포", "분류기별 혼동행렬 (시드 <seed>)", "탐색 예산별 분류기 3시드 평균(점 = 시드)", "신뢰도 곡선: 구간 평균 예측 확률 대 실제 지연율 (시드 <seed>, <보정 조건>)"
- 캡션:
  > 왼쪽은 날씨 미사용 P6_clean 모델의 라벨 255,001행 교차검증 예측(OOF)이며, 비율은 세 시드(같은 행의 재분할 3회)의 칸 수를 합쳐 계산했습니다. 원본 시각이 모두 없는 3,031행(실제 지연 519행)은 지연 recall이 3시드 평균 약 10%로 낮았고, 날짜 귀속 키가 불완전해 날씨 평가 집단(180,332행)에는 포함되지 않습니다. 오른쪽 분류기 비교는 사전 선언한 유한 후보 안의 비교입니다. 시험한 모든 예산에서 Random Forest가 앞섰지만 예산이 모델·실험마다 다르고 고정한 축도 있어 어느 알고리즘이 일반적으로 낫다는 근거가 아닙니다. 신뢰도 곡선의 보정기는 외부 학습 데이터 안에서만 적합했으며, 보정은 ECE를 줄였지만 LogLoss·Macro F1과 분류기 순서는 거의 바꾸지 않았습니다. 두 영역은 집단·모델이 달라 직접 비교하지 않습니다. 정적 교차검증 결과이며 미래 운항 성능을 뜻하지 않습니다. 출처: output/tableau/d3_*.csv, model_*.csv

### 배치 스케치

```text
+--------------------------------------------------------------+--------------+
| 제목                                                         | dimension    |
+---------------------------------+----------------------------+ seed         |
| 부제(OOF)                       | 부제(분류기)               | 최소 행 수   |
| D3-그룹 recall·FPR (560x300)    | D3-예산별 분류기 비교 (380x300)| 보정 조건 |
| D3-3,031행 강조 (270x200) | D3-오류 안정성 (290x200) | [D3-보정 신뢰도 곡선 ⇄ D3-분류기 혼동행렬] (380x200) |
+---------------------------------+----------------------------+--------------+
| 캡션                                                                        |
+-----------------------------------------------------------------------------+
```

두 영역 사이에 RULE 색 세로 구분선(2px)을 둡니다. 오른쪽 아래 칸은 **표시/숨기기 단추가 있는 세로 컨테이너 두 개**(기본 표시: D3-보정 신뢰도 곡선, 단추 문구 "혼동행렬 보기"/"신뢰도 곡선 보기")로 두 시트를 번갈아 보입니다. 이전 안의 D3-분류기 지표 시트는 D3-예산별 분류기 비교가 대신합니다(`model_metrics_long`은 대시보드 2에서만 사용).

### 툴팁

```text
D3-그룹: <dimension 별칭> · <group>
         지연 recall <[시드합산 recall]> · 오경보율 <[시드합산 FPR]> · precision <[시드합산 precision]>
         시드당 행 <[행 수(시드당)]> · 실제 지연 비율 <[지연율]>
         세 시드 합산 칸: TN <SUM(tn)> / FP <SUM(fp)> / FN <SUM(fn)> / TP <SUM(tp)>
D3-분류기: <variant> · 시드 <seed> · <[칸 유형]> <SUM(n)>행 (평가 180,332행)
D3-예산별: <model_label> · <search_budget> (<n_configurations>개 설정) · <metric 별칭>
         3시드 평균 <MIN([3시드 평균])> · 실험 <experiment> · 날씨 사용 조건 · 보정 없음
         (시드점 툴팁에는 시드 <seed> · 해당 시드 값 <AVG(value)>를 추가)
D3-신뢰도: <model_label> · <calibration_arm_ko> · 시드 <seed> · 구간 <bin>
         평균 예측 확률 <[구간 평균 예측 확률]> · 실제 지연율 <[구간 실제 지연율]>
         구간 행 <SUM(n)> · 실제 지연 <SUM(n_delayed)> · <binning_ko>
```

(`[지연율]`은 이 원본에서 `SUM([n_delayed]) / SUM([n_rows])`로 같은 식을 만듭니다.)

---

## 5. Tableau Public 게시

1. 세 대시보드를 하나의 통합 문서에 두거나(권장: 대시보드 3개 + 스토리 없음), 대시보드별 통합 문서로 나눕니다. 통합 문서 이름 예: `Airplane 항공편 지연 탐색 (포트폴리오)`.
2. 모든 데이터 원본을 **추출(Extract)** 로 바꿉니다(Tableau Public은 추출만 게시). 원본 목록에 `flights_rowlevel.csv`가 없는지 다시 확인합니다.
3. 사용하지 않는 시트·필드는 숨기고(필드 숨기기), 각 대시보드의 탭 이름을 `1 지연 패턴`, `2 날씨와 지연`, `3 모델 오류`로 정합니다.
4. **파일 > Tableau Public에 저장(Save to Tableau Public As…)**. 로그인 창이 뜨면 **사용자 본인이 직접 로그인**합니다. 에이전트는 계정 정보를 입력하지 않습니다.
5. 게시 후 브라우저에서 열어 설정: 설명(아래 문구), "Show Sheets as Tabs" 켜기, 데이터 다운로드 허용은 **끄기**(집계 CSV라도 저장소 링크로 안내).
   - 설명 문구: `라벨 255,001편과 날씨 결합 180,332편의 관측 지연 비율, 그리고 정적 교차검증 모델의 오류 분포를 보여 주는 포트폴리오 대시보드입니다. 비율은 연관이며 원인이나 미래 성능을 뜻하지 않습니다. 데이터 설명: GitHub Peter-jackson12/Airplane의 output/tableau/README_DATA_KO.md`
6. 게시 URL(예: `https://public.tableau.com/app/profile/<사용자>/viz/<통합문서>/<대시보드>`)을 조율 에이전트에 보고합니다.

## 6. 스크린샷 내보내기

- Tableau Desktop에서 각 대시보드를 열고 **대시보드 > 이미지 내보내기** (또는 Tableau Public 페이지의 다운로드 > 이미지).
- 크기: 대시보드 고정 크기 그대로 **1200 × 800 px**, PNG. 가능하면 2배(2400 × 1600)로 내보낸 뒤 1600 × 1067로 축소해 README의 다른 그림(가로 1600)과 폭을 맞춥니다. 파일당 600 KB 이하 권장.
- 파일명과 위치:
  - `assets/readme/tableau_delay_patterns.png` (대시보드 1)
  - `assets/readme/tableau_weather_delay.png` (대시보드 2)
  - `assets/readme/tableau_model_errors.png` (대시보드 3)
- 필터 기본값 상태(공항 역할 origin, time_role departure, 구간 피처 p01i 또는 tmpf, seed 42, dimension raw_time_pattern, 최소 행 수 100, 노선 Top N 20)에서 캡처합니다. 툴팁·선택 강조가 떠 있지 않게 합니다.
- PNG는 바이너리이므로 줄바꿈 규칙 대상이 아닙니다. `assets/readme/sources.json`(SVG 그림 영수증)에는 추가하지 않습니다 — 그 영수증과 테스트는 생성기가 만든 SVG만 다룹니다.

## 7. README 반영 (조율 에이전트가 편집)

README의 **2절 끝, `<a id="engineering"></a>` 바로 위**에 짧은 하위 절을 추가합니다. 새 앵커는 기존 앵커와 겹치지 않게 `<a id="dashboards"></a>`를 씁니다. 상단 내비게이션 줄에 `· [대시보드](#dashboards)`를 추가할지는 선택입니다.

삽입 예시(링크·이미지 경로는 실제 파일이 있을 때만 넣습니다 — README 링크 검사 테스트가 경로 존재를 확인함):

```markdown
<a id="dashboards"></a>
### 인터랙티브 대시보드 (Tableau Public)

[Tableau Public에서 보기](<게시 URL>) · [데이터 사전](output/tableau/README_DATA_KO.md) · [추출 스크립트](scripts/build_tableau_extracts.py)

| 대시보드 | 집단·분모 | 내용 |
|---|---|---|
| 1 지연 패턴 | 라벨 255,001행 | 항공사·공항·예정 시각·월·노선별 실제 지연 비율 |
| 2 날씨와 지연 | 날씨 평가 180,332행 | 날씨 구간별 실제 지연 비율, 날씨 유무 혼동행렬(3시드) |
| 3 모델 오류 | OOF 255,001행 × 3시드 / 180,332행 | 그룹별 놓침·오경보, 원본 시각 결측 3,031행, 분류기 3종 혼동행렬 |

![라벨 255001행에서 항공사 공항 시간대 월 노선별 실제 지연 비율을 보여 주는 Tableau 대시보드](assets/readme/tableau_delay_patterns.png)

구간·그룹별 비율은 관측된 연관이며 원인이나 미래 성능을 뜻하지 않습니다. 모델 영역은 위 결과와 같은 정적 교차검증 근거를 다시 그린 것입니다(새 학습 없음).
```

- 이미지 alt 텍스트는 20자 이상 서술형으로(기존 README 규칙과 같게) 씁니다. 세 장을 모두 넣으면 길어지므로 대표 1장 + 나머지는 `<details>` 안에 넣는 것을 권장합니다(`<details>`/`</details>` 균형 테스트 있음).
- 같은 변경에서 [AGENTS.md](../AGENTS.md)의 코드 지도(`scripts/`)와 실행 근거 지도에 `scripts/build_tableau_extracts.py`, `output/tableau/`, 이 문서를 추가합니다.
- README 편집 후 `PYTHONUTF8=1 uv run --locked --offline python -m pytest -q`를 실행합니다(앵커 중복, 내부 링크, 링크 경로 존재, `<details>` 균형 검사).

## 8. 체크리스트

- [ ] manifest `all_checks_passed = true` 확인
- [ ] 데이터 원본 15개 중 사용한 것만 연결, `flights_rowlevel.csv` 미연결
- [ ] 모든 지연율·recall·FPR이 `SUM(...)/SUM(...)` 계산 필드로 표시됨(행 열의 비율을 AVG하지 않음)
- [ ] 각 대시보드 부제에 집단·분모(255,001 / 180,332 / 255,001 × 3 / 3,031)가 있음
- [ ] 대시보드 1 KPI가 255,001 / 45,000 / 17.6%로 표시됨
- [ ] 대시보드 2 구간 막대의 행 수 합(역할별)이 180,332, 혼동행렬 칸 합이 180,332
- [ ] 대시보드 2 혼동행렬 시드 42 값이 `output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_runs.csv`와 같음
- [ ] 대시보드 3 `raw_time_pattern = both_missing`의 시드합산 recall ≈ 10.0%
- [ ] 대시보드 3 예산별 비교의 3시드 평균이 README 2절 표와 같음(예: LightGBM 24개 설정 Macro F1 0.6005, Random Forest 9개 설정 0.6085), 보정 실험 행은 이 시트에 없음
- [ ] 대시보드 3 신뢰도 곡선이 시드 하나·`equal_frequency_15`로 표시되고 구간 `n` 합이 180,332, 대각선(완전 보정선) 표시
- [ ] 미량 강수 구간 라벨에 "0.0001 표기, 측정량 아님" 표시
- [ ] 캡션에 인과·미래 성능을 부정하는 문장, 10분 가정, 시드 = 재분할 문구가 있음
- [ ] 색상이 1.1 토큰만 사용(빨강/초록 없음)
- [ ] 고정 크기 1200 × 800, 한글 깨짐 없음
- [ ] Tableau Public 게시(사용자 로그인), 데이터 다운로드 비허용, 설명 문구 입력
- [ ] PNG 3장 `assets/readme/tableau_*.png` 저장
- [ ] 게시 URL과 스크린샷 경로를 조율 에이전트에 보고

## 9. 하지 말 것

- `data/tableau/flights_rowlevel.csv` 또는 `data/` 아래 파일을 Tableau Public에 게시하거나 커밋하지 않습니다.
- `.twb`/`.twbx` 파일을 저장소에 커밋하지 않습니다(필요하면 로컬 보관만).
- 서로 다른 집단(255,001 / 180,332 / 3,031)의 지연율·점수를 한 축·한 표에서 비교하지 않습니다. 특히 D3 OOF(날씨 미사용, 255,001행)와 분류기 비교(날씨 사용, 180,332행)를 섞지 않습니다.
- `delay_rate`, `recall_delayed` 같은 행별 비율 열을 `AVG`하거나 `SUM`하지 않습니다.
- "날씨 때문에", "~하면 지연이 줄어든다", "예측 정확도 X%로 미래 지연을 예측" 같은 인과·미래 표현, "혼잡도"(`Traffic`) 표현을 쓰지 않습니다.
- 탐색 예산이 다른 분류기 결과를 "최적 모델"·"더 우수한 알고리즘"으로 표현하지 않습니다. 예산(`search_budget`)을 항상 함께 표시하고, 서로 다른 `experiment`의 행을 합산하지 않습니다(보정 실험의 `none` 행은 다른 실험과 값이 같거나 거의 같은 재적합 결과).
- 신뢰도 곡선의 `mean_probability`·`observed_rate`를 `AVG`하지 않습니다(`SUM(sum_probability)/SUM(n)`, `SUM(n_delayed)/SUM(n)` 사용).
- 시드 SD를 신뢰구간·유의성이라고 부르지 않습니다. Wilson 구간은 "참고 구간"으로만 표기합니다.
- 월 축에 연도를 붙이지 않습니다(대시보드 1의 월은 2018·2019년 혼합). 연-월은 대시보드 2의 날짜 귀속 집단에서만 씁니다.
- 원본 시각 결측 칸을 숨기거나 다른 시각에 합치지 않습니다.
- CSV를 Tableau 밖에서 편집하거나 새 수치를 손으로 입력하지 않습니다. 수정이 필요하면 생성기를 고쳐 다시 실행합니다.
- 외부 데이터(공항 좌표, 날씨 등)를 인터넷에서 새로 가져와 붙이지 않습니다.
- Tableau Public 로그인 정보를 에이전트가 입력하지 않습니다.
- README의 기존 수치·결론 문장을 바꾸지 않습니다. 새 하위 절만 추가합니다.
