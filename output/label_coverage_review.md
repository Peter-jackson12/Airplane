# 라벨 유무 분포 진단 — 실제 전체 데이터

## 범위와 해석

원본 1,000,000행, 19열. 라벨 255,001행 (25.5001%), 미라벨 744,999행.
라벨행 양성률 17.6470%. ID 중복 0, 완전 행 중복 0.
분석 단위는 CSV의 한 행이다. ID가 유일해도 실제 운항편의 유일성을 입증하지는 않는다.
미라벨의 정답은 채우지 않았다. 분석은 원본 컬럼 기준이며 전처리 후 모델 입력의 동등성 검사는 아니다.

## 직접 측정한 결과

타깃을 제외한 결측률의 최대 절대 차이는 **Airline: +0.1235%p (미라벨-라벨)**다.
주요 저차원 분포(Month/Airline/출발·도착 공항)의 최대 TV는 **0.018086**다.
TV=0.5×구성비 절대 차이 합이며 0이면 관측 분포가 같다. 통계적 유의수준이나 대표성 합격선이 아니다.

| column | categories | total_variation | max_share_gap_pp | unlabeled_rows_in_unseen_categories | unlabeled_unseen_pct |
| --- | --- | --- | --- | --- | --- |
| Tail_Number | 6430 | 0.071678 | 0.018586 | 407 | 0.054631 |
| Origin_Airport | 374 | 0.018086 | 0.266471 | 0 | 0.0 |
| Destination_Airport | 375 | 0.017221 | 0.271511 | 4 | 0.000537 |
| Month | 12 | 0.012418 | 0.453404 | 0 | 0.0 |
| Airline | 29 | 0.010563 | 0.53568 | 0 | 0.0 |
| Destination_State | 53 | 0.009776 | 0.232699 | 0 | 0.0 |
| Carrier_Code(IATA) | 12 | 0.009604 | 0.549023 | 0 | 0.0 |
| Origin_State | 53 | 0.008663 | 0.355379 | 0 | 0.0 |
| Day_of_Month | 31 | 0.007702 | 0.157545 | 0 | 0.0 |

![결측 및 월별 라벨 보유율](label_coverage/coverage_overview.png)

![상위 범주 구성비](label_coverage/category_shares.png)

## 표본 1,000행 이상 그룹의 라벨 보유율 차이

희소 그룹의 극단적 비율을 주요 증거로 쓰지 않기 위한 표시 기준이다. 전체 그룹은 CSV에 보존했다.

| column | category | total_n | labeled_n | label_rate_pct | gap_from_overall_pp |
| --- | --- | --- | --- | --- | --- |
| Destination_Airport | EUG | 1088 | 226 | 20.7721 | -4.728 |
| Airline | Empire Airlines Inc. | 1056 | 308 | 29.1667 | 3.6666 |
| Origin_Airport | ANC | 2383 | 675 | 28.3256 | 2.8255 |
| Origin_Airport | MAF | 1215 | 344 | 28.3128 | 2.8127 |
| Airline | Commutair Aka Champlain Enterprises, Inc. | 7143 | 1630 | 22.8195 | -2.6806 |
| Origin_Airport | EUG | 1049 | 294 | 28.0267 | 2.5266 |
| Destination_Airport | CID | 1272 | 296 | 23.2704 | -2.2297 |
| Origin_Airport | COS | 1477 | 409 | 27.6913 | 2.1912 |
| Destination_Airport | EWR | 23756 | 5542 | 23.3288 | -2.1713 |
| Origin_State | Alaska | 4476 | 1233 | 27.5469 | 2.0468 |
| Destination_Airport | MAF | 1295 | 304 | 23.4749 | -2.0252 |
| Destination_State | New Jersey | 21910 | 5145 | 23.4824 | -2.0177 |

## 결론의 경계

측정된 구성비 차이는 관측된 변수에서의 차이이며, 라벨 선정 과정의 원인을 알려주지 않는다.
고유값이 많은 Tail_Number는 작은 그룹과 표본 크기 차이 때문에 TV가 커질 수 있다. 이를 곧바로 선택 편향으로 판정하지 않는다.
분포가 가까워도 MCAR(완전 무작위 결측), 미라벨의 지연율, 성능 일반화는 입증되지 않는다.
따라서 기존 CV 결과의 직접 적용 범위는 라벨 255,001행이다. 미라벨로의 확대는 가정으로 남긴다.
Month 비교는 계절별 구성 진단이다. 연도가 없어 연속 시계열·장기 추세·미래 검증이라고 부르지 않는다.
HHMM 숫자의 평균·분위수는 원시 표기 비교용이며 평균 시각이나 경과시간으로 해석하지 않는다.
그룹별 observed_delay_rate는 관측 라벨 안에서만 계산했고 인과효과가 아니다.

## 위험과 조치

대표성 미확인: 중요도 중간, 한계의 존재는 확실하나 편향의 방향·크기는 미확인. 라벨 생성/선정 규칙을 확보하기 전 재가중·미라벨 정답 추정을 하지 않는다.
연도·실측/예정 시각 규약 미확인: 미래 예측 주장에는 중요도 높음. 현재는 정적 데이터 과제로 범위를 제한한다.
측정된 분포 차이 자체는 오류로 판정하지 않는다. 전체 구성비·결측률·미관측 범주 비율을 다음 데이터와 비교할 기준으로 보존한다.

## 근거와 재현

원본 `data/train.csv`, SHA256 `e996797b3fffab9484c9005457370662ec2b08516eecaf3eed9d2b72061edd84`.
실행 시각 `2026-09-16T02:51:38.541858+00:00`. counts/구성비 합계 검증을 코드에서 실행했다.
`uv run --offline python -u notebooks/build_label_coverage_review.py`

실행 노트북: [label_coverage_review.ipynb](../notebooks/label_coverage_review.ipynb).
세부 근거: [그룹 분포](label_coverage/group_distributions.csv), [결측률](label_coverage/missingness.csv),
[수치 분포](label_coverage/numeric_distributions.csv), [원본 스키마](label_coverage/schema.csv), [지문](label_coverage/provenance.json).
