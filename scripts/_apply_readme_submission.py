"""One-time README presentation edit. Removed before the documentation PR merges."""
from pathlib import Path
import hashlib
import re

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'README.md'
original = path.read_text(encoding='utf-8')
appendix_marker = '<a id="evidence-appendix"></a>'
appendix = original[original.index(appendix_marker):]

prefix = '''# Airplane · 항공편 지연 분류와 데이터 전처리

[![Git-only CI](https://github.com/Peter-jackson12/Airplane/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Peter-jackson12/Airplane/actions/workflows/ci.yml)
[평가 코드](src/cv.py) · [잠금 의존성](uv.lock) · [검증된 결과](#5-최신검증결과) · [실행 재개](#7-코드구조와재현)

**불확실한 값을 채우는 것보다, 무엇을 알고 있는지 구분하고 같은 조건에서 검증하는 프로젝트입니다.**

원본 항공편 데이터의 결측·시각·항공사 식별을 점검하고, **전처리의 타당성과 실제 예측 성능을 분리해 검증**합니다. 기존 입력의 LightGBM 분류·OOF·확률 보정 분석은 완료했으며, 날씨를 새 정보원으로 추가하는 확장 실험을 진행하고 있습니다.

작성: Peter-jackson12 TF · 문서 기준: 2026-09-21 · 제출 형태: GitHub README + 코드 + 연결된 실행 근거

<a id="submission-overview"></a>
### 30초 요약

| 데이터 규모 | 직접 평가한 집단 | 전처리 비교 실험 | 날짜 귀속 채택 |
|---:|---:|---:|---:|
| **1,000,000행** | **라벨 255,001행** | **10조건 × 3시드 × 5-fold** | **706,759행** |

분모와 범위는 [데이터 설명](#2-데이터와분석범위), [10조건 결과](output/preprocessing_full_summary.csv), [날짜 귀속 집계](output/baseline_recovery_v2_row_date_attribution_20260918_status_summary.csv)에서 확인합니다. 날짜 귀속 집단과 모델 평가 집단은 같은 모집단이 아닙니다.

> **확인한 결론:** 전처리의 의미와 검증 구조는 개선했지만 **Macro F1 향상은 확인하지 못했습니다.** 확률 보정으로 ECE가 감소해도 분류 성능이 개선되는 것은 아니었습니다.
>
> **아직 확인하지 않은 결론:** 전체 날씨 결합과 동일 조건 모델 비교는 미완료입니다. **날씨 추가의 성능 향상·실시간 운영 성능을 주장하지 않습니다.**

<a id="project-status"></a>
### 검증이 끝난 부분과 진행 중인 부분

| 영역 | 현재 근거 | 다음 판정 경계 |
|---|---|---|
| 원본 품질·전처리 | 유일 대응 대치, 시각 의미, 원본 결측 이력 검증 | 의미 개선 ≠ 성능 향상 |
| 모델·OOF·보정 | 10조건 × 3시드 비교 및 별도 OOF·보정 분석 | 전체 Phase·미래 운영 성능 검증은 아님 |
| 날짜 귀속 | 2018/2019 BTS 12개월 대조, 706,759행 채택 | 293,241행은 미검사가 아니라 보류 |
| 날씨 표본 | 21행·기존 300행·stratafix 300행을 각각 검증 | 서로 다른 표본을 합산하지 않음 |
| 전체 날씨 수집 | bulk/shard/resume 구현, 부분 실수집·복구 로컬 보고 | 전체 완료 manifest는 아직 Git에 미반영 |
| 날씨 모델 비교 | 동일 라벨 행·분할·시드·nested 평가 계약 | 전체 결합 → 대조 실험 → 최종 결론 |

**이 표는 실시간 다운로드 모니터가 아닙니다.** 최신 로컬 진행률을 추정해 채우지 않으며, 완료 수치는 해당 실행 근거가 반영된 뒤 갱신합니다. [완료 결과를 반영할 위치](#submission-completion)

**평가자:** [30초 요약](#submission-overview) → [전체 구조](#4-현재파이프라인) → [결과 그래프](#5-최신검증결과) → [한계](#6-한계와다음단계)  
**실행자·에이전트:** [AGENTS.md](AGENTS.md) → [현재 상태](#project-status) → [재현·운영 경계](#7-코드구조와재현) → [근거 지도](#8-문서안내)

**목차:** [1. 목적](#1-목적과문제정의) · [2. 데이터](#2-데이터와분석범위) · [3. 전처리](#3-전처리결정과근거) · [4. 파이프라인](#4-현재파이프라인) · [5. 결과](#5-최신검증결과) · [6. 날씨·다음 단계](#6-한계와다음단계) · [7. 재현](#7-코드구조와재현) · [8. 근거 지도](#8-문서안내) · [9. 접기 부록](#evidence-appendix) · [발표 흐름](#presentation-route)

---

'''
text = prefix + original[original.index('<a id="1-목적과문제정의"></a>'):]


def replace_once(old: str, new: str) -> None:
    global text
    if text.count(old) != 1:
        raise ValueError(f'Expected one edit target, found {text.count(old)}: {old[:90]}')
    text = text.replace(old, new, 1)


replace_once('### 분석에서 중요했던 네 가지 문제', '''![원본 100만 행 중 날짜 귀속 채택 706759행, 결측 키의 단일 후보 291308행 및 기타 1933행은 보류](assets/readme/date_attribution.svg)

**그림 1. 채택과 보류를 구분한 날짜 귀속.** 막대는 모두 같은 원본 100만 행의 서로 겹치지 않는 집단입니다. 후보 연도가 하나여도 키가 결측이면 채택하지 않습니다. [원본 집계 CSV](output/baseline_recovery_v2_row_date_attribution_20260918_status_summary.csv) · [그림의 수치·출처 해시](assets/readme/sources.json)

### 분석에서 중요했던 네 가지 문제''')

replace_once('## 4. 현재 파이프라인\n', '''## 4. 현재 파이프라인

### 전체 작업을 한 장으로 보기

```mermaid
flowchart TD
    A["원본 100만 행 / Delay 라벨·미라벨 구분"] --> B["결측·시각·항공사 식별 점검"]
    B --> C["기존 입력 전처리 + LightGBM"]
    C --> D["nested 평가 / OOF / 보정 비교 완료"]
    A --> E["BTS 대조 / 채택한 행만 날짜 귀속"]
    E --> F["공항-관측소 매핑 / UTC 예측 시점"]
    F --> G["월별 bulk 날씨 수집 / shard별 저장"]
    G --> H["전체 완료 검증 / finalize"]
    H --> I["예측 시점까지 가용한 날씨만 결합"]
    D --> J["같은 평가 조건의 날씨 없음 vs 있음 비교"]
    I --> J
```

**핵심 질문은 다운로드 속도가 아니라 새 정보가 지연 분류에 도움이 되는가입니다.** 상단의 기존 입력 분석과 하단의 날씨 확장은 별도 경로이며, 마지막 동일 조건 비교로 연결합니다. `finalize`는 수집 완료 검사이지 모델 실험의 완료가 아닙니다.

### 라벨 누수를 막는 평가 경계
''')

replace_once('![전처리 변경 효과](output/preprocessing_full_effects.png)\n\n그림은 동일 시드끼리 계산한 변경−기준의 평균 ±1 표본 SD이며 신뢰구간이 아닙니다.', '''![동일 nested 프로토콜의 P4, P4_clean, P6_fixed, P6_clean Macro F1 평균과 3시드 표준편차 비교](assets/readme/model_comparison.svg)

**그림 2. 전처리 10조건 중 네 기준 조건의 Macro F1.** 점은 3시드 평균, 오차막대는 ±1 표본 SD이며 신뢰구간이 아닙니다. 차이를 읽기 위한 확대 축임을 명시했으며, 오차막대 겹침만으로 동등성·유의성을 판정하지 않습니다. 모든 조건과 paired delta는 [기존 개별 변경 효과 그림](output/preprocessing_full_effects.png)에서도 확인합니다.''')

replace_once('표는 동일 라벨 255,001행·nested 평가의 3시드 평균입니다.', '''![P6_clean 공유형 보정의 ECE와 LogLoss 비교: Isotonic의 ECE 감소가 LogLoss 개선으로 이어지지는 않음](assets/readme/calibration_tradeoff.svg)

**그림 3. 보정 지표의 상충 관계.** P6_clean·공유형·전체 라벨 집단의 세 보정 조건만 표시했습니다. 두 축 모두 낮을수록 좋고, 정확한 Macro F1은 위 표에서 함께 읽습니다. ECE는 비율을 100배 한 %p 단위입니다. [원본 수준값 CSV](output/baseline_recovery_v2_calibration_20260917_level_summary.csv) · [그림 출처](assets/readme/sources.json)

표는 동일 라벨 255,001행·nested 평가의 3시드 평균입니다.''')

replace_once('### 날씨 결합과 비교의 고정 계약', '''![stratafix 수집 가능 268행에서 가용성 지연 가정별 출발 및 도착 날씨 결합률, 60분 가정에서 각각 150행과 157행 결합](assets/readme/weather_latency.svg)

**그림 4. 데이터가 있어도 예측 시점에 가용하지 않으면 사용할 수 없습니다.** 분모는 stratafix의 수집 가능 268행입니다. 보류 32행을 포함한 전체 300행 분모와 구분하며, 0/10/30/60분은 실제 수신 지연의 실측값이 아닌 가정입니다. 이는 결합률 그래프이지 날씨 모델의 성능 그래프가 아닙니다. [민감도 CSV](output/baseline_recovery_v2_weather_expanded_stratafix_20260918_latency_sensitivity.csv)

### 날씨 결합과 비교의 고정 계약''')

replace_once('**다음 우선순위입니다. 전체 네트워크 실행은 아직 하지 않았고, 먼저 실행 plan을 고정한 뒤 작은 shard부터 검증합니다.**', '**진행 중인 확장 단계입니다. bulk 수집·부분 실수집·복구 경로는 구현됐으며, 전체 수집 완료와 행별 결합은 별도의 근거로 판정합니다. 최신 로컬 진행률은 이 문서에서 추정하지 않습니다.**')
replace_once('**이번 작업은 1~2번을 완료했습니다.** 전체 706,759행 수집·모델 재학습으로 바로 확대하지 않았습니다.', '**검증 완료 근거는 1~2번까지 확보했습니다.** 이후 전체 수집 경로를 구현해 로컬 실행으로 확장했으며, 전체 수집 완료·행별 결합·모델 재학습의 완료는 아직 이 README에서 선언하지 않습니다.')

replace_once('<a id="7-코드구조와재현"></a>', '''<a id="weather-glossary"></a>
### 용어를 작업 단위로 읽기

| 용어 | 이 프로젝트에서의 뜻 |
|---|---|
| request group | 같은 IEM network·UTC 월·최대 20 station으로 만든 한 번의 논리적 요청 |
| HTTP attempt | 실제 호출 한 번; 실패와 재시도도 각각 예산을 소비 |
| shard | request group을 기본 50개씩 나눈 저장·재개 단위; 새로운 모델이나 데이터 종류가 아님 |
| checkpoint | 어디까지 처리했는지와 누적 예산·시도 이력을 저장한 진행 기록 |
| manifest / fingerprint | 결과의 경로·해시·집계 / 입력과 요청 규칙이 같은지 판별하는 식별값 |
| finalize / join | 전체 요청 완료·파일 무결성 검증 / 항공편 행과 가용한 날씨 관측 결합 |
| OOF / TE | 학습에 쓰지 않은 fold의 예측 / 라벨 기반 범주 인코딩; TE는 inner 경계를 지킴 |

<a id="submission-completion"></a>
### 남은 결과를 반영할 위치와 완료 기준

빈 성능 그래프나 예상 다운로드 수치를 실제 결과처럼 넣지 않습니다. 아래 항목은 **결과 미반영**이며, 데이터가 없어서 0이라고 표시한 것이 아닙니다.

| 후속 결과 | 필요한 근거 | 반영 위치·완료 조건 |
|---|---|---|
| 전체 수집 통계·shard별 요청/재시도/용량 | 전체 fetch manifest와 각 shard manifest·checkpoint | 6~7절; 전 shard 성공, 요청 집합 일치, 캐시 재검증 후 실측 그래프 추가 |
| 전체 행별 날씨 결합률·제외 사유 | 전체 join 및 진단 결과 | 6절; ID·행수·순서와 observed/available 시간 경계 확인 |
| 날씨 없음 vs 있음 성능 | 동일 라벨 행·시드·fold·nested 계약의 paired 결과 | 5절; Macro F1·LogLoss·AUC, 시드별 차이와 한계 보고 |
| 제출 최종 결론 | 위 근거와 코드 revision | 상단 요약·현재 상태·그림 출처를 함께 갱신; 향상 자체는 완료 조건이 아님 |

<a id="7-코드구조와재현"></a>''')

replace_once('## 7. 코드 구조와 재현\n', '''## 7. 코드 구조와 재현

> **수집 중인 실행 환경:** 문서 개편을 반영하려고 돌아가는 프로세스를 중단하거나 중간에 pull하지 않습니다. 문서와 그림 작업은 GitHub의 별도 브랜치에서 수행하며, 실행 환경 동기화는 현재 수집이 정상 종료하거나 멈춘 뒤 기존 작업을 보존해 진행합니다.

<a id="reliability-design"></a>
### 실패를 숨기지 않는 수집·복구 구조

```mermaid
flowchart TD
    A["고정된 plan과 이전 shard 검증"] --> B{"검증된 기존 캐시가 있는가?"}
    B -->|있음| C["해시·스키마·시간창 검사 / 시도 이력 보존"]
    B -->|없음| D["HTTP 전에 요청·바이트 예산 예약 저장"]
    D --> E["같은 요청으로 제한된 재시도 / 누적 예산 반영"]
    E --> F["캐시 검증 / checkpoint 완료 상태 저장"]
    C --> G{"shard 전체 성공인가?"}
    F --> G
    G -->|성공| H["결과 재검증 후 다음 shard"]
    G -->|미완료·실패| I["중단 / 이후 shard 시작하지 않음"]
```

| 보호하는 경계 | 구현과 검증 근거 | 보장하지 않는 것 |
|---|---|---|
| 다른 요청을 같은 작업으로 이어받지 않음 | [고정 plan·fingerprint](notebooks/fetch_weather_full_sharded.py) | 실패한 station을 임의 대체하지 않음 |
| 실패·재시도도 예산에 포함 | [예약·누적 accounting](notebooks/fetch_weather_sample_expanded.py) | 미측정 바이트를 0으로 가정하지 않음 |
| 완료 기록도 다시 확인 | [순차 runner](notebooks/run_weather_full_collection.py), [회귀](tests/test_weather_full_collection_runner.py) | exit 0 하나만으로 전체 완료를 선언하지 않음 |
| HTTP 성공 후 완료 저장이 끊겨도 이력 유지 | [checkpoint 복구 회귀](tests/test_weather_expanded_pipeline.py) | 컨테이너·OS·스토리지 무장애를 주장하지 않음 |

파일 교체 재시도는 **HTTP 재호출과 별개**입니다. 현행 checkpoint 교체는 최초 시도를 포함해 최대 5회, 실패 사이 0.1초 간격이며 계속 실패하면 예외를 전파합니다. 원자적 rename이 전원 장애까지 포함한 영속성 보장을 뜻하지는 않습니다. 원인 프로세스를 특정하지 못한 Windows 접근 거부를 백신 탓으로 단정하지 않습니다.

### 공개 저장소와 로컬 데이터의 경계

GitHub에는 코드·테스트·문서·공개 가능한 집계 근거를 둡니다. 원본 `data/train.csv`, 대용량 날씨 캐시와 사고 snapshot은 로컬에 유지합니다. 인증키·쿠키·개인정보는 커밋하지 않으며, `.gitignore`를 보안 접근제어로 간주하지 않습니다. 이 README와 그림은 **Git에 있는 집계 파일만으로** 생성·검증할 수 있습니다.
''')

replace_once('### 코드 지도', '''<a id="readme-figures"></a>
### README 그림의 재현과 출처 검증

다음 명령은 공개 집계 CSV만 읽습니다. IEM 호출·원본 데이터 로드·모델 학습을 하지 않습니다. 그림 파일은 정적 SVG이며 한글 설명과 정확한 표를 함께 제공해 이미지 없이도 결론을 읽을 수 있게 했습니다.

```powershell
# 그림·실제 사용값·출처 및 그림 해시 생성
uv run --locked --offline python scripts/build_readme_assets.py

# 읽기 전용: 출처, 필터, 표시 수치, SVG 해시가 바뀌지 않았는지 검사
uv run --locked --offline python scripts/build_readme_assets.py --check

# README 링크·기존 계약·그림 재현성 검사
uv run --locked --offline python -m pytest -q tests/test_readme_contract.py tests/test_readme_presentation.py
```

[그림 생성기](scripts/build_readme_assets.py) · [수치·필터·SHA-256 출처표](assets/readme/sources.json) · [문서 회귀 검사](tests/test_readme_presentation.py)

### 코드 지도''')

replace_once(appendix_marker, '''<a id="presentation-route"></a>
### 발표할 때는 이 순서로 설명합니다

**문제 → 판단 → 검증 → 확장.** “결측을 많이 채우면 더 좋은 모델일까?”로 시작해, 모호한 대치를 중단하고 시각의 의미를 바로잡은 이유를 3절에서 보여줍니다. 이어 5절의 같은 조건 비교에서 **의미 개선이 Macro F1 향상으로 이어지지 않았음**을 설명하고, 보정 지표와 탐지 성능의 차이를 짚습니다. 마지막으로 새 정보원인 날씨도 예측 시점의 가용성을 지켜야 한다는 6절로 연결합니다. 수집·복구 구조는 이 실험을 재현 가능하게 만드는 근거로 설명합니다.

**질문에 대한 근거 위치:** “누수는?” → 4절의 inner/outer 경계와 전체 입력 묶음의 한계. “날씨 효과는?” → 아직 미검증, 완료 기준표. “왜 보류했나?” → 2절 날짜 귀속과 6절 매핑 규칙. “중단되면?” → 7절 checkpoint·순차 runner와 회귀 테스트.

기존 분석·명령·실행 회차는 아래 **같은 README 안**에 남아 있습니다. AGENTS와 README를 읽는 기존 컨트롤타워 방식은 유지합니다.

''' + appendix_marker)

# Preserve the long evidence appendix byte-for-byte, not merely its headings.
assert text[text.index(appendix_marker):] == appendix
old_ids = set(re.findall(r'<a\s+id="([^"]+)"', original))
new_ids = re.findall(r'<a\s+id="([^"]+)"', text)
assert old_ids <= set(new_ids) and len(new_ids) == len(set(new_ids))
path.write_text(text, encoding='utf-8')
print('README presentation applied; all explicit anchors and the entire evidence appendix preserved.')
print('Appendix SHA256:', hashlib.sha256(appendix.encode()).hexdigest())
