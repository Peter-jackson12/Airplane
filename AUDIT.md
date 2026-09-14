# 항공 지연 예측 파이프라인 감사 보고서

**대상**: `run_baseline.py`, `run_tuned.py`, `run_target_encoded.py`, `run_hybrid.py`, `run_pseudo_labeling.py`, `run_advanced_features.py`, `run_phase7_weather_model.py` (총 7종, 1,852행)
**작성일**: 2026-09-14
**범위**: 코드 정적 분석 + 데이터 사실 확인 (모델 재학습은 수행하지 않음)
**작성 대상 독자**: 이 프로젝트의 모델링 결과를 검토·재현해야 하는 사람 (본인 및 리뷰어)

---

## 0. 요약 판정

| 항목 | 판정 |
|---|---|
| OOF Target Encoding의 fold 경계 | **fold 내부에서 계산됨 — 정상** (TE를 사용하는 5개 스크립트 전부) |
| Pseudo-label의 fold 경계 침범 | **누수 확정 (HIGH)** — teacher 앙상블이 전체 라벨을 봤고, 그 산출물이 모든 fold의 학습셋에 동일하게 주입됨 |
| Traffic(혼잡도) 피처의 타깃 누수 | **타깃 누수 아님** — 집계 대상이 `Route`의 `count`이며 `Delay`가 개입하는 경로가 전혀 없음. 단, 전체 데이터 기준 transductive 집계라는 별도 이슈는 존재 |
| 평가 프로토콜 일관성 | **불일치 다수 — 현재 상태의 Phase 간 성능 비교는 무효** |
| 추가 발견 (전 스크립트 공통) | 검증 fold를 early stopping의 `eval_set`으로 사용 (경미한 낙관 편향), 임계값을 채점 대상 OOF 벡터에서 직접 선택 (winner's curse) |

**가장 중요한 결론 하나**: 7개 스크립트는 `random_state=42`만 공유할 뿐 **피처셋·하이퍼파라미터·임계값 탐색 범위·지표 집계 방식이 모두 조금씩 다르다.** 따라서 현재 보고되고 있는 "Phase N → Phase N+1 성능 향상"은 어떤 변경 때문인지 귀속(attribution)이 불가능하다. 이것은 누수보다 실무적으로 더 큰 문제다.

---

## 1. 공통 모듈 `src/features.py` 설계안

### 1.1 중복 현황

7개 스크립트에서 실질적으로 동일한 코드 블록이 반복된다.

| 로직 | 중복 횟수 | 대표 위치 |
|---|---|---|
| `Carrier_Code(IATA)` → `Airline` 결측 대치 | 7/7 | [run_baseline.py:39-58](run_baseline.py#L39-L58), [run_hybrid.py:37-46](run_hybrid.py#L37-L46) |
| HHMM 파싱 (`//100`, `%100`, `-1` 센티널) | 7/7 | [run_target_encoded.py:49-53](run_target_encoded.py#L49-L53) |
| 자정 넘김 보정 소요시간 | 7/7 | [run_hybrid.py:62-70](run_hybrid.py#L62-L70) |
| `Route` 결합 피처 | 7/7 | [run_tuned.py:85-89](run_tuned.py#L85-L89) |
| `Air_Speed_Proxy` | 7/7 | [run_baseline.py:128](run_baseline.py#L128) |
| Sin/Cos 시간 인코딩 | 7/7 | [run_hybrid.py:73-75](run_hybrid.py#L73-L75) |
| `drop_cols` 리스트 | 7/7 | [run_pseudo_labeling.py:80-92](run_pseudo_labeling.py#L80-L92) |
| 라벨/미라벨 분리 + `target_map` | 7/7 | [run_hybrid.py:127-132](run_hybrid.py#L127-L132) |
| `LabelEncoder` → `category` 변환 | 7/7 | [run_advanced_features.py:185-188](run_advanced_features.py#L185-L188) |
| `get_smoothed_target_encoding` (완전 동일 본문) | 5/7 | [run_target_encoded.py:108-124](run_target_encoded.py#L108-L124), [run_hybrid.py:108-119](run_hybrid.py#L108-L119), [run_pseudo_labeling.py:96-107](run_pseudo_labeling.py#L96-L107), [run_advanced_features.py:151-162](run_advanced_features.py#L151-L162), [run_phase7_weather_model.py:92-98](run_phase7_weather_model.py#L92-L98) |
| StratifiedKFold + LGBM 학습 루프 | 7/7 | — |
| 임계값 그리드 탐색 | 6/7 | [run_hybrid.py:206-215](run_hybrid.py#L206-L215) |
| Confusion Matrix 출력 | 5/7 | — |
| 소요시간 역산 복원 | 2/7 | [run_advanced_features.py:81-113](run_advanced_features.py#L81-L113) |
| Traffic 혼잡도 피처 | 2/7 | [run_advanced_features.py:117-122](run_advanced_features.py#L117-L122) |

**즉 7개 파일 중 실질적으로 고유한 코드는 각 파일당 20~60행 수준이고, 나머지 180~250행은 복사본이다.**

### 1.2 설계 원칙

리팩터링의 목표는 "행 수 줄이기"가 아니라 **2절에서 지적하는 누수를 구조적으로 재발 불가능하게 만드는 것**이다. 따라서 세 가지를 강제한다.

1. **fold를 넘나드는 모든 연산은 반드시 fold 인덱스를 명시적 인자로 받는다.** 타깃을 사용하는 함수는 "이 라벨을 써도 되는 행 집합"을 인자로 받지 않으면 호출할 수 없게 만든다.
2. **피처 생성 단계를 `target-free`(fold 밖에서 1회) / `target-dependent`(fold 안에서 K회)로 물리적으로 분리한다.** 모듈 안에서 두 그룹을 주석이 아니라 별도 섹션·별도 호출 규약으로 둔다.
3. **프로토콜 상수(fold 수, seed, 임계값 그리드, 지표 집계 방식)는 스크립트가 아니라 모듈이 소유한다.** 스크립트가 덮어쓰려면 명시적으로 인자를 넘기게 하고, 그 사실이 결과 리포트에 자동으로 찍히게 한다.

### 1.3 함수 시그니처 설계안

```python
# src/features.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

# =============================================================================
# 상수 — 7개 스크립트에 흩어져 있던 매직 리터럴을 한 곳으로
# =============================================================================

TARGET_COL = "Delay"
TARGET_MAP = {"Not_Delayed": 0, "Delayed": 1}

#: TE / category 인코딩 대상. baseline만 State·Carrier_Code를 추가로 포함했다(§3 참조).
CAT_COLS: tuple[str, ...] = (
    "Tail_Number", "Route", "Origin_Airport", "Destination_Airport", "Airline",
)

#: 제거 컬럼 프리셋. 스크립트별 drop_cols 차이를 이름으로 고정한다(§3.2).
DROP_PRESETS: dict[str, tuple[str, ...]] = {
    # run_baseline.py:140-147
    "baseline": ("ID", "Cancelled", "Diverted",
                 "Origin_Airport_ID", "Destination_Airport_ID", "Carrier_ID(DOT)"),
    # run_tuned / run_target_encoded / run_hybrid / run_pseudo_labeling
    "pruned": ("ID", "Cancelled", "Diverted",
               "Origin_Airport_ID", "Destination_Airport_ID", "Carrier_ID(DOT)",
               "Carrier_Code(IATA)", "Origin_State", "Destination_State", "Day_of_Month"),
    # run_advanced_features / run_phase7_weather_model
    "advanced": ("ID", "Cancelled", "Diverted",
                 "Origin_Airport_ID", "Destination_Airport_ID", "Carrier_ID(DOT)",
                 "Carrier_Code(IATA)", "Origin_State", "Destination_State",
                 "Day_of_Month", "Dep_Minute", "Arr_Minute"),
}


# =============================================================================
# A. 로딩 — 타깃 비의존
# =============================================================================

def load_data(
    path: str | Path,
    *,
    with_weather: bool = False,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """원본 CSV를 로드한다.

    with_weather=True 이면 data/train_with_weather.csv 스키마(29열)를 기대하고,
    사전 계산되어 들어 있는 Dep_Hour 열을 즉시 제거한다.
    (현재 run_phase7_weather_model.py:54 는 이 열을 조용히 덮어쓰고 있어,
     병합 시점 정의와 학습 시점 정의가 다를 때 추적이 불가능하다.)
    """


# =============================================================================
# B. 타깃 비의존 전처리 — fold 밖에서 1회만 호출 (전체 df 대상, 누수 없음)
# =============================================================================

def impute_cross(
    df: pd.DataFrame,
    *,
    pairs: Sequence[tuple[str, str]] = (
        ("Carrier_Code(IATA)", "Airline"),
        ("Origin_Airport", "Origin_State"),
        ("Destination_Airport", "Destination_State"),
    ),
    bidirectional: bool = False,
) -> pd.DataFrame:
    """식별자 간 1:1 매핑을 이용한 상호 결측 대치.

    pairs 의 각 (key, value) 에 대해 key -> value 방향으로 채운다.
    bidirectional=True 이면 value -> key 방향도 수행한다
    (run_baseline.py:46-58 만 Carrier<->Airline 양방향을 수행했다).

    누수 판정: Delay 를 참조하지 않으므로 타깃 누수 없음. 전체 df 기준 집계이나
    "IATA 코드와 항공사명의 대응"은 데이터에 독립적인 상수이므로 transductive 이슈도 없다.
    """


def parse_hhmm(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """HHMM 정수/실수/결측 → (hour, minute). 결측·파싱실패는 -1 센티널.

    7개 스크립트에 7벌 존재하던 동일 로직의 단일 구현.
    """


def build_time_features(
    df: pd.DataFrame,
    *,
    keep_minute: bool = False,
    midnight_wrap: bool = True,
) -> pd.DataFrame:
    """Dep_Hour / Arr_Hour / (선택) Dep_Minute·Arr_Minute / Estimated_Duration 생성.

    midnight_wrap=True 이면 도착<출발 인 경우 +1440분 보정.
    양쪽 시각이 모두 유효할 때만 Estimated_Duration 을 채우고, 아니면 NaN.
    """


def restore_time_missing(
    df: pd.DataFrame,
    *,
    route_col: str = "Route",
    fill_duration: bool = True,
    fill_hours: bool = True,
) -> pd.DataFrame:
    """소요시간 및 단측 결측 시각의 역산 복원.

    fill_duration : Route 중앙값 → 전역 중앙값 순으로 Estimated_Duration 결측 채움
                    (run_advanced_features.py:83-92, run_phase7_weather_model.py:64-65)
    fill_hours    : Duration 을 이용해 한쪽만 결측인 Arr_Hour / Dep_Hour 를 역산
                    (run_advanced_features.py:96-113 에만 존재. Phase 7 은 누락 — §3.3)

    ★ 두 플래그를 분리한 이유: 현재 Phase 6 과 Phase 7 은 fill_duration 은 같고
      fill_hours 만 다른데, 이 차이가 Traffic 피처의 의미를 바꿔버린다(§2.3).
      플래그로 분리해야 그 차이가 호출부에 드러난다.

    누수 판정: Delay 미참조 → 타깃 누수 없음. 단 Route 중앙값은 전체 데이터 집계이므로
    엄밀히는 transductive (§2.4). fold 별로 계산하고 싶으면 C 섹션으로 옮길 것.
    """


def build_traffic_features(
    df: pd.DataFrame,
    *,
    date_keys: Sequence[str] = ("Month", "Day_of_Month"),
    origin_hour_col: str = "Dep_Hour",
    dest_hour_col: str = "Arr_Hour",
    exclude_missing_hour: bool = True,
) -> pd.DataFrame:
    """공항×날짜×시간대 운항 편수(혼잡도) 피처 Origin_Traffic / Dest_Traffic.

    exclude_missing_hour=True 이면 hour == -1 인 행을 집계에서 제외하고 해당 행의
    Traffic 을 NaN 으로 둔다.
    ★ 현행 두 스크립트는 이 옵션이 없어서 -1 이 하나의 거대한 버킷으로 집계된다(§2.3).

    누수 판정: 집계 대상이 Route 의 count 이고 Delay 가 어디에도 개입하지 않으므로
    타깃 누수 아님. 운항 스케줄은 예측 시점에 이미 확정된 정보이므로 전체 데이터
    기준 집계라도 배포 시 재현 가능하다 (§2.3 상세).
    """


def build_cyclic_features(
    df: pd.DataFrame,
    *,
    hour_cols: Sequence[str] = ("Dep_Hour",),
    include_cos: bool = True,
    missing: Literal["nan", "noon", "keep"] = "nan",
) -> pd.DataFrame:
    """시각의 24시간 주기 삼각함수 인코딩.

    missing="nan"  : hour<0 → NaN  (baseline / tuned / target_encoded / hybrid / pseudo)
    missing="noon" : hour<0 → 12   (advanced / phase7 — run_advanced_features.py:125)

    ★ 이 한 줄의 차이가 Phase 5→6 비교를 오염시킨다. 결측 시각 행이 "정오 출발"로
      취급되면서 Sin/Cos 분포가 바뀌므로 반드시 명시적 인자로 노출해야 한다.

    include_cos=False 는 run_tuned.py:82 재현용 (Sin 만 생성).
    """


def prune_columns(
    df: pd.DataFrame,
    *,
    preset: Literal["baseline", "pruned", "advanced"] = "pruned",
    extra: Iterable[str] = (),
) -> pd.DataFrame:
    """DROP_PRESETS 기반 컬럼 제거. 존재하지 않는 컬럼은 무시."""


def split_labeled(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """(X_labeled, y_labeled, X_unlabeled) 반환. 인덱스는 모두 reset 된다.

    ★ run_baseline.py:152-157 만 reset_index 를 하지 않는다. 위치 기반 iloc 만 쓰므로
      fold 분할 결과 자체는 동일하지만, 반환 규약을 통일해 둔다.
    """


def encode_categoricals(
    X: pd.DataFrame,
    *,
    cat_cols: Sequence[str] = CAT_COLS,
    fit_frames: Sequence[pd.DataFrame] = (),
    na_token: str = "MISSING",
) -> tuple[pd.DataFrame, dict[str, "LabelEncoder"]]:
    """문자열 범주 → 정수코드 → pandas 'category' dtype.

    fit_frames 가 비어 있으면 X 만으로 vocabulary 를 만든다(6개 스크립트).
    run_pseudo_labeling.py:133-144 만 미라벨 데이터까지 합쳐 fit 하므로
    fit_frames=(X_unlabeled,) 로 재현한다.

    누수 판정: Delay 미참조 → 타깃 누수 없음. 단 vocabulary 크기가 달라지면
    LightGBM 의 범주 분기 후보 수가 달라지므로 Phase 4↔5 비교가 오염된다(§3.2).
    """


# =============================================================================
# C. 타깃 의존 피처 — 반드시 fold 안에서만 호출
# =============================================================================

def smoothed_target_encode(
    fit_keys: pd.Series,
    fit_target: pd.Series,
    apply_keys: pd.Series,
    *,
    m: float = 20.0,
) -> np.ndarray:
    """베이지안 평활 타깃 인코딩. 단일 방향(fit → apply)만 수행한다.

    5벌 중복된 get_smoothed_target_encoding 의 대체. 기존 함수는 (train_enc, test_enc)
    튜플을 반환해 "학습행 자신의 라벨이 자기 인코딩에 들어간다"는 사실을 숨겼다.
    이 함수는 한 번에 한 대상만 인코딩하므로 호출부에서 그 사실이 드러난다.

    smooth = (count*mean + m*global_mean) / (count + m)
    fit_keys 에 없는 범주는 global_mean 으로 대치.
    """


def oof_target_encode(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    outer_train_idx: np.ndarray,
    outer_valid_idx: np.ndarray,
    cols: Sequence[str] = CAT_COLS,
    m: float = 20.0,
    inner_splits: int = 0,
    seed: int = 42,
    prefix: str = "TE_",
    drop_original: bool = False,
    extra_fit_rows: tuple[pd.DataFrame, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """outer fold 하나에 대한 TE 를 수행하고 (X_train_te, X_valid_te) 를 반환한다.

    핵심 계약
    ---------
    * outer_train_idx / outer_valid_idx 를 **필수 키워드 인자**로 요구한다.
      → fold 밖에서 TE 를 계산하는 실수가 시그니처 레벨에서 불가능해진다.
    * inner_splits=0 (현행 5개 스크립트의 동작): 학습행은 자기 자신의 라벨이 포함된
      통계로 인코딩된다. 검증 추정치를 낙관시키지는 않지만 모델이 TE 를 과신하게 만든다(§2.2).
    * inner_splits>=5 (권장): 학습행 인코딩도 내부 K-fold 로 산출한다.
    * drop_original=True  → run_target_encoded.py:191-192 재현 (원본 범주 제거)
      drop_original=False → hybrid / advanced / phase7 재현 (원본 범주 유지)
    * extra_fit_rows: pseudo-label 등 추가 학습행을 TE 통계에 포함시킬 때 사용.
      **인자로 명시하지 않으면 절대 포함되지 않는다** — run_pseudo_labeling.py:257-258 의
      암묵적 오염을 방지하기 위한 설계(§2.1-B).
    """


# =============================================================================
# D. 준지도 — 누수를 구조적으로 막는 시그니처
# =============================================================================

def make_pseudo_labels(
    teacher_predict: Callable[[pd.DataFrame], np.ndarray],
    X_unlabeled: pd.DataFrame,
    *,
    neg_percentile: float = 10.0,
    pos_percentile: float = 98.0,
    percentile_basis: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """고확신 미라벨 행을 선별해 (pseudo_X, pseudo_y) 반환.

    ★ teacher_predict 는 "모델 리스트"가 아니라 "이미 특정 fold 의 학습 데이터만으로
      적합된 예측 함수" 하나를 받는다. 호출부가 fold 루프 안에서 fold 전용 teacher 를
      넘기도록 강제하여 run_pseudo_labeling.py 의 fold-간 누수를 구조적으로 차단한다(§2.1).

    percentile_basis 를 별도로 받는 이유: 분위수 임계값 자체도 전체 라벨 정보가 스며든
    통계이므로, fold 별 teacher 예측 벡터로 계산해야 한다.
    """


# =============================================================================
# E. 평가 프로토콜 — 모듈이 소유한다
# =============================================================================

@dataclass(frozen=True)
class CVConfig:
    """평가 프로토콜 전체를 하나의 값으로 고정. 리포트에 그대로 직렬화한다."""
    n_splits: int = 5
    shuffle: bool = True
    seed: int = 42
    early_stopping_rounds: int = 40
    #: True 이면 early stopping 을 채점 대상 fold 가 아닌 내부 holdout 으로 수행 (§2.5)
    inner_early_stopping: bool = False
    threshold_grid: tuple[float, float, float] = (0.10, 0.70, 0.01)
    #: True 이면 임계값을 K-1 fold 에서 고르고 나머지 fold 에 적용 (§2.6)
    nested_threshold: bool = False
    metric_aggregation: Literal["pooled_oof", "fold_mean"] = "pooled_oof"


def make_folds(y: pd.Series, cfg: CVConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedKFold 분할을 리스트로 materialize 한다.

    동일 cfg + 동일 y 길이/순서라면 7개 스크립트 전부 동일한 분할을 얻는다(확인 완료, §3.1).
    """


def tune_threshold(
    y_true: pd.Series,
    probs: np.ndarray,
    cfg: CVConfig,
    *,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[float, float]:
    """(best_threshold, macro_f1) 반환.

    cfg.nested_threshold=False 는 현행 6개 스크립트 동작(낙관 편향 포함, §2.6).
    True 이면 folds 를 이용해 K-1 fold 에서 임계값을 고르고 남은 fold 에서만 채점한다.
    """


def evaluate_oof(
    y_true: pd.Series,
    probs: np.ndarray,
    cfg: CVConfig,
    *,
    fold_assignments: np.ndarray | None = None,
    label: str = "",
) -> dict:
    """LogLoss / ROC-AUC / F1@0.5 / F1@best / Confusion Matrix 를 dict 로 반환.

    cfg.metric_aggregation="pooled_oof"  : OOF 벡터 전체로 1회 계산 (6개 스크립트)
    cfg.metric_aggregation="fold_mean"   : fold 별 계산 후 평균 (run_baseline.py:238-249)
    ★ Macro-F1 은 두 방식의 값이 서로 다르다. 반드시 같은 방식으로 맞춰야 비교 가능(§3.4).

    반환 dict 에는 cfg 전체와 피처 목록 해시가 포함되어, 실행 결과만 보고도
    어떤 프로토콜로 얻은 수치인지 사후 확인이 가능하다.
    """
```

### 1.4 호출부 재현 대조표

리팩터링이 동작 보존임을 확인하기 위한 매핑이다. 7개 스크립트는 아래 조합으로 모두 재현된다.

| 스크립트 | `impute_cross` | `prune_columns` | `build_cyclic_features` | `restore_time_missing` | `build_traffic_features` | TE |
|---|---|---|---|---|---|---|
| baseline | `bidirectional=True`, 3쌍 | `baseline` | `missing="nan"`, cos O | 미사용 | 미사용 | 없음 |
| tuned | 1쌍 (Carrier→Airline) | `pruned` | `missing="nan"`, **cos X** | 미사용 | 미사용 | **없음** (docstring과 불일치) |
| target_encoded | 1쌍 | `pruned` | `missing="nan"`, cos O | 미사용 | 미사용 | `m=20`, `drop_original=True` |
| hybrid | 1쌍 | `pruned` | `missing="nan"`, cos O | 미사용 | 미사용 | `m=20`, `drop_original=False` |
| pseudo_labeling | 1쌍 | `pruned` | `missing="nan"`, cos O | 미사용 | 미사용 | `m=20`, `drop_original=False` + `extra_fit_rows` |
| advanced_features | 1쌍 | `advanced` | **`missing="noon"`**, cos O | `duration=T, hours=T` | 사용 | `m=25`, `drop_original=False` |
| phase7_weather | 1쌍 | `advanced` | **`missing="noon"`**, cos O | `duration=T, **hours=F**` | 사용 | `m=25`, `drop_original=False` |

---

## 2. 데이터 누수 감사

### 2.1 `run_pseudo_labeling.py` — pseudo-label의 fold 경계 침범

#### (A) 판정: **누수 확정 (HIGH)**

근거가 되는 세 지점을 인용한다.

**① Teacher 앙상블이 전체 라벨을 소비한다** — [run_pseudo_labeling.py:171-201](run_pseudo_labeling.py#L171-L201)

```python
171:    for fold, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled), 1):
172:        X_tr = X_labeled.iloc[train_idx].copy()
...
201:        unlabeled_preds += model.predict_proba(X_unlab_fold)[:, 1] / 5.0
```

`unlabeled_preds`는 5개 fold 모델 예측의 **단순 평균**이다. fold `j`의 teacher는 `train_idx(j)` = 전체 라벨의 4/5로 학습되므로, 임의의 라벨 행 `i`는 정확히 **5개 teacher 중 4개의 학습셋에 포함**된다. 따라서 `unlabeled_preds`의 모든 원소는 255,001개 라벨 **전부**의 함수다.

**② 임계값도 전체 라벨 정보의 함수다** — [run_pseudo_labeling.py:210-214](run_pseudo_labeling.py#L210-L214)

```python
210:    thresh_neg = np.percentile(unlabeled_preds, 10)
211:    thresh_pos = np.percentile(unlabeled_preds, 98)
213:    high_pos_idx = np.where(unlabeled_preds >= thresh_pos)[0]
214:    high_neg_idx = np.where(unlabeled_preds <= thresh_neg)[0]
```

`pseudo_X` / `pseudo_y`의 **선별 자체**가 ①의 벡터에서 나온다. 이 두 객체는 fold 루프 **밖에서 단 한 번** 만들어져 동결된다.

**③ 동결된 pseudo 집합이 모든 fold의 학습셋에 동일하게 주입된다** — [run_pseudo_labeling.py:242-249](run_pseudo_labeling.py#L242-L249)

```python
242:    for fold, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled), 1):
243:        X_tr_orig = X_labeled.iloc[train_idx].copy()
...
248:        X_tr_aug = pd.concat([X_tr_orig, pseudo_X], ignore_index=True)
249:        y_tr_aug = pd.concat([y_tr_orig, pseudo_y], ignore_index=True)
```

**결론**: fold `k`의 student가 학습하는 `y_tr_aug`에는, fold `k`의 **검증 행 라벨이 teacher를 거쳐 압축·재부호화된 정보**가 들어 있다. `final_oof_probs[val_idx]`([run_pseudo_labeling.py:271](run_pseudo_labeling.py#L271))는 따라서 순수한 out-of-fold 추정치가 아니다. **판정: 누수.**

#### (B) 가장 대역폭이 큰 경로는 TE다

[run_pseudo_labeling.py:252-261](run_pseudo_labeling.py#L252-L261)

```python
252:        for col in cat_cols:
253:            X_tr_aug[col] = X_tr_aug[col].astype("category")
254:            X_vl_orig[col] = X_vl_orig[col].astype("category")
256:            # 타깃 인코딩
257:            tr_enc, vl_enc = get_smoothed_target_encoding(
258:                X_tr_aug[col], y_tr_aug, X_vl_orig[col], m=20.0
259:            )
...
261:            X_vl_orig[f"TE_{col}"] = vl_enc
```

검증 행에 붙는 `TE_Tail_Number`, `TE_Route` 값이 `y_tr_aug`(= 진짜 라벨 + pseudo 라벨)로 계산된다. 즉 **오염된 라벨이 검증 행의 피처 값에 직접 반영된다.** 이것이 "teacher 파라미터 → student 트리"라는 간접 경로보다 훨씬 직접적이다.

**참고 — 사용자 질문 ①에 대한 답**: TE 계산 자체는 fold **안**에서 이루어진다. 기계적 위치는 올바르다. 문제는 위치가 아니라 **입력 라벨 집합이 fold를 넘어섰다는 것**이다.

#### (C) 낙관 편향 크기 추정

정량화를 위해 데이터에서 다음 사실을 직접 확인했다.

| 사실 | 값 | 확인 방법 |
|---|---|---|
| 라벨 행 수 | 255,001 (Delayed 45,000 = 17.65%) | `train.csv` 전수 집계 |
| 미라벨 행 수 | 744,999 | 동일 |
| pseudo 양성 (상위 2%) | 약 14,900행 | `744,999 × 0.02` |
| pseudo 음성 (하위 10%) | 약 74,500행 | `744,999 × 0.10` |
| fold별 실제 학습행 | 204,001행 | `255,001 × 0.8` |
| **증강 비율** | **약 30.5%** | `89,400 / 293,401` |
| Tail_Number 고유값 | 6,430 | 전수 집계 |
| Tail_Number당 평균 라벨 행 | 40.4행 (fold당 약 8행) | 동일 |
| 미라벨↔라벨 **완전 중복 항공편** | 20행 (0.003%) | `(Month, Day, Origin, Dest, Tail, Dep_Time)` 키 대조 |
| 미라벨↔라벨 (Month, Day, Tail) 공유 | 13.45% | 동일 |

**증폭 요인**

- pseudo 행이 학습셋의 30%를 차지한다 — 적은 비중이 아니다.
- 누수 경로가 TE라는 **그룹 단위 직접 매핑**이다. Tail_Number 기준 fold당 검증 행이 약 8행(그룹 평균 40행의 20%)이므로, 그룹 통계의 상당 지분이 검증 행에서 유래한 정보로 재구성될 수 있다.

**억제 요인 (중요)**

- **완전 중복 항공편이 사실상 없다(0.003%).** 따라서 행 단위 암기(memorization)는 불가능하고, 누수는 Tail_Number / Route / Month 수준의 **그룹 단위**로만 흐른다.
- **`Day_of_Month`가 제거되어 있다** ([run_pseudo_labeling.py:90](run_pseudo_labeling.py#L90)). 13.45%에 달하는 "같은 날 같은 기체" 쌍은 지연 전파(delay propagation) 때문에 라벨 상관이 매우 높지만, 모델이 특정 날짜를 식별할 수 없으므로 **이 강한 경로는 닫혀 있다.** 이것이 누수 규모를 크게 제한한다.
- 정보가 **이진 임계값**(상위 2% / 하위 10%)을 통과하며 손실된다. 게다가 선별된 행은 정의상 **예측이 쉬운 극단 구간**이라, macro-F1이 실제로 결정되는 애매한 중간 구간에는 거의 기여하지 않는다.
- student의 TE는 204,001개의 **진짜** 라벨도 함께 쓴다. 오염분은 소수 지분이다.

**추정치**

> ROC-AUC 낙관 편향 **+0.002 ~ +0.008**, Macro-F1 낙관 편향 **+0.002 ~ +0.006**
> (중앙 추정: AUC +0.004 내외)

Phase 5가 Phase 4 대비 이 범위를 넘지 않는 폭으로 "개선"되었다면, **그 개선은 전부 누수로 설명될 수 있으며 실질 개선은 0 또는 음수일 가능성이 높다.** 준지도 학습이 오히려 성능을 떨어뜨리는 경우는 흔하다.

**정확한 측정 절차** (위 추정치를 대체할 유일한 방법):
teacher를 fold 루프 **안으로** 옮겨, fold `k`의 teacher를 `train_idx(k)`만으로 학습시키고, 그 teacher의 예측으로 fold `k` 전용 `pseudo_X_k` / `pseudo_y_k`를 만든 뒤 fold `k`의 student에만 주입한다(분위수 임계값도 fold별로 재계산). 이 정직한 버전과 현행 버전의 OOF 지표 차이가 곧 누수 크기다. §1.3의 `make_pseudo_labels(teacher_predict, ...)` 시그니처가 이 구조를 강제한다.

#### (D) 부수적 발견

- [run_pseudo_labeling.py:153](run_pseudo_labeling.py#L153)에서 계산한 `pos_weight`(= 4.667)가 [run_pseudo_labeling.py:263](run_pseudo_labeling.py#L263)의 student에도 그대로 쓰인다. 증강 후 실제 음/양 비율은 약 4.76으로 달라지므로 의도한 클래스 가중이 미세하게 어긋난다. 영향은 작다.
- [run_pseudo_labeling.py:191](run_pseudo_labeling.py#L191) `teacher_models.append((model, X_tr, y_tr))` — 5개 fold의 학습 프레임(각 20만 행)을 리스트에 계속 보관하지만 이후 전혀 사용되지 않는다. 순수한 메모리 낭비다.
- [run_pseudo_labeling.py:133-144](run_pseudo_labeling.py#L133-L144)의 `LabelEncoder`는 6개 다른 스크립트와 달리 미라벨 데이터까지 합쳐 fit한다. 타깃 누수는 아니지만(§2.4), 범주 vocabulary 크기가 달라져 LightGBM의 범주형 분기 후보가 바뀐다 → Phase 4↔5 비교를 오염시키는 요인이 하나 더 있다는 뜻이다.

### 2.2 OOF Target Encoding — fold 안/밖 판정

사용자 질문 ①에 대한 스크립트별 전수 판정이다.

| 스크립트 | TE 존재 | 계산 위치 | 판정 |
|---|---|---|---|
| `run_baseline.py` | 없음 | — | 해당 없음 |
| `run_tuned.py` | **없음** | — | **docstring 오류** (아래 참조) |
| `run_target_encoded.py` | 있음 | fold 루프 [176-189](run_target_encoded.py#L176-L189) 내부 | ✅ 정상 |
| `run_hybrid.py` | 있음 | fold 루프 [173-186](run_hybrid.py#L173-L186) 내부 | ✅ 정상 |
| `run_pseudo_labeling.py` (teacher) | 있음 | fold 루프 [171-182](run_pseudo_labeling.py#L171-L182) 내부 | ✅ 위치는 정상 |
| `run_pseudo_labeling.py` (student) | 있음 | fold 루프 [252-261](run_pseudo_labeling.py#L252-L261) 내부 | ⚠️ 위치는 정상이나 **입력 라벨이 오염**(§2.1-B) |
| `run_advanced_features.py` | 있음 | fold 루프 [217-228](run_advanced_features.py#L217-L228) 내부 | ✅ 정상 |
| `run_phase7_weather_model.py` | 있음 | fold 루프 [149-157](run_phase7_weather_model.py#L149-L157) 내부 | ✅ 정상 |

**`run_tuned.py`의 docstring 오류 (중요)**

[run_tuned.py:1-5](run_tuned.py#L1-L5)는 다음과 같이 선언한다.

```python
1: """항공편 운항 지연(Flight Delay) 예측 2차 개선 파이프라인
2: - OOF Target Encoding (Tail_Number, Route)
```

그러나 **파일 전체에 target encoding 코드가 존재하지 않는다.** `get_smoothed_target_encoding` 함수도, `TE_` 컬럼 생성도 없다(파일 전수 검색으로 확인). 2차 개선의 실제 내용은 (a) 컬럼 pruning, (b) 하이퍼파라미터 확대(`num_leaves` 47→63 등), (c) 임계값 최적화 도입 세 가지뿐이다.

→ **Phase 2의 성능 향상을 "Target Encoding 덕분"으로 서술한 보고서가 있다면 그 서술은 사실과 다르다.** Phase 2와 Phase 3의 실제 차이는 "TE 도입"이 아니라 "TE 도입 + 원본 범주 컬럼 제거"이며, Phase 3([run_target_encoded.py:191-192](run_target_encoded.py#L191-L192))은 원본 고카디널리티 컬럼을 **삭제**하므로 오히려 정보를 잃었을 수 있다. Phase 4(hybrid)가 그것을 되돌린 것이 AUC "반등"([run_hybrid.py:224](run_hybrid.py#L224) 주석)의 설명으로 더 자연스럽다.

**공통 설계 결함 (누수는 아님)**

5개 TE 스크립트 모두 학습 행을 자기 자신의 라벨이 포함된 통계로 인코딩한다. 예: [run_target_encoded.py:121](run_target_encoded.py#L121)

```python
121:    train_encoded = train_s.map(smooth).fillna(global_mean).values
```

`smooth`는 `train_s`/`target` 전체로 만들어졌으므로, 어떤 학습 행의 `TE_Tail_Number`에는 그 행 자신의 라벨이 들어 있다. **이것은 OOF 검증 추정치를 낙관시키지 않는다** — 검증 행의 인코딩은 여전히 학습 라벨만으로 계산되기 때문이다. 그러나 모델이 학습 시점에 TE 피처를 실제보다 신뢰할 수 있게 인식하여 TE에 과도하게 의존하게 만들고, 그 결과 **검증 성능이 낮아지고 피처 중요도가 왜곡된다.** 특히 `Tail_Number`(6,430 범주, 그룹당 40행)에서 `m=20`은 그룹 자신의 라벨에 대한 평활이 약한 편이다. 개선 여지이지 누수 항목은 아니다.

### 2.3 Traffic(혼잡도) 피처 — 타깃 누수 판정

사용자 질문 ③에 대한 답이다. 해당 코드는 두 스크립트에만 있다.

[run_advanced_features.py:117-122](run_advanced_features.py#L117-L122)

```python
117:    df["Origin_Traffic"] = df.groupby(
118:        ["Month", "Day_of_Month", "Origin_Airport", "Dep_Hour"]
119:    )["Route"].transform("count")
120:    df["Dest_Traffic"] = df.groupby(
121:        ["Month", "Day_of_Month", "Destination_Airport", "Arr_Hour"]
122:    )["Route"].transform("count")
```

**판정: 타깃 누수 아님.**

이유는 단순하고 결정적이다. **집계 대상 컬럼이 `Route`이고 집계 함수가 `count`다.** `Delay`는 `groupby` 키에도, 집계 대상에도, 필터 조건에도 등장하지 않는다. 라벨에서 이 피처로 향하는 정보 경로가 코드상 존재하지 않는다. 타깃 인코딩(그룹별 `Delay` 평균)이었다면 fold 밖 계산은 치명적 누수였겠지만, 여기서는 단순 편수 계수다.

**다만 세 가지 단서를 붙인다.**

**(a) Transductive 집계이나 실무적으로 정당하다.** 검증 행의 `Origin_Traffic`은 같은 공항·같은 날·같은 시간대의 **다른 행들(검증 행 포함, 미라벨 행 포함)의 존재**를 이용해 계산된다. 형식적으로는 fold 밖 정보를 쓴 것이다. 그러나 운항 스케줄은 예측 시점(출발 며칠~몇 시간 전)에 **이미 확정·공개된 정보**다. 배포 환경에서도 동일하게 계산할 수 있으므로, 이 transductive 성격이 만들어내는 낙관 편향은 **0으로 본다.** 같은 논리가 [run_advanced_features.py:83-86](run_advanced_features.py#L83-L86)의 `route_median_dur`에도 적용된다.

**(b) `Cancelled` / `Diverted`는 전 데이터에서 상수 0이다.** 확인 결과 100만 행 전부 `Cancelled=0`, `Diverted=0`이다. 따라서 편수 계수에 결항편이 섞이는 문제는 실제로 없다. 동시에, 여러 스크립트가 이 두 컬럼을 `drop_cols`에 넣으며 "누수 및 불필요 컬럼"([run_baseline.py:139](run_baseline.py#L139))이라 주석했지만, 실제로는 **분산이 0인 무정보 컬럼**이다. 제거는 옳지만 이유 설명은 틀렸다.

**(c) 결측 시각 버킷 문제 — 이쪽이 실제 결함이다.** `Dep_Hour`가 `-1`인 행들이 `groupby`에서 **하나의 거대한 버킷**으로 묶여, 해당 공항·해당 날짜의 "시각 미상 항공편 총수"가 `Origin_Traffic`으로 들어간다. 이는 혼잡도가 아니라 **결측 밀도**를 측정하는 피처다. 누수는 아니지만 의미가 오염된 피처이고, `exclude_missing_hour` 옵션으로 분리해야 한다(§1.3).

그리고 이 문제는 Phase 6과 Phase 7에서 **서로 다르게** 나타난다. Phase 6은 [run_advanced_features.py:96-113](run_advanced_features.py#L96-L113)에서 단측 결측 시각을 먼저 역산 복원한 **뒤** traffic을 계산한다. 반면 Phase 7은 [run_phase7_weather_model.py:63-69](run_phase7_weather_model.py#L63-L69)에서 `Estimated_Duration`만 채우고 **`Arr_Hour`/`Dep_Hour` 역산 복원 블록을 통째로 누락한 채** traffic을 계산한다.

→ **`Origin_Traffic`/`Dest_Traffic`은 Phase 6과 Phase 7에서 서로 다른 것을 측정하는 피처다.** Phase 7의 `-1` 버킷이 훨씬 크다. 날씨 피처와 무관한 이 차이가 두 Phase의 비교를 오염시킨다(§3.3).

### 2.4 타깃 비의존 전역 통계 — 전수 확인

리뷰어가 반드시 묻는 항목이므로 명시적으로 판정해 둔다. 아래는 모두 fold 분할 **밖**에서 전체 데이터로 계산되지만, **어느 것도 `Delay`를 참조하지 않는다.**

| 연산 | 위치 | 판정 |
|---|---|---|
| Carrier↔Airline / Airport↔State 매핑 대치 | [run_baseline.py:39-84](run_baseline.py#L39-L84) 외 6개 | 타깃 누수 없음 |
| `LabelEncoder.fit_transform` (전체 라벨 데이터) | [run_hybrid.py:143-146](run_hybrid.py#L143-L146) 외 | 타깃 누수 없음 |
| `LabelEncoder` (라벨+미라벨 합쳐 fit) | [run_pseudo_labeling.py:133-144](run_pseudo_labeling.py#L133-L144) | 타깃 누수 없음 (단 비교 오염, §2.1-D) |
| `route_median_dur` / `global_median_dur` | [run_advanced_features.py:83-92](run_advanced_features.py#L83-L92), [run_phase7_weather_model.py:64-65](run_phase7_weather_model.py#L64-L65) | 타깃 누수 없음 |
| `Origin_Traffic` / `Dest_Traffic` | §2.3 | 타깃 누수 없음 |
| `pos_weight` (`y` 전체 기반) | [run_hybrid.py:154](run_hybrid.py#L154) 외 | 타깃 사용하나 스칼라 1개. 편향 무시 가능 |

### 2.5 Early stopping이 채점 fold를 사용한다 — 7/7 전부

```python
# run_hybrid.py:187-196 (다른 6개도 동일 패턴)
188:            X_train, y_train,
189:            eval_set=[(X_val, y_val)],
192:            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
195:        val_probs = model.predict_proba(X_val)[:, 1]
196:        oof_probs[val_idx] = val_probs
```

트리 개수라는 하이퍼파라미터가 **최종 점수를 매기는 바로 그 데이터**에서 선택된다. **판정: 경미한 낙관 편향 확정.** fold당 검증 행이 51,000개로 크고 선택 대상이 매끄러운 1차원 곡선이므로 크기는 작다 — **ROC-AUC +0.000 ~ +0.002, LogLoss −0.001 ~ −0.003** 수준으로 추정한다.

7개 전부에 동일하게 존재하므로 Phase 간 비교를 크게 왜곡하지는 않는다. 단 `stopping_rounds`가 30 / 35 / 40으로 제각각이라(§3.1) 추출되는 편향의 양이 스크립트마다 미세하게 다르다.

### 2.6 임계값을 채점 대상 OOF 벡터에서 직접 선택 — 6/7

```python
# run_advanced_features.py:246-255
246:    thresholds = np.arange(0.15, 0.50, 0.01)
250:    for th in thresholds:
251:        preds = (oof_probs >= th).astype(int)
252:        score = f1_score(y, preds, average="macro")
254:            best_f1 = score
```

`best_f1`은 동일한 255,001행에서 35~60개 임계값에 대해 계산한 F1의 **최댓값**이다. 이를 그대로 모델 성능으로 보고하면 winner's curse가 포함된다. **판정: 낙관 편향 확정.**

크기 추정: n=255,001, 양성 45,000에서 macro-F1의 표준오차는 약 0.0015~0.002이고, 인접 임계값 간 상관이 매우 높아 유효 독립 시행 수는 3~6회 정도다. 따라서 **+0.001 ~ +0.002** 수준.

절대값은 작지만, **보고되는 Phase 간 개선 폭 자체가 이 크기와 같은 자릿수**라는 점이 문제다. [run_phase7_weather_model.py:196](run_phase7_weather_model.py#L196)이 "과거 최고 0.5746"과 비교하는 방식으로는, 0.002 이하의 차이를 실질 개선으로 해석할 근거가 없다.

### 2.7 Phase 7 전용 — 배포 타당성 및 데이터 결함

CV 누수는 아니지만 결론의 타당성에 직결되므로 기록한다.

**(a) 날씨 피처 절반이 전부 결측이다.** `train_with_weather.csv` 100만 행 전수 확인:

| 컬럼 | 유효 비율 | 고유값 수 |
|---|---|---|
| `Weather_Origin_wspd` (풍속) | 55.77% | — |
| `Weather_Origin_prcp` (강수) | 51.57% | 142 |
| `Weather_Origin_coco` (기상코드) | 53.56% | 18 |
| **`Weather_Origin_snow` (적설)** | **0.00%** | **0** |
| **`Weather_Origin_wpgt` (돌풍)** | **0.00%** | **0** |

[run_phase7_weather_model.py:3](run_phase7_weather_model.py#L3)의 docstring이 명시한 "풍속, **돌풍, 적설**, 강수"에서 **돌풍과 적설은 단 한 건도 수집되지 않았다.** 해당 관측소들이 이 변수를 제공하지 않기 때문이다. [run_phase7_weather_model.py:211](run_phase7_weather_model.py#L211)의 날씨 중요도 출력은 이 전량 결측 컬럼들을 중요도 0으로 표시하게 된다. 마지막 커밋 메시지 "날씨 데이터 결합 실패"와 부합한다.

**(b) 도착지 날씨가 없다.** 병합은 `Origin_Airport` 한쪽만 수행한다([merge_weather_pipeline.py:136-142](merge_weather_pipeline.py#L136-L142)). 컬럼 접두어가 `Weather_Origin_`인 것이 그 증거다. 지연은 도착지 기상에도 강하게 좌우되므로 신호의 절반이 빠져 있다.

**(c) 관측치 vs 예보.** 병합된 값은 출발 예정 시각의 **실측 관측치**다. 현실의 예측 시점(출발 수 시간~수 일 전)에는 관측치가 아니라 예보만 존재한다. CV 누수는 아니지만 **배포 시 성능은 여기서 측정된 것보다 낮다.**

**(d) 연도 가정이 코드상 미검증이다.** [merge_weather_pipeline.py:20](merge_weather_pipeline.py#L20) `TARGET_YEAR = 2022`는 "확정!"이라 주석되어 있으나, 근거를 만드는 [find_flight_year.py](find_flight_year.py)는 구글 검색 링크를 출력하는 **수동 확인 보조 도구**일 뿐 연도를 프로그램적으로 검증하지 않는다. 연도가 틀리면 월·일만 맞는 엉뚱한 날씨가 붙어 피처가 사실상 노이즈가 된다. 이 가정을 검증하는 코드가 저장소 어디에도 없다.

**(e) 참고 — 이상 없음으로 확인된 항목.** `train_with_weather.csv`는 `utf-8-sig`(BOM 포함)로 저장되어 있으나 pandas가 BOM을 자동 제거하므로 첫 컬럼은 `ID`로 정상 인식되고 `drop_cols`도 정상 동작한다. 또한 조인 후에도 행 수가 정확히 1,000,000으로 유지되어 조인 키 중복에 의한 행 증식은 없다.

---

## 3. 평가 프로토콜 대조표

### 3.1 프로토콜 상수

| 항목 | baseline | tuned | target_encoded | hybrid | pseudo_labeling | advanced | phase7 |
|---|---|---|---|---|---|---|---|
| 데이터 파일 | train.csv | train.csv | train.csv | train.csv | train.csv | train.csv | **train_with_weather.csv** |
| `n_splits` | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| `shuffle` | True | True | True | True | True | True | True |
| KFold `random_state` | 42 | 42 | 42 | 42 | 42 | 42 | 42 |
| LGBM `random_state` | 42 | 42 | 42 | 42 | 42 | 42 | 42 |
| `learning_rate` | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | **0.04** | **0.04** |
| `num_leaves` | **47** | 63 | 63 | 63 | 63 | **79** | **79** |
| `max_depth` | **7** | 8 | 8 | 8 | 8 | **9** | **9** |
| `min_child_samples` | 기본 | 기본 | 기본 | 기본 | 기본 | **40** | **40** |
| `colsample_bytree` | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | **0.75** | **0.75** |
| `n_estimators` | 800 | 1000 | 1000 | 1000 | **800** | **1200** | **1200** |
| `stopping_rounds` | 40 | 40 | 40 | 40 | **30** | **35** | **35** |
| TE 평활 `m` | — | — | 20.0 | 20.0 | 20.0 | **25.0** | **25.0** |
| **임계값 그리드** | **없음 (0.5 고정)** | **0.10~0.69 (60점)** | **0.10~0.69 (60점)** | **0.15~0.49 (35점)** | **0.15~0.49 (35점)** | **0.15~0.49 (35점)** | **0.15~0.49 (35점)** |
| **지표 집계** | **fold 평균** | pooled OOF | pooled OOF | pooled OOF | pooled OOF | pooled OOF | pooled OOF |

**일치하는 것은 `n_splits=5`, `shuffle=True`, `random_state=42` 세 가지뿐이다.** 나머지는 전부 스크립트마다 다르다.

**긍정적 확인 하나**: 모든 스크립트가 동일한 255,001행을 동일한 순서로 필터링하고 동일한 `StratifiedKFold(5, shuffle=True, random_state=42)`를 적용하므로, **7개 스크립트의 fold 분할은 실제로 동일하다.** Phase 7도 `train_with_weather.csv`가 좌측 조인으로 행 수·순서를 보존했으므로(100만 행 유지 확인) 동일하다. 즉 fold 분할 자체는 비교 가능한 유일한 축이다.

### 3.2 피처셋 차이

| 피처/처리 | baseline | tuned | target_enc | hybrid | pseudo | advanced | phase7 |
|---|---|---|---|---|---|---|---|
| `Origin_State`/`Destination_State` | **유지** | 제거 | 제거 | 제거 | 제거 | 제거 | 제거 |
| `Carrier_Code(IATA)` | **유지** | 제거 | 제거 | 제거 | 제거 | 제거 | 제거 |
| `Day_of_Month` | **유지** | 제거 | 제거 | 제거 | 제거 | 제거(집계 후) | 제거(집계 후) |
| `Dep_Minute`/`Arr_Minute` | **유지** | 제거 | 제거 | 제거 | 제거 | 제거 | 제거 |
| `Cos_Dep_Hour` | 유지 | **제거** | 유지 | 유지 | 유지 | 유지 | 유지 |
| Airport↔State 상호 대치 | **수행** | 미수행 | 미수행 | 미수행 | 미수행 | 미수행 | 미수행 |
| Carrier↔Airline 양방향 | **양방향** | 단방향 | 단방향 | 단방향 | 단방향 | 단방향 | 단방향 |
| 결측 시각 → Sin/Cos | NaN | NaN | NaN | NaN | NaN | **12(정오)** | **12(정오)** |
| Duration 결측 역산 | 미수행 | 미수행 | 미수행 | 미수행 | 미수행 | **수행** | **수행** |
| 단측 결측 시각 역산 | 미수행 | 미수행 | 미수행 | 미수행 | 미수행 | **수행** | **미수행** |
| Traffic 혼잡도 | 미사용 | 미사용 | 미사용 | 미사용 | 미사용 | **사용** | **사용** |
| 원본 범주 컬럼 | 유지 | 유지 | **제거** | 유지 | 유지 | 유지 | 유지 |
| TE 컬럼 | 없음 | **없음** | 5개 | 5개 | 5개 | 5개 | 5개 |
| LabelEncoder fit 범위 | 라벨만 | 라벨만 | 라벨만 | 라벨만 | **라벨+미라벨** | 라벨만 | 라벨만 |
| 날씨 피처 | — | — | — | — | — | — | **10개 (2개는 전량 결측)** |

### 3.3 Phase 간 비교 가능성 판정

| 비교 | 동시에 변한 요인 | 판정 |
|---|---|---|
| Phase 1 → 2 (baseline → tuned) | 피처 8종 제거 + `Cos_Dep_Hour` 제거 + 상호대치 축소 + 하이퍼파라미터 3종 + **지표 집계 방식(fold평균 → pooled)** + 임계값 탐색 도입 | **무효.** 특히 지표 집계 방식이 달라 Macro-F1은 애초에 같은 양이 아니다 |
| Phase 2 → 3 (tuned → target_encoded) | TE 도입 + 원본 범주 컬럼 **제거** + `Cos_Dep_Hour` 복원 | **부분 무효.** "TE 효과"와 "원본 범주 상실 효과"가 뒤섞여 있다 |
| Phase 3 → 4 (target_encoded → hybrid) | 원본 범주 컬럼 복원 + **임계값 그리드 60점 → 35점** | **조건부 유효.** 단일 피처 변경이라 해석 가능하나, 그리드가 달라 F1 비교는 불공정 |
| Phase 4 → 5 (hybrid → pseudo_labeling) | pseudo-label 증강(**누수 포함**) + `n_estimators` 1000→800 + `stopping_rounds` 40→30 + LabelEncoder fit 범위 변경 | **무효.** 누수(§2.1) + 하이퍼파라미터 3종 동시 변경 |
| Phase 5 → 6 (pseudo → advanced) | Traffic 피처 + Duration/시각 역산 + Sin/Cos 결측처리 변경 + **하이퍼파라미터 6종 전면 변경** + TE `m` 20→25 + pseudo-label 제거 | **무효.** 변경 요인이 10개 이상 |
| Phase 6 → 7 (advanced → phase7) | 날씨 10개 + `Weather_Available` + **단측 시각 역산 누락**(→ Traffic 피처 의미 변화, §2.3c) | **부분 무효.** 하이퍼파라미터는 완전히 동일하므로 가장 깨끗한 비교이지만, 시각 역산 누락이라는 날씨와 무관한 차이가 섞여 있다 |

**종합 판정: 현재 저장소 상태에서 Phase 1~7의 성능 수치를 순차 비교하여 "개선 추이"로 제시하는 것은 통계적으로 무효다.** 각 단계에서 여러 요인이 동시에 변경되었으므로, 관측된 지표 변화를 특정 기법(Target Encoding, Pseudo-Labeling, 혼잡도 피처, 날씨 결합)에 귀속시킬 수 없다.

### 3.4 임계값 그리드 불일치의 구체적 영향

세 그룹으로 갈린다.

- **광역 그리드** `np.arange(0.10, 0.70, 0.01)` — tuned, target_encoded (60점)
- **협역 그리드** `np.arange(0.15, 0.50, 0.01)` — hybrid, pseudo, advanced, phase7 (35점)
- **탐색 없음** — baseline (0.5 고정)

두 가지 문제가 동시에 발생한다.

1. **협역 그리드는 0.50 이상을 탐색하지 않는다.** `scale_pos_weight ≈ 4.67`이 적용되어 예측 확률이 위로 밀려 있으므로 최적 임계값은 0.5 미만일 가능성이 높지만, 이는 **가정이지 보장이 아니다.** `best_thresh` 초기값이 0.5로 설정되어 있어([run_hybrid.py:207](run_hybrid.py#L207)) 그리드 밖의 값이 반환될 수도 없다. 최적점이 경계에 붙어 있으면 협역 그리드 스크립트들은 과소 보고된다.
2. **광역 그리드는 winner's curse를 더 많이 얻는다.** 60점 탐색이 35점 탐색보다 낙관 편향이 크다(§2.6). 즉 Phase 2·3은 Phase 4~7보다 **구조적으로 유리한 조건에서 채점되었다.**

두 효과는 방향이 반대이므로 상쇄될 수도, 누적될 수도 있다. 어느 쪽인지 알 수 없다는 것이 곧 **F1 기준 Phase 비교가 무효라는 뜻이다.**

`run_baseline.py`는 임계값 탐색을 아예 하지 않고 0.5 고정 F1을 fold 평균으로 보고한다([run_baseline.py:232-248](run_baseline.py#L232-L248)). 이 값은 다른 6개의 `best_f1`과 **다른 종류의 양**이므로 같은 표에 나란히 놓아서는 안 된다.

---

## 4. 권고 조치 (우선순위 순)

1. **[필수] Phase 5 재실행** — teacher를 fold 루프 안으로 이동하여 누수를 제거하고, 정직한 수치를 현행 수치와 병기한다. 차이가 곧 §2.1-C 추정치의 실측값이다.
2. **[필수] `run_tuned.py` docstring 수정** — 존재하지 않는 Target Encoding 서술을 제거한다. 보고서에 같은 서술이 있다면 함께 수정한다.
3. **[필수] 프로토콜 고정 후 전 Phase 재실행** — `src/features.py` + 단일 `CVConfig`로 통일하고, 하이퍼파라미터는 한 세트로 고정한 채 **피처셋만** Phase별로 바꾼다. 이것이 비교 가능성을 회복하는 유일한 방법이다.
4. **[중요] 임계값 선택을 nested로 전환** — K-1 fold에서 임계값을 고르고 남은 fold에서 채점한다(`CVConfig.nested_threshold=True`).
5. **[중요] Phase 7 날씨 데이터 재수집** — `snow`/`wpgt` 전량 결측을 해소하거나 해당 변수 포기를 명시하고, 도착지 날씨를 추가하며, `TARGET_YEAR=2022` 가정을 프로그램적으로 검증한다.
6. **[중요] Phase 7에 단측 시각 역산 복원 추가** — Phase 6과 Traffic 피처 정의를 일치시켜야 날씨 효과를 분리 측정할 수 있다.
7. **[권장] Early stopping을 내부 holdout으로 이전** — 학습 fold를 다시 쪼개 early stopping 전용 셋을 만든다.
8. **[권장] TE에 inner K-fold 적용** — 학습 행이 자기 라벨로 인코딩되는 문제를 해소한다(`oof_target_encode(inner_splits=5)`).
9. **[권장] Traffic 피처의 `-1` 버킷 분리** — `exclude_missing_hour=True`.
10. **[정리] `Cancelled`/`Diverted` 주석 수정** — "누수 컬럼"이 아니라 전 데이터 상수 0인 무정보 컬럼이다.

---

## 부록 A. 감사 중 데이터에서 직접 확인한 사실

코드 판독만으로는 확정할 수 없어 실제로 측정한 항목이다.

| 확인 항목 | 결과 |
|---|---|
| `train.csv` 행 수 | 1,000,000 |
| 라벨 분포 | Not_Delayed 210,001 / Delayed 45,000 / 미라벨 744,999 |
| 양성 비율 | 17.65% (→ `scale_pos_weight` ≈ 4.667) |
| `Cancelled` / `Diverted` 분포 | 전 행 0 (분산 0) |
| `Tail_Number` 고유값 / 결측 | 6,430 / 0 |
| `Airline` 결측 (원본 → 대치 후) | 108,920 → 11,864 |
| `Origin_Airport` / `Destination_Airport` 고유값 | 374 / 375 |
| 미라벨↔라벨 완전 중복 항공편 | 20행 (0.003%) |
| 미라벨↔라벨 (Month, Day, Tail) 공유 | 13.45% |
| `train_with_weather.csv` 행 수 / 열 수 | 1,000,000 / 29 (행 증식 없음) |
| 날씨 결합률 (전체 / 라벨 행) | 55.77% / 55.17% |
| `Weather_Origin_snow` / `Weather_Origin_wpgt` | **양쪽 모두 100% 결측** |
| BOM(`utf-8-sig`) 처리 | pandas가 자동 제거 — `ID` 컬럼 정상 인식, 드롭 정상 동작 |

## 부록 B. 이 감사에서 수행하지 않은 것

- 모델 재학습을 전혀 수행하지 않았다. 따라서 §2.1-C·§2.5·§2.6의 낙관 편향 수치는 **코드 구조와 데이터 통계에 근거한 추정치**이며 실측값이 아니다. 각 항목에 측정 절차를 함께 기재했다.
- `README.md` 및 보고서 템플릿에 기록된 실제 실행 결과 수치와의 대조는 범위에 포함하지 않았다.
- 코드는 수정하지 않았다.
