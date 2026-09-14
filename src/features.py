"""항공편 지연 예측 — 공통 전처리 / 피처 엔지니어링 모듈.

`AUDIT.md` §1 의 시그니처 설계안 구현. 7개 `run_*.py` 스크립트에 복사되어 있던
전처리·피처엔지니어링 로직을 단일 모듈로 통합한다.

설계 원칙 (AUDIT.md §1.2)
-------------------------
1. fold 를 넘나드는 모든 연산은 fold 인덱스를 **명시적 인자로** 받는다.
2. 피처 생성을 target-free (섹션 B, fold 밖 1회) / target-dependent (섹션 C·D,
   fold 안 K회) 로 물리적으로 분리한다.
3. 평가 프로토콜 상수는 스크립트가 아니라 모듈(`src/cv.py`)이 소유한다.

이 모듈의 존재 이유는 "중복 제거"가 아니라 **누수의 구조적 차단**이다.
섹션 C·D 의 함수는 fold 인덱스나 fold 전용 콜러블 없이는 호출 자체가 불가능하다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Literal, NamedTuple, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

__all__ = [
    "TARGET_COL",
    "TARGET_MAP",
    "CAT_COLS",
    "DROP_PRESETS",
    "load_data",
    "impute_cross",
    "parse_hhmm",
    "build_time_features",
    "restore_time_missing",
    "build_traffic_features",
    "build_cyclic_features",
    "prune_columns",
    "split_labeled",
    "encode_categoricals",
    "smoothed_target_encode",
    "oof_target_encode",
    "TEResult",
    "make_pseudo_labels",
    "PseudoLabels",
    "fit_fold_teacher",
]

# =============================================================================
# 상수 — 7개 스크립트에 흩어져 있던 매직 리터럴을 한 곳으로
# =============================================================================

TARGET_COL = "Delay"
TARGET_MAP: dict[str, int] = {"Not_Delayed": 0, "Delayed": 1}

#: TE / category 인코딩 대상 (AUDIT.md §1.3).
CAT_COLS: tuple[str, ...] = (
    "Tail_Number",
    "Route",
    "Origin_Airport",
    "Destination_Airport",
    "Airline",
)

#: 제거 컬럼 프리셋. 스크립트별 drop_cols 차이를 이름으로 고정한다 (AUDIT.md §3.2).
#: Cancelled / Diverted 는 "누수 컬럼"이 아니라 전 행 0 인 상수 컬럼이다 (AUDIT.md §2.3-b).
DROP_PRESETS: dict[str, tuple[str, ...]] = {
    # run_baseline.py:140-147
    "baseline": (
        "ID",
        "Cancelled",
        "Diverted",
        "Origin_Airport_ID",
        "Destination_Airport_ID",
        "Carrier_ID(DOT)",
    ),
    # run_tuned / run_target_encoded / run_hybrid / run_pseudo_labeling
    "pruned": (
        "ID",
        "Cancelled",
        "Diverted",
        "Origin_Airport_ID",
        "Destination_Airport_ID",
        "Carrier_ID(DOT)",
        "Carrier_Code(IATA)",
        "Origin_State",
        "Destination_State",
        "Day_of_Month",
    ),
    # run_advanced_features / run_phase7_weather_model
    "advanced": (
        "ID",
        "Cancelled",
        "Diverted",
        "Origin_Airport_ID",
        "Destination_Airport_ID",
        "Carrier_ID(DOT)",
        "Carrier_Code(IATA)",
        "Origin_State",
        "Destination_State",
        "Day_of_Month",
        "Dep_Minute",
        "Arr_Minute",
    ),
}

#: 시각 결측 센티널. 7개 스크립트가 공유하던 -1 규약.
MISSING_HOUR = -1


# =============================================================================
# A. 로딩 — 타깃 비의존
# =============================================================================


def load_data(
    path: str | Path,
    *,
    with_weather: bool = False,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """원본 CSV 를 로드한다.

    Parameters
    ----------
    with_weather
        True 이면 `train_with_weather.csv` 스키마를 기대하고, 병합 시점에 계산되어
        들어 있는 `Dep_Hour` 열을 **즉시 제거**한다. 이 열을 남겨 두면
        `build_time_features()` 가 조용히 덮어써서 병합 시점 정의와 학습 시점 정의가
        어긋나도 추적할 수 없다 (run_phase7_weather_model.py:54 의 문제).

    Notes
    -----
    `train_with_weather.csv` 는 utf-8-sig(BOM) 로 저장되어 있으나 pandas 가 BOM 을
    자동 제거하므로 첫 컬럼은 `ID` 로 정상 인식된다 (AUDIT.md §2.7-e 확인 완료).
    """
    df = pd.read_csv(Path(path), usecols=list(usecols) if usecols else None)
    if with_weather and "Dep_Hour" in df.columns:
        df = df.drop(columns=["Dep_Hour"])
    return df


# =============================================================================
# B. 타깃 비의존 전처리 — fold 밖에서 1회만 호출 (전체 df 대상, 누수 없음)
#
# 이 섹션의 함수는 `Delay` 를 일절 참조하지 않는다. 따라서 fold 분할 전에
# 전체 데이터로 계산해도 타깃 누수가 발생하지 않는다 (AUDIT.md §2.4 전수 판정).
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

    `pairs` 의 각 `(key, value)` 에 대해 key -> value 방향으로 채운다.
    `bidirectional=True` 이면 value -> key 방향도 수행한다
    (run_baseline.py:46-58 만 Carrier<->Airline 양방향을 수행했다).

    누수 판정: `Delay` 미참조 → 타깃 누수 없음. "IATA 코드와 항공사명의 대응"은
    데이터에 독립적인 상수이므로 transductive 이슈도 없다.
    """
    out = df.copy()
    for key, value in pairs:
        if key not in out.columns or value not in out.columns:
            continue
        both = out.dropna(subset=[key, value])

        fwd = both.drop_duplicates(subset=[key]).set_index(key)[value].to_dict()
        out[value] = out[value].fillna(out[key].map(fwd))

        if bidirectional:
            bwd = both.drop_duplicates(subset=[value]).set_index(value)[key].to_dict()
            out[key] = out[key].fillna(out[value].map(bwd))
    return out


def parse_hhmm(s: pd.Series, *, strict: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """HHMM 정수/실수/결측 → `(hour, minute)`. 결측·파싱실패는 `-1` 센티널.

    7개 스크립트에 7벌 존재하던 동일 로직의 단일 구현.

    Parameters
    ----------
    strict
        False (기본, 원본 동작 보존): `2470` 같은 비현실적 값도 hour=24, minute=70 으로
        그대로 통과시킨다.
        True: hour>23 또는 minute>59 인 행을 `-1` 로 무효화한다. 원본 스크립트에는
        없던 동작이므로 재현 실험 시에는 반드시 False 를 유지할 것.
    """
    val = pd.to_numeric(s, errors="coerce").fillna(-1).to_numpy(dtype=float)
    ok = val >= 0
    hour = np.where(ok, (val // 100), MISSING_HOUR).astype(int)
    minute = np.where(ok, (val % 100), MISSING_HOUR).astype(int)
    if strict:
        bad = (hour > 23) | (minute > 59)
        hour = np.where(bad, MISSING_HOUR, hour)
        minute = np.where(bad, MISSING_HOUR, minute)
    return hour, minute


def build_time_features(
    df: pd.DataFrame,
    *,
    dep_col: str = "Estimated_Departure_Time",
    arr_col: str = "Estimated_Arrival_Time",
    keep_minute: bool = False,
    midnight_wrap: bool = True,
    strict: bool = False,
) -> pd.DataFrame:
    """`Dep_Hour` / `Arr_Hour` / (선택) 분 / `Estimated_Duration` 생성.

    `midnight_wrap=True` 이면 도착<출발 인 경우 +1440분 보정한다.
    양쪽 시각이 **모두 유효할 때만** `Estimated_Duration` 을 채우고, 아니면 NaN.
    """
    out = df.copy()
    dep_h, dep_m = parse_hhmm(out[dep_col], strict=strict)
    arr_h, arr_m = parse_hhmm(out[arr_col], strict=strict)

    out["Dep_Hour"] = dep_h
    out["Arr_Hour"] = arr_h
    # 분 단위는 restore_time_missing(fill_hours=True) 의 역산에 필요하다.
    out["Dep_Minute"] = dep_m
    out["Arr_Minute"] = arr_m

    valid = (dep_h >= 0) & (arr_h >= 0)
    dep_total = dep_h * 60 + dep_m
    arr_total = arr_h * 60 + arr_m
    duration = arr_total - dep_total
    if midnight_wrap:
        duration = np.where(arr_total < dep_total, duration + 1440, duration)
    out["Estimated_Duration"] = np.where(valid, duration, np.nan)

    if not keep_minute:
        # 역산이 끝난 뒤 prune_columns(preset="advanced") 가 제거하지만,
        # 호출부가 분 단위를 원하지 않으면 여기서 바로 떨군다.
        out = out.drop(columns=["Dep_Minute", "Arr_Minute"])
    return out


def restore_time_missing(
    df: pd.DataFrame,
    *,
    route_col: str = "Route",
    fill_duration: bool = True,
    fill_hours: bool = True,
) -> pd.DataFrame:
    """소요시간 및 단측 결측 시각의 역산 복원.

    Parameters
    ----------
    fill_duration
        Route 중앙값 → 전역 중앙값 순으로 `Estimated_Duration` 결측을 채운다
        (run_advanced_features.py:83-92, run_phase7_weather_model.py:64-65).
    fill_hours
        Duration 을 이용해 한쪽만 결측인 `Arr_Hour` / `Dep_Hour` 를 역산한다
        (run_advanced_features.py:96-113 **에만** 존재. Phase 7-Ex 는 누락).

    두 플래그를 분리한 이유
    -----------------------
    Phase 6 과 Phase 7-Ex 는 `fill_duration` 은 같고 `fill_hours` 만 다른데,
    이 차이가 `build_traffic_features()` 의 `-1` 버킷 크기를 바꿔 Traffic 피처의
    **의미 자체를 다르게 만든다** (AUDIT.md §2.3-c). 플래그로 분리해야 호출부에
    그 차이가 드러난다.

    누수 판정: `Delay` 미참조 → 타깃 누수 없음. Route 중앙값은 전체 데이터 집계라
    엄밀히는 transductive 이나, 운항 스케줄은 예측 시점에 확정된 정보이므로
    배포 시 재현 가능하다.
    """
    out = df.copy()
    if "Estimated_Duration" not in out.columns:
        raise KeyError(
            "Estimated_Duration 이 없습니다. build_time_features() 를 먼저 호출하세요."
        )

    if fill_duration:
        if route_col in out.columns:
            route_median = out.groupby(route_col, observed=True)[
                "Estimated_Duration"
            ].transform("median")
            out["Estimated_Duration"] = out["Estimated_Duration"].fillna(route_median)
        global_median = out["Estimated_Duration"].median()
        out["Estimated_Duration"] = out["Estimated_Duration"].fillna(global_median)

    if fill_hours:
        for col in ("Dep_Minute", "Arr_Minute"):
            if col not in out.columns:
                raise KeyError(
                    f"{col} 이 없습니다. fill_hours=True 에는 분 단위가 필요합니다. "
                    "build_time_features(keep_minute=True) 로 호출하세요."
                )
        dep_h = out["Dep_Hour"].to_numpy()
        arr_h = out["Arr_Hour"].to_numpy()
        # 두 마스크는 상호 배타적이다 (한쪽은 Dep>=0, 다른 쪽은 Dep==-1).
        # 원본과 달리 갱신 전에 한꺼번에 계산해 순서 의존성을 없앤다.
        arr_missing = (arr_h == MISSING_HOUR) & (dep_h >= 0)
        dep_missing = (dep_h == MISSING_HOUR) & (arr_h >= 0)

        if arr_missing.any():
            calc = (
                out.loc[arr_missing, "Dep_Hour"] * 60
                + out.loc[arr_missing, "Dep_Minute"]
                + out.loc[arr_missing, "Estimated_Duration"]
            ) % 1440
            out.loc[arr_missing, "Arr_Hour"] = (calc // 60).astype(int)
            out.loc[arr_missing, "Arr_Minute"] = (calc % 60).astype(int)

        if dep_missing.any():
            calc = (
                out.loc[dep_missing, "Arr_Hour"] * 60
                + out.loc[dep_missing, "Arr_Minute"]
                - out.loc[dep_missing, "Estimated_Duration"]
            ) % 1440
            out.loc[dep_missing, "Dep_Hour"] = (calc // 60).astype(int)
            out.loc[dep_missing, "Dep_Minute"] = (calc % 60).astype(int)

    return out


def build_traffic_features(
    df: pd.DataFrame,
    *,
    date_keys: Sequence[str] = ("Month", "Day_of_Month"),
    origin_hour_col: str = "Dep_Hour",
    dest_hour_col: str = "Arr_Hour",
    origin_airport_col: str = "Origin_Airport",
    dest_airport_col: str = "Destination_Airport",
    exclude_missing_hour: bool = True,
    add_missing_flag: bool = True,
) -> pd.DataFrame:
    """공항×날짜×시간대 운항 편수(혼잡도) `Origin_Traffic` / `Dest_Traffic`.

    `exclude_missing_hour=True` (기본, 권장)
        `hour == -1` 인 행을 **집계에서 제외**하고 해당 행의 Traffic 을 NaN 으로 둔다.
        추가로 `Origin_Hour_Missing` / `Dest_Hour_Missing` 플래그 컬럼을 만들어
        "시각 결측"이라는 정보를 Traffic 값과 분리해 보존한다.

    `exclude_missing_hour=False`
        run_advanced_features.py / run_phase7_weather_model.py 의 현행 동작 재현용.
        `-1` 이 하나의 거대한 버킷으로 묶여 혼잡도가 아니라 **결측 밀도**를 측정하게
        된다 (AUDIT.md §2.3-c). 재현 실험 외에는 사용하지 말 것.

    누수 판정
    ---------
    집계 대상이 편수(count)이고 `Delay` 가 groupby 키·집계 대상·필터 어디에도
    등장하지 않으므로 **타깃 누수가 아니다**. 전체 데이터 기준 집계이나 운항
    스케줄은 예측 시점에 이미 확정·공개된 정보이므로 배포 시 재현 가능하다
    (AUDIT.md §2.3-a).
    """
    if not df.index.is_unique:
        raise ValueError(
            "인덱스가 고유하지 않습니다. 집계 결과 정렬이 어긋납니다. "
            "df = df.reset_index(drop=True) 후 호출하세요."
        )
    out = df.copy()
    date_keys = list(date_keys)

    specs = (
        ("Origin_Traffic", "Origin_Hour_Missing", origin_airport_col, origin_hour_col),
        ("Dest_Traffic", "Dest_Hour_Missing", dest_airport_col, dest_hour_col),
    )

    for name, flag_name, airport_col, hour_col in specs:
        missing_cols = [c for c in (*date_keys, airport_col, hour_col) if c not in out.columns]
        if missing_cols:
            raise KeyError(f"{name} 생성에 필요한 컬럼이 없습니다: {missing_cols}")

        keys = [*date_keys, airport_col, hour_col]
        is_missing = out[hour_col] == MISSING_HOUR

        if add_missing_flag:
            out[flag_name] = is_missing.astype("int8")

        if exclude_missing_hour:
            valid = out.loc[~is_missing]
            counts = valid.groupby(keys, observed=True, dropna=False).transform("size")
            # transform 은 원본 컬럼 수만큼 동일 값을 돌려주므로 첫 열만 취한다.
            counts = counts.iloc[:, 0] if isinstance(counts, pd.DataFrame) else counts
            # 인덱스 정렬로 결측 시각 행에는 자동으로 NaN 이 들어간다.
            out[name] = counts.astype("float64")
        else:
            counts = out.groupby(keys, observed=True, dropna=False).transform("size")
            counts = counts.iloc[:, 0] if isinstance(counts, pd.DataFrame) else counts
            out[name] = counts.astype("float64")

    return out


def build_cyclic_features(
    df: pd.DataFrame,
    *,
    hour_cols: Sequence[str] = ("Dep_Hour",),
    include_cos: bool = True,
    missing: Literal["nan", "noon", "keep"] = "nan",
) -> pd.DataFrame:
    """시각의 24시간 주기 삼각함수 인코딩.

    `missing="nan"`  : hour<0 → NaN  (baseline / tuned / target_encoded / hybrid / pseudo)
    `missing="noon"` : hour<0 → 12   (advanced / phase7-Ex — run_advanced_features.py:125)
    `missing="keep"` : 원값(-1) 그대로 사용

    이 한 줄의 차이가 Phase 5→6 비교를 오염시킨다. 결측 시각 행이 "정오 출발"로
    취급되면서 Sin/Cos 분포가 바뀌므로 반드시 명시적 인자로 노출한다.

    `include_cos=False` 는 run_tuned.py:82 재현용 (Sin 만 생성).
    """
    if missing not in ("nan", "noon", "keep"):
        raise ValueError(f"missing 은 'nan'|'noon'|'keep' 중 하나여야 합니다: {missing!r}")

    out = df.copy()
    for col in hour_cols:
        if col not in out.columns:
            raise KeyError(f"{col} 이 없습니다. build_time_features() 를 먼저 호출하세요.")
        hours = out[col].to_numpy(dtype=float)
        if missing == "nan":
            hours = np.where(hours >= 0, hours, np.nan)
        elif missing == "noon":
            hours = np.where(hours >= 0, hours, 12.0)

        stem = col.removesuffix("_Hour")
        angle = 2.0 * np.pi * hours / 24.0
        out[f"Sin_{stem}_Hour"] = np.sin(angle)
        if include_cos:
            out[f"Cos_{stem}_Hour"] = np.cos(angle)
    return out


def prune_columns(
    df: pd.DataFrame,
    *,
    preset: Literal["baseline", "pruned", "advanced"] = "pruned",
    extra: Iterable[str] = (),
) -> pd.DataFrame:
    """`DROP_PRESETS` 기반 컬럼 제거. 존재하지 않는 컬럼은 무시한다."""
    if preset not in DROP_PRESETS:
        raise ValueError(
            f"알 수 없는 preset: {preset!r}. 사용 가능: {sorted(DROP_PRESETS)}"
        )
    targets = set(DROP_PRESETS[preset]) | set(extra)
    return df.drop(columns=[c for c in df.columns if c in targets])


def split_labeled(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    target_map: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """`(X_labeled, y_labeled, X_unlabeled)` 반환. 인덱스는 모두 reset 된다.

    라벨 판정 기준은 7개 스크립트 공통 규약을 따른다 —
    `notna()` 이면서 공백만으로 이루어지지 않은 행.
    """
    mapping = TARGET_MAP if target_map is None else target_map
    raw = df[target_col]
    labeled_mask = raw.notna() & (raw.astype("string").str.strip() != "")

    labeled = df[labeled_mask].copy().reset_index(drop=True)
    unlabeled = df[~labeled_mask].copy().reset_index(drop=True)

    y = labeled[target_col].map(mapping)
    if y.isna().any():
        bad = sorted(set(labeled.loc[y.isna(), target_col].astype(str)))[:5]
        raise ValueError(f"target_map 에 없는 라벨 값이 있습니다: {bad}")

    X_labeled = labeled.drop(columns=[target_col])
    X_unlabeled = unlabeled.drop(columns=[target_col])
    return X_labeled, y.astype(int), X_unlabeled


def encode_categoricals(
    X: pd.DataFrame,
    *,
    cat_cols: Sequence[str] = CAT_COLS,
    fit_frames: Sequence[pd.DataFrame] = (),
    na_token: str = "MISSING",
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """문자열 범주 → 정수코드 → pandas `category` dtype.

    `fit_frames` 가 비어 있으면 `X` 만으로 vocabulary 를 만든다 (6개 스크립트).
    run_pseudo_labeling.py:133-144 만 미라벨 데이터까지 합쳐 fit 하므로
    `fit_frames=(X_unlabeled,)` 로 재현한다.

    누수 판정: `Delay` 미참조 → 타깃 누수 없음. 단 vocabulary 크기가 달라지면
    LightGBM 의 범주 분기 후보 수가 달라져 Phase 4↔5 비교가 오염된다 (AUDIT.md §2.1-D).
    """
    out = X.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in cat_cols:
        if col not in out.columns:
            continue
        parts = [_as_token_series(out[col], na_token)]
        parts.extend(
            _as_token_series(f[col], na_token) for f in fit_frames if col in f.columns
        )
        vocab = pd.concat(parts, ignore_index=True)

        enc = LabelEncoder().fit(vocab)
        out[col] = pd.Categorical(enc.transform(_as_token_series(out[col], na_token)))
        encoders[col] = enc

    return out, encoders


def _as_token_series(s: pd.Series, na_token: str) -> pd.Series:
    """결측을 `na_token` 문자열로 치환한 문자열 Series.

    pandas 3 에서 `.astype(str)` 은 NaN 을 보존하므로(문자열 "nan" 으로 바꾸지 않음)
    `astype("string")` 을 거쳐 결측을 명시적으로 채운다.
    """
    return s.astype("string").fillna(na_token).astype(str)


# =============================================================================
# C. 타깃 의존 피처 — 반드시 fold 안에서만 호출
#
# 이 섹션 아래의 함수는 `y` 를 읽는다. 따라서 "어떤 행의 라벨을 써도 되는가"를
# 호출부가 명시하지 않으면 호출 자체가 실패하도록 설계했다.
# =============================================================================


def smoothed_target_encode(
    fit_keys: pd.Series,
    fit_target: pd.Series,
    apply_keys: pd.Series,
    *,
    m: float = 20.0,
) -> np.ndarray:
    """베이지안 평활 타깃 인코딩. **단일 방향(fit → apply)만** 수행한다.

    5벌 중복된 `get_smoothed_target_encoding` 의 대체.

    기존 함수는 `(train_enc, test_enc)` 튜플을 한 번에 돌려주어 "학습행 자신의 라벨이
    자기 인코딩에 들어간다"는 사실을 숨겼다 (AUDIT.md §2.2 말미). 이 함수는 한 번에
    한 대상만 인코딩하므로 그 사실이 호출부에 드러난다.

        smooth = (count * mean + m * global_mean) / (count + m)

    `fit_keys` 에 없는 범주는 `global_mean` 으로 대치한다.
    """
    if len(fit_keys) != len(fit_target):
        raise ValueError(
            f"fit_keys({len(fit_keys)}) 와 fit_target({len(fit_target)}) 의 길이가 다릅니다."
        )
    if m < 0:
        raise ValueError(f"평활 계수 m 은 음수일 수 없습니다: {m}")

    keys = pd.Series(np.asarray(fit_keys), name="_key")
    target = pd.Series(np.asarray(fit_target, dtype=float), name="_y")

    global_mean = float(target.mean())
    stats = target.groupby(keys, observed=True).agg(["count", "mean"])
    smooth = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)

    applied = pd.Series(np.asarray(apply_keys)).map(smooth)
    return applied.fillna(global_mean).to_numpy(dtype=float)


class TEResult(NamedTuple):
    """`oof_target_encode()` 의 반환값.

    `AUDIT.md` §1.3 은 `(X_train_te, X_valid_te)` 2-튜플을 제안했으나, `extra_fit_rows`
    를 쓰면 학습 프레임의 행 수가 늘어나 `y` 를 호출부가 따로 맞춰야 한다. 그 수작업이
    바로 run_pseudo_labeling.py:248-249 에서 오염이 발생한 지점이므로, `y_train` 을
    함께 돌려주어 X 와 y 가 어긋날 여지를 없앤다. 앞 두 필드 순서는 설계안과 동일하므로
    `X_tr, X_va = oof_target_encode(...)[:2]` 형태로도 쓸 수 있다.
    """

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_train: pd.Series


def oof_target_encode(
    X: pd.DataFrame,
    y: pd.Series,
    outer_train_idx: np.ndarray,
    outer_valid_idx: np.ndarray,
    *,
    cols: Sequence[str] = CAT_COLS,
    m: float = 20.0,
    inner_splits: int = 5,
    seed: int = 42,
    prefix: str = "TE_",
    drop_original: bool = False,
    extra_fit_rows: tuple[pd.DataFrame, pd.Series] | None = None,
) -> TEResult:
    """outer fold 하나에 대한 Target Encoding 을 수행한다.

    핵심 계약
    ---------
    * `outer_train_idx` / `outer_valid_idx` 는 **기본값 없는 필수 위치 인자**다.
      둘 중 하나라도 빠뜨리면 `TypeError` 가 난다. fold 밖에서 TE 를 계산하는 실수가
      시그니처 레벨에서 불가능해진다 (AUDIT.md §1.2 원칙 1).
    * 두 인덱스가 겹치면 `ValueError`. 검증 행의 라벨이 자기 인코딩에 들어가는
      직접 누수를 런타임에서도 막는다.
    * `inner_splits >= 2` (기본 5, 권장): 학습행 인코딩도 내부 K-fold 로 산출하여
      **학습행이 자기 자신의 라벨로 인코딩되지 않게** 한다.
      `inner_splits = 0`: 현행 5개 스크립트의 동작 재현. 검증 추정치를 낙관시키지는
      않지만 모델이 TE 를 과신하게 만든다 (AUDIT.md §2.2 말미).
    * `drop_original=True`  → run_target_encoded.py:191-192 재현 (원본 범주 제거)
      `drop_original=False` → hybrid / advanced / phase7-Ex 재현 (원본 범주 유지)
    * `extra_fit_rows`: pseudo-label 등 추가 학습행을 TE 통계와 학습 프레임에 포함한다.
      **인자로 명시하지 않으면 절대 포함되지 않는다.** run_pseudo_labeling.py:257-258 의
      암묵적 오염을 막기 위한 설계다. 넘기는 행이 현재 fold 의 학습 데이터만으로
      만들어졌는지는 호출부 책임이며, `make_pseudo_labels()` 가 그것을 보장한다.

    검증 행 인코딩은 항상 학습 측 라벨(= outer_train + extra)만으로 계산된다.
    """
    train_idx = np.asarray(outer_train_idx)
    valid_idx = np.asarray(outer_valid_idx)

    if train_idx.ndim != 1 or valid_idx.ndim != 1:
        raise ValueError("fold 인덱스는 1차원 배열이어야 합니다.")
    if train_idx.size == 0 or valid_idx.size == 0:
        raise ValueError("fold 인덱스가 비어 있습니다.")

    overlap = np.intersect1d(train_idx, valid_idx)
    if overlap.size:
        raise ValueError(
            f"outer_train_idx 와 outer_valid_idx 가 {overlap.size}개 행에서 겹칩니다. "
            "검증 행의 라벨이 자기 인코딩에 들어가는 직접 누수입니다."
        )
    if inner_splits == 1 or inner_splits < 0:
        raise ValueError(f"inner_splits 는 0 이거나 2 이상이어야 합니다: {inner_splits}")

    X_train = X.iloc[train_idx].copy().reset_index(drop=True)
    y_train = y.iloc[train_idx].copy().reset_index(drop=True)
    X_valid = X.iloc[valid_idx].copy().reset_index(drop=True)

    if extra_fit_rows is not None:
        extra_X, extra_y = extra_fit_rows
        if len(extra_X) != len(extra_y):
            raise ValueError(
                f"extra_fit_rows 의 X({len(extra_X)}) 와 y({len(extra_y)}) 길이가 다릅니다."
            )
        X_train = pd.concat([X_train, extra_X], ignore_index=True)
        y_train = pd.concat(
            [y_train, pd.Series(np.asarray(extra_y), dtype=y_train.dtype)],
            ignore_index=True,
        )
        # concat 으로 풀린 category dtype 을 복원한다 (PLAN.md §3-5).
        for col in cols:
            if col in X_train.columns and isinstance(
                X.iloc[:0][col].dtype, pd.CategoricalDtype
            ):
                X_train[col] = X_train[col].astype("category")

    present = [c for c in cols if c in X_train.columns]
    for col in present:
        # 검증 행: 학습 측 라벨 전체로 인코딩 (누수 없음).
        X_valid[f"{prefix}{col}"] = smoothed_target_encode(
            X_train[col], y_train, X_valid[col], m=m
        )

        if inner_splits == 0:
            X_train[f"{prefix}{col}"] = smoothed_target_encode(
                X_train[col], y_train, X_train[col], m=m
            )
        else:
            enc = np.empty(len(X_train), dtype=float)
            inner = StratifiedKFold(
                n_splits=inner_splits, shuffle=True, random_state=seed
            )
            for in_tr, in_va in inner.split(X_train, y_train):
                enc[in_va] = smoothed_target_encode(
                    X_train[col].iloc[in_tr],
                    y_train.iloc[in_tr],
                    X_train[col].iloc[in_va],
                    m=m,
                )
            X_train[f"{prefix}{col}"] = enc

    if drop_original and present:
        X_train = X_train.drop(columns=present)
        X_valid = X_valid.drop(columns=present)

    return TEResult(X_train=X_train, X_valid=X_valid, y_train=y_train)


# =============================================================================
# D. 준지도 — 누수를 구조적으로 막는 시그니처
# =============================================================================


class PseudoLabels(NamedTuple):
    """`make_pseudo_labels()` 의 반환값. `X, y` 순으로 언패킹된다."""

    X: pd.DataFrame
    y: pd.Series
    thresh_neg: float
    thresh_pos: float


def make_pseudo_labels(
    teacher_predict: Callable[[pd.DataFrame], np.ndarray],
    X_unlabeled: pd.DataFrame,
    *,
    neg_percentile: float = 10.0,
    pos_percentile: float = 98.0,
) -> PseudoLabels:
    """고확신 미라벨 행을 선별해 `(pseudo_X, pseudo_y, thresh_neg, thresh_pos)` 반환.

    누수 차단 설계 (AUDIT.md §2.1, PLAN.md §3-6)
    --------------------------------------------
    * 첫 인자는 **모델도, 모델 리스트도 아닌 콜러블 하나**다. "이미 특정 fold 의 학습
      데이터만으로 적합된 예측 함수"를 받는다. 호출부는 fold 루프 안에서 그 fold 전용
      teacher 를 만들어 넘겨야 한다 (`fit_fold_teacher()` 참조).
    * 이 함수의 시그니처에는 **`y` 도 `X_labeled` 도 없다.** 함수 본문에서 전체 라벨
      데이터에 접근할 방법이 구조적으로 존재하지 않는다.
    * 분위수 임계값은 이 함수 안에서, 넘겨받은 콜러블의 출력만으로 계산된다.
      즉 **fold 마다 다시 계산된다.** AUDIT.md §1.3 초안에 있던 `percentile_basis`
      인자는 의도적으로 제거했다 — 미리 계산한 벡터를 주입할 수 있으면 5-fold 앙상블
      평균을 그대로 흘려보낼 수 있어, 막으려던 바로 그 누수의 우회로가 된다.

    run_pseudo_labeling.py:201 이 저지른 "5개 fold 모델 예측의 평균"을 넘기려는 시도는
    `TypeError` 로 거부된다.
    """
    if isinstance(teacher_predict, (list, tuple, set, dict)):
        raise TypeError(
            "teacher 모델/예측 컬렉션을 넘길 수 없습니다. 여러 fold 모델의 앙상블 평균은 "
            "전체 라벨을 소비하므로 fold 경계를 넘는 누수가 됩니다 (AUDIT.md §2.1). "
            "해당 fold 의 학습 데이터만으로 적합된 예측 콜러블 하나를 넘기세요."
        )
    if not callable(teacher_predict):
        raise TypeError(
            f"teacher_predict 는 콜러블이어야 합니다: {type(teacher_predict).__name__}"
        )
    if not 0.0 <= neg_percentile < pos_percentile <= 100.0:
        raise ValueError(
            f"0 <= neg_percentile({neg_percentile}) < pos_percentile({pos_percentile}) <= 100 "
            "이어야 합니다."
        )

    probs = np.asarray(teacher_predict(X_unlabeled), dtype=float)
    if probs.shape != (len(X_unlabeled),):
        raise ValueError(
            f"teacher_predict 는 길이 {len(X_unlabeled)} 의 1차원 확률 배열을 반환해야 "
            f"합니다. 받은 shape: {probs.shape}"
        )

    thresh_neg = float(np.percentile(probs, neg_percentile))
    thresh_pos = float(np.percentile(probs, pos_percentile))

    pos_idx = np.flatnonzero(probs >= thresh_pos)
    neg_idx = np.flatnonzero(probs <= thresh_neg)

    pseudo_X = pd.concat(
        [X_unlabeled.iloc[pos_idx], X_unlabeled.iloc[neg_idx]], ignore_index=True
    )
    pseudo_y = pd.Series(
        np.concatenate([np.ones(pos_idx.size, dtype=int), np.zeros(neg_idx.size, dtype=int)]),
        dtype=int,
    )
    return PseudoLabels(
        X=pseudo_X, y=pseudo_y, thresh_neg=thresh_neg, thresh_pos=thresh_pos
    )


def fit_fold_teacher(
    X_labeled: pd.DataFrame,
    y_labeled: pd.Series,
    train_idx: np.ndarray,
    *,
    fit_predict: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray],
) -> Callable[[pd.DataFrame], np.ndarray]:
    """fold 전용 teacher 예측 콜러블을 만든다. `make_pseudo_labels()` 의 짝.

    `train_idx` 가 가리키는 행의 라벨만 클로저에 갇히므로, 반환된 콜러블은 나머지
    fold 의 라벨에 접근할 수 없다. 안전한 경로를 가장 쓰기 쉬운 경로로 만들기 위한
    헬퍼다.

    Parameters
    ----------
    fit_predict
        `(X_tr, y_tr, X_apply) -> probs` 형태. 모델 선택은 호출부에 맡긴다.

    Examples
    --------
    >>> for fold, (tr, va) in enumerate(folds):            # doctest: +SKIP
    ...     teacher = fit_fold_teacher(X, y, tr, fit_predict=my_lgbm)
    ...     pseudo = make_pseudo_labels(teacher, X_unlabeled)
    """
    train_idx = np.asarray(train_idx)
    if train_idx.size == 0:
        raise ValueError("train_idx 가 비어 있습니다.")
    if train_idx.size >= len(X_labeled):
        raise ValueError(
            f"train_idx 가 전체 라벨 데이터({len(X_labeled)}행)를 모두 포함합니다"
            f"({train_idx.size}행). teacher 는 fold 의 학습 파트만 봐야 합니다 "
            "(AUDIT.md §2.1)."
        )

    X_tr = X_labeled.iloc[train_idx].copy().reset_index(drop=True)
    y_tr = y_labeled.iloc[train_idx].copy().reset_index(drop=True)

    def _predict(X_apply: pd.DataFrame) -> np.ndarray:
        return np.asarray(fit_predict(X_tr, y_tr, X_apply), dtype=float)

    return _predict
