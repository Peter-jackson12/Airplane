"""Phase 1~6 통일 프로토콜 재측정 — 정직한 기준선 확정.

`PLAN.md` §4-0 P0-2 / P0-3 실행 스크립트. `AUDIT.md` §3 이 "Phase 간 비교가 통계적으로
무효"라고 판정한 원인을 전부 제거한 상태에서 Phase 1~6 을 다시 잰다.

통일한 것
---------
* **평가 프로토콜**: 단일 `CVConfig` 인스턴스를 전 Phase 가 공유한다. fold 분할,
  임계값 그리드, early stopping 인내 라운드, 지표 집계 방식이 모두 같아진다.
* **지표 집계**: 전 Phase pooled OOF. 기존 Phase 1 만 fold 평균이었다 (AUDIT §3-9).
* **임계값**: 전 Phase nested 선택. 기존 6개 스크립트의 "전체 OOF 최댓값"은
  winner's curse 이므로 `naive_macro_f1` 로 따로 기록만 한다 (AUDIT §2.6).
* **트리/임계값 선택**: outer-train 내부 holdout에서 선택한다. Teacher도 student
  inner-train 안에서만 적합한다. P5_leaky는 의도적인 누수 대조군이다.
* **LightGBM 하이퍼파라미터**: 전 Phase 동일 고정 (Phase 4 설정 기준). Phase 별로
  **피처셋만** 교체한다.
* **TE 평활 계수 `m`**: 전 Phase 20.0 고정. 기존 Phase 6 만 25.0 이었다 (AUDIT §3.3).
* **범주 vocabulary**: 전 Phase 라벨+미라벨 합쳐 fit. 기존에는 Phase 5 만 그랬다
  (AUDIT §2.1-D). 타깃 비의존이므로 누수가 아니며, 통일하면 혼입 요인 하나가 준다.
* **TE 내부 K-fold**: 전 Phase `inner_splits=5` 고정. 기존 5개 스크립트는 0 에 해당한다.
  이 변경의 단독 효과를 분리하기 위해 Phase 4 만 `inner_splits=0` 으로도 추가 실행한다.

재측정 대상에서 제외한 것
-------------------------
Phase 7-Ex (기상 결합). 원천 데이터 결함(적설·돌풍 100% 결측, 결합률 55.77%,
도착지 기상 부재)으로 폐기되었으므로 기준선에 포함하지 않는다 (PLAN.md §3-12).

산출물
------
* `output/baseline_recovery_v2.csv` — Phase 단위 원자적 저장(스키마/실험 지문 검사). 중단되어도 이어서 실행된다.
* `output/baseline_recovery_v2.md` — 구프로토콜(§2-1) 대 신프로토콜 대조표 + 해석.

사용법
------
    uv run python rerun_all_phases.py                 # 전체 실행 (재개 지원)
    uv run python rerun_all_phases.py --sample 30000  # 축소 스모크 테스트
    uv run python rerun_all_phases.py --phases P4,P5_honest
    uv run python rerun_all_phases.py --force         # 같은 실험의 선택 Phase를 재실행해 교체
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.cv import CVConfig, evaluate_oof, make_folds, run_fold_nested_grid
from src.run_store import digest, file_digest, read_rows, upsert_row, save_oof, read_oof
from src.oof import OOF_SCHEMA_VERSION, build_oof_rows
from src.features import (
    build_cyclic_features,
    build_time_features,
    build_traffic_features,
    encode_categoricals,
    impute_cross,
    build_speed_features,
    load_data,
    make_pseudo_labels,
    prune_columns,
    restore_time_missing,
    split_labeled,
)

warnings.filterwarnings("ignore")

# Windows 콘솔 기본 코드페이지(cp949)는 em-dash 등을 인코딩하지 못해 실행이 죽는다.
# 리다이렉트/파이프 환경에서도 동일하게 동작하도록 스트림을 UTF-8 로 고정한다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover - 비표준 스트림 대비
        pass

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
OUTPUT_DIR = ROOT / "output"
CSV_PATH = OUTPUT_DIR / "baseline_recovery_v2.csv"
MD_PATH = OUTPUT_DIR / "baseline_recovery_v2.md"
RUN_METADATA: dict = {}
RUN_ID = ""
SAVE_OOF = False


# =============================================================================
# 고정 프로토콜 — 전 Phase 공유
# =============================================================================

#: 단일 인스턴스. 모든 Phase 가 이 객체 하나를 그대로 쓴다.
#: stopping_rounds는 레거시 API 호환용이다. Student/teacher 모두 nested grid를 사용한다.
CFG = CVConfig(
    n_splits=5,
    shuffle=True,
    seed=42,
    stopping_rounds=40,
    threshold_grid=(0.10, 0.70, 0.01),
    metric_aggregation="pooled_oof",
    inner_holdout_frac=0.2,
)

#: 전 Phase 동일 고정. Phase 4(run_hybrid.py:156-169) 설정을 기준으로 삼는다.
#: `n_estimators` 는 더 이상 여기서 고정하지 않는다 — `run_fold_nested_grid()` 가
#: `N_ESTIMATORS_GRID` 를 스윕해 fold마다 명시적으로 확정한다(es_protocol_final.md
#: "최종 권장 n_estimators 선택 절차"). `scale_pos_weight` 는 지정하지 않는다
#: (es_protocol_final.md 작업 5 — Macro F1 차이는 작은 3시드 실험에서 불확실하며, LogLoss 는 spw 없음이
#: 압도적으로 우세해 제거가 확정되었다. 기본값 1.0 이 적용된다).
LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

#: `es_protocol_final.md` 작업 1~3 이 쓴 그리드 그대로. 5-fold x 3seed 15회 중
#: 14회가 정확히 k=50 을 선택했다(1회만 k=75) — 이미 이 조도로도 fold/시드 간
#: 선택이 안정적이었다. 작은 표본에서 결정론적이라고 일반화하지 않는다.
N_ESTIMATORS_GRID: list[int] = [10, 25, 50, 75, 100, 150, 300, 600]

TE_SMOOTHING_M = 20.0
TE_INNER_SPLITS = 5

#: 기존 PLAN.md §2-1 기록치 (구프로토콜). 대조표 작성용.
OLD_PROTOCOL: dict[str, dict] = {
    "P1": dict(thr=0.50, logloss=0.4636, auc=0.6366, f1=0.4516, tp=0, agg="fold 평균"),
    "P2": dict(thr=0.22, logloss=0.4635, auc=0.6357, f1=0.5724, tp=13039, agg="pooled"),
    "P3": dict(thr=0.22, logloss=0.4635, auc=0.6043, f1=0.5521, tp=14895, agg="pooled"),
    "P4": dict(thr=0.22, logloss=0.4633, auc=0.6107, f1=0.5579, tp=14386, agg="pooled"),
    "P5_leaky": dict(thr=0.24, logloss=0.4587, auc=0.6416, f1=0.5746, tp=13527, agg="pooled"),
    "P6_legacy": dict(thr=0.21, logloss=0.4629, auc=0.6345, f1=0.5721, tp=15866, agg="pooled"),
}


# =============================================================================
# Phase 정의 — 피처셋만 달라진다
# =============================================================================

BASELINE_CATS = (
    "Origin_Airport", "Origin_State", "Destination_Airport", "Destination_State",
    "Airline", "Carrier_Code(IATA)", "Tail_Number", "Route",
)
STANDARD_CATS = (
    "Tail_Number", "Route", "Origin_Airport", "Destination_Airport", "Airline",
)


@dataclass(frozen=True)
class PhaseSpec:
    key: str
    label: str
    #: 전처리
    bidirectional_impute: bool = False
    prune_preset: str = "pruned"
    drop_extra: tuple[str, ...] = ("Dep_Minute", "Arr_Minute")
    cyclic_missing: str = "nan"
    include_cos: bool = True
    restore_duration: bool = False
    restore_hours: bool = False
    traffic: bool = False
    traffic_exclude_missing: bool = True
    cat_cols: tuple[str, ...] = STANDARD_CATS
    #: 타깃 의존
    te: bool = False
    te_drop_original: bool = False
    inner_splits: int = TE_INNER_SPLITS
    pseudo: str | None = None  # None | "leaky" | "honest"
    #: [미사용] es_protocol_final.md 작업 5 로 scale_pos_weight 제거가 확정되어
    #: `model_factory()` 는 이 값을 더 이상 읽지 않는다. 원본 스크립트가 전부
    #: scale_pos_weight 를 적용했다는 사실을 Phase 정의에 문서로 남기기 위해
    #: 필드만 유지한다(값을 바꿔도 학습에 아무 영향이 없다).
    use_spw: bool = True
    note: str = ""
    #: 개선 전처리는 별도 Phase 키로 선택하여 과거 피처 정의를 보존한다.
    safe_preprocessing: bool = False
    unique_imputation: bool | None = None
    safe_ratio: bool | None = None
    explicit_time_missing: bool | None = None

    def feature_signature(self) -> tuple:
        """피처 빌드 캐시 키. 타깃 의존 설정은 제외한다."""
        return (
            self.bidirectional_impute, self.prune_preset, self.drop_extra,
            self.cyclic_missing, self.include_cos, self.restore_duration,
            self.restore_hours, self.traffic, self.traffic_exclude_missing,
            self.cat_cols, self.safe_preprocessing, self.unique_imputation,
            self.safe_ratio, self.explicit_time_missing,
        )


PHASES: list[PhaseSpec] = [
    PhaseSpec(
        key="P1", label="Phase 1: Baseline",
        bidirectional_impute=True, prune_preset="baseline", drop_extra=(),
        cat_cols=BASELINE_CATS,
        note="State/Carrier_Code/Day_of_Month/분 단위 유지. TE 없음.",
    ),
    PhaseSpec(
        key="P2", label="Phase 2: Tuned (Pruning)",
        include_cos=False,
        note="노이즈 피처 Pruning + Cos 제거. TE 없음(원 docstring은 오류).",
    ),
    PhaseSpec(
        key="P3", label="Phase 3: Target Encoded",
        te=True, te_drop_original=True,
        note="원본 범주 컬럼을 TE 수치로 완전 치환.",
    ),
    PhaseSpec(
        key="P4", label="Phase 4: Hybrid",
        te=True, te_drop_original=False,
        note="원본 category dtype + TE 동시 투입. 하이퍼파라미터 기준 Phase.",
    ),
    PhaseSpec(
        key="P4_te0", label="Phase 4: Hybrid (inner_splits=0)",
        te=True, te_drop_original=False, inner_splits=0,
        note="TE 내부 K-fold 도입의 단독 효과 분리용. 기존 5개 스크립트의 TE 방식.",
    ),
    PhaseSpec(
        key="P4_nospw", label="Phase 4: Hybrid (scale_pos_weight 미적용)",
        te=True, te_drop_original=False, use_spw=False,
        note="호환용 별칭. 현재 P4와 동일하며 spw 비교 조건이 아니다.",
    ),
    PhaseSpec(
        key="P5_leaky", label="Phase 5: Pseudo-Labeling (누수 재현)",
        te=True, pseudo="leaky",
        note="teacher = 5개 fold 모델 예측 평균. 기존 구현 재현.",
    ),
    PhaseSpec(
        key="P5_honest", label="Phase 5: Pseudo-Labeling (정직)",
        te=True, pseudo="honest",
        note="fold 내부 teacher. 분위수도 fold별 재계산.",
    ),
    PhaseSpec(
        key="P6_legacy", label="Phase 6: Advanced (Traffic 레거시)",
        prune_preset="advanced", drop_extra=(), cyclic_missing="noon",
        restore_duration=True, restore_hours=True,
        traffic=True, traffic_exclude_missing=False,
        te=True,
        note="Dep_Hour==-1 을 하나의 버킷으로 집계. 기존 구현 재현.",
    ),
    PhaseSpec(
        key="P6_fixed", label="Phase 6: Advanced (Traffic 결측 제외)",
        prune_preset="advanced", drop_extra=(), cyclic_missing="noon",
        restore_duration=True, restore_hours=True,
        traffic=True, traffic_exclude_missing=True,
        te=True,
        note="Dep_Hour==-1 행을 집계에서 제외 + 결측 플래그 분리 (AUDIT §3-8 수정).",
    ),
]

# 동일한 모델 선택 프로토콜에서 전처리 묶음의 효과를 비교할 신규 조건.
# 기존 P4/P6 정의와 과거 결과를 덮어쓰지 않는다.
PHASES.extend([
    replace(next(s for s in PHASES if s.key == base), key=key, label=label,
            safe_preprocessing=True, cyclic_missing="nan",
            note="유일 대응만 대치; local-clock proxy 명시; 0분 분모 NaN; 시각 결측 플래그")
    for base, key, label in [
        ("P4", "P4_clean", "Phase 4: 전처리 개선"),
        ("P6_fixed", "P6_clean", "Phase 6: 전처리 개선"),
    ]
])

# One-component changes from each unchanged reference; do not infer component
# contributions by comparing the combined clean pipeline with old reports.
for _base in ("P4", "P6_fixed"):
    _spec = next(s for s in PHASES if s.key == _base)
    for _suffix, _field in [("impute", "unique_imputation"),
                             ("ratio", "safe_ratio"),
                             ("missing", "explicit_time_missing")]:
        PHASES.append(replace(_spec, key=f"{_base}_{_suffix}",
                              label=f"{_spec.label} / {_suffix} only",
                              note=f"단일 변경: {_field}", **{_field: True}))

#: `best_iterations`(ES `best_iteration_`) 대신 `selected_n_estimators`(nested grid 가
#: 고른 fold별 n_estimators) 와 `grid_scores`(fold별 `{k: inner-holdout LogLoss}`) 를
#: 기록한다 — 조건 A/B(ES) 에서 조건 C(nested grid) 로 전환한 구조 변경을 CSV 스키마에
#: 그대로 반영한다. `scale_pos_weight` 는 항상 "-" 로 기록된다(작업 3, 전 Phase 제거).
CSV_FIELDS = [
    "run_id", "run_metadata",
    "phase_key", "label", "n_rows", "n_features", "log_loss", "roc_auc",
    "f1_at_050", "macro_f1_nested", "naive_macro_f1", "threshold_optimism",
    "deployment_threshold", "per_fold_thresholds", "tn", "fp", "fn", "tp",
    "recall", "selected_n_estimators", "grid_scores", "pseudo_counts", "pseudo_total",
    "inner_splits", "scale_pos_weight", "traffic_mode", "te_mode", "pseudo_mode",
    "elapsed_sec", "timestamp", "note", "protocol", "lgbm_params",
    "oof_path", "oof_sha256", "oof_columns",
]


# =============================================================================
# 피처 빌드 — 타깃 비의존 구간
# =============================================================================


def build_features(raw: pd.DataFrame, spec: PhaseSpec):
    """`spec` 이 지정한 피처셋을 만들어 `(X_lab, y, X_unlab)` 을 돌려준다.

    이 함수는 `Delay` 를 `split_labeled()` 에서 단 한 번만 건드린다. 그 전 단계는
    전부 타깃 비의존이다. 전체 데이터 집계/vocabulary를 사용하는 기존 실험
    계약은 유지하며, 실제 배포 시점에 해당 자료를 사용할 수 있는지는 별도 문제다.
    """
    unique = spec.safe_preprocessing if spec.unique_imputation is None else spec.unique_imputation
    safe_ratio = spec.safe_preprocessing if spec.safe_ratio is None else spec.safe_ratio
    explicit_missing = spec.safe_preprocessing if spec.explicit_time_missing is None else spec.explicit_time_missing
    df = impute_cross(raw, bidirectional=spec.bidirectional_impute,
                      conflict="unique" if unique else "first")

    # Route 는 traffic / TE 보다 먼저 있어야 한다.
    df["Route"] = (
        df["Origin_Airport"].astype(str) + "_" + df["Destination_Airport"].astype(str)
    )

    # 분 단위는 restore_time_missing(fill_hours=True) 에 필요하므로 항상 만든 뒤,
    # 필요 없는 Phase 에서는 prune 단계에서 떨군다.
    df = build_time_features(df, keep_minute=True)
    if explicit_missing:
        for col in ("Dep_Hour", "Arr_Hour"):
            df[f"{col}_Originally_Missing"] = df[col].lt(0).astype("int8")
        df["Local_Time_Gap_Originally_Missing"] = df.Estimated_Duration.isna().astype("int8")

    if spec.restore_duration or spec.restore_hours:
        df = restore_time_missing(
            df,
            fill_duration=spec.restore_duration,
            fill_hours=spec.restore_hours,
        )

    if spec.traffic:
        df = build_traffic_features(
            df,
            exclude_missing_hour=spec.traffic_exclude_missing,
            add_missing_flag=spec.traffic_exclude_missing,
        )

    df = build_cyclic_features(
        df, missing="nan" if explicit_missing else spec.cyclic_missing, include_cos=spec.include_cos,
        add_missing_flag=explicit_missing,
    )

    # 7개 스크립트 공통 파생. 복원된 Duration 을 쓰도록 restore 뒤에 계산한다.
    df = build_speed_features(df, safe=safe_ratio)

    df = prune_columns(df, preset=spec.prune_preset, extra=spec.drop_extra)

    X_lab, y, X_unlab = split_labeled(df)

    # 범주 인코딩: 라벨+미라벨을 한 번에 인코딩해 동일한 category dtype 을 보장한다.
    # 따로 인코딩하면 categories 배열이 달라져 pseudo 행 concat 시 dtype 이 풀린다.
    n_lab = len(X_lab)
    combined = pd.concat([X_lab, X_unlab], ignore_index=True)
    combined, encoders = encode_categoricals(combined, cat_cols=spec.cat_cols)
    X_lab = combined.iloc[:n_lab].reset_index(drop=True)
    X_unlab = combined.iloc[n_lab:].reset_index(drop=True)
    # Preserve how MISSING was encoded for export; this metadata is not a model feature.
    X_lab.attrs["missing_category_codes"] = {
        col: int(enc.transform(["MISSING"])[0])
        for col, enc in encoders.items() if "MISSING" in enc.classes_
    }

    return X_lab, y, X_unlab


def model_factory() -> lgb.LGBMClassifier:
    """새 LightGBM 분류기를 만든다. `run_fold()`/`run_fold_nested_grid()` 양쪽이
    기대하는 "인자 없이 호출하는 콜러블" 계약을 그대로 만족한다.

    `scale_pos_weight` 는 더 이상 계산하지 않는다 (es_protocol_final.md 작업 5,
    `LGBM_PARAMS` 독스트링 참고) — LightGBM 기본값 1.0 이 전 Phase 에 적용된다.
    """
    return lgb.LGBMClassifier(**LGBM_PARAMS)


# =============================================================================
# Phase 5 — teacher 두 갈래
# =============================================================================


def _teacher_fit_predict(X_tr, y_tr, X_apply, spec: PhaseSpec, fold: int):
    """Teacher도 clean inner TE + nested grid를 사용한다.

    Honest 경로의 X_tr은 student inner-train만 포함한다. Teacher의 자체
    holdout은 그 안에서 분리되므로 student holdout 라벨을 참조하지 않는다.
    """
    fit = run_fold_nested_grid(
        model_factory, X_tr, y_tr, X_apply, CFG, fold=fold,
        n_estimators_grid=N_ESTIMATORS_GRID,
        te_cols=spec.cat_cols if spec.te else (), te_m=TE_SMOOTHING_M,
        te_inner_splits=spec.inner_splits, te_drop_original=spec.te_drop_original,
    )
    return fit.valid_probs


def _honest_pseudo_factory(X_unlab, spec, fold):
    """student의 inner 분할 이후에만 호출되는 pseudo 생성자."""
    def build(X_inner, y_inner):
        teacher = lambda apply: _teacher_fit_predict(X_inner, y_inner, apply, spec, fold)
        pseudo = make_pseudo_labels(
            teacher, X_unlab, neg_percentile=10.0, pos_percentile=98.0
        )
        return pseudo.X, pseudo.y
    return build


def _leaky_ensemble_pseudo_labels(X_lab, y, X_unlab, folds, spec):
    """[재현 전용] 기존 `run_pseudo_labeling.py` 의 누수 구조를 그대로 재현한다.

    이 함수는 **의도적으로 잘못된 구현**이다. `src.features.make_pseudo_labels()` 는
    이 경로를 `TypeError` 로 거부하도록 설계되어 있으므로, 누수 크기를 실측하기 위해
    모듈을 우회한다. 실제 모델링에는 절대 쓰지 말 것.

    재현하는 누수 (AUDIT.md §2.1):
      1. teacher 예측이 5개 fold 모델의 **평균** → 임의의 라벨 행이 5개 중 4개
         teacher 의 학습셋에 포함되므로, 결과 벡터가 전체 라벨의 함수가 된다.
      2. 분위수 임계값을 그 벡터에서 fold 루프 **밖에서** 한 번만 계산한다.
      3. 그렇게 동결된 pseudo 집합을 모든 fold 의 학습셋에 동일하게 주입한다.
    """
    print("   [leaky] teacher 5-fold 앙상블 평균 산출 중 (누수 재현)...")
    ensemble = np.zeros(len(X_unlab))
    for fold, (tr, _) in enumerate(folds):
        probs = _teacher_fit_predict(
            X_lab.iloc[tr].reset_index(drop=True),
            y.iloc[tr].reset_index(drop=True),
            X_unlab,
            spec, fold,
        )
        ensemble += probs / len(folds)
        print(f"      teacher fold {fold + 1}/{len(folds)} 완료")

    # fold 루프 밖에서 단 한 번 — 이것이 누수의 전달 경로다.
    thresh_neg = float(np.percentile(ensemble, 10))
    thresh_pos = float(np.percentile(ensemble, 98))
    pos_idx = np.flatnonzero(ensemble >= thresh_pos)
    neg_idx = np.flatnonzero(ensemble <= thresh_neg)

    pseudo_X = pd.concat(
        [X_unlab.iloc[pos_idx], X_unlab.iloc[neg_idx]], ignore_index=True
    )
    pseudo_y = pd.Series(
        np.concatenate([np.ones(pos_idx.size, int), np.zeros(neg_idx.size, int)]),
        dtype=int,
    )
    print(
        f"   [leaky] 동결된 pseudo 집합 {len(pseudo_X):,}건 "
        f"(양성 {pos_idx.size:,} / 음성 {neg_idx.size:,}), "
        f"임계 {thresh_neg:.4f} / {thresh_pos:.4f}"
    )
    return pseudo_X, pseudo_y


# =============================================================================
# Phase 실행
# =============================================================================


def run_phase(spec: PhaseSpec, raw: pd.DataFrame, cache: dict) -> dict:
    started = time.perf_counter()
    print(f"\n{'=' * 78}\n▶ {spec.key} — {spec.label}\n{'=' * 78}")
    if spec.note:
        print(f"   {spec.note}")

    sig = spec.feature_signature()
    if sig not in cache:
        t0 = time.perf_counter()
        cache.clear()  # 100만 행 프레임을 여러 벌 들고 있지 않는다
        cache[sig] = build_features(raw, spec)
        print(f"   피처 빌드 {time.perf_counter() - t0:.1f}s")
    else:
        print("   피처 빌드 캐시 재사용")
    X_lab, y, X_unlab = cache[sig]

    folds = make_folds(y, CFG)
    print(
        f"   라벨 {len(X_lab):,}행 | 피처 {X_lab.shape[1]}개 | "
        f"미라벨 {len(X_unlab):,}행 | scale_pos_weight 미적용(고정, es_protocol_final.md 작업 5)"
    )

    frozen_pseudo = None
    if spec.pseudo == "leaky":
        frozen_pseudo = _leaky_ensemble_pseudo_labels(X_lab, y, X_unlab, folds, spec)

    # TE 를 쓰지 않는 Phase 는 `te_cols=()` 로 grid 선택 경로만 태운다 — TE 유무와
    # 무관하게 모든 Phase 가 동일한 nested n_estimators 선택 절차를 거친다.
    te_cols = spec.cat_cols if spec.te else ()

    oof = np.full(len(y), np.nan)
    selected_n_estimators: list[int] = []
    grid_scores_per_fold: list[dict[int, float]] = []
    pseudo_counts: list[int] = []
    thresholds: list[float] = []
    feature_names: list[str] = []

    for fold, (tr, va) in enumerate(folds):
        t0 = time.perf_counter()
        extra = None

        if spec.pseudo == "leaky":
            extra = frozen_pseudo


        # X_tr/X_va 는 TE 인코딩 **전** 원본 카테고리 상태로 넘긴다 — TE 계산 자체를
        # run_fold_nested_grid() 내부에서 outer/inner 2단계로 수행하기 때문이다
        # (es_protocol_final.md 조건 C, audit_addendum_te_leak.md §2 가 지적한
        # ES-inner-holdout TE 리키지를 구조적으로 차단한다). pseudo 행(`extra`)은
        # `extra_fit_rows` 로 전달되어 항상 inner-train 쪽에만 추가되므로,
        # 예전의 `holdout_eligible` 위치-필터링 없이도 pseudo 행이 grid 선택용
        # inner-holdout 채점에 섞이는 일이 구조적으로 없다.
        X_tr_raw = X_lab.iloc[tr].reset_index(drop=True)
        y_tr_raw = y.iloc[tr].reset_index(drop=True)
        X_va_raw = X_lab.iloc[va].reset_index(drop=True)

        selection = {}
        fit = run_fold_nested_grid(
            model_factory, X_tr_raw, y_tr_raw, X_va_raw, CFG,
            fold=fold,
            n_estimators_grid=N_ESTIMATORS_GRID,
            te_cols=te_cols,
            te_m=TE_SMOOTHING_M,
            te_inner_splits=spec.inner_splits,
            te_drop_original=spec.te_drop_original,
            extra_fit_rows=extra,
            extra_fit_factory=(_honest_pseudo_factory(X_unlab, spec, fold)
                               if spec.pseudo == "honest" else None),
            selection_metadata=selection,
        )
        thresholds.append(selection["threshold"])
        if spec.pseudo:
            pseudo_counts.append(selection["pseudo_count"])
        oof[va] = fit.valid_probs
        selected_n_estimators.append(fit.selected_n_estimators)
        grid_scores_per_fold.append(fit.grid_scores)
        if not feature_names:
            feature_names = list(getattr(fit.model, "feature_name_", []))
        print(
            f"   fold {fold + 1}/{len(folds)} | 학습 {fit.n_inner_train:,}행 "
            f"(holdout {fit.n_inner_holdout:,}) | n_estimators {fit.selected_n_estimators} "
            f"| {time.perf_counter() - t0:.1f}s"
        )

    assert not np.isnan(oof).any(), "모든 행이 정확히 한 번 채점되어야 한다"

    res = evaluate_oof(y, oof, folds, CFG, label=spec.label,
                       feature_names=feature_names, per_fold_thresholds=thresholds)
    res["protocol"].update(protocol_description())
    elapsed = time.perf_counter() - started

    row = {
        "phase_key": spec.key,
        "label": spec.label,
        "n_rows": res["n_rows"],
        "n_features": res["n_features"],
        "log_loss": round(res["log_loss"], 6),
        "roc_auc": round(res["roc_auc"], 6),
        "f1_at_050": round(res["f1_at_050"], 6),
        "macro_f1_nested": round(res["macro_f1"], 6),
        "naive_macro_f1": round(res["naive_macro_f1"], 6),
        "threshold_optimism": round(res["threshold_optimism"], 6),
        "deployment_threshold": res["deployment_threshold"],
        "per_fold_thresholds": json.dumps(res["per_fold_thresholds"]),
        "tn": res["confusion_matrix"]["tn"],
        "fp": res["confusion_matrix"]["fp"],
        "fn": res["confusion_matrix"]["fn"],
        "tp": res["confusion_matrix"]["tp"],
        "recall": round(res["recall"], 6),
        "selected_n_estimators": json.dumps(selected_n_estimators),
        "grid_scores": json.dumps(grid_scores_per_fold),
        "pseudo_counts": json.dumps(pseudo_counts),
        "pseudo_total": int(np.sum(pseudo_counts)) if pseudo_counts else 0,
        "inner_splits": spec.inner_splits,
        "scale_pos_weight": "-",
        "traffic_mode": (
            "-" if not spec.traffic
            else ("exclude_missing" if spec.traffic_exclude_missing else "legacy")
        ),
        "te_mode": (
            "none" if not spec.te
            else ("drop_original" if spec.te_drop_original else "hybrid")
        ),
        "pseudo_mode": spec.pseudo or "-",
        "elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": spec.note,
        "protocol": json.dumps(res["protocol"], ensure_ascii=False),
        "lgbm_params": json.dumps(LGBM_PARAMS),
        "oof_path": "", "oof_sha256": "", "oof_columns": "",
    }

    if SAVE_OOF:
        if not RUN_ID:
            raise ValueError("Experiment identity has not been initialized")
        oof_rows = build_oof_rows(raw, y, oof, folds, thresholds, selected_n_estimators,
                                 run_id=RUN_ID, phase_key=spec.key, seed=CFG.seed, features=X_lab)
        path, sha = save_oof(oof_rows, CSV_PATH.parent / f"{CSV_PATH.stem}_oof", spec.key)
        row.update(oof_path=path.relative_to(CSV_PATH.parent).as_posix(), oof_sha256=sha,
                   oof_columns=json.dumps(list(oof_rows.columns)))

    print(
        f"   ★ LogLoss {row['log_loss']:.4f} | AUC {row['roc_auc']:.4f} | "
        f"Macro F1(nested) {row['macro_f1_nested']:.4f} "
        f"(naive {row['naive_macro_f1']:.4f}, 차이 {row['threshold_optimism']:+.4f}) "
        f"| {elapsed:.0f}s"
    )
    return row


# =============================================================================
# CSV 즉시 append / 재개
# =============================================================================


def load_done(force: bool) -> dict[str, dict]:
    # --force controls execution, never bypasses schema/protocol validation.
    if not RUN_ID:
        raise ValueError("Experiment identity has not been initialized")
    rows = read_rows(CSV_PATH, CSV_FIELDS, RUN_ID)
    if SAVE_OOF:
        for row in rows.values():
            read_oof(CSV_PATH, row)
    return rows


def append_row(row: dict) -> None:
    if not RUN_ID:
        raise ValueError("Experiment identity has not been initialized")
    row.update(run_id=RUN_ID, run_metadata=json.dumps(RUN_METADATA, ensure_ascii=False))
    upsert_row(CSV_PATH, CSV_FIELDS, RUN_ID, row)


def configure_run(sample: int | None, output_prefix: str | None) -> None:
    global CSV_PATH, MD_PATH, RUN_METADATA, RUN_ID
    if sample is not None and sample <= 0:
        raise ValueError("sample must be positive")
    prefix = output_prefix or (
        f"baseline_recovery_v2_sample_{sample}" if sample else "baseline_recovery_v2"
    )
    if Path(prefix).name != prefix or any(c in prefix for c in '/\\:.'):
        raise ValueError("output-prefix must be a plain filename stem")
    if not prefix.startswith("baseline_recovery_v2"):
        raise ValueError("output-prefix must start with baseline_recovery_v2")
    CSV_PATH, MD_PATH = OUTPUT_DIR / f"{prefix}.csv", OUTPUT_DIR / f"{prefix}.md"
    RUN_METADATA = {
        "protocol_version": "nested-grid-inner-te-teacher-threshold-v2",
        "save_oof": SAVE_OOF, "oof_schema_version": OOF_SCHEMA_VERSION,
        "cv": protocol_description(), "grid": N_ESTIMATORS_GRID, "params": LGBM_PARAMS,
        "te_m": TE_SMOOTHING_M, "phases": [asdict(s) for s in PHASES],
        "sample": sample, "data_sha256": file_digest(DATA_PATH),
        "code_sha256": {name: file_digest(ROOT / name) for name in
                        ("src/cv.py", "src/features.py", "src/run_store.py", "src/oof.py", "rerun_all_phases.py")},
        "versions": {name: version(name) for name in
                     ("lightgbm", "pandas", "numpy", "scikit-learn")},
        "threshold_selection": "outer_train_holdout",
        "teacher_scope": "student_inner_train",
    }
    RUN_ID = digest(RUN_METADATA)


def protocol_description() -> dict:
    return {**CFG.describe(), "inner_early_stopping": False,
            "tree_selection": "nested_grid", "n_estimators_grid": N_ESTIMATORS_GRID,
            "threshold_selection": "outer_train_holdout",
            "teacher_scope": "student_inner_train", "teacher_selection": "nested_grid",
            "stopping_rounds_used": False}


def print_header(specs: list[PhaseSpec], sample: int | None) -> None:
    print("=" * 78)
    print("Phase 1~6 통일 프로토콜 재측정 — 정직한 기준선 확정")
    print("PLAN.md §4-0 P0-2 / P0-3 · AUDIT.md §3 대응")
    print("=" * 78)
    print(f"\n[실행 시각] {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    if sample:
        print(f"[!] 축소 스모크 모드: 상위 {sample:,}행만 사용 — 결과를 인용하지 말 것")

    print("\n[CVConfig.describe()]")
    for k, v in protocol_description().items():
        print(f"  {k:24} = {v}")

    print("\n[LightGBM 하이퍼파라미터 — 전 Phase 동일 고정 (Phase 4 기준)]")
    for k, v in LGBM_PARAMS.items():
        print(f"  {k:24} = {v}")
    print(f"  {'scale_pos_weight':24} = 미적용 (전 Phase 고정, es_protocol_final.md 작업 5)")
    print(f"  {'n_estimators':24} = nested grid 선택 (아래 grid, ES 아님)")
    print(f"  {'n_estimators_grid':24} = {N_ESTIMATORS_GRID}")

    print("\n[TE 설정 — 전 Phase 동일 고정]")
    print(f"  {'smoothing_m':24} = {TE_SMOOTHING_M}  (기존 Phase 6 만 25.0 이었음)")
    print(f"  {'inner_splits':24} = {TE_INNER_SPLITS}  (기존 5개 스크립트는 0 에 해당)")
    print("  ※ P4_te0 만 inner_splits=0 으로 추가 실행 — TE 방식 변경의 단독 효과 분리용")

    print(f"\n[실행 대상] {len(specs)}개")
    for s in specs:
        print(f"  {s.key:12} {s.label}")
    print(f"\n[제외] Phase 7-Ex — 원천 데이터 결함으로 폐기 (PLAN.md §3-12)")
    print("=" * 78)


def main() -> int:
    global CFG, SAVE_OOF
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=None, help="축소 스모크용 행 수")
    ap.add_argument("--seed", type=int, default=42, help="CV와 LightGBM에 동일 적용할 시드")
    ap.add_argument("--phases", type=str, default=None, help="쉼표 구분 phase_key")
    ap.add_argument("--force", action="store_true", help="같은 실험의 선택 Phase를 재실행해 교체")
    ap.add_argument("--report-only", action="store_true", help="CSV 로 리포트만 재생성")
    ap.add_argument("--output-prefix", help="output/ 아래 새 결과 파일 이름(확장자 제외)")
    ap.add_argument("--save-oof", action="store_true", help="행별 OOF·원본 결측 이력 저장 및 재개 검증")
    args = ap.parse_args()
    SAVE_OOF = args.save_oof
    CFG = replace(CFG, seed=args.seed)
    LGBM_PARAMS["random_state"] = args.seed

    if not DATA_PATH.exists():
        print(f"[!] 데이터 파일이 없습니다: {DATA_PATH}")
        return 1
    try:
        configure_run(args.sample, args.output_prefix)
        done = load_done(args.force)
    except ValueError as exc:
        print(f"[!] {exc}")
        return 1

    if args.report_only:
        if not done:
            print(f"[!] {CSV_PATH} 가 없습니다.")
            return 1
        write_report(done)
        print(f">> 리포트 재생성 완료: {MD_PATH}")
        return 0

    wanted = set(args.phases.split(",")) if args.phases else None
    specs = [s for s in PHASES if wanted is None or s.key in wanted]
    if wanted and wanted - {s.key for s in PHASES}:
        print(f"[!] 알 수 없는 phase: {wanted}")
        return 1

    print_header(specs, args.sample)

    if not DATA_PATH.exists():
        print(f"[!] 데이터 파일이 없습니다: {DATA_PATH}")
        return 1

    print(f"\n>> 데이터 로드: {DATA_PATH}")
    t0 = time.perf_counter()
    raw = load_data(DATA_PATH)
    if args.sample:
        raw = raw.head(args.sample).reset_index(drop=True)
    print(f"   {raw.shape[0]:,}행 × {raw.shape[1]}열 | {time.perf_counter() - t0:.1f}s")

    cache: dict = {}
    failed = False
    for spec in specs:
        if spec.key in done and not args.force:
            print(f"\n▶ {spec.key} — 이미 완료됨, 건너뜀 (--force 로 재실행)")
            continue
        try:
            row = run_phase(spec, raw, cache)
        except Exception as exc:  # noqa: BLE001 - 한 Phase 실패가 전체를 막지 않게
            print(f"\n[!] {spec.key} 실패: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
            failed = True
            continue
        append_row(row)
        done[spec.key] = row
        print(f"   >> CSV checkpoint 완료: {CSV_PATH}")

    write_report(done)
    print(f"\n>> 대조표 저장: {MD_PATH}")
    print(f">> 원본 CSV   : {CSV_PATH}")
    return 1 if failed else 0


# =============================================================================
# 리포트
# =============================================================================


def _f(row: dict, key: str, nd: int = 4) -> str:
    try:
        return f"{float(row[key]):.{nd}f}"
    except (KeyError, TypeError, ValueError):
        return "—"


def write_report(done: dict[str, dict]) -> None:
    order = [s.key for s in PHASES if s.key in done]
    rows = [done[k] for k in order]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    L: list[str] = []
    A = L.append
    A("# 기준선 복구 — Phase 1~6 통일 프로토콜 재측정 결과\n")
    A(f"> 생성 시각: {now}  ")
    A(f"> 원본 데이터: `{CSV_PATH.name}`  ")
    A(f"> 실험 ID: `{RUN_ID}`  ")
    if RUN_METADATA.get("sample") is not None:
        A(f"> **스모크 테스트: 원본 앞 {RUN_METADATA['sample']:,}행. 성능 비교에 인용하지 말 것.**  ")
    A("> P5_leaky는 의도적 누수 대조군이며 성능 순위에서 제외한다.  ")
    A("> 근거 문서: [AUDIT.md](../AUDIT.md) §3, [PLAN.md](../PLAN.md) §4-0 P0-2·P0-3\n")
    A("---\n")

    A("## 1. 적용한 통일 프로토콜\n")
    if rows:
        proto = json.loads(rows[0]["protocol"])
        A("| 항목 | 값 |")
        A("| :--- | :--- |")
        for k, v in proto.items():
            A(f"| `{k}` | `{v}` |")
        A("")
        A("**LightGBM 하이퍼파라미터 (전 Phase 동일 고정, Phase 4 설정 기준)**\n")
        A("```json")
        A(json.dumps(json.loads(rows[0]["lgbm_params"]), indent=2))
        A("```\n")
    A("추가 고정 사항:\n")
    A(f"* TE 평활 계수 `m` = {TE_SMOOTHING_M} — 기존 Phase 6 만 25.0 이었다.")
    A(f"* TE 내부 K-fold `inner_splits` = {TE_INNER_SPLITS} — 기존 5개 스크립트는 0 에 해당한다.")
    A("* 범주 vocabulary 는 전 Phase 라벨+미라벨 합쳐 fit — 기존에는 Phase 5 만 그랬다.")
    A("* `scale_pos_weight` 는 전 Phase 미적용(es_protocol_final.md 작업 5, 기본값 1.0).")
    A(f"* `n_estimators` 는 ES 대신 nested grid(`{N_ESTIMATORS_GRID}`)로 fold마다 선택한다 ")
    A("  (es_protocol_final.md 조건 C — TE-inner 리키지 차단, `run_fold_nested_grid()`).")
    A("* Phase 7-Ex 는 원천 데이터 결함으로 재측정 대상에서 제외.\n")
    A("---\n")

    A("## 2. 구프로토콜 대 신프로토콜 대조표\n")
    A("`macro_f1_nested` 가 보고용 값이다. `naive` 는 기존 6개 스크립트 방식(전체 OOF ")
    A("최댓값)으로, winner's curse 를 포함하므로 성능 수치로 인용해서는 안 된다.\n")
    A("| Phase | LogLoss 구 | LogLoss 신 | AUC 구 | AUC 신 | F1 구 | **F1 신(nested)** | F1 신(naive) | naive−보고 F1 | TP 구 | TP 신 |")
    A("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        old = OLD_PROTOCOL.get(r["phase_key"])
        if old:
            A(
                f"| {r['label']} | {old['logloss']:.4f} | {_f(r, 'log_loss')} "
                f"| {old['auc']:.4f} | {_f(r, 'roc_auc')} "
                f"| {old['f1']:.4f} | **{_f(r, 'macro_f1_nested')}** "
                f"| {_f(r, 'naive_macro_f1')} | +{_f(r, 'threshold_optimism')} "
                f"| {old['tp']:,} | {int(r['tp']):,} |"
            )
        else:
            A(
                f"| {r['label']} | — | {_f(r, 'log_loss')} | — | {_f(r, 'roc_auc')} "
                f"| — | **{_f(r, 'macro_f1_nested')}** | {_f(r, 'naive_macro_f1')} "
                f"| +{_f(r, 'threshold_optimism')} | — | {int(r['tp']):,} |"
            )
    A("")
    A("> 구프로토콜 Phase 1 의 F1 은 **fold 평균** 집계이고 임계값 탐색도 없었다(0.50 고정). ")
    A("> 신프로토콜 값과는 애초에 같은 종류의 양이 아니므로 증감 해석에 쓸 수 없다.\n")

    A("### 2-1. 실행 세부\n")
    A("| Phase | 피처 수 | 배포 임계값 | fold별 임계값 | fold별 선택 n_estimators(nested grid) | pseudo 선별 | 소요(초) |")
    A("| :--- | ---: | ---: | :--- | :--- | :--- | ---: |")
    for r in rows:
        pc = json.loads(r["pseudo_counts"] or "[]")
        pc_s = "—" if not pc else ", ".join(f"{c:,}" for c in pc)
        A(
            f"| {r['label']} | {r['n_features']} | {_f(r, 'deployment_threshold', 2)} "
            f"| {r['per_fold_thresholds']} | {r.get('selected_n_estimators', '—')} | {pc_s} "
            f"| {_f(r, 'elapsed_sec', 1)} |"
        )
    A("")
    A("---\n")
    A(_interpretation(done))

    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(L), encoding="utf-8")


SIG = 0.002  # AUDIT §3-10: 0.002 이하 차이는 실질 개선으로 보지 않는다


def _interpretation(done: dict[str, dict]) -> str:
    """단일 실행의 기술 통계만 보고한다. 통계적 유의성/인과 결론을 자동 생성하지 않는다."""
    lines = [
        "## 3. 해석 범위", "",
        "트리 수와 임계값은 각 outer-train 내부 holdout에서 선택했다.",
        "혼동행렬과 recall도 각 fold의 사전 선택 임계값으로 계산한다.",
        "배포 임계값은 fold별 임계값의 중앙값이며 전체 데이터 재학습 시 별도 검증이 필요하다.",
        "naive F1과 보고 F1의 차이는 진단값이며 편향의 인과 추정치가 아니다.",
        "단일 시드 결과나 0.002 기준만으로 유의성/동률을 판정하지 않는다.",
        "후속 비교는 동일 시드/분할의 Phase 간 차이를 짝지어 보고해야 한다.",
        "구프로토콜 수치는 역사적 기록이며 새 수치와의 차이를 피처 효과로 귀속하지 않는다.",
        "", "| Phase | Macro F1 | LogLoss | AUC |", "| :--- | ---: | ---: | ---: |",
    ]
    rows = [r for r in done.values() if r["phase_key"] not in {"P5_leaky", "P4_nospw"}]
    for r in sorted(rows, key=lambda r: (-float(r["macro_f1_nested"]), float(r["log_loss"]))):
        lines.append(f"| {r['label']} | {_f(r, 'macro_f1_nested')} | {_f(r, 'log_loss')} | {_f(r, 'roc_auc')} |")
    lines.extend(["", "P5_leaky는 outer-valid 라벨의 간접 유입을 의도적으로 유지한 진단 대조군이다.",
                  "P4_nospw는 현재 P4와 동일한 호환용 별칭이므로 순위에서 제외했다."])
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
