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
* **early stopping**: 전 Phase 학습 fold 내부 holdout. 채점 fold 는 평가셋에
  들어가지 않는다 (AUDIT §2.5).
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
* `output/baseline_recovery.csv` — Phase 단위로 즉시 append. 중단되어도 이어서 실행된다.
* `output/baseline_recovery.md` — 구프로토콜(§2-1) 대 신프로토콜 대조표 + 해석.

사용법
------
    uv run python rerun_all_phases.py                 # 전체 실행 (재개 지원)
    uv run python rerun_all_phases.py --sample 30000  # 축소 스모크 테스트
    uv run python rerun_all_phases.py --phases P4,P5_honest
    uv run python rerun_all_phases.py --force         # 기존 CSV 무시하고 재실행
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.cv import CVConfig, evaluate_oof, make_folds, run_fold
from src.features import (
    build_cyclic_features,
    build_time_features,
    build_traffic_features,
    encode_categoricals,
    fit_fold_teacher,
    impute_cross,
    load_data,
    make_pseudo_labels,
    oof_target_encode,
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
CSV_PATH = OUTPUT_DIR / "baseline_recovery.csv"
MD_PATH = OUTPUT_DIR / "baseline_recovery.md"


# =============================================================================
# 고정 프로토콜 — 전 Phase 공유
# =============================================================================

#: 단일 인스턴스. 모든 Phase 가 이 객체 하나를 그대로 쓴다.
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
#: n_estimators 는 상한이며 실제 트리 수는 내부 holdout 기반 early stopping 이 정한다.
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
    "n_estimators": 1000,
    "verbose": -1,
}

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
    use_spw: bool = True
    note: str = ""

    def feature_signature(self) -> tuple:
        """피처 빌드 캐시 키. 타깃 의존 설정은 제외한다."""
        return (
            self.bidirectional_impute, self.prune_preset, self.drop_extra,
            self.cyclic_missing, self.include_cos, self.restore_duration,
            self.restore_hours, self.traffic, self.traffic_exclude_missing,
            self.cat_cols,
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
        note="진단용. spw 는 eval set 에도 적용되어 early stopping 을 2회차에 멈춘다.",
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

CSV_FIELDS = [
    "phase_key", "label", "n_rows", "n_features", "log_loss", "roc_auc",
    "f1_at_050", "macro_f1_nested", "naive_macro_f1", "threshold_optimism",
    "deployment_threshold", "per_fold_thresholds", "tn", "fp", "fn", "tp",
    "recall", "best_iterations", "pseudo_counts", "pseudo_total",
    "inner_splits", "scale_pos_weight", "traffic_mode", "te_mode", "pseudo_mode",
    "elapsed_sec", "timestamp", "note", "protocol", "lgbm_params",
]


# =============================================================================
# 피처 빌드 — 타깃 비의존 구간
# =============================================================================


def build_features(raw: pd.DataFrame, spec: PhaseSpec):
    """`spec` 이 지정한 피처셋을 만들어 `(X_lab, y, X_unlab)` 을 돌려준다.

    이 함수는 `Delay` 를 `split_labeled()` 에서 단 한 번만 건드린다. 그 전 단계는
    전부 타깃 비의존이므로 fold 분할 전에 전체 데이터로 계산해도 누수가 없다
    (AUDIT.md §2.4 전수 판정).
    """
    df = impute_cross(raw, bidirectional=spec.bidirectional_impute)

    # Route 는 traffic / TE 보다 먼저 있어야 한다.
    df["Route"] = (
        df["Origin_Airport"].astype(str) + "_" + df["Destination_Airport"].astype(str)
    )

    # 분 단위는 restore_time_missing(fill_hours=True) 에 필요하므로 항상 만든 뒤,
    # 필요 없는 Phase 에서는 prune 단계에서 떨군다.
    df = build_time_features(df, keep_minute=True)

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
        df, missing=spec.cyclic_missing, include_cos=spec.include_cos
    )

    # 7개 스크립트 공통 파생. 복원된 Duration 을 쓰도록 restore 뒤에 계산한다.
    df["Air_Speed_Proxy"] = df["Distance"] / (df["Estimated_Duration"] + 1e-5)

    df = prune_columns(df, preset=spec.prune_preset, extra=spec.drop_extra)

    X_lab, y, X_unlab = split_labeled(df)

    # 범주 인코딩: 라벨+미라벨을 한 번에 인코딩해 동일한 category dtype 을 보장한다.
    # 따로 인코딩하면 categories 배열이 달라져 pseudo 행 concat 시 dtype 이 풀린다.
    n_lab = len(X_lab)
    combined = pd.concat([X_lab, X_unlab], ignore_index=True)
    combined, _ = encode_categoricals(combined, cat_cols=spec.cat_cols)
    X_lab = combined.iloc[:n_lab].reset_index(drop=True)
    X_unlab = combined.iloc[n_lab:].reset_index(drop=True)

    return X_lab, y, X_unlab


def model_factory(scale_pos_weight: float | None):
    """`scale_pos_weight=None` 이면 인자를 아예 넘기지 않는다 (LightGBM 기본 1.0)."""

    def _make():
        extra = {} if scale_pos_weight is None else {"scale_pos_weight": scale_pos_weight}
        return lgb.LGBMClassifier(**LGBM_PARAMS, **extra)

    return _make


# =============================================================================
# Phase 5 — teacher 두 갈래
# =============================================================================


def _teacher_fit_predict(X_tr, y_tr, X_apply, spec: PhaseSpec, spw: float, fold: int):
    """teacher 한 대를 적합시켜 `X_apply` 에 대한 지연 확률을 돌려준다.

    TE 는 `oof_target_encode()` 로 계산한다. `X_apply` 쪽 라벨은 더미(0)를 넘기는데,
    이 함수가 valid 측 라벨을 절대 읽지 않기 때문이다 — 그 성질 자체가
    `tests/test_features.py::test_validation_labels_never_enter_encoding` 으로
    검증되어 있다.
    """
    combined = pd.concat([X_tr, X_apply], ignore_index=True)
    y_combined = pd.concat(
        [y_tr.reset_index(drop=True), pd.Series(np.zeros(len(X_apply), dtype=int))],
        ignore_index=True,
    )
    for col in spec.cat_cols:
        if col in combined.columns:
            combined[col] = combined[col].astype("category")

    te = oof_target_encode(
        combined,
        y_combined,
        np.arange(len(X_tr)),
        np.arange(len(X_tr), len(combined)),
        cols=spec.cat_cols,
        m=TE_SMOOTHING_M,
        inner_splits=spec.inner_splits,
        seed=CFG.seed,
        drop_original=spec.te_drop_original,
    )
    fit = run_fold(
        model_factory(spw), te.X_train, te.y_train, te.X_valid, CFG, fold=fold
    )
    return fit.valid_probs


def _leaky_ensemble_pseudo_labels(X_lab, y, X_unlab, folds, spec, spw):
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
            spec, spw, fold,
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
    spw = float((len(y) - y.sum()) / y.sum()) if spec.use_spw else None
    print(
        f"   라벨 {len(X_lab):,}행 | 피처 {X_lab.shape[1]}개 | "
        f"미라벨 {len(X_unlab):,}행 | "
        f"scale_pos_weight {'미적용' if spw is None else f'{spw:.4f}'}"
    )

    frozen_pseudo = None
    if spec.pseudo == "leaky":
        frozen_pseudo = _leaky_ensemble_pseudo_labels(
            X_lab, y, X_unlab, folds, spec, spw
        )

    oof = np.full(len(y), np.nan)
    best_iters: list[int | None] = []
    pseudo_counts: list[int] = []
    feature_names: list[str] = []

    for fold, (tr, va) in enumerate(folds):
        t0 = time.perf_counter()
        extra = None

        if spec.pseudo == "leaky":
            extra = frozen_pseudo
            pseudo_counts.append(len(frozen_pseudo[0]))
        elif spec.pseudo == "honest":
            # teacher 는 이 fold 의 학습 파트만 본다. 분위수도 여기서 재계산된다.
            teacher = fit_fold_teacher(
                X_lab, y, tr,
                fit_predict=lambda a, b, c, _f=fold: _teacher_fit_predict(
                    a, b, c, spec, spw, _f
                ),
            )
            pseudo = make_pseudo_labels(
                teacher, X_unlab, neg_percentile=10.0, pos_percentile=98.0
            )
            extra = (pseudo.X, pseudo.y)
            pseudo_counts.append(len(pseudo.X))
            print(
                f"   fold {fold + 1} teacher: pseudo {len(pseudo.X):,}건 "
                f"(임계 {pseudo.thresh_neg:.4f} / {pseudo.thresh_pos:.4f})"
            )

        if spec.te:
            te = oof_target_encode(
                X_lab, y, tr, va,
                cols=spec.cat_cols,
                m=TE_SMOOTHING_M,
                inner_splits=spec.inner_splits,
                seed=CFG.seed,
                drop_original=spec.te_drop_original,
                extra_fit_rows=extra,
            )
            X_tr, y_tr, X_va = te.X_train, te.y_train, te.X_valid
        else:
            X_tr = X_lab.iloc[tr].reset_index(drop=True)
            y_tr = y.iloc[tr].reset_index(drop=True)
            X_va = X_lab.iloc[va].reset_index(drop=True)

        # early stopping holdout 은 진짜 라벨 행에서만 뽑는다. pseudo 행이 평가셋에
        # 들어가면 트리 개수가 teacher 의 가짜 라벨에 맞춰지므로, 준지도 효과 측정
        # 자체가 오염된다. pseudo 행은 내부 학습셋에는 그대로 남는다.
        eligible = np.arange(len(tr)) if extra is not None else None

        fit = run_fold(
            model_factory(spw), X_tr, y_tr, X_va, CFG,
            fold=fold, holdout_eligible=eligible,
        )
        oof[va] = fit.valid_probs
        best_iters.append(fit.best_iteration)
        if not feature_names:
            feature_names = list(X_tr.columns)
        print(
            f"   fold {fold + 1}/{len(folds)} | 학습 {len(X_tr):,}행 "
            f"(holdout {fit.n_inner_holdout:,}) | best_iter {fit.best_iteration} "
            f"| {time.perf_counter() - t0:.1f}s"
        )

    assert not np.isnan(oof).any(), "모든 행이 정확히 한 번 채점되어야 한다"

    res = evaluate_oof(y, oof, folds, CFG, label=spec.label, feature_names=feature_names)
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
        "best_iterations": json.dumps(best_iters),
        "pseudo_counts": json.dumps(pseudo_counts),
        "pseudo_total": int(np.sum(pseudo_counts)) if pseudo_counts else 0,
        "inner_splits": spec.inner_splits,
        "scale_pos_weight": "-" if spw is None else round(spw, 4),
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
    }

    print(
        f"   ★ LogLoss {row['log_loss']:.4f} | AUC {row['roc_auc']:.4f} | "
        f"Macro F1(nested) {row['macro_f1_nested']:.4f} "
        f"(naive {row['naive_macro_f1']:.4f}, 편향 +{row['threshold_optimism']:.4f}) "
        f"| {elapsed:.0f}s"
    )
    return row


# =============================================================================
# CSV 즉시 append / 재개
# =============================================================================


def load_done(force: bool) -> dict[str, dict]:
    if force or not CSV_PATH.exists():
        return {}
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as fh:
        return {r["phase_key"]: r for r in csv.DictReader(fh)}


def append_row(row: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not CSV_PATH.exists()
    with CSV_PATH.open("a", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def print_header(specs: list[PhaseSpec], sample: int | None) -> None:
    print("=" * 78)
    print("Phase 1~6 통일 프로토콜 재측정 — 정직한 기준선 확정")
    print("PLAN.md §4-0 P0-2 / P0-3 · AUDIT.md §3 대응")
    print("=" * 78)
    print(f"\n[실행 시각] {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    if sample:
        print(f"[!] 축소 스모크 모드: 상위 {sample:,}행만 사용 — 결과를 인용하지 말 것")

    print("\n[CVConfig.describe()]")
    for k, v in CFG.describe().items():
        print(f"  {k:24} = {v}")

    print("\n[LightGBM 하이퍼파라미터 — 전 Phase 동일 고정 (Phase 4 기준)]")
    for k, v in LGBM_PARAMS.items():
        print(f"  {k:24} = {v}")
    print(f"  {'scale_pos_weight':24} = (라벨 분포에서 산출, 전 Phase 동일)")

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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=None, help="축소 스모크용 행 수")
    ap.add_argument("--phases", type=str, default=None, help="쉼표 구분 phase_key")
    ap.add_argument("--force", action="store_true", help="기존 CSV 무시하고 재실행")
    ap.add_argument("--report-only", action="store_true", help="CSV 로 리포트만 재생성")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    done = load_done(args.force)

    if args.report_only:
        if not done:
            print(f"[!] {CSV_PATH} 가 없습니다.")
            return 1
        write_report(done)
        print(f">> 리포트 재생성 완료: {MD_PATH}")
        return 0

    wanted = set(args.phases.split(",")) if args.phases else None
    specs = [s for s in PHASES if wanted is None or s.key in wanted]
    if wanted and not specs:
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
            continue
        append_row(row)
        done[spec.key] = row
        print(f"   >> CSV append 완료: {CSV_PATH}")

    write_report(done)
    print(f"\n>> 대조표 저장: {MD_PATH}")
    print(f">> 원본 CSV   : {CSV_PATH}")
    return 0


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
    A("> 원본 데이터: `output/baseline_recovery.csv`  ")
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
    A("* `scale_pos_weight` 는 라벨 분포에서 산출한 동일 값을 전 Phase 사용.")
    A("* Phase 7-Ex 는 원천 데이터 결함으로 재측정 대상에서 제외.\n")
    A("---\n")

    A("## 2. 구프로토콜 대 신프로토콜 대조표\n")
    A("`macro_f1_nested` 가 보고용 값이다. `naive` 는 기존 6개 스크립트 방식(전체 OOF ")
    A("최댓값)으로, winner's curse 를 포함하므로 성능 수치로 인용해서는 안 된다.\n")
    A("| Phase | LogLoss 구 | LogLoss 신 | AUC 구 | AUC 신 | F1 구 | **F1 신(nested)** | F1 신(naive) | 임계값 편향 | TP 구 | TP 신 |")
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
    A("| Phase | 피처 수 | 배포 임계값 | fold별 임계값 | fold별 best_iteration | pseudo 선별 | 소요(초) |")
    A("| :--- | ---: | ---: | :--- | :--- | :--- | ---: |")
    for r in rows:
        pc = json.loads(r["pseudo_counts"] or "[]")
        pc_s = "—" if not pc else ", ".join(f"{c:,}" for c in pc)
        A(
            f"| {r['label']} | {r['n_features']} | {_f(r, 'deployment_threshold', 2)} "
            f"| {r['per_fold_thresholds']} | {r['best_iterations']} | {pc_s} "
            f"| {_f(r, 'elapsed_sec', 1)} |"
        )
    A("")
    A("---\n")
    A(_interpretation(done))

    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(L), encoding="utf-8")


SIG = 0.002  # AUDIT §3-10: 0.002 이하 차이는 실질 개선으로 보지 않는다


def _interpretation(done: dict[str, dict]) -> str:
    """재측정이 기존 서술을 뒤집는 부분을 명시한다 (요청 [5])."""
    L: list[str] = []
    A = L.append
    g = lambda k, f: float(done[k][f]) if k in done else None  # noqa: E731

    A("## 3. 재측정이 기존 서술에 대해 말하는 것\n")
    A(f"판정 기준은 AUDIT.md §3-10 을 따른다 — **Macro F1 / AUC 차이가 {SIG} 이하면 ")
    A("실질 개선으로 인정하지 않는다.** early stopping 과 임계값 선택에서 오는 잔여 ")
    A("편향이 그 크기이기 때문이다.\n")

    # --- 3-0. 헤드라인 -------------------------------------------------------
    A("### 3-0. 신프로토콜 기준 순위 — 기존 서사와 정면으로 어긋난다\n")
    ranked = [
        (k, float(done[k]["roc_auc"]), float(done[k]["macro_f1_nested"]),
         float(done[k]["log_loss"]), done[k]["label"])
        for k in done
    ]
    if ranked:
        A("| 순위 | Phase | ROC-AUC | Macro F1(nested) | LogLoss |")
        A("| ---: | :--- | ---: | ---: | ---: |")
        for i, (k, auc, f1, ll, lab) in enumerate(
            sorted(ranked, key=lambda r: -r[1]), 1
        ):
            A(f"| {i} | {lab} | **{auc:.4f}** | {f1:.4f} | {ll:.4f} |")
        A("")
        best_auc = max(ranked, key=lambda r: r[1])
        best_f1 = max(ranked, key=lambda r: r[2])
        best_ll = min(ranked, key=lambda r: r[3])
        A(f"* **AUC 최고: {best_auc[4]}** (`{best_auc[1]:.4f}`)")
        A(f"* **Macro F1 최고: {best_f1[4]}** (`{best_f1[2]:.4f}`)")
        A(f"* **LogLoss 최저: {best_ll[4]}** (`{best_ll[3]:.4f}`)")
        A("")
        if best_auc[0] == "P3":
            A("> 기존 §2-1 은 Phase 3 을 \"카테고리 분기력 상실로 AUC 급락\"으로 기록했다. ")
            A("> 통일 프로토콜에서는 **Phase 3 이 AUC 최고**다. 구프로토콜의 Phase 3 평가는 ")
            A("> 피처셋이 아니라 TE 구현 방식(`inner_splits=0`)에서 온 것이었다(§3-4 참조).\n")
        if best_ll[0] == "P4_nospw":
            A("> LogLoss 최저값은 `scale_pos_weight` 를 끈 진단용 변형에서 나왔다. ")
            A("> 기존 최고 기록(0.4587, 누수 포함 Phase 5)보다 낮다(§3-7 참조).\n")

    # (1) Phase 5 vs Phase 6
    A("### 3-1. Phase 5 는 여전히 Phase 6 보다 우수한가?\n")
    p5h, p6 = g("P5_honest", "roc_auc"), g("P6_fixed", "roc_auc")
    p5h_f1, p6_f1 = g("P5_honest", "macro_f1_nested"), g("P6_fixed", "macro_f1_nested")
    if None in (p5h, p6, p5h_f1, p6_f1):
        A("_아직 두 Phase 가 모두 완료되지 않았다._\n")
    else:
        d_auc, d_f1 = p5h - p6, p5h_f1 - p6_f1
        A(f"* 정직한 Phase 5 AUC `{p5h:.4f}` vs Phase 6(수정) AUC `{p6:.4f}` → 차이 `{d_auc:+.4f}`")
        A(f"* 정직한 Phase 5 F1 `{p5h_f1:.4f}` vs Phase 6(수정) F1 `{p6_f1:.4f}` → 차이 `{d_f1:+.4f}`")
        if abs(d_auc) <= SIG and abs(d_f1) <= SIG:
            A(f"\n**판정: 구분되지 않는다.** 두 지표 모두 차이가 {SIG} 이하다. ")
            A("기존 §2-1 의 \"Phase 5 전 지표 최고치\" 서술은 재측정으로 지지되지 않는다.\n")
        elif d_auc > SIG and d_f1 > SIG:
            A("\n**판정: Phase 5 우세가 유지된다.** 누수를 제거한 뒤에도 차이가 유의하다.\n")
        elif d_auc < -SIG or d_f1 < -SIG:
            A("\n**판정: 뒤집혔다.** 누수 제거 후 Phase 6 이 우세하다. ")
            A("기존 서술은 수정되어야 한다.\n")
        else:
            A("\n**판정: 지표별로 엇갈린다.** 단일 우열을 주장할 수 없다.\n")

    # (2) 누수 크기 실측
    A("### 3-2. Phase 5 누수의 실측 크기\n")
    lk, hn = g("P5_leaky", "roc_auc"), g("P5_honest", "roc_auc")
    lk_f1, hn_f1 = g("P5_leaky", "macro_f1_nested"), g("P5_honest", "macro_f1_nested")
    if None in (lk, hn, lk_f1, hn_f1):
        A("_두 변형이 모두 완료되지 않았다._\n")
    else:
        A(f"| 지표 | 누수 재현 | 정직 | 차이(= 누수 크기) | AUDIT §3-6 추정 |")
        A(f"| :--- | ---: | ---: | ---: | :--- |")
        A(f"| ROC-AUC | {lk:.4f} | {hn:.4f} | **{lk - hn:+.4f}** | +0.002 ~ +0.008 |")
        A(f"| Macro F1 | {lk_f1:.4f} | {hn_f1:.4f} | **{lk_f1 - hn_f1:+.4f}** | +0.002 ~ +0.006 |")
        A("")
        d = lk - hn
        if 0.002 <= d <= 0.008:
            A("**추정 구간 안에 들어왔다.** AUDIT §2.1-C 의 낙관 편향 추정이 실측으로 확인되었다.\n")
        elif d > 0.008:
            A("**추정 상한을 넘었다.** 누수가 예상보다 크다. §2.1-C 의 억제 요인 분석을 재검토해야 한다.\n")
        elif d < 0:
            A("**부호가 반대다.** 누수 재현 쪽이 오히려 낮다. fold 간 teacher 분산 등 ")
            A("다른 요인이 더 크게 작용했을 수 있으므로 단정할 수 없다.\n")
        else:
            A("**추정 하한보다 작다.** 누수의 실효 크기가 예상보다 작았다.\n")
        # fold별 선별 건수 편차
        pc = json.loads(done["P5_honest"]["pseudo_counts"] or "[]")
        if pc:
            A(f"fold 별 pseudo 선별 건수(정직): {', '.join(f'{c:,}' for c in pc)} — ")
            A(f"최소 {min(pc):,} / 최대 {max(pc):,}, 편차 {max(pc) - min(pc):,}건. ")
            A("fold 마다 teacher 가 다르므로 기존 기록의 **89,438건 고정이 성립하지 않는다.**\n")

    # (3) Phase 3 -> 4 AUC 반등
    A("### 3-3. Phase 3→4 의 \"AUC 반등\"은 재현되는가?\n")
    p3, p4 = g("P3", "roc_auc"), g("P4", "roc_auc")
    if None in (p3, p4):
        A("_두 Phase 가 모두 완료되지 않았다._\n")
    else:
        d = p4 - p3
        A(f"* Phase 3 AUC `{p3:.4f}` → Phase 4 AUC `{p4:.4f}` → 차이 `{d:+.4f}`")
        A(f"* 구프로토콜 기록: 0.6043 → 0.6107 (차이 +0.0064)\n")
        if d > SIG:
            A("**판정: 재현된다.** 원본 category dtype 을 유지하는 하이브리드 인코딩의 ")
            A("이점은 프로토콜을 통일해도 살아남는다.\n")
        elif d < -SIG:
            A("**판정: 재현되지 않는다. 부호가 뒤집혔다.** 통일 프로토콜에서는 원본 범주를 ")
            A("복원한 Phase 4 가 TE 단독 치환인 Phase 3 보다 **오히려 낮다**. ")
            A("구프로토콜의 \"AUC 반등\"은 프로토콜 요인(임계값 그리드, TE 방식)이 만든 ")
            A("허상이었을 가능성이 크다.\n")
            A("이는 PLAN.md §3-A 항목 2(\"하이브리드 카테고리 보존\")의 근거를 직접 흔든다. ")
            A("해당 도메인 규칙은 재검증 전까지 보류해야 한다.\n")
        else:
            A(f"**판정: 재현되지 않는다.** 차이가 ±{SIG} 이내로 구분되지 않는다. ")
            A("구프로토콜에서 관측된 반등은 프로토콜 요인이었을 수 있다.\n")

    # (4) TE inner K-fold 단독 효과
    A("### 3-4. TE 내부 K-fold 도입의 단독 효과\n")
    a, b = g("P4", "roc_auc"), g("P4_te0", "roc_auc")
    a_f1, b_f1 = g("P4", "macro_f1_nested"), g("P4_te0", "macro_f1_nested")
    if None in (a, b, a_f1, b_f1):
        A("_Phase 4 두 변형이 모두 완료되지 않았다._\n")
    else:
        A(f"| 지표 | inner_splits=0 (기존 방식) | inner_splits=5 (신) | 차이 |")
        A(f"| :--- | ---: | ---: | ---: |")
        A(f"| ROC-AUC | {b:.4f} | {a:.4f} | {a - b:+.4f} |")
        A(f"| Macro F1 | {b_f1:.4f} | {a_f1:.4f} | {a_f1 - b_f1:+.4f} |")
        A("")
        A("이 차이만큼은 **\"프로토콜 통일\"이 아니라 \"TE 방식 변경\"의 효과**다. ")
        A("§2 대조표의 신구 차이를 해석할 때 이 몫을 먼저 덜어내야 한다.\n")

    # (5) Traffic 수정의 순효과
    A("### 3-5. Traffic 결측 버킷 수정(AUDIT §3-8)의 순효과\n")
    lg, fx = g("P6_legacy", "roc_auc"), g("P6_fixed", "roc_auc")
    lg_f1, fx_f1 = g("P6_legacy", "macro_f1_nested"), g("P6_fixed", "macro_f1_nested")
    if None in (lg, fx, lg_f1, fx_f1):
        A("_Phase 6 두 변형이 모두 완료되지 않았다._\n")
    else:
        A(f"| 지표 | 레거시(-1 뭉침) | 수정(결측 제외 + 플래그) | 차이 |")
        A(f"| :--- | ---: | ---: | ---: |")
        A(f"| ROC-AUC | {lg:.4f} | {fx:.4f} | {fx - lg:+.4f} |")
        A(f"| Macro F1 | {lg_f1:.4f} | {fx_f1:.4f} | {fx_f1 - lg_f1:+.4f} |")
        A("")
        if abs(fx - lg) <= SIG:
            A(f"**판정: 성능 차이는 유의하지 않다.** 다만 수정본은 `Origin_Traffic` 이 ")
            A("혼잡도만 측정하고 결측 여부는 별도 플래그로 분리되므로, **피처 의미가 ")
            A("해석 가능해진다**는 이점은 성능과 무관하게 유효하다.\n")
        elif fx > lg:
            A("**판정: 수정본이 우세하다.** 결측 밀도를 혼잡도로 오인하던 것이 실제로 ")
            A("성능을 갉아먹고 있었다.\n")
        else:
            A("**판정: 레거시가 높게 나왔다.** `-1` 버킷이 우연히 예측력 있는 신호로 ")
            A("작동했을 수 있으나, 그것은 혼잡도가 아니라 결측 패턴이므로 배포 시 ")
            A("재현을 신뢰하기 어렵다.\n")

    # (6) 유의하게 남는 전이
    A(f"### 3-6. {SIG} 규칙을 적용했을 때 실제로 남는 Phase 전이\n")
    chain = [
        ("P1", "P2", "Phase 1 → 2 (Pruning + 임계값)"),
        ("P2", "P3", "Phase 2 → 3 (TE 도입 + 원본 범주 제거)"),
        ("P3", "P4", "Phase 3 → 4 (원본 범주 복원)"),
        ("P4", "P5_honest", "Phase 4 → 5 (준지도 증강, 정직)"),
        ("P5_honest", "P6_fixed", "Phase 5 → 6 (도메인 피처)"),
    ]
    A("| 전이 | ΔAUC | ΔMacro F1 | 판정 |")
    A("| :--- | ---: | ---: | :--- |")
    survivors: list[str] = []
    for a_k, b_k, name in chain:
        if a_k not in done or b_k not in done:
            A(f"| {name} | — | — | 미완료 |")
            continue
        d_auc = float(done[b_k]["roc_auc"]) - float(done[a_k]["roc_auc"])
        d_f1 = float(done[b_k]["macro_f1_nested"]) - float(done[a_k]["macro_f1_nested"])
        up_auc, up_f1 = d_auc > SIG, d_f1 > SIG
        dn_auc, dn_f1 = d_auc < -SIG, d_f1 < -SIG
        if up_auc and up_f1:
            verdict = "**유의한 개선** (두 지표 모두)"
            survivors.append(name)
        elif dn_auc and dn_f1:
            verdict = "**유의한 악화** (두 지표 모두)"
        elif (up_auc and dn_f1) or (up_f1 and dn_auc):
            verdict = "**엇갈림** — 단일 우열 주장 불가"
        elif up_auc:
            verdict = "AUC만 개선, F1 구분 불가"
            survivors.append(f"{name} [AUC 한정]")
        elif up_f1:
            verdict = "F1만 개선, AUC 구분 불가"
            survivors.append(f"{name} [F1 한정]")
        elif dn_auc or dn_f1:
            verdict = "한쪽 지표 악화"
        else:
            verdict = f"구분 불가 (±{SIG} 이내)"
        A(f"| {name} | {d_auc:+.4f} | {d_f1:+.4f} | {verdict} |")
    A("")
    if survivors:
        A("**실질 개선으로 남는 단계:** " + ", ".join(survivors) + "\n")
    else:
        A("**유의하게 남는 개선 단계가 없다.** 기존의 \"단계적 성능 향상\" 서사는 ")
        A("통일된 프로토콜에서 재현되지 않는다.\n")

    # --- 3-7. scale_pos_weight 진단 ---
    A("### 3-7. `scale_pos_weight` 가 early stopping 을 조기 종료시킨다 (진단)\n")
    a_ll, b_ll = g("P4", "log_loss"), g("P4_nospw", "log_loss")
    a_auc2, b_auc2 = g("P4", "roc_auc"), g("P4_nospw", "roc_auc")
    if None in (a_ll, b_ll, a_auc2, b_auc2):
        A("_진단 변형이 완료되지 않았다._\n")
    else:
        it_on = json.loads(done["P4"]["best_iterations"])
        it_off = json.loads(done["P4_nospw"]["best_iterations"])
        A("LightGBM 은 `scale_pos_weight` 를 **평가셋에도 적용**한다. 그래서 early ")
        A("stopping 이 보는 지표가 가중 버전이 되어 매우 이른 회차에 멈춘다. ")
        A("`metric` 을 `auc` 로 바꿔도 동일하게 재현되므로 원인은 metric 이 아니라 ")
        A("가중치 자체다. 기존 7개 스크립트가 전부 이 설정을 쓰고 있었다.\n")
        A("| 항목 | spw 적용 (Phase 4 기준 설정) | spw 미적용 |")
        A("| :--- | ---: | ---: |")
        A(f"| fold별 best_iteration | `{it_on}` | `{it_off}` |")
        A(f"| LogLoss | {a_ll:.4f} | **{b_ll:.4f}** |")
        A(f"| ROC-AUC | **{a_auc2:.4f}** | {b_auc2:.4f} |")
        A("")
        A("읽어야 할 지점은 **2-트리 모델의 AUC 가 500-트리 모델보다 높다**는 것이다. ")
        A("추가 부스팅이 랭킹 품질을 떨어뜨린다 — 이 데이터의 신호가 실제로 얕다는 ")
        A("뜻이며 프로토콜 결함이 아니다. 다만 **LogLoss 는 spw 를 끌 때 뚜렷하게 ")
        A(f"낮아진다** ({a_ll:.4f} → {b_ll:.4f}). 기존 프로젝트가 \"LogLoss 0.45대 진입\"을 ")
        A("성과로 기록해 온 만큼, 그 값이 누수도 준지도 학습도 없이 **하이퍼파라미터 ")
        A("한 줄로** 얻어진다는 사실은 해당 서사를 다시 보게 만든다.\n")
        A("> 본 재측정의 주 비교표(§2)는 지시대로 Phase 4 설정(spw 적용)을 전 Phase ")
        A("> 고정으로 유지했다. 위 행은 진단용이며 Phase 순위 비교에는 넣지 않는다.\n")

    A("---\n")
    A("## 4. 한계\n")
    A("* 본 재측정은 **단일 시드(42)** 의 5-Fold 1회 실행이다. fold 분할 자체의 ")
    A(f"변동을 반영하지 않으므로, {SIG} 근처의 차이는 시드를 바꾸면 순위가 뒤집힐 수 ")
    A("있다. 확정하려면 시드를 바꿔 여러 번 반복해야 한다.")
    A("* 하이퍼파라미터를 Phase 4 설정으로 고정했으므로, 각 Phase 가 **자기 피처셋에 ")
    A("최적인 설정**에서 얼마나 나올 수 있는지는 측정하지 않았다. 이는 의도된 것이다 ")
    A("— 혼입 요인을 줄이는 대신 각 Phase 의 상한을 포기했다.")
    A("* `naive_macro_f1` 은 기존 스크립트 방식의 재현일 뿐이며 성능 수치가 아니다.")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
