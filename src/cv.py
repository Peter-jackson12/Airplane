"""항공편 지연 예측 — 교차 검증 프로토콜 모듈.

`AUDIT.md` §1.3 섹션 E / §2.5 / §2.6, `PLAN.md` §3-9, §3-10, §4-0 P0-2·P0-4 구현.

이 모듈의 목적
--------------
7개 `run_*.py` 스크립트는 `random_state=42` 외에는 평가 프로토콜을 공유하지 않았다.
임계값 그리드가 3갈래(+ v0 시제품까지 4갈래)로 갈리고, Phase 1 만 지표를 fold 평균으로
집계하며, 7개 전부가 채점 fold 를 early stopping 의 `eval_set` 으로 썼다. 그 결과
Phase 간 성능 비교가 통계적으로 무효가 되었다 (AUDIT.md §3).

따라서 프로토콜을 **스크립트가 아니라 모듈이 소유한다.** `CVConfig` 하나를 만들어
넘기면 fold 분할·early stopping·임계값 선택·지표 집계가 모두 그 설정으로 고정되고,
`describe()` 로 결과에 그대로 직렬화된다.

두 가지 편향은 옵션이 아니라 **기본 동작으로 제거**했다.
1. early stopping 은 학습 fold 를 다시 쪼갠 내부 holdout 에서 수행한다.
   채점 fold(`X_valid`)는 `eval_set` 에 들어가지 않는다 (AUDIT.md §2.5).
2. 임계값은 nested 방식으로 고른다. K-1 fold 의 OOF 에서 고르고 남은 fold 에서
   채점한다. 전체 OOF 벡터에서 최댓값을 고르는 방식은 winner's curse 이므로
   점수로 보고하지 않는다 (AUDIT.md §2.6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, NamedTuple, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

__all__ = [
    "CVConfig",
    "make_folds",
    "run_fold",
    "FoldFit",
    "tune_threshold_nested",
    "ThresholdResult",
    "evaluate_oof",
]


# =============================================================================
# 프로토콜 정의
# =============================================================================


@dataclass(frozen=True)
class CVConfig:
    """평가 프로토콜 전체를 하나의 값으로 고정한다.

    Attributes
    ----------
    n_splits
        outer fold 수. 7개 스크립트 공통값 5 를 기본으로 둔다.
    seed
        fold 분할 및 내부 holdout 분할 시드.
    stopping_rounds
        early stopping 인내 라운드. 기존 스크립트는 30/35/40 으로 제각각이었다
        (AUDIT.md §3.1). 여기서 하나로 고정한다.
    threshold_grid
        `(start, stop, step)`. `stop` 은 **포함**된다. 기존 스크립트의
        `0.10~0.69`(60점) / `0.15~0.49`(35점) / `0.10~0.40`(31점) 분기를 끝낸다.
    metric_aggregation
        `"pooled_oof"` 고정. fold 평균 집계(run_baseline.py:238-249)는 Macro-F1 이
        pooled 값과 다른 양이 되므로 허용하지 않는다 (AUDIT.md §3.4).
    inner_holdout_frac
        학습 fold 에서 early stopping 전용 holdout 으로 떼어낼 비율.
    """

    n_splits: int = 5
    shuffle: bool = True
    seed: int = 42
    stopping_rounds: int = 40
    threshold_grid: tuple[float, float, float] = (0.10, 0.70, 0.01)
    metric_aggregation: Literal["pooled_oof"] = "pooled_oof"
    inner_holdout_frac: float = 0.2

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError(f"n_splits 는 2 이상이어야 합니다: {self.n_splits}")
        if self.metric_aggregation != "pooled_oof":
            raise ValueError(
                f"metric_aggregation 은 'pooled_oof' 로 고정입니다: "
                f"{self.metric_aggregation!r}. fold 평균 집계는 Macro-F1 이 pooled 값과 "
                "다른 양이 되어 Phase 간 비교를 깨뜨립니다 (AUDIT.md §3.4)."
            )
        start, stop, step = self.threshold_grid
        if step <= 0:
            raise ValueError(f"threshold_grid 의 step 은 양수여야 합니다: {step}")
        if not 0.0 <= start < stop <= 1.0:
            raise ValueError(
                f"0 <= start({start}) < stop({stop}) <= 1 이어야 합니다."
            )
        if not 0.0 < self.inner_holdout_frac < 0.5:
            raise ValueError(
                f"inner_holdout_frac 은 (0, 0.5) 범위여야 합니다: {self.inner_holdout_frac}"
            )
        if self.stopping_rounds < 1:
            raise ValueError(f"stopping_rounds 는 1 이상이어야 합니다: {self.stopping_rounds}")

    def thresholds(self) -> np.ndarray:
        """`threshold_grid` 를 실제 후보 배열로 전개한다. `stop` 포함."""
        start, stop, step = self.threshold_grid
        n = int(round((stop - start) / step)) + 1
        return np.round(start + step * np.arange(n), 10)

    def describe(self) -> dict[str, Any]:
        """결과 dict 에 그대로 실어 보낼 프로토콜 서술."""
        out = asdict(self)
        out["n_thresholds"] = int(self.thresholds().size)
        out["inner_early_stopping"] = True
        out["nested_threshold"] = True
        return out


def make_folds(
    y: pd.Series, cfg: CVConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedKFold 분할을 리스트로 materialize 한다.

    동일 `cfg` + 동일 길이/순서의 `y` 라면 어느 스크립트에서 호출하든 같은 분할을
    얻는다. 이것이 Phase 간 비교가 성립하기 위한 최소 조건이다.
    """
    y_arr = np.asarray(y)
    skf = StratifiedKFold(
        n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.seed
    )
    return [
        (np.asarray(tr), np.asarray(va))
        for tr, va in skf.split(np.zeros(len(y_arr)), y_arr)
    ]


# =============================================================================
# fold 실행 — early stopping 을 채점 fold에서 분리
# =============================================================================


def _lightgbm_or_none():
    """lightgbm 을 선택적으로 import 한다. 테스트용 대역 추정기를 위해 분리."""
    try:
        import lightgbm as lgb
    except ImportError:  # pragma: no cover - lightgbm 은 선언된 의존성이다
        return None
    return lgb


def _fit_parameters(model: Any) -> set[str]:
    """`model.fit()` 이 받는 인자 이름 집합. 시그니처를 못 읽으면 빈 집합."""
    import inspect

    try:
        return set(inspect.signature(model.fit).parameters)
    except (TypeError, ValueError):  # pragma: no cover - C 확장 추정기 대비
        return set()


class FoldFit(NamedTuple):
    """`run_fold()` 의 반환값."""

    model: Any
    valid_probs: np.ndarray
    best_iteration: int | None
    n_inner_train: int
    n_inner_holdout: int


def run_fold(
    model_factory: Callable[[], Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    cfg: CVConfig,
    *,
    fold: int = 0,
    holdout_eligible: np.ndarray | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> FoldFit:
    """한 fold 를 학습하고 채점 fold 의 예측 확률을 돌려준다.

    누수 차단 (AUDIT.md §2.5)
    -------------------------
    학습 fold `X_train` 을 다시 층화 분할하여 **내부 holdout** 을 만들고, early
    stopping 은 그 holdout 으로만 수행한다. 채점 대상인 `X_valid` 는 평가셋에
    **절대** 들어가지 않으며, 오직 `predict_proba()` 에만 쓰인다. 기존 7개 스크립트는
    `eval_set=[(X_val, y_val)]` 로 트리 개수를 고른 뒤 같은 `X_val` 로 OOF 점수를 매겼다.

    이 보장은 **구조적**이다. holdout 은 `X_train` 에서만 잘라내므로 조건 분기가
    개입할 여지가 없다. (인덱스 겹침으로 런타임 검사를 하려던 초기 구현은 폐기했다 —
    `reset_index()` 를 거친 프레임끼리는 인덱스 라벨이 행을 식별하지 못해 거짓 양성이
    난다. 검증은 `tests/test_features.py::test_valid_rows_never_reach_eval_set` 가
    평가셋에 실제로 들어간 행을 내용으로 대조하여 수행한다.)

    fold 분할 자체의 정합성(train ∩ valid = ∅)은 `oof_target_encode()` 와
    `make_folds()` 가 보증한다.

    준지도 학습 시의 2차 오염 (`holdout_eligible`)
    ----------------------------------------------
    기본 동작은 `X_train` **전체**를 층화 분할해 holdout 을 뽑는다. 그런데 Phase 5
    처럼 `X_train` 에 pseudo-label 행이 섞여 있으면, early stopping 이 teacher 가
    만들어낸 **가짜 라벨에 대해** 트리 개수를 최적화하게 된다. 그러면 준지도 학습의
    효과를 측정하려는 실험 자체가 오염된다 — 성능 지표가 "진짜 라벨을 얼마나 잘
    맞히는가"가 아니라 "teacher 를 얼마나 잘 모방하는가"를 일부 반영하기 때문이다.

    `holdout_eligible` 에 **진짜 라벨 행의 위치 인덱스**를 넘기면 holdout 은 그
    부분집합에서만 추출된다. pseudo 행은 holdout 후보에서 빠지지만 **내부 학습셋에는
    계속 포함**되므로, 증강 효과는 그대로 살리면서 채점 기준만 진짜 라벨로 유지한다.

    Parameters
    ----------
    model_factory
        인자 없이 새 추정기를 만들어 돌려주는 콜러블. fold 간 상태 공유를 막기 위해
        모델 인스턴스가 아니라 팩토리를 받는다.
    holdout_eligible
        early stopping holdout 후보로 쓸 행의 **위치 인덱스**(0-based, `X_train` 기준).
        `None`(기본)이면 전체 행이 후보다. 지정하면 그 부분집합에서만 holdout 을 뽑고,
        나머지 행은 전부 내부 학습셋으로 들어간다.
    fit_kwargs
        `model.fit()` 에 그대로 전달할 추가 인자 (예: `categorical_feature`).
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train({len(X_train)}) 과 y_train({len(y_train)}) 의 길이가 다릅니다."
        )

    n_rows = len(X_train)
    idx = np.arange(n_rows)

    if holdout_eligible is None:
        inner_tr, inner_ho = train_test_split(
            idx,
            test_size=cfg.inner_holdout_frac,
            random_state=cfg.seed + fold,
            stratify=np.asarray(y_train),
        )
    else:
        eligible = np.unique(np.asarray(holdout_eligible, dtype=int))
        if eligible.size == 0:
            raise ValueError("holdout_eligible 이 비어 있습니다.")
        if eligible.min() < 0 or eligible.max() >= n_rows:
            raise ValueError(
                f"holdout_eligible 이 X_train 범위(0..{n_rows - 1})를 벗어납니다: "
                f"[{eligible.min()}, {eligible.max()}]"
            )
        # holdout 크기는 전체 학습 행 기준으로 잡되, 후보 집합을 넘을 수는 없다.
        n_holdout = int(round(n_rows * cfg.inner_holdout_frac))
        n_holdout = max(1, min(n_holdout, eligible.size - 1))

        y_eligible = np.asarray(y_train)[eligible]
        # 후보 안에 한쪽 클래스만 있으면 층화가 불가능하므로 무작위 추출로 내린다.
        stratify = y_eligible if np.unique(y_eligible).size > 1 else None
        kept, inner_ho = train_test_split(
            eligible,
            test_size=n_holdout,
            random_state=cfg.seed + fold,
            stratify=stratify,
        )
        # 후보에서 빠진 행(= pseudo 행 등)은 전부 내부 학습셋으로.
        inner_tr = np.setdiff1d(idx, inner_ho, assume_unique=False)
        del kept

    X_in, y_in = X_train.iloc[inner_tr], y_train.iloc[inner_tr]
    X_ho, y_ho = X_train.iloc[inner_ho], y_train.iloc[inner_ho]

    model = model_factory()
    kwargs = dict(fit_kwargs or {})
    callbacks = list(kwargs.pop("callbacks", []))

    lgb = _lightgbm_or_none()
    if lgb is not None and isinstance(model, lgb.LGBMModel):
        callbacks.append(
            lgb.early_stopping(stopping_rounds=cfg.stopping_rounds, verbose=False)
        )

    # 평가셋 전달 방식은 추정기마다 다르다. lightgbm 4.7 은 eval_set 을 deprecate 하고
    # eval_X / eval_y 를 쓴다. 시그니처를 보고 맞춰 넘기되, 어느 쪽도 받지 않는
    # 추정기라면 early stopping 없이 그대로 적합한다.
    fit_params = _fit_parameters(model)
    if "eval_X" in fit_params:
        kwargs.update(eval_X=X_ho, eval_y=y_ho)
    elif "eval_set" in fit_params:
        kwargs.update(eval_set=[(X_ho, y_ho)])
    if callbacks and "callbacks" in fit_params:
        kwargs["callbacks"] = callbacks

    model.fit(X_in, y_in, **kwargs)

    valid_probs = np.asarray(model.predict_proba(X_valid), dtype=float)[:, 1]
    return FoldFit(
        model=model,
        valid_probs=valid_probs,
        best_iteration=getattr(model, "best_iteration_", None),
        n_inner_train=int(inner_tr.size),
        n_inner_holdout=int(inner_ho.size),
    )


# =============================================================================
# 임계값 — nested 선택
# =============================================================================


class ThresholdResult(NamedTuple):
    """`tune_threshold_nested()` 의 반환값.

    Attributes
    ----------
    nested_f1
        **보고해야 할 값.** fold k 의 임계값을 나머지 K-1 fold 에서 고르고 fold k 에서만
        채점한 뒤 pooled 로 계산한 Macro-F1. winner's curse 가 제거되어 있다.
    deployment_threshold
        실제 배포에 쓸 임계값. fold 별 선택값의 중앙값이다. 전체 OOF 최댓값을 쓰지
        않는 이유는 그것이 곧 winner's curse 의 정의이기 때문이다.
    naive_threshold / naive_f1
        기존 6개 스크립트 방식(전체 OOF 에서 최댓값)의 재현. **비교·진단용이며 성능
        수치로 인용해서는 안 된다.**
    optimism
        `naive_f1 - nested_f1`. AUDIT.md §2.6 이 +0.001~+0.002 로 추정한 낙관 편향의
        실측값이다.
    """

    nested_f1: float
    deployment_threshold: float
    per_fold_thresholds: list[float]
    naive_threshold: float
    naive_f1: float
    optimism: float


def tune_threshold_nested(
    y_true: pd.Series,
    oof_probs: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    cfg: CVConfig,
) -> ThresholdResult:
    """임계값을 nested 방식으로 선택하고 편향 없는 Macro-F1 을 돌려준다.

    절차 (AUDIT.md §2.6)
    --------------------
    fold `k` 마다:
      1. 나머지 K-1 fold 의 검증 행만 모아 임계값 후보를 훑고 최적값을 고른다.
      2. 그 임계값을 fold `k` 의 행에**만** 적용해 예측 라벨을 확정한다.
    모든 fold 의 예측 라벨을 pooled 하여 Macro-F1 을 한 번 계산한다.

    각 행의 임계값이 그 행을 보지 않고 결정되므로, 기존 방식의 낙관 편향이 사라진다.
    """
    y = np.asarray(y_true)
    probs = np.asarray(oof_probs, dtype=float)
    if y.shape != probs.shape:
        raise ValueError(
            f"y_true({y.shape}) 와 oof_probs({probs.shape}) 의 형태가 다릅니다."
        )
    if len(folds) < 2:
        raise ValueError(
            f"nested 임계값 선택에는 fold 가 2개 이상 필요합니다: {len(folds)}"
        )

    grid = cfg.thresholds()
    preds = np.zeros_like(y, dtype=int)
    per_fold: list[float] = []

    for _, valid_idx in folds:
        valid_idx = np.asarray(valid_idx)
        outer = np.setdiff1d(np.arange(y.size), valid_idx, assume_unique=False)
        best_th, best_score = float(grid[0]), -1.0
        for th in grid:
            score = f1_score(y[outer], (probs[outer] >= th).astype(int), average="macro")
            if score > best_score:
                best_score, best_th = float(score), float(th)
        per_fold.append(best_th)
        preds[valid_idx] = (probs[valid_idx] >= best_th).astype(int)

    nested_f1 = float(f1_score(y, preds, average="macro"))

    naive_th, naive_f1 = float(grid[0]), -1.0
    for th in grid:
        score = f1_score(y, (probs >= th).astype(int), average="macro")
        if score > naive_f1:
            naive_f1, naive_th = float(score), float(th)

    return ThresholdResult(
        nested_f1=nested_f1,
        deployment_threshold=float(np.median(per_fold)),
        per_fold_thresholds=per_fold,
        naive_threshold=naive_th,
        naive_f1=naive_f1,
        optimism=float(naive_f1 - nested_f1),
    )


# =============================================================================
# 지표 집계
# =============================================================================


def evaluate_oof(
    y_true: pd.Series,
    oof_probs: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    cfg: CVConfig,
    *,
    label: str = "",
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """OOF 벡터를 pooled 로 채점해 결과 dict 를 돌려준다.

    반환 dict 에는 `cfg.describe()` 와 피처 목록 해시가 포함되므로, 결과만 보고도
    어떤 프로토콜로 얻은 수치인지 사후에 확인할 수 있다 (AUDIT.md §1.3 섹션 E).

    `macro_f1` 키는 **nested 값**이다. 기존 스크립트가 보고하던 naive 최댓값은
    `naive_macro_f1` 로 분리해 두었고, 두 값의 차이가 `threshold_optimism` 이다.
    """
    y = np.asarray(y_true)
    probs = np.asarray(oof_probs, dtype=float)

    th = tune_threshold_nested(y, probs, folds, cfg)
    preds_at_deploy = (probs >= th.deployment_threshold).astype(int)
    cm = confusion_matrix(y, preds_at_deploy, labels=[0, 1])

    result: dict[str, Any] = {
        "label": label,
        "n_rows": int(y.size),
        "positive_rate": float(y.mean()),
        "log_loss": float(log_loss(y, probs)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "f1_at_050": float(f1_score(y, (probs >= 0.5).astype(int), average="macro")),
        "macro_f1": th.nested_f1,
        "naive_macro_f1": th.naive_f1,
        "threshold_optimism": th.optimism,
        "deployment_threshold": th.deployment_threshold,
        "per_fold_thresholds": th.per_fold_thresholds,
        "naive_threshold": th.naive_threshold,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "recall": float(cm[1, 1] / max(cm[1, 0] + cm[1, 1], 1)),
        "protocol": cfg.describe(),
    }
    if feature_names is not None:
        names = list(feature_names)
        result["n_features"] = len(names)
        result["feature_hash"] = f"{hash(tuple(sorted(names))) & 0xFFFFFFFF:08x}"
    return result
