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

검증 경계:
1. early stopping 은 학습 fold 를 다시 쪼갠 내부 holdout 에서 수행한다.
   채점 fold(`X_valid`)는 `eval_set` 에 들어가지 않는다 (AUDIT.md §2.5).
2. 새 runner는 outer-train 내부 holdout에서 임계값을 선택한다. 호환용
   tune_threshold_nested의 K-1 fold OOF 방식에는 교차 fold 라벨 의존성이 남는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, NamedTuple, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .features import oof_target_encode, smoothed_target_encode

__all__ = [
    "CVConfig",
    "make_folds",
    "run_fold",
    "FoldFit",
    "run_fold_nested_grid",
    "FoldFitGrid",
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


def _make_inner_split(
    n_rows: int,
    y_train: pd.Series,
    cfg: CVConfig,
    fold: int,
    holdout_eligible: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """outer-train 위치 인덱스(0..n_rows-1)를 inner-train/inner-holdout 으로 나눈다.

    `run_fold()` 와 `run_fold_nested_grid()` 가 완전히 동일한 분할 공식을 쓰도록
    공유하는 헬퍼다. 두 함수 중 하나만 이 로직을 바꾸면 두 early-stopping 계열
    (콜백 기반 vs 그리드 기반) 이 서로 다른 inner-holdout 을 보게 되어, 트리 수
    선택 절차의 비교 자체가 성립하지 않는다 — 그래서 별도 함수로 분리했다.

    Parameters
    ----------
    n_rows
        outer-train 의 전체 행 수 (`len(X_train)`).
    y_train
        outer-train 라벨. 층화 분할에 쓰인다.
    holdout_eligible
        `run_fold()` 의 동명 인자와 동일한 의미 — `None` 이면 전체 행이 후보,
        지정하면 그 위치 인덱스 부분집합에서만 inner-holdout 을 뽑는다(준지도
        pseudo 행을 채점 후보에서 제외하기 위한 용도, `run_fold()` 독스트링 참고).

    Returns
    -------
    (inner_tr, inner_ho)
        `X_train` 기준 0-based 위치 인덱스 배열.
    """
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

    return inner_tr, inner_ho


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
    inner_tr, inner_ho = _make_inner_split(n_rows, y_train, cfg, fold, holdout_eligible)

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


def _set_n_estimators(model: Any, k: int) -> Any:
    """`model` 의 `n_estimators` 를 그리드 값으로 덮어쓴다.

    `model_factory()` 의 기존 계약("인자 없이 새 추정기를 만든다")을 그대로 유지하기
    위해, 그리드 탐색 쪽에서 만들어진 모델의 파라미터를 덮어쓰는 방식을 택했다
    (`run_fold_nested_grid()` 독스트링 참고).
    """
    if hasattr(model, "set_params"):
        try:
            model.set_params(n_estimators=k)
            return model
        except (ValueError, TypeError):  # pragma: no cover - 비표준 추정기 대비
            pass
    if hasattr(model, "n_estimators"):
        model.n_estimators = k
        return model
    raise TypeError(
        f"{type(model).__name__} 은 n_estimators 를 설정할 수 없습니다. "
        "run_fold_nested_grid() 는 이 값을 그리드마다 바꿔가며 재학습하므로 "
        "model_factory() 가 만드는 추정기가 이를 지원해야 합니다."
    )


class FoldFitGrid(NamedTuple):
    """`run_fold_nested_grid()` 의 반환값. `FoldFit` 과 대칭이다.

    Attributes
    ----------
    selected_n_estimators
        inner-holdout LogLoss 가 최소인 그리드 점. 이 fold 의 최종 트리 수.
    grid_scores
        `{n_estimators: inner-holdout LogLoss}` 전체 — 그리드 전 구간을 그대로
        남겨 사후 검토(예: 경계값에 몰렸는지)가 가능하게 한다.
    """

    model: Any
    valid_probs: np.ndarray
    selected_n_estimators: int
    grid_scores: dict[int, float]
    n_inner_train: int
    n_inner_holdout: int


def run_fold_nested_grid(
    model_factory: Callable[[], Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    cfg: CVConfig,
    *,
    fold: int = 0,
    n_estimators_grid: Sequence[int],
    te_cols: Sequence[str] = (),
    te_m: float = 20.0,
    te_inner_splits: int = 5,
    te_drop_original: bool = False,
    holdout_eligible: np.ndarray | None = None,
    extra_fit_rows: tuple[pd.DataFrame, pd.Series] | None = None,
    extra_fit_factory: Callable[[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]] | None = None,
    selection_metadata: dict[str, Any] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> FoldFitGrid:
    """early stopping 대신 nested grid 로 `n_estimators` 를 고정 선택한다.

    `output/es_protocol_final.md` "최종 권장 `n_estimators` 선택 절차"(작업 2-C,
    조건 "clean") 의 구현이다. `run_fold()` 와 원칙은 완전히 같다 — 학습 fold 를
    다시 쪼갠 내부 holdout 으로만 트리 수를 정하고, 채점 대상인 `X_valid` 는 그
    결정에 전혀 관여하지 않는다. 다른 점은 그 결정 메커니즘 하나뿐이다 — ES
    콜백(`stopping_rounds` 기반 patience) 대신, `n_estimators_grid` 각 점을 명시
    학습해 inner-holdout LogLoss 가 최소인 점을 고른다.

    기존 ES 실험은 TE 사전 인코딩에 의한 inner 검증 오염을 포함했다.
    순수 잡음이나 ES 자체의 결함으로 일반화하지 않는다. 이 함수는 clean inner
    경계에서 후보를 비교한다. 과거 3시드 결과는 통계적 유의성 보장이 아니다.

    **핵심 차이 — `X_train` 은 TE 인코딩 전 원본 카테고리여야 한다**
    -------------------------------------------------------------
    `run_fold()` 는 이미 TE(Target Encoding)가 끝난 `X_train` 을 받는다. 그 TE 는
    보통 outer-train 전체(이 fold 의 학습 파트, 예: ~204,000행)를 대상으로 한 번
    계산된다. 그런데 이 함수는 그 outer-train 을 다시 80/20 으로 쪼개 inner-train/
    inner-holdout 을 만드는데, **outer 경계에서 이미 계산된 TE 는 inner 경계를
    모른다.** `oof_target_encode()` 가 outer-train 전체를 `inner_splits`(TE 자체의
    K-fold, ES 의 inner-holdout 과는 다른 개념) 로 회전시키며 각 행의 `TE_*` 를
    계산하므로, inner-holdout의 TE 통계에 다른 inner-holdout 라벨이 사용되며
    inner-train 피처에도
    holdout 라벨 정보가 유입될 수 있다. Train 라벨로 holdout을 인코딩하는 것은
    정상이며, 반대 방향이 누수다 (`output/audit_addendum_te_leak.md`
    §2). 트리 수가 늘어날수록 모델이 이 간접 정보를 정교하게 활용해
    inner-holdout LogLoss 를 실제 일반화 능력과 무관하게 계속 낮출 수 있어, 그리드
    선택이 "가장 많이 외운" 지점(그리드 최댓값)을 고르는 정반대 결과를 낸다 —
    실측으로 그 최댓값 지점의 outer-valid 성능이 그리드 최저치였다
    (`es_protocol_final.md` 작업 2-C "naive 버전은 실패").

    그래서 이 함수는 TE 계산 자체를 내부에서, **inner-train/inner-holdout 분할
    이후** 수행한다(`oof_target_encode()` 를 outer-train 대신 inner 경계로 호출).
    이렇게 하면 inner-holdout 의 `TE_*` 는 inner-train 라벨만으로 계산되어,
    `run_fold()` 가 outer 경계에서 이미 보장하던 것과 원칙적으로 동일한 차단을
    한 단계 안쪽에 적용한 것이 된다.

    Parameters
    ----------
    X_train, X_valid
        **원본 카테고리 상태** (TE 인코딩 전) 의 outer-train / outer-valid.
        `run_fold()` 에 넘기는 `X_train`/`X_valid` 와 달리 `TE_*` 컬럼이 없어야
        한다 — 있으면 그 값은 outer 경계 TE 이므로 이 함수가 다시 계산하는
        inner 경계 TE 와 섞여 의미가 불분명해진다.
    n_estimators_grid
        탐색할 `n_estimators` 후보. 빈 시퀀스는 허용하지 않는다.
    te_cols
        TE 를 적용할 컬럼. 빈 튜플(기본값)이면 TE 를 전혀 쓰지 않는 Phase 도
        동일한 grid-선택 경로를 그대로 탈 수 있다(`oof_target_encode()` 를 빈
        `cols` 로 호출하면 TE 컬럼 없이 inner 분할만 수행한다).
    te_m, te_inner_splits, te_drop_original
        `oof_target_encode()` 동명 인자에 그대로 대응한다.
    holdout_eligible
        `run_fold()` 와 동일한 의미 — `_make_inner_split()` 을 공유하므로 준지도
        pseudo 행을 inner-holdout 후보에서 제외하는 방식도 동일하다.
    extra_fit_rows
        `oof_target_encode()` 동명 인자에 그대로 전달된다. 항상 inner-train
        쪽에만 추가되므로(그 함수의 계약), pseudo 행이 inner-holdout 채점에
        섞이는 일은 구조적으로 없다.
    extra_fit_factory
        분할 이후 inner-train 원본 피처/라벨만 받는 pseudo 생성 콜러블.
        extra_fit_rows와 동시 사용 불가. 외부 holdout 라벨을 캡처하지 않아야 한다.
    selection_metadata
        전달하면 선택 모델의 inner-holdout 확률로 고른 threshold와 pseudo_count를
        저장한다. Outer-valid 라벨은 이 함수에 전달되지 않는다.
    fit_kwargs
        `model.fit()` 에 전달할 추가 인자. `callbacks` 키는 제거한다 — 이
        경로는 early stopping 콜백을 쓰지 않는다(그리드 각 점을 조기종료 없이
        끝까지 학습한다).

    Returns
    -------
    FoldFitGrid
        선택된 `n_estimators` 로 이미 학습된 모델(재학습 없이 grid 스윕에서
        재사용)과, 그 모델로 outer-valid 를 채점한 확률.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train({len(X_train)}) 과 y_train({len(y_train)}) 의 길이가 다릅니다."
        )
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
           or k <= 0 for k in n_estimators_grid):
        raise ValueError("n_estimators_grid 는 양의 정수여야 합니다.")
    grid = sorted(set(int(k) for k in n_estimators_grid))
    if not grid:
        raise ValueError("n_estimators_grid 가 비어 있습니다.")
    if any(str(c).startswith("TE_") for X in (X_train, X_valid) for c in X.columns):
        raise ValueError("TE 사전 인코딩 데이터는 nested grid에 전달할 수 없습니다.")

    n_rows = len(X_train)
    inner_tr, inner_ho = _make_inner_split(n_rows, y_train, cfg, fold, holdout_eligible)

    if extra_fit_factory is not None:
        if extra_fit_rows is not None:
            raise ValueError("extra_fit_rows 와 extra_fit_factory 는 동시에 지정할 수 없습니다.")
        # Teacher/pseudo 생성자는 student holdout 라벨에 접근하지 못한다.
        extra_fit_rows = extra_fit_factory(
            X_train.iloc[inner_tr].copy().reset_index(drop=True),
            y_train.iloc[inner_tr].copy().reset_index(drop=True),
        )

    # 2단계: TE 를 inner 경계에서 계산한다. drop_original 은 여기서는 항상 False 로
    # 호출해 원본 컬럼을 남겨 둔다 — outer-valid 인코딩(5단계)에 같은 모집단의
    # 원본 카테고리 값이 필요하기 때문이다. 실제 컬럼 제거는 아래에서 한꺼번에
    # 수행한다.
    te = oof_target_encode(
        X_train, y_train, inner_tr, inner_ho,
        cols=te_cols, m=te_m, inner_splits=te_inner_splits,
        seed=cfg.seed, drop_original=False,
        extra_fit_rows=extra_fit_rows,
    )
    X_in_full, y_in = te.X_train, te.y_train
    X_ho_full = te.X_valid
    # inner-holdout 라벨. te.X_valid 와 동일한 순서(inner_ho 슬라이스)로 뽑는다.
    y_ho = y_train.iloc[inner_ho].reset_index(drop=True)

    # 5단계: outer-valid 도 같은 inner-train 모집단(X_in_full 의 원본 컬럼 + y_in)
    # 으로 단방향 인코딩한다. extra_fit_rows 가 있으면 X_in_full 에 이미 포함되어
    # 있으므로 그 라벨도 자연히 반영된다.
    present = [c for c in te_cols if c in X_valid.columns]
    X_va_full = X_valid.copy().reset_index(drop=True)
    for col in present:
        X_va_full[f"TE_{col}"] = smoothed_target_encode(
            X_in_full[col], y_in, X_va_full[col], m=te_m
        )

    if te_drop_original and present:
        X_in_model = X_in_full.drop(columns=present)
        X_ho_model = X_ho_full.drop(columns=present)
        X_va_model = X_va_full.drop(columns=present)
    else:
        X_in_model, X_ho_model, X_va_model = X_in_full, X_ho_full, X_va_full

    # 3단계: 그리드 스윕. early stopping 콜백은 쓰지 않는다 — 각 k 를 끝까지 학습.
    kwargs = dict(fit_kwargs or {})
    kwargs.pop("callbacks", None)
    if "eval_set" in kwargs or "eval_sample_weight" in kwargs:
        raise ValueError("nested grid 에 외부 eval_set 을 전달할 수 없습니다.")

    grid_scores: dict[int, float] = {}
    best_k: int | None = None
    best_score = np.inf
    best_model: Any = None

    for k in grid:
        model = model_factory()
        _set_n_estimators(model, k)
        fit_params = _fit_parameters(model)
        call_kwargs = {kk: vv for kk, vv in kwargs.items() if kk in fit_params}
        model.fit(X_in_model, y_in, **call_kwargs)

        ho_probs = np.asarray(model.predict_proba(X_ho_model), dtype=float)[:, 1]
        score = float(log_loss(y_ho, ho_probs, labels=[0, 1]))
        grid_scores[k] = score

        # 4단계: LogLoss 최소점을 선택한다(작업 0 — 동급 1순위 지표, threshold 없이
        # 계산 가능해 nested 하게 쓸 수 있는 유일한 지표).
        if score < best_score:
            best_score, best_k, best_model = score, k, model

    assert best_model is not None and best_k is not None  # grid 가 비어 있지 않으므로 항상 참

    if selection_metadata is not None:
        # k 와 임계값 모두 outer-train 내부에서 결정한다. 이 holdout 점수는
        # 튜닝용이며 성능 추정치로 보고하지 않는다. outer-valid 는 마지막에만 채점한다.
        selected_probs = np.asarray(best_model.predict_proba(X_ho_model))[:, 1]
        scores = [f1_score(y_ho, selected_probs >= th, average="macro")
                  for th in cfg.thresholds()]
        selection_metadata.update(
            threshold=float(cfg.thresholds()[int(np.argmax(scores))]),
            pseudo_count=0 if extra_fit_rows is None else len(extra_fit_rows[0]),
        )

    valid_probs = np.asarray(best_model.predict_proba(X_va_model), dtype=float)[:, 1]
    return FoldFitGrid(
        model=best_model,
        valid_probs=valid_probs,
        selected_n_estimators=best_k,
        grid_scores=grid_scores,
        n_inner_train=int(len(X_in_model)),
        n_inner_holdout=int(len(X_ho_model)),
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
        채점한 뒤 pooled 로 계산한 Macro-F1. 교차 fold 라벨 의존성은 남는다.
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
    """레거시 cross-fold OOF 임계값 선택. 완전한 label-independent nested CV는 아니다.

    절차 (AUDIT.md §2.6)
    --------------------
    fold `k` 마다:
      1. 나머지 K-1 fold 의 검증 행만 모아 임계값 후보를 훑고 최적값을 고른다.
      2. 그 임계값을 fold `k` 의 행에**만** 적용해 예측 라벨을 확정한다.
    모든 fold 의 예측 라벨을 pooled 하여 Macro-F1 을 한 번 계산한다.

    자기 fold OOF는 제외하지만 다른 fold 모델에 자기 fold 라벨이 사용될 수 있다.
    엄격한 평가는 outer-train 내부에서 임계값을 선택해 evaluate_oof에 전달한다.
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
    per_fold_thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """OOF 벡터를 pooled 로 채점해 결과 dict 를 돌려준다.

    반환 dict 에는 `cfg.describe()` 와 피처 목록 해시가 포함되므로, 결과만 보고도
    어떤 프로토콜로 얻은 수치인지 사후에 확인할 수 있다 (AUDIT.md §1.3 섹션 E).

    per_fold_thresholds는 outer-train 내부에서 미리 선택한 값이어야 한다.
    전달된 경우 F1/혼동행렬/recall 모두 fold별 임계값으로 계산한다.
    미전달 시 호환용 cross-fold OOF 선택을 사용하며 완전한 label independence는 없다.
    `macro_f1` 키는 fold별 임계값 점수다. 기존 스크립트가 보고하던 naive 최댓값은
    `naive_macro_f1` 로 분리해 두었고, 두 값의 차이가 `threshold_optimism` 이다.
    """
    y = np.asarray(y_true)
    probs = np.asarray(oof_probs, dtype=float)

    th = tune_threshold_nested(y, probs, folds, cfg)
    preds_at_deploy = (probs >= th.deployment_threshold).astype(int)
    if per_fold_thresholds is not None:
        thresholds = [float(t) for t in per_fold_thresholds]
        if len(thresholds) != len(folds) or any(
            not np.isfinite(t) or not 0 <= t <= 1 for t in thresholds
        ):
            raise ValueError("각 fold 에 유효한 사전 선택 임계값 하나가 필요합니다.")
        preds_at_deploy = np.zeros_like(y, dtype=int)
        for (_, va), threshold in zip(folds, thresholds):
            preds_at_deploy[va] = probs[va] >= threshold
        score = float(f1_score(y, preds_at_deploy, average="macro"))
        th = ThresholdResult(score, float(np.median(thresholds)), thresholds,
                             th.naive_threshold, th.naive_f1, th.naive_f1 - score)
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
    result["protocol"]["threshold_selection"] = (
        "outer_train_holdout" if per_fold_thresholds is not None else "cross_fold_oof_legacy"
    )
    if feature_names is not None:
        names = list(feature_names)
        result["n_features"] = len(names)
        result["feature_hash"] = f"{hash(tuple(sorted(names))) & 0xFFFFFFFF:08x}"
    return result
