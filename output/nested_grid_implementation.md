# Nested n_estimators 선택 구현 — `output/es_protocol_final.md` 조건 C 실장

> 이 문서는 **읽기 전용 진단이 아니라 구현 결과 보고서**다. `output/es_diagnosis.md`,
> `output/es_protocol_final.md`, `output/audit_addendum_te_leak.md` 세 문서가 3중
> 검증으로 확정한 결론(§요약 참고)을 재검토 없이 코드로 옮긴 작업 1~4의 결과를
> 기록한다. 브랜치: `feat/nested-n-estimators` (base: `master`).
>
> **작업 5(Phase 1~6 전체 재실행)는 이번 범위에 포함되지 않는다 — 실행하지 않았다.**

---

## 요약

| 항목 | 결과 |
| :--- | :--- |
| 브랜치 | `feat/nested-n-estimators` (origin 에 push 하지 않음) |
| 커밋 | 2개 (아래 §5) |
| `tests/test_features.py` | **124 / 124 통과** (`uv run pytest tests/test_features.py -v`) |
| 작업 4 회귀(Phase 4, seed=42, 전체 255,001행) | **목표치와 오차 ±0.00005 이내로 일치** — 재현 성공, 추가 디버깅 불필요 |
| 수정 금지 목록 | 전부 미수정 확인 (`git status` — `output/baseline_recovery.csv`/`.md` 변경 없음) |

---

## 작업 1 — `run_fold_nested_grid()` / `FoldFitGrid`

### 위치와 시그니처

파일: `src/cv.py`

- `_make_inner_split()` — **175~239행**. `run_fold()` 와 `run_fold_nested_grid()` 가
  공유하는 inner-train/inner-holdout 분할 헬퍼로 추출했다. `run_fold()`(**243~337행**)
  본문의 인라인 분할 로직을 이 헬퍼 호출로 교체했고(동작 변화 없음 — 리팩터링
  전후 `tests/test_features.py::TestRunFold` 전체가 그대로 통과함으로 확인), 두
  함수가 서로 다른 inner-holdout 을 보는 사고를 원천적으로 막았다.
- `_set_n_estimators()` — **340~360행**. `model_factory()` 의 "인자 없이 새 추정기를
  만든다" 기존 계약을 유지하면서, 그리드 값으로 만들어진 모델의 `n_estimators` 를
  덮어쓴다(`set_params` 우선, 실패 시 속성 직접 대입).
- `class FoldFitGrid(NamedTuple)` — **363~380행**. 필드: `model`, `valid_probs`,
  `selected_n_estimators`, `grid_scores`(`{n_estimators: inner-holdout LogLoss}`),
  `n_inner_train`, `n_inner_holdout`.
- `def run_fold_nested_grid(...)` — **383~553행**. 최종 시그니처:

```python
def run_fold_nested_grid(
    model_factory: Callable[[], Any],
    X_train: pd.DataFrame,   # outer-train, TE 인코딩 전 원본 카테고리
    y_train: pd.Series,
    X_valid: pd.DataFrame,   # outer-valid, TE 인코딩 전 원본 카테고리
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
    fit_kwargs: dict[str, Any] | None = None,
) -> FoldFitGrid:
```

지시된 제안 시그니처 대비 **`extra_fit_rows` 를 추가**했다 — `oof_target_encode()`
동명 인자를 그대로 전달하는 통로다. 이걸 추가한 이유는 작업 3(§4)에서 설명한다.

### 내부 절차 (지시된 5단계 그대로 구현)

1. `_make_inner_split(len(X_train), y_train, cfg, fold, holdout_eligible)` 로
   `run_fold()` 와 동일한 공식(`train_test_split(..., random_state=cfg.seed+fold,
   stratify=y_train)`)의 inner-train/inner-holdout 을 얻는다.
2. `oof_target_encode(X_train, y_train, inner_tr, inner_ho, cols=te_cols, ...,
   drop_original=False, extra_fit_rows=extra_fit_rows)` 를 **inner 경계**에서 호출한다.
   `drop_original` 은 이 호출에서는 항상 `False` 로 고정하고(5단계에 필요한 원본
   카테고리 값을 보존하기 위해), 실제 컬럼 제거는 4~5단계 결과에 나중에 일괄 적용한다.
3. `n_estimators_grid` 각 `k` 에 대해 `model_factory()` → `_set_n_estimators(model, k)`
   → 조기종료 콜백 없이 inner-train 으로 학습 → inner-holdout LogLoss 계산.
4. LogLoss 최소 `k*` 선택(이미 학습된 모델 재사용, 재학습 없음).
5. `X_valid` 를 inner-train 라벨 기준 `smoothed_target_encode()` 로 단방향 인코딩하고
   `k*` 모델로 채점.

독스트링(383~470행 부근)에 `audit_addendum_te_leak.md` §2 메커니즘을 요약 인용해
"왜 TE 를 inner 경계에서 재계산해야 하는지" 를 설명해 두었다.

---

## 작업 2 — 누수 차단 단위 테스트

파일: `tests/test_features.py`, `class TestRunFoldNestedGrid` (**988행~**).

`uv run pytest tests/test_features.py -v` 전체 실행 결과: **124 passed, 0 failed**
(신규 7건 포함 — 기존 117건 전부 그대로 통과, 회귀 없음).

새로 추가한 7개 테스트:

| 테스트 | 검증 내용 |
| :--- | :--- |
| `test_inner_holdout_te_depends_only_on_inner_train_labels` | **핵심 계약.** `_make_inner_split()` + `oof_target_encode()` 를 `run_fold_nested_grid()` 내부와 동일하게 호출한 뒤, inner-holdout 행 하나의 라벨을 뒤집어도 inner-holdout TE(`X_valid` 쪽) 전체가 불변임을 확인. (inner-holdout 라벨은 애초에 인코딩 계산에 참조되지 않으므로 이것이 수학적으로 옳은 "무누수" 성질이다.) |
| `test_inner_train_labels_unaffected_by_holdout_flip` | inner-train 자체의 내부 K-fold 회전 인코딩도 inner-holdout 라벨 변경에 무관함을 확인. |
| `test_outer_precomputed_te_leaks_inner_train_labels_into_holdout` | **대조군(회귀 테스트).** 현재 `run_fold()` 가 받는 방식(outer-train 전체 선계산 TE → 80/20 재분할)을 의도적으로 재현해, inner-train 쪽이 될 행의 라벨을 바꾸면 inner-holdout 이 될 행의 TE 값이 실제로 바뀜을 확인 — `audit_addendum_te_leak.md` §2 리키지의 존재 자체를 회귀적으로 고정한다. |
| `test_outer_valid_content_does_not_affect_grid_selection` | `X_valid` 의 카테고리/수치 값을 완전히 다른 내용으로 덮어써도 `selected_n_estimators`/`grid_scores` 가 완전히 동일함을 확인(채점 대상이 grid 선택에 관여하지 않음). |
| `test_returns_foldfitgrid_with_expected_fields` | 기본 동작 스모크 — 반환 타입/필드/shape. |
| `test_empty_grid_rejected` | 빈 grid 는 `ValueError`. |
| `test_works_without_target_encoding` | `te_cols=()` (기본값) — TE 미사용 Phase 도 동일 경로로 정상 동작. |

작업 지시의 "선택 사항" 대조 테스트(outer 선계산 방식이 실제로 리키지를 일으킴)도
구현해 포함시켰다.

---

## 작업 3 — `rerun_all_phases.py` 구조 변경

### 변경 내역

- `run_phase()` 의 fold 루프 진입 전 TE 1회 계산(구 470~479행 부근,
  `te = oof_target_encode(X_lab, y, tr, va, ...)`) 을 제거하고, fold 루프 안에서
  `run_fold_nested_grid(model_factory, X_tr_raw, y_tr_raw, X_va_raw, CFG, fold=fold,
  n_estimators_grid=N_ESTIMATORS_GRID, te_cols=te_cols, ...)` 를 호출하도록 바꿨다.
  `X_tr_raw`/`X_va_raw` 는 `X_lab.iloc[tr]`/`X_lab.iloc[va]` — TE 인코딩 전 원본
  카테고리 그대로다.
- `te_cols = spec.cat_cols if spec.te else ()` — `spec.te=False` Phase(P1, P2)는
  `te_cols=()` 로 동일한 nested-grid 경로를 타되 TE 는 전혀 계산하지 않는다.
- `scale_pos_weight` 계산(`spw = float((len(y)-y.sum())/y.sum())`)과
  `model_factory(spw)` 호출을 완전히 제거했다. `model_factory()` 는 이제 인자를
  받지 않는 일반 콜러블이다 (`run_fold()`/`run_fold_nested_grid()` 양쪽의
  "인자 없이 호출" 계약을 그대로 만족).
- `PhaseSpec.use_spw` 필드는 유지하되 독스트링에 "[미사용]" 명시 — 원본 스크립트가
  전부 spw 를 적용했다는 사실을 문서로만 남긴다.
- `N_ESTIMATORS_GRID = [10, 25, 50, 75, 100, 150, 300, 600]` 상수를 추가했다
  (`es_protocol_final.md` 가 쓴 그리드 그대로).
- `CFG.stopping_rounds` 필드는 값 자체는 유지하되(원본 스크립트/`run_fold()` 재현용),
  주석으로 "새 경로에서는 미사용" 명시.
- `CSV_FIELDS` 의 `best_iterations` 를 `selected_n_estimators`/`grid_scores` 로
  교체했다. `scale_pos_weight` 열은 항상 `"-"` 로 기록된다.
- `holdout_eligible` 을 이용한 pseudo 행 위치-필터링 로직(`eligible = np.arange(len(tr))
  if extra is not None else None`)을 완전히 제거했다 — 아래 §4 참고.
- `write_report()`/`_interpretation()` 의 스키마 참조(`best_iterations` →
  `selected_n_estimators`)와 §3-7(`scale_pos_weight` 진단) 서술을 갱신해, 이제는
  `scale_pos_weight` 가 전 Phase 에서 구조적으로 제거되었다는 사실과 정합되게
  고쳤다(이 절은 작업 5 를 실행하지 않는 한 실제로 렌더링되지 않지만, 코드가
  깨지거나 사실과 어긋나는 채로 방치하지 않았다).

### 스모크 검증 (전체 8개 Phase 정의, 20,000행 표본)

`P1, P2, P3, P4, P4_te0, P5_leaky, P5_honest, P6_legacy, P6_fixed` 전부를 20,000행
표본으로 개별 실행해 예외 없이 완료됨을 확인했다(출력 로그는 대화 중 확인, 파일로
남기지 않음). 특히 pseudo 두 종(`P5_leaky`, `P5_honest`)에서 `n_inner_train` 값이
"진짜 fold-train 중 inner-train 몫 + pseudo 전체" 와 정확히 일치함을 확인해,
`extra_fit_rows` 가 항상 inner-train 쪽에만 반영되고(홀드아웃 후보에서 구조적으로
배제) 그 결과가 이전 `holdout_eligible` 방식과 동등함을 검증했다.

---

## 작업 4 — 회귀 테스트 결과 (Phase 4, seed=42, 전체 255,001행)

`rerun_all_phases.run_phase()` 를 스크래치패드 드라이버로 직접 호출했다
(`append_row()`/`write_report()`/`main()` 은 호출하지 않아 `output/baseline_recovery.csv`/
`.md` 는 전혀 건드리지 않았다 — 실행 후 `git status` 로 미변경 확인).

| 지표 | 목표(es_protocol_final.md 조건 C) | 실측 | 오차 |
| :--- | ---: | ---: | ---: |
| pooled AUC | 0.6425 | **0.642545** | +0.000045 |
| pooled LogLoss | 0.4477 | **0.447748** | +0.000048 |
| Macro F1 (nested, pooled) | 0.5755 | **0.575528** | +0.000028 |
| fold별 선택 n_estimators | `[50,50,50,50,50]` 근처(±1 그리드 스텝 허용) | **`[50, 50, 50, 50, 50]`** | 정확히 일치 |

세 지표 모두 허용 오차(±0.001)를 한 자릿수 이상 여유 있게 통과했고, fold별
n_estimators 도 그리드 스텝 이탈 없이 5-fold 전부 정확히 50을 선택했다.
**추가 디버깅이 필요 없었다** — 첫 실행에서 바로 재현되었다. 소요 시간은 55초
(fold당 약 9~10초, 그리드 8점 × inner-train 163,200행).

fold별 inner-holdout LogLoss 그리드(전 구간)도 `es_protocol_final.md` 작업 2-C 의
U자형과 일치하는 형태(`k=50` 부근이 최솟값, 그 좌우로 완만히 증가)를 보였다 —
원시 수치는 `output/nested_grid_implementation.md` 작성용 스크래치패드
(`task4_result.json`, 프로젝트에는 남기지 않음)에 있다.

---

## Pseudo-Labeling 처리 방식 (작업 3의 회색지대)

**결정: teacher 는 기존 `run_fold()`(ES 기반) 를 그대로 유지하고, student(최종
fold 학습, `run_phase()` 본문의 fold 루프)만 `run_fold_nested_grid()` 로 전환했다.**

근거(`rerun_all_phases.py` `_teacher_fit_predict()` 독스트링에도 동일하게 남겼다):

1. **비용**: nested grid 는 fold 하나당 그리드 점 수(8개)만큼 모델을 학습한다.
   `P5_leaky` 는 teacher 자체가 이미 fold 마다(5회) 통째로 재학습되므로, teacher 에도
   grid 를 적용하면 비용이 8배 추가로 곱해진다. 이번 작업의 목적(구조 변경의 정확성
   검증)에 비해 정당화되지 않는다고 판단했다.
2. **오염 경로가 다르다**: `audit_addendum_te_leak.md` 가 실측한 TE-inner 리키지는
   **채점 대상 모델**(`evaluate_oof()` 에 들어가는 최종 확률)의 트리 수 선택에
   영향을 준다. teacher 의 출력은 그 자체로 채점되지 않고 `make_pseudo_labels()` 의
   분위수 임계값(상위 2% / 하위 10%)을 통과하는지만 결정하는 데 쓰인다 — teacher
   확률의 세부 랭킹이 흔들려도 극단 분위수 선별은 비교적 완만하게만 영향받는다.
3. student 쪽 성능(`evaluate_oof()` 로 보고되는 pooled AUC/LogLoss/Macro F1)이
   이번 회귀 테스트(작업 4)의 판정 대상이므로, teacher 를 바꾸지 않아도 "nested
   grid 구조 변경"의 핵심 효과는 그대로 검증된다.

이 결정 덕분에 `run_fold_nested_grid()` 의 `extra_fit_rows` 파라미터(제안 시그니처에는
없던 추가)가 자연스럽게 필요해졌다 — pseudo 행이 student 의 TE 계산에 항상
**inner-train 쪽에만** 추가되도록(`oof_target_encode()` 의 동명 계약을 그대로
위임) 만들었고, 그 결과 예전의 `holdout_eligible` 위치-필터링(`np.arange(len(tr))`)이
구조적으로 불필요해졌다(스모크 검증에서 확인).

**재검토 여지**: teacher 도 nested grid 로 옮기면 pseudo-label 품질 자체가 더
안정될 가능성이 있다 — 다만 이번 판단은 비용 대비 효과와 범위(구조 변경 검증)를
근거로 한 것이며, 사람이 재검토할 수 있는 회색지대임을 명시한다.

---

## `git diff --stat`

```
$ git diff --stat master...HEAD
 rerun_all_phases.py    | 215 +++++++++++++++++++-------------
 src/cv.py              | 323 +++++++++++++++++++++++++++++++++++++++++++------
 tests/test_features.py | 222 +++++++++++++++++++++++++++++++++
 3 files changed, 641 insertions(+), 119 deletions(-)
```

## `git log --oneline` (이 브랜치에서 만든 커밋)

```
75d5969 refactor: rerun_all_phases.py를 nested n_estimators 경로로 전환
01e40e1 feat: nested n_estimators 선택 함수 run_fold_nested_grid() 신설
```

(base: `master` @ `e2094e7`. origin 에는 push 하지 않았다.)

---

## `PLAN.md`/`README.md`/`AUDIT.md` — 다음에 사람이 판단할 목록

이번 작업에서 이 세 파일은 읽기만 했다(수정 금지 목록). 정독 중 갱신이 필요해
보이는 지점을 아래에 나열한다 — 직접 손대지 않았다.

1. **`AUDIT.md` 14행 / §2.2 요약표** — "OOF Target Encoding 의 fold 경계 — fold
   내부에서 계산됨, 정상" 판정은 **outer fold 경계**에 한해서만 여전히 유효하다.
   `audit_addendum_te_leak.md` 가 지적한 **inner(ES-holdout) 경계**의 리키지는
   이 표에 별도 행으로 반영되어 있지 않다. 이번 구현으로 그 리키지 자체는
   `run_fold_nested_grid()` 경로에서 구조적으로 해소되었지만(작업 4 검증 완료),
   `AUDIT.md` 본문에는 아직 그 사실이 기록되지 않았다 — `audit_addendum_te_leak.md`
   부록을 본문에 병합하거나 상호 참조를 추가할지는 사람이 판단할 사안이다.
2. **`PLAN.md` §3-A 항목 1**(불균형 대응을 `scale_pos_weight` 대신 임계값 조정으로
   처리) — 이번 구현으로 `scale_pos_weight` 가 코드에서 완전히 제거되어 이 방침이
   실제로 100% 실행되었다. `PLAN.md` 가 이 항목을 "계획"이 아니라 "완료"로 갱신할
   시점인지는 Phase 1~6 재실행(작업 5, 미실행) 이후 판단하는 편이 안전해 보인다.
3. **`PLAN.md` §4-0** (P0-2/P0-3/P0-4 등 체크리스트 항목) — `rerun_all_phases.py` 의
   구조가 이번 작업으로 다시 한번 바뀌었다(ES → nested grid). 이 항목들의 "완료"
   판정 기준이 어느 버전의 `rerun_all_phases.py` 를 가리키는지 재확인이 필요해
   보인다.
4. **`README.md`** — 프로젝트 개요/재현 방법 섹션에 `rerun_all_phases.py` 실행법이
   설명되어 있다면, `scale_pos_weight` 관련 서술이나 "early stopping 기반" 이라는
   표현이 남아 있는지 확인이 필요하다(이번 세션에서는 `README.md` 를 정독하지
   않았다 — 수정 금지 목록에 있어 열람만 했고, 관련 서술 유무를 특정하지 못했다).
5. **`output/baseline_recovery.csv`/`.md`** — 이번 구조 변경 이후 Phase 1~6 을
   재실행(작업 5)하면 CSV 스키마가 `best_iterations` → `selected_n_estimators`/
   `grid_scores` 로 바뀌어 있으므로, 기존 CSV 와 컬럼 구조가 다르다는 점을
   실행 전에 인지해야 한다(`load_done()`/`--force` 재개 로직이 컬럼 불일치를
   런타임에서 검사하지 않는다).

---

## 최종 확인 사항 (지시된 최종 보고 항목)

- **현재 브랜치**: `feat/nested-n-estimators`
- **커밋 상태**: 위 2개 커밋, origin 에 push 하지 않음. `git status` 는
  `output/audit_addendum_te_leak.md`/`output/te_leak_check_fold2.png` 만 untracked로
  표시(이전 진단 작업 산출물, 손대지 않음). 그 외 clean.
- **회귀 테스트 통과 여부**: `tests/test_features.py` 124/124 통과. 작업 4 수치
  회귀도 목표 대비 오차 ±0.00005 이내로 통과.
- **본 보고서 경로**: `output/nested_grid_implementation.md` (본 파일)

**작업 5(Phase 1~6 전체 재실행)는 이 보고서 작성으로 종료하며 진행하지 않았다.**
별도 승인 후 진행 대상이다.
