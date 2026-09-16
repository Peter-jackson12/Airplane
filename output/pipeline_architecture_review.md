# 파이프라인 구조 감사 — P5 라벨 흐름 / 임계값 선택 / 출력·재개 로직

> **성격 고지**: 이 문서는 읽기 전용 감사 보고서다. 코드 수정, 커밋/푸시, Phase
> 재실행을 하지 않았다. 아래 모든 항목은 다음 세 범주로 표시한다(`CLAUDE.md` §2
> 관례).
> - **[사실]** — `Read`/`Grep`으로 코드를 직접 읽어 확인한 내용 (파일:행 명시).
> - **[실행]** — 이 세션에서 실제로 실행해 나온 결과 (스크래치패드 스크립트,
>   `python -c ...`). 프로젝트 코드는 수정하지 않았고, 실행도 `src.cv`/
>   `src.features`의 기존 함수를 그대로 import 해서 호출했을 뿐이다.
> - **[추론]** — 위 둘로부터 유도한 판단. 확정이 아님을 표현에 명시한다.
>
> **작업 상태 [사실]**: 브랜치 `feat/nested-n-estimators`, `origin`과 동기화됨,
> `git status` — untracked `.codex/`(무관한 다른 도구 설정) 외 clean. 이 세션에서
> 코드/기존 문서를 변경하지 않았다.

---

## 0. 읽은 자료

`output/es_diagnosis.md`, `output/es_protocol_final.md`,
`output/audit_addendum_te_leak.md`, `output/nested_grid_implementation.md`,
`src/cv.py`(705행 전체), `src/features.py`(810행 전체),
`rerun_all_phases.py`(1048행 전체), `tests/test_features.py`(구조 및 관련 클래스
발췌), `PLAN.md`/`README.md`/`AUDIT.md`(관련 절 발췌 확인)를 읽었다. 이 문서가
인용하는 4개 진단 문서의 결론 자체는 재검증하지 않고 전제로 삼았다 — 이번
감사의 대상은 그 결론들이 아니라 **그 결론을 구현한 현재 코드**다.

---

## 1. 발견 사항 요약 (심각도순)

| # | 심각도 | 분류 | 한 줄 요약 | 위치 |
|---|---|---|---|---|
| F1 | **HIGH** | 확정 결함 (재현됨) | `--force` 재실행 시 CSV 헤더(30열)와 신규 스키마(31열)가 어긋나 기존 파일에 append하면 열 밀림으로 데이터가 조용히 오염된다 | `rerun_all_phases.py:609-616`, `output/baseline_recovery.csv` |
| F2 | **HIGH** | 확정 결함 | `--force` 없이 그대로 재실행하면 `load_done()`이 구프로토콜(스크린 붕괴, spw 적용) 행을 "완료됨"으로 오인해 전부 skip하고, `write_report()`가 구프로토콜 수치로 새 타임스탬프의 보고서를 재생성한다 | `rerun_all_phases.py:602-606, 660, 691-693, 706` |
| F3 | **HIGH** | 설계상 위험 (구조는 확정, 영향 크기 미확인) | P5(leaky/honest) 에서 teacher 의 ES-inner-holdout 분할과 student 의 nested-grid inner-holdout 분할이 **동일 파티션**이다 — teacher 가 student 의 그리드 채점 대상을 실질적으로 미리 봤다 | `rerun_all_phases.py:347-396, 493-531`, `src/cv.py:175-240` |
| F4 | MEDIUM | 확정 결함 (범위 한정) | teacher 학습 경로(`_teacher_fit_predict`→`run_fold()`)는 `audit_addendum_te_leak.md`가 학생 경로에서 지적해 고친 "TE 선계산 후 ES 분할" 리키지를 그대로 가지고 있다 — 수정이 student 에만 적용됐다 | `rerun_all_phases.py:384-396` |
| F5 | MEDIUM | 설계상 위험 | `tune_threshold_nested()`의 "nested" 독립성은 각 fold **자신의** 확률이 자기 임계값에 영향을 주지 않는다는 것만 보장한다 — 다른 fold의 OOF 생성 모델이 그 fold의 라벨로 학습됐다는 경로는 차단하지 않는다 | `src/cv.py:588-646` |
| F6 | LOW/문서 | 미확인 → 정정 필요 | `es_protocol_final.md` 작업 2-B의 "fold 2 진짜 전역 최솟값 라운드 499" 서술은 이후 `audit_addendum_te_leak.md`가 그 자체를 TE-inner 리키지의 산물로 재규명했다 — 두 문서를 따로 읽으면 "499가 진짜 신호"로 오독할 위험이 있다 | `es_protocol_final.md` 작업2-B, `audit_addendum_te_leak.md` §3 |
| F7 | LOW | 테스트 커버리지 공백 | `test_fold_threshold_ignores_own_rows`는 "자기 행" 독립성만 검증한다 — F5(라벨 출처 독립성)를 검증하는 테스트는 없다 | `tests/test_features.py:1237-1247` |
| F8 | LOW | 테스트 커버리지 공백 | F3(teacher/student holdout 파티션 동일성)을 검증·회귀 고정하는 테스트가 없다 | `tests/test_features.py` 전체(해당 클래스 없음) |

---

## 2. P5 라벨 흐름 감사

### 2.1 경로 재구성 [사실]

**P5_leaky** (`rerun_all_phases.py:399-442`, `_leaky_ensemble_pseudo_labels`):
fold 루프 **밖에서** 5개 fold 각각의 teacher(그 fold의 `tr`만으로 학습, 397-421행)
예측을 평균해 **한 번만** pseudo 집합을 얼리고(423-441행), 그 동일 집합을
**모든 fold**의 학습에 주입한다(474행, `frozen_pseudo`). 이 함수 자체가
"의도적으로 잘못된 구현"이라고 독스트링에 명시되어 있다(403-404행).

**판정 [사실]**: fold `k`를 채점할 때 쓰는 pseudo 집합은 teacher `j`(`j≠k`)의
예측을 포함하고, teacher `j`는 `tr_j`(fold `j`의 outer-train)로 학습되는데
`tr_j`는 fold `k`의 outer-valid 행을 포함한다(5-fold 분할에서 자명). 즉
**fold k 자신의 채점 라벨이 앙상블 평균을 거쳐 fold k 자신의 학습셋에
재유입된다.** `AUDIT.md` §2.1이 지적한 원 누수를 정확히 재현한 것이며,
`make_pseudo_labels()`가 리스트/튜플/딕셔너리 입력을 `TypeError`로 거부하도록
설계돼 있어(`src/features.py:728-733`) 정식 API로는 이 경로를 만들 수
없다 — 그래서 이 함수는 `make_pseudo_labels()`를 **우회**해서 직접 구현했다
(403-404행에 그렇게 명시됨). 실제 모델링에는 쓰이지 않고 누수 크기 측정
전용이라는 문서상 전제(`PhaseSpec` 232-234행 note)는 코드와 일치한다.

**P5_honest** (`rerun_all_phases.py:493-509`): fold 루프 **안에서** 매 fold마다
`fit_fold_teacher(X_lab, y, tr, ...)`로 그 fold의 `tr`만 보는 teacher를 새로
만들고(`src/features.py:769-809`), 그 fold 전용 분위수로 pseudo를 다시 계산한다
(`make_pseudo_labels`, 501-503행). **teacher가 va(채점 fold)의 행을 보는 경로는
없다** — `fit_fold_teacher`는 `train_idx`(=tr)만 클로저에 담고
(`src/features.py:803-807`), `tr`과 `va`는 `StratifiedKFold` 분할로 항상
서로소다(`src/cv.py:123-138`, `tests/test_features.py:802-805`
`test_train_valid_disjoint`로 회귀 고정됨). **이 경로는 확인한 범위에서
"AUDIT.md §2.1 원 누수"로부터 안전하다.**

### 2.2 새로 발견한 경로 — teacher와 student가 같은 inner-holdout을 공유한다

`run_fold_nested_grid()`(student, `src/cv.py:383-553`)와 `run_fold()`
(teacher, `_teacher_fit_predict`가 호출, `src/cv.py:243-337`)는 둘 다
`_make_inner_split()`(`src/cv.py:175-240`)를 공유한다. 이 헬퍼는
`(n_rows, y_train, cfg, fold, holdout_eligible)`에만 의존해
`train_test_split(idx, test_size=cfg.inner_holdout_frac, random_state=cfg.seed+fold, stratify=y_train)`을
계산한다(208-213행).

**[사실] 확인한 호출 인자**: P5_honest의 fold 루프(`rerun_all_phases.py:486-531`)
에서, 같은 `fold` 값으로
- teacher 쪽: `fit_fold_teacher(X_lab, y, tr, ...)` → `_teacher_fit_predict`
  내부에서 `oof_target_encode(combined, ..., outer_train_idx=np.arange(len(X_tr)), ...)`
  가 `te.X_train`(=`X_tr` 그대로, 길이 `len(tr)`) 을 만들고
  `run_fold(model_factory, te.X_train, te.y_train, te.X_valid, CFG, fold=fold)`
  가 내부에서 `_make_inner_split(len(tr), te.y_train, CFG, fold, None)` 호출.
- student 쪽: `run_fold_nested_grid(model_factory, X_tr_raw, y_tr_raw, X_va_raw, CFG, fold=fold, ...)`
  가 내부에서 `_make_inner_split(len(tr), y_tr_raw, CFG, fold, None)` 호출.

두 호출의 `n_rows`(=`len(tr)`), `y_train`(둘 다 `y.iloc[tr]`을 순서 보존한 채
`reset_index`한 것과 동일한 값), `cfg`(같은 `CFG` 인스턴스), `fold`(같은 정수)가
전부 동일하다.

**[실행] 검증**: 합성 데이터(200행, 이진 라벨, `Tail_Number` 3범주)로 위 두
경로를 `src.cv`/`src.features`의 실제 함수만 그대로 호출해 재현했다.

```
student inner_ho[:10] : [5, 8, 24, 30, 32, 49, 50, 51, 53, 68]
teacher inner_ho[:10] : [5, 8, 24, 30, 32, 49, 50, 51, 53, 68]
IDENTICAL inner_ho partitions: True
IDENTICAL inner_tr partitions: True
```

**두 inner-holdout 파티션이 완전히 동일하다.** 즉 P5_leaky/P5_honest 양쪽 모두
"student가 nested grid로 n_estimators를 고를 때 쓰는 inner-holdout(H)"과
"teacher가 자기 early stopping에 쓰는 inner-holdout"이 **같은 fold에서 정확히
같은 행 집합**이다.

**라벨 재유입 경로 [사실 + 추론]**:

1. teacher는 `H`의 라벨로 `.fit()`을 직접 하지 않는다 — `run_fold()`는 `X_in`
   (inner-train)으로만 `model.fit()`하고 `X_ho`(=H)는 `eval_set`으로만 쓴다
   (`src/cv.py:304, 322-328`). 이 자체는 정상이다.
2. 그러나 `X_in`의 `TE_*` 값은 `oof_target_encode()`가 **`H`가 갈라지기 전의
   `tr` 전체**를 `inner_splits`(TE 자체의 K-fold 회전)로 계산한다
   (`_teacher_fit_predict` 384-394행, `oof_target_encode` 내부
   `src/features.py:665-681`) — 이는 `audit_addendum_te_leak.md` §2가 이미
   실측한 바로 그 메커니즘("ES-inner-holdout 경계에서는 TE가 다시 계산되지
   않는다")이 teacher 학습에도 그대로 적용됨을 뜻한다(§2.3 참고). 즉 `X_in`의
   `TE_*`는 `H`의 라벨 정보를 간접적으로 포함할 수 있다.
3. teacher의 `best_iteration_`(조기 종료 시점) 자체도 `H`(=`X_ho`)의
   `binary_logloss`로 결정되는데, 이 `binary_logloss`가 앞서 언급한 TE-inner
   리키지로 인해 실제 일반화 능력과 무관하게 계속 개선되는 것처럼 보일 수 있다
   — `audit_addendum_te_leak.md` §3의 정량 실측(리키지 있음 조건에서 전역
   최솟값이 499라운드까지 밀리는 것)과 같은 메커니즘이다. 다만 그 정량치
   자체를 teacher 학습에 대해 재측정하지는 않았다(**추론**).
4. teacher 예측은 `X_unlab`(진짜 미라벨 풀)에 대해서만 계산되고
   (`make_pseudo_labels`), 그 pseudo 집합은 `extra_fit_rows`로
   `oof_target_encode()`에 전달되어 **항상 inner-train 쪽에만** 추가된다
   (`src/features.py:640-656`이 `X_train = pd.concat([X_train, extra_X])`로
   train_idx 슬라이스 뒤에 붙이며, `H`가 이미 분리된 뒤에 일어나는 연산이라
   pseudo 행이 `H` 자체에 섞이는 경로는 없다 — 이 부분은 `nested_grid_implementation.md`
   의 회귀 테스트 `test_inner_holdout_te_depends_only_on_inner_train_labels`
   (`tests/test_features.py:1021`)로 확인됨).
5. 그렇게 pseudo 행이 섞인 `inner-train ∪ pseudo`로 학습된 각 그리드 점 모델이
   바로 **`H`(=2와 3에서 teacher를 거쳐 간접적으로 자신의 라벨 정보가 흘러든
   그 자리)**에서 채점되어 `n_estimators`가 선택된다.

**결론 [추론, 미확인]**: `H`의 실제 라벨이 (2)(3)을 거쳐 teacher의 학습 상태에
스며들고, teacher의 산출물(pseudo-label)이 (4)를 거쳐 student의 inner-train에
들어간 뒤, (5)에서 바로 그 `H`에 대해 채점되어 그리드가 선택된다 — **경로
자체는 구조적으로 존재한다(1~5는 전부 코드 추적/실행으로 확인됨).** 다만 이
경로가 실제로 n_estimators 선택이나 최종 outer-valid 지표를 **얼마나** 바꾸는지는
측정하지 않았다. 참고할 만한 정황 근거는 있다 — ①
`_teacher_fit_predict`가 쓰는 `run_fold()`는 spw 없이 `metric=binary_logloss`
조합이므로 `es_diagnosis.md`가 실측한 결정론적 붕괴({2,3})는 없지만,
`es_protocol_final.md` 작업 1·3이 실측한 **조건 A(ES) 특유의 큰 fold간·시드간
분산**(best_iteration 54~567대 요동)은 그대로 적용된다는 점, ②
`audit_addendum_te_leak.md` §3이 같은 메커니즘(TE 선계산 후 ES 분할)에서 fold 2
하나에 대해 outer-valid AUC −0.0210, LogLoss +0.0101 크기의 실측 영향을
보였다는 점(다만 이는 *student* 경로에 대한 수치이며 teacher에 대해 재측정한
값이 아니다). 이 둘을 근거로 "teacher 쪽 잔여 리키지의 영향이 무시할 만하다"고
단정할 수 없다 — `nested_grid_implementation.md`가 스스로 "회색지대"로 남긴
지점(§ Pseudo-Labeling 처리 방식)이 이번 추적으로 더 구체적인 메커니즘을
얻었을 뿐, 크기는 여전히 미확인이다.

**va(진짜 채점 대상, outer-valid)에 대한 영향 — 없음 [사실]**: 위 경로 전체는
`tr`(outer-train) 내부에서만 일어난다. `va`의 행은 teacher 학습(`fit_fold_teacher`가
`tr`만 클로저에 담음, `src/features.py:803-804`)에도, `extra_fit_rows`
(항상 inner-train에만 추가, 위 4)에도 전혀 등장하지 않는다. 즉 **outer-valid
자신의 라벨이 자신의 점수에 재유입되는 직접 누수는 이번 추적 범위에서
발견되지 않았다.** F3은 "채점 지표 자체의 부정직"이 아니라 "n_estimators
선택 메커니즘의 독립성 가정 위반"으로 분류해야 정확하다.

### 2.3 `holdout_eligible`/`extra_fit_rows`가 실제로 막는 것 [사실]

`run_phase()`는 `run_fold_nested_grid()` 호출 시 `holdout_eligible`을 넘기지
않는다(`rerun_all_phases.py:522-531`에 해당 인자 없음 → 기본값 `None`). 즉
현재 코드에서 이 인자는 **호출되지 않는 죽은 경로**다.
`nested_grid_implementation.md` §3이 "예전 `holdout_eligible` 위치-필터링이
구조적으로 불필요해졌다"고 적은 것과 일치한다 — `extra_fit_rows`가
`oof_target_encode()` 내부에서 `train_idx` 슬라이스 뒤에만 붙기 때문에
(`src/features.py:640-656`), pseudo 행이 애초에 `H`의 후보가 될 수 없어
`holdout_eligible`로 따로 걸러낼 필요가 없어진 것은 맞다. **이 특정 채널(=pseudo
행 자체가 홀드아웃에 물리적으로 섞이는 것)은 확실히 차단된다.** 그러나 §2.2가
보여준 채널(teacher가 `H`를 미리 봤다는 사실 자체)은 `holdout_eligible`/
`extra_fit_rows`가 다루는 층위보다 한 단계 위(teacher 학습 자체)에 있어 이
메커니즘으로는 차단되지 않는다.

### 2.4 기존 테스트가 보장하는 범위 [사실]

| 테스트 | 보장 범위 | F3을 잡는가 |
|---|---|---|
| `test_fit_fold_teacher_closure_sees_only_fold_train`(468-545행 내, 533-544) | teacher 클로저가 `tr`(fold 학습 파트)만 본다 — `va`를 안 본다는 것만 확인 | 아니오 — `tr` 내부에서 student의 `H`와 겹치는지는 검사 대상이 아님 |
| `test_pseudo_labels_differ_across_folds`(1342-1363) | fold마다 teacher가 달라 pseudo 집합도 달라짐(동결 누수 재발 방지) | 아니오 |
| `test_full_fold_loop_produces_valid_oof`(1365-1406) | teacher→pseudo→TE→학습 전 과정이 **`oof_target_encode()`+`run_fold()`** 조합으로 도는지(옛 outer-경계 방식) | 아니오 — 이 e2e 테스트는 `run_fold_nested_grid()`를 쓰지 않아 현재 `rerun_all_phases.py`의 실제 P5 경로(§2.2)를 재현하지 않는다 |
| `test_inner_holdout_te_depends_only_on_inner_train_labels`(1021) 외 `TestRunFoldNestedGrid` 6종 | student 쪽 nested-grid의 TE-inner 경계 자체가 깨끗한지 | 부분적 — student 자신의 경계는 검증하지만 teacher와의 파티션 공유는 검증 범위 밖 |

**결론**: F3(teacher/student inner-holdout 파티션 동일성)과 F4(teacher 경로의
TE-inner 리키지 잔존)를 직접 겨냥한 테스트는 현재 없다.

---

## 3. 임계값 선택 감사

### 3.1 호출 경로 [사실]

`evaluate_oof()`(`src/cv.py:654-704`) → `tune_threshold_nested(y, oof_probs, folds, cfg)`
(674행) → `src/cv.py:588-646`. `oof_probs`는 `run_phase()`가 fold 루프에서
채운 pooled OOF 벡터(`rerun_all_phases.py:480, 532`) — 즉 행 `i`의 확률은
`i`가 속한 fold의 모델(그 fold를 outer-valid로 채점한 모델)이 만든 값이다.

`tune_threshold_nested()`의 절차(603-604행 문서, 616-629행 구현): fold `k`마다
`outer = 전체 인덱스 \ valid_idx_k`를 잡고, `probs[outer]`(즉 **fold k가 아닌
다른 모든 fold의 OOF 확률**)에서만 그리드를 훑어 fold `k`의 임계값을 고른 뒤,
그 임계값을 `preds[valid_idx_k]`에만 적용한다.

### 3.2 실제로 보장되는 것과 보장되지 않는 것

**보장되는 것 [사실, `test_fold_threshold_ignores_own_rows` 1237-1247행으로
회귀 고정]**: fold `k`의 **자기 자신의 확률 값**을 아무리 바꿔도 fold `k`의
임계값은 바뀌지 않는다. 이것이 `AUDIT.md` §2.6이 지적한 winner's curse(전체
OOF에서 직접 최댓값을 고르는 것)를 제거한다 — `naive_f1 ≥ nested_f1`이
항상 성립한다는 `test_nested_f1_not_greater_than_naive`(1217-1222행)가 이를
수학적으로 확인한다.

**보장되지 않는 것 [사실 + 추론]**: `probs[outer]`를 만든 모델들(fold
`j≠k`의 모델)은 각각 `outer-train_j`(=전체 \ fold `j`)로 학습되는데, `outer-train_j`는
**fold `k`의 행을 포함한다**(`j≠k`이므로 자명, `make_folds()`의 정의상 각
행은 정확히 하나의 fold에서만 outer-valid가 된다 —
`test_folds_partition_all_rows_exactly_once`, 796-800행). 즉 fold `k`의
임계값을 고르는 데 쓰인 `probs[outer]`의 각 성분은, **fold `k`의 라벨로 학습된
모델**이 만든 값이다(그 값 자체는 자기 자신의 outer-valid 행에 대한 것이라 그
쪽은 OOF로 깨끗하지만, 모델이라는 매개체 자체는 fold `k`의 라벨을 안다).

이것이 실무적으로 문제가 되려면 "fold `k`의 라벨을 학습에 쓴 모델의 예측
경향"과 "fold `k` 자신에 최적인 임계값" 사이에 상관이 있어야 한다. 두 가능한
경로:
1. **fold 수가 적을수록(K=5) 경로가 강해진다** — 한 모델의 학습 데이터
   80%가 다른 4개 fold와 공유되므로, 5개 모델의 예측 분포가 서로 독립이라고
   보기 어렵다(전부 같은 하이퍼파라미터, 같은 시드 계열로 학습된 유사 모델).
2. `deployment_threshold`는 `per_fold_thresholds`의 **중앙값**이라
   (`ThresholdResult.deployment_threshold`, 641행) 개별 fold 하나의 편향이
   희석되긴 하지만, 5개 fold 임계값 자체가 서로 완전히 독립적인 추정치는
   아니라는 뜻이다.

**[추론, 미확인]**: 이 경로가 실제로 유의미한 낙관 편향을 만드는지는
측정하지 않았다. `es_protocol_final.md` 작업 4가 보인 실측(9회 실행, 45개
fold별 임계값이 전부 0.22~0.23의 좁은 띠에 있고 그리드 하한에 물리지
않음)은 이 편향의 상한이 크지 않을 가능성을 시사하지만, 그 실측은 "그리드가
안전한가"를 본 것이지 "이 특정 교차-fold 경로가 있는가/없는가"를 직접 겨냥한
실험이 아니다.

### 3.3 엄격한 대안과의 차이

**현재 방식**: 임계값 선택에 쓰는 "K-1 fold의 OOF"가 애초에 **outer 교차검증
루프 자체가 만들어낸 부산물**이다 — 별도의 홀드아웃을 추가로 파지 않는다.
비용은 0(이미 계산된 OOF를 재사용)이지만, 위에서 설명한 대로 "fold `k`의
임계값 선택에 관여하는 모델 중 누구도 fold `k`의 라벨을 학습에 쓰지 않았다"는
더 강한 독립성은 보장하지 않는다.

**엄격한 대안**: `run_fold_nested_grid()`가 `n_estimators`에 대해 이미 하고
있는 것과 같은 원리를 임계값에도 적용하는 것 — 즉 각 outer fold 내부에서
**추가로** inner-holdout을 만들어(또는 기존 ES/그리드용 inner-holdout을
재사용해) 그 fold **자신의 outer-train만으로** 임계값을 고르고, outer-valid에는
그 임계값만 적용한다. 이렇게 하면 fold `k`의 임계값 선택에 관여하는 모든
데이터(모델 파라미터 + 임계값 그리드 평가 모두)가 fold `k`의 라벨을 전혀
보지 않은 것으로 좁혀진다 — 다만 이는 fold마다 별도의 임계값 탐색 절차를
요구하므로 현재의 "OOF 재사용" 방식보다 구현이 복잡해지고, `deployment_threshold`
정의(현재는 fold별 값의 중앙값)도 다시 검토해야 한다. **이번 세션에서는
구현하지 않았다** — 지시에 따라 서술만 한다.

---

## 4. 출력 및 재개(resume) 로직 감사

### 4.1 현재 `output/baseline_recovery.csv`/`.md`의 실제 상태 [사실]

```
$ git log --oneline -- output/baseline_recovery.csv
cf8bbbd feat: Phase 1~6 통일 프로토콜 재측정 — 정직한 기준선 확정
```
현재 파일은 이 한 커밋에서 만들어진 **구프로토콜 결과**다 — CSV의
`scale_pos_weight` 열에 `4.6667`(P1~P4_te0) 값이 그대로 있고, `best_iterations`
열에 `P1`=`[2, 2, 2, 2, 2]` 같은 `es_diagnosis.md` §1이 지적한 붕괴 패턴이
그대로 남아 있으며(1행 실측: `P1,...,"[2, 2, 2, 2, 2]",...,4.6667,...`),
`lgbm_params` JSON에 `"n_estimators": 1000`이 고정돼 있다(현재 `LGBM_PARAMS`엔
이 키가 없음). 즉 **이 파일은 `es_diagnosis.md`/`es_protocol_final.md`/
`audit_addendum_te_leak.md`가 폐기를 권고한 바로 그 실행 결과다.**
CSV 헤더는 30개 필드다(직접 카운트, §4.2 참고). `git status`는 clean —
이 상태로 커밋되어 있고, 이 세션은 손대지 않았다.

### 4.2 기본 실행(무옵션) — 구프로토콜 결과가 "완료됨"으로 오인된다 [사실 + 실행]

```
$ cut -d, -f1 output/baseline_recovery.csv
phase_key
P1
P2
P3
P4
P4_te0
P4_nospw
P5_leaky
P5_honest
P6_legacy
P6_fixed
```
현재 CSV에는 `PHASES`(`rerun_all_phases.py:199-257`)의 **10개 key 전부**가 이미
들어 있다. `main()`(651-709행)의 로직:
```python
done = load_done(args.force)          # force=False → CSV 그대로 읽음(602-606행)
...
for spec in specs:
    if spec.key in done and not args.force:   # 691행 — 10개 전부 True
        print(f"...이미 완료됨, 건너뜀...")
        continue
    ...
write_report(done)                     # 706행 — 루프 결과와 무관하게 항상 호출
```
**결론 [사실]**: `uv run python rerun_all_phases.py`를 `--force` 없이 그대로
실행하면 10개 Phase 전부 "이미 완료됨"으로 skip되고, 학습은 **한 번도
일어나지 않은 채** `write_report(done)`이 구프로토콜 수치로
`output/baseline_recovery.md`를 **새 타임스탬프로 재생성**한다. 파일 상단의
"생성 시각"만 갱신되므로, 이 보고서를 처음 보는 사람은 방금 새 프로토콜로
재실행된 결과라고 오인하기 매우 쉽다 — 유일한 단서는 "2-1. 실행 세부" 표의
`selected_n_estimators` 열이 전부 `—`로 나온다는 것뿐이다(`r.get('selected_n_estimators', '—')`,
792행) — 구 CSV에 이 필드가 없기 때문이다. **`load_done()`은 phase_key
일치만으로 "완료"를 판정하며, 그 행이 어느 프로토콜/코드 버전에서 나왔는지
검사하지 않는다.**

### 4.3 `--force` 재실행 — CSV 파일이 열 밀림으로 손상된다 [사실 + 실행 재현]

`append_row()`(609-616행)는 파일이 이미 존재하면(`is_new=False`) 헤더를 다시
쓰지 않고 `csv.DictWriter(fh, fieldnames=CSV_FIELDS)`로 그냥 이어 쓴다.
`--force`를 주면 `load_done()`이 기존 행을 무시하고(602-603행) 전 Phase를
다시 학습하지만, 결과 행은 여전히 **같은 파일**(`output/baseline_recovery.csv`)에
append된다 — 파일을 새로 만들거나 헤더를 갱신하지 않는다.

**필드 수 직접 확인 [실행]**:
```
old header (실제 CSV 1행)         : 30개 필드 (…, best_iterations, pseudo_counts, …)
CSV_FIELDS (현재 코드, 263-270행) : 31개 필드 (…, selected_n_estimators, grid_scores, pseudo_counts, …)
```
`best_iterations` 한 필드가 `selected_n_estimators`+`grid_scores` 두 필드로
바뀌어 net **+1열**이고, 그 뒤의 모든 필드(`pseudo_counts`부터 `lgbm_params`까지
13개)가 한 칸씩 밀린다.

**재현 [실행]** — 동일한 상황(기존 30열 헤더 파일에 31열 fieldnames로
append)을 최소 예제로 만들어 `csv.DictReader`로 다시 읽었다:
```python
# 기존 파일: 헤더 a,b,c + 데이터 1,2,3
# 새로 append: DictWriter(fieldnames=['a','x','b','c']).writerow({'a':'A','x':'NEW','b':'B','c':'C'})
# 결과 (DictReader로 다시 읽으면):
{'a': '1', 'b': '2', 'c': '3'}                       # 기존 행 — 정상
{'a': 'A', 'b': 'NEW', 'c': 'B', None: ['C']}        # 새로 append된 행 — 밀림
```
헤더가 `a,b,c`인 채로 4개 값을 쓴 행을 다시 읽으면, `b` 열에 실제로는 새
필드(`x`)의 값이 들어가고 `c` 열에는 실제 `b` 값이 들어가며, 진짜 마지막
필드 값은 눈에 보이지 않는 `None` 키 아래 리스트로 빠진다 — **예외 없이,
조용히** 일어난다. `output/baseline_recovery.csv`에 그대로 적용하면: 새로
append된 각 Phase 행에서 `pseudo_counts` 열에는 실제로 `grid_scores`(거대한
JSON 딕셔너리)가, `inner_splits` 열에는 실제 `pseudo_total`이, `scale_pos_weight`
열에는 실제 `inner_splits`가 들어가는 식으로 **13개 열이 전부 의미가
어긋나고**, 실제 `lgbm_params` 값은 `None` 키 리스트에 숨어 CSV를 순회하는
어떤 코드도 정상적으로 읽지 못한다.

**결론 [사실]**: 이 손상은 `rerun_all_phases.py` 자체가 `write_report()`에
넘기는 `done`(메모리 내 dict, 방금 실행한 새 행만 담김)에는 영향을 주지
않으므로 **그 실행에서 생성되는 `baseline_recovery.md` 자체는 정상**이다.
피해자는 **그 다음에** `output/baseline_recovery.csv`를 다시 읽는 모든
경로다 — 다음 번 `load_done()` 호출(예: 일부 Phase만 추가로 돌리는 재실행),
또는 CSV를 직접 열어 보는 사람/노트북. `load_done()`의 `{r["phase_key"]: r for r in csv.DictReader(fh)}`는
동일 key가 여러 번 나오면 **마지막 것**으로 덮이므로 딕셔너리 자체는
런타임에서 안 죽지만, 그 "마지막 것"의 필드 값이 위처럼 밀려 있다.

### 4.4 재개 검증에 필요한 식별 정보 — 현재 불충분 [사실]

`load_done()`이 신뢰하는 것은 `phase_key` 문자열 하나뿐이다. 같은 `phase_key`
아래 다음 항목이 달라져도 이를 감지하는 코드가 없다:

| 식별 축 | 현재 CSV에 기록되는가 | `load_done()`이 검사하는가 |
|---|---|---|
| 데이터 파일/행 수 | `n_rows`로 기록됨 | 아니오 |
| 코드 버전(ES vs nested-grid 메커니즘) | 간접적으로만 — 컬럼 자체가 다름(`best_iterations` vs `selected_n_estimators`) | 아니오 |
| 시드 | `protocol` JSON(`CVConfig.describe()`)에 `seed` 포함 | 아니오 |
| fold 분할 | `protocol`에 `n_splits`/`shuffle`/`seed` 포함 — 분할 자체를 바꾸려면 이 값을 바꿔야 하므로 간접 식별 가능 | 아니오 |
| TE 설정(`inner_splits`, `m`) | `inner_splits` 열은 있음, `m`은 없음(전 Phase 고정값이라 로그 안 함) | 아니오 |
| n_estimators 그리드 | 기록 안 됨(그리드 자체, `N_ESTIMATORS_GRID`) — `grid_scores`의 key 집합으로 간접 역산은 가능 | 아니오 |
| 임계값 그리드 | `protocol.threshold_grid` | 아니오 |

특히 **`CVConfig.describe()`가 만드는 `protocol` JSON은 ES 시절과 nested-grid
시절이 완전히 동일하다** — `n_splits=5, seed=42, stopping_rounds=40,
threshold_grid=(0.10,0.70,0.01), inner_holdout_frac=0.2`는 두 코드 버전
모두에서 같은 값이다(`CVConfig` 자체는 이번 구조 변경으로 바뀌지 않았다 —
`stopping_rounds` 필드가 값은 유지한 채 "새 경로에서는 미사용"이 됐을 뿐,
`rerun_all_phases.py:96-100` 주석 참고). 즉 **가장 신뢰할 만해 보이는
`protocol` 필드조차 두 프로토콜을 구분하지 못한다.** 실제로 구분 가능한
유일한 필드는 컬럼 존재 여부(`best_iterations` vs `selected_n_estimators`)와
`lgbm_params`의 `n_estimators` 키 유무, `scale_pos_weight` 값 타입(`4.6667`
숫자 vs `"-"` 문자열)인데, 이들은 전부 **사람이 육안으로 대조해야만** 드러나고
코드가 자동으로 검사하지 않는다.

### 4.5 `baseline_recovery_v2` 제안

§4.2~4.4의 근본 원인은 "이 CSV/MD 쌍이 어떤 프로토콜의 결과인지"를 파일
자신이 말해주지 않는다는 것이다. 최소 변경 두 가지(적용은 사용자 판단,
이번 세션은 서술만 함):

1. **출력 경로 분리(권장, 최소 변경)**: `CSV_PATH`/`MD_PATH`를
   `output/baseline_recovery_v2.csv`/`.md`처럼 프로토콜 버전이 드러나는 새
   이름으로 바꾼다. 기존 `output/baseline_recovery.csv`/`.md`는 "ES
   프로토콜, 구프로토콜(spw 적용)"의 기록으로 **그대로 보존**하고, 새 파일이
   "nested-grid 프로토콜, spw 미적용"의 단일 진원지가 된다. `load_done()`이
   구 파일을 참조할 가능성 자체가 사라지므로 §4.2/§4.3 두 문제 모두 원천
   차단된다. `OLD_PROTOCOL` 딕셔너리(139-146행, §2 대조표용 구프로토콜
   수치)는 이 이름 변경과 무관하게 그대로 유지하면 된다 — 그 딕셔너리는
   `PLAN.md §2-1` 원본 스크립트 기록치를 인용하는 것이지 `baseline_recovery.csv`
   를 다시 읽는 게 아니다.
2. **스키마/프로토콜 지문 검사(선택, 근본적)**: CSV 행에 코드 버전을 식별하는
   필드(예: 그리드 선택 메커니즘 이름 + `N_ESTIMATORS_GRID`/`LGBM_PARAMS`의
   해시)를 추가하고, `load_done()`이 파일 헤더 또는 그 필드를 현재
   `CSV_FIELDS`/현재 설정과 대조해 불일치 시 예외를 내거나 그 행을 무시하도록
   바꾼다. 1번보다 작업량이 크지만 향후 또 다른 구조 변경이 생겨도 재발을
   막는다.

이번 세션은 코드를 고치지 않았으므로 어느 쪽도 적용하지 않았다 — **다음
전체 재실행 전에 둘 중 최소 1번은 반드시 결정해야 한다**(§6).

---

## 5. 이전 보고 서술 검증

### 5.1 fold 2의 "54"와 "499" — 서로 다른 것을 가리킨다 [사실, 문서 간 대조]

- **54** = `es_diagnosis.md`/`es_protocol_final.md` 조건 A(표준 ES,
  `stopping_rounds=40`, spw 없음)에서 patience 기반 early stopping이 **실제로
  멈춘 지점**(`best_iteration_=54`). `es_protocol_final.md` 작업 1은 이 지점의
  outer-valid 성능(AUC 0.6446)이 그 fold의 그리드-오라클 최적점(n=50, AUC
  0.6450)과 사실상 동일하다고 실측했다 — 즉 **54는 우연히 최적점 근처에 멈춘
  "짧은" 케이스**로 분류됐다.
- **499** = 같은 fold를 **조기종료 없이 700라운드까지 강제 학습**시켰을 때,
  그 시점의 (당시 계산 방식 기준) inner-holdout `binary_logloss`가 전역
  최솟값을 찍은 라운드(`es_protocol_final.md` 작업 2-B "부수 발견",
  `argmin=499, min=0.444431`). 이 수치는 애초에 "ES가 54에서 너무 일찍
  멈췄고 사실 더 학습했으면 좋았다"는 근거로 제시되어 `min_delta` 도입(조건
  B)의 동기가 됐다.
- **`audit_addendum_te_leak.md` §3의 재규명 [사실, 이후 문서가 스스로 정정]**:
  같은 fold, 같은 `argmin=499, min=0.444431`을 "TE를 outer-train 전체에
  선계산한 뒤 80/20 분할"(리키지 있음, 조건 A와 동일 구성)로 강제 재현해
  일치를 확인한 뒤, TE를 inner 경계에서 다시 계산하는 clean 버전으로
  바꾸면 **전역 최솟값이 라운드 54로 이동**한다는 것을 보였다(§3 표: clean
  버전 전역 최솟값 라운드 54, outer-valid LogLoss 0.4471/AUC 0.6446 —
  오라클 최적점과 사실상 동일).

**정정 [사실 기반 결론]**: "499"는 진짜로 더 학습해서 얻은 개선 신호가
아니라, TE-inner 경계 리키지가 만든 **허상의 개선 곡선**이었다. 그 fold의
진짜 전역 최적점은(리키지를 제거하면) 54 근처이며, 이는 애초에 ES가
멈췄던 지점과 사실상 같다 — "ES가 54에서 멈춰 손해를 봤다"는 §2-B의
동기 서술은 이제 근거가 사라졌다. `es_protocol_final.md`는 이 정정을
반영하지 않은 채(그 문서가 `audit_addendum_te_leak.md`보다 먼저 작성됨)
남아 있고, `audit_addendum_te_leak.md` §5는 "결론 자체(조건 C 채택)는
바뀌지 않는다"고만 적어 두었다 — **개별 수치(499)의 해석까지 정정한다고
명시하지는 않았다.** 두 문서를 따로 읽으면 "499가 놓친 진짜 신호"라고
오독할 위험이 남아 있으므로, 이번 감사에서 명시적으로 정정해 둔다. (조건
B의 최종 채택 여부에는 영향 없음 — `es_protocol_final.md` 작업 3의 시드
분산 실측으로 조건 B는 별도 사유(AUC 시드 표준편차 0.00241 > 0.002 기준)로
이미 기각됐다.)

### 5.2 정상 경로 대 누수 경로 [사실, §2 재정리]

- **정상**: (a) outer 경계 — `outer_valid`의 `TE_*`는 항상 `outer_train`
  라벨로만 계산된다(`AUDIT.md` §2.2, `oof_target_encode()` 661-663행).
  (b) student의 inner 경계(현재 코드) — `run_fold_nested_grid()`가 TE를
  `_make_inner_split()` **이후** 계산하므로 `inner-holdout`의 `TE_*`는
  `inner-train` 라벨만으로 계산된다(`src/cv.py:490-509`,
  `test_inner_holdout_te_depends_only_on_inner_train_labels`로 회귀 고정).
- **누수**: outer 경계에서 TE를 **먼저** 계산한 뒤 그 결과를 다시 80/20으로
  쪼개는 경로 — `run_fold()`가 받는 `X_train`이 이미 TE 인코딩된 상태이므로
  내부 80/20 분할이 TE의 K-fold 회전 경계와 어긋난다
  (`audit_addendum_te_leak.md` §2). **현재 이 경로는 두 곳에 남아 있다**:
  (i) 원본 `run_*.py` 7개 스크립트(재현/측정 전용, 실운영 경로 아님),
  (ii) **`_teacher_fit_predict()`(`rerun_all_phases.py:384-396`) — 현재도
  실제로 실행되는 P5_leaky/P5_honest의 teacher 학습 경로.** (ii)는 이번
  감사에서 재확인한 사실이며(§2.2, F4), "student 경로는 고쳤으니 이
  리키지는 프로젝트에서 사라졌다"고 서술하면 부정확하다.

### 5.3 "고정 트리 수만으로 누수가 차단된다"는 설명의 정확한 범위 [사실]

`audit_addendum_te_leak.md` §4는 이미 "조기종료가 실제로 inner-holdout을
보는 경우에만" 리키지가 발생하며, "`n_estimators`를 고정하고 조기종료를
쓰지 않는 방식(조건 C)은 애초에 inner-holdout을 학습 중 참조하지 않으므로
이 리키지의 영향을 받지 않는다"고 정확히 명시했다. 이번 감사가 덧붙이는
것은: 이 보호는 **"조기종료 콜백을 아예 안 쓴다"는 조건**에 걸려 있지,
**"트리 수를 고정한다"는 것 자체**에 걸려 있지 않다는 점이다. 실제로
`run_fold_nested_grid()`의 그리드 각 점(`k`)도 결국 "그 `k`로 고정한
트리 수"로 학습되지만, 그 자체가 리키지를 막는 게 아니라 — **TE를 매
grid 호출 전에 inner 경계로 다시 계산했기 때문에** 막힌 것이다
(`src/cv.py:486-495`). 반대로 F4가 보여주듯, teacher는 `run_fold()`(ES
콜백 사용)로 학습되므로 "고정 트리 수"라는 성질이 없고, 조기종료가
`H`를 직접 참조하므로 §2.2/F4의 리키지 채널이 여전히 열려 있다. 즉
"트리 수를 고정하면 안전하다"가 아니라 "**inner-holdout을 학습 루프
안에서 참조하는 메커니즘(ES 콜백이든 grid 채점이든)을 쓰면서 그 TE를
outer 경계에서 선계산하면 위험하다**"가 정확한 일반화이며, 두 조건을
같이 봐야 한다.

### 5.4 3-시드 표준편차와 0.002 기준의 정확한 의미 [사실]

`es_protocol_final.md` 작업 3은 스스로 다음 두 가지를 명시했고, 이번 감사는
이를 재확인한다(재실행하지 않음, 문서 재독):

1. **n=3은 표본이 작다** — 작업 3 자신이 "n=3 시드로 추정한 std라 표본
   오차가 크다... 최소 5~10개 시드로 std를 다시 추정해야 한다"고 명시했다
   (작업 3, "결론 — 몇 회 반복해야 하는가" 절). 3개 표본의 표준편차는 그
   자체로 신뢰구간이 넓어, "조건 C의 std가 0.002의 1/3"이라는 결론을
   과도하게 정밀한 수치로 인용해서는 안 된다 — 방향(A/B ≫ C)은 표본이 작아도
   비교적 견고해 보이지만, 절대값(0.00059 등)을 소수점 5자리까지 믿을
   근거는 약하다.
2. **0.002는 그 자체가 추정치다** — `AUDIT.md` §2.6의 "임계값 winner's
   curse 크기 +0.001~+0.002" 추정에서 온 관례적 기준이며, 별도의 통계적
   유의성 검정(가설검정, 신뢰구간)이 아니다. 작업 3이 "표준편차가 0.002보다
   작다/크다"를 비교한 것은 **엄밀한 유의성 판정이 아니라 실무적 휴리스틱
   대조**로 읽어야 한다. 특히 조건 A/B의 "range"(최댓값-최솟값)가 이미
   0.002~0.006 수준이라는 관찰(작업 3)은 표준편차 비교보다 오히려 더
   설득력 있는 근거로, "0.002 규칙 자체가 A/B의 실제 노이즈를 과소평가한다"는
   결론과도 일관된다.

**결론 [추론]**: 조건 C가 A/B보다 fold간·시드간 분산이 작다는 **방향성**
결론은 이번 감사에서 재검토한 코드 구조(§2.3, TE-inner 경계를 매 grid
호출마다 다시 닫는 방식)와 정합적이며 신뢰할 만하다. 다만 "0.002의
1/3 이하"라는 구체적 배율이나 "단일 시드로 통계적으로 방어 가능하다"는
문구는 n=3 표본에서 나온 것임을 인용할 때마다 함께 밝혀야 한다 — 이후
Phase 1~6 전체 재실행 보고서가 이 배율을 근거로 "충분하다"고 단정하면
안 된다.

---

## 6. 종합 — 확정 결함 / 설계상 위험 / 미확인

### 확정 결함 (코드 실행/직접 대조로 재현됨)
- **F1** `--force` 재실행 시 CSV 열 밀림 (§4.3, 실행 재현 완료).
- **F2** 무옵션 재실행 시 구프로토콜 결과가 "완료"로 오인되어 새 타임스탬프의
  보고서가 재생성됨 (§4.2, 코드 로직으로 확정).
- **F4** teacher 학습 경로가 `audit_addendum_te_leak.md`의 TE-inner 리키지를
  그대로 가지고 있음 (§2.2/§5.3, 코드 대조로 확정). *영향 범위는 teacher의
  tree-count 선택과 그로부터 파생되는 pseudo-label 품질에 한정 — outer-valid
  라벨 자체는 건드리지 않음(§2.2 말미).*

### 설계상 위험 (구조는 확인됨, 실제 영향 크기는 미측정)
- **F3** teacher/student inner-holdout 파티션 동일성 (§2.2, 실행으로 구조는
  확정했으나 이것이 P5의 보고 수치를 얼마나 왜곡하는지는 미측정).
- **F5** 임계값 선택의 교차-fold 라벨 경로 (§3.2, 메커니즘은 확인, 크기는
  미측정 — `es_protocol_final.md` 작업 4의 임계값 안정성 실측이 간접적으로
  상한을 시사하나 이 경로를 직접 겨냥한 실험은 아님).

### 미확인 → 문서 정정 필요
- **F6** "fold 2 진짜 최솟값 499" 서술은 `audit_addendum_te_leak.md`가 이미
  반증했으나 `es_protocol_final.md` 본문에는 정정 표시가 없음 (§5.1).

### 테스트 커버리지 공백
- **F7** 임계값 선택의 교차-fold 라벨 독립성을 검증하는 테스트 없음.
- **F8** teacher/student inner-holdout 파티션 동일성을 검증(또는 이를 의도된
  설계로 인정하고 회귀 고정)하는 테스트 없음.

---

## 7. 필요한 최소 수정 및 추가 테스트 제안 (서술만, 미구현)

이번 세션에서는 구현하지 않는다. Chief Architect가 작업지시서를 작성할 때
참고할 수 있도록 범위만 정리한다.

1. **F1/F2 (출력 로직)**: §4.5의 두 옵션 중 최소 "출력 경로 분리"
   (`baseline_recovery_v2.csv`/`.md`)를 먼저 적용하는 것이 리스크 대비
   구현량이 가장 작다. 근본 해결(스키마 지문 검사)은 별도 작업으로 분리
   가능.
2. **F3/F4 (P5 teacher 경로)**: 두 가지 독립적 선택지가 있다 — (a) teacher도
   `run_fold_nested_grid()`로 옮겨 student와 동일한 TE-inner 경계 차단을
   적용하되, `_make_inner_split()`이 fold당 **한 번만** 호출되도록(teacher용과
   student용이 우연히 같은 파티션을 또 공유하지 않도록) 별도의 파생 시드를
   쓰는 방안(예: `cfg.seed + fold + 1000`), 또는 (b) teacher가 애초에 `H`를
   보지 않도록 teacher 학습 전에 `tr`을 먼저 (student와 동일한 규칙으로)
   `tr_inner`/`H`로 나누고 teacher를 `tr_inner`만으로 학습시키는 방안. (b)가
   §2.2가 지적한 문제를 구조적으로 없애는 더 직접적인 방법이다. 비용
   증가분(`nested_grid_implementation.md`가 우려한 "8배") 여부는 어느
   선택지를 쓰느냐에 따라 다르다 — (b)는 그리드 스윕이 아니라 단순 재분할이라
   비용 증가가 없다.
3. **F5 (임계값 선택)**: §3.3의 "엄격한 outer-train 내부 선택" 방식을 도입할지
   여부는 실측(현재 방식과의 차이가 실제로 존재하는지) 후 결정할 사안 —
   구현 전에 먼저 두 방식의 차이를 작은 합성 실험으로 정량화하는 것을
   권장한다.
4. **F7 테스트 제안**: `tune_threshold_nested()`에 대해, "fold `j`의 확률을
   만든 모델이 fold `k`의 라벨을 학습에 썼다"는 상황을 합성 데이터로
   구성해(예: fold `j`의 확률이 fold `k`의 라벨과 상관되도록 의도적으로
   설계), 그 상관이 `deployment_threshold`/`nested_f1`에 미치는 영향을
   측정하는 감도 테스트.
5. **F8 테스트 제안**: `_make_inner_split()`을 이용해 "같은 `(n_rows, y, cfg, fold)`
   조합이면 항상 동일한 파티션을 돌려준다"는 성질 자체를 명시적으로
   문서화하는 계약 테스트를 추가하고, P5 경로에 대해서는 "teacher 학습
   인덱스와 student inner-holdout 인덱스가 겹치지 않아야 한다"는 **원하는
   최종 상태**를 먼저 실패하는 테스트로 작성해 두면(TDD), 위 2번 수정의
   완료 기준이 된다.

---

## 8. 전체 재실행 전 필수 해결 항목

Chief Architect/사용자가 Phase 1~6 전체 재실행을 승인하기 전에, 최소한
다음을 결정/해결해야 한다고 판단한다(우선순위순):

1. **[필수] F1/F2 해결** — 최소한 출력 경로를 `baseline_recovery_v2.*`로
   분리하거나, 기존 `output/baseline_recovery.csv`를 재실행 전에 별도
   위치로 백업(rename)한다. 이것 없이 재실행하면 결과 파일이 신뢰할 수
   없거나(F2) 손상될(F1, `--force` 사용 시) 위험이 확정적이다.
2. **[강력 권장] F3/F4에 대한 입장 결정** — 수정하고 재실행할지, 현재
   상태를 "알려진 한계"로 명시하고 P5 결과에 그 한계를 주석으로 남긴 채
   재실행할지 결정한다. 최소한 P5_leaky/P5_honest의 보고서 절(`_interpretation()`
   §3-1, §3-2)에 이 한계를 각주로 추가하는 것을 권장한다.
3. **[권장] F6 반영** — `es_protocol_final.md`에 "작업 2-B의 499는 이후
   `audit_addendum_te_leak.md` §3에서 TE-inner 리키지의 산물로 재규명됨"이라는
   정정 각주를 추가할지 Chief Architect가 판단한다(이번 세션은 기존 진단
   문서 수정 금지 대상이라 직접 고치지 않았다).
4. **[선택] F5/F7/F8** — 전체 재실행의 시간·비용을 늘리지 않는 범위이므로,
   재실행과 독립적으로 이후 스프린트에서 처리해도 무방하다고 판단한다.

이 문서 자체는 코드/기존 결과를 변경하지 않았다 — 위 항목의 실제 수정과
Phase 1~6 재실행은 `CLAUDE.md` §3의 승인 게이트를 거쳐 별도로 진행한다.
