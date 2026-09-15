# AUDIT.md 부록 — ES-inner-holdout 경계의 Target Encoding 리키지

> `AUDIT.md`는 수정하지 않는다. 이 문서는 그 §2(데이터 누수 감사)가 다루지 않은
> 사각지대 하나를 별도로 기록하기 위한 부록이다. 검증 스크립트와 원 데이터는
> `output/es_protocol_final.md` 작업 1·2(특히 2-B, 2-C)와 동일한 Phase 4 설정
> (TE + 원본 category 동시 투입, `inner_splits=5`, `te_drop_original=False`,
> `scale_pos_weight` 없음, seed=42)을 그대로 재사용했다. 실측 대상은 fold=1
> (0-indexed, 1-indexed로 "fold 2" — `es_diagnosis.md`/`es_protocol_final.md`
> 조건 A에서 `best_iteration=54`로 멈춘 바로 그 fold)이다.

---

## 1. 정정 대상 명시 — `AUDIT.md`의 판정이 다루지 않는 경계

`AUDIT.md` 14행(요약표)과 §2.2는 다음과 같이 판정한다.

> OOF Target Encoding의 fold 경계 — **fold 내부에서 계산됨, 정상** (TE를 사용하는
> 5개 스크립트 전부)

이 판정은 **`oof_target_encode(X_lab, y, outer_train_idx, outer_valid_idx, ...)`가
경계로 삼는 outer fold 검증 행**에 대해서는 지금도 옳다. 검증 행(`outer_valid_idx`)의
`TE_*`는 항상 `outer_train_idx` 라벨만으로 계산되며, 검증 행 자신의 라벨이 자기
인코딩에 들어가는 직접 누수는 없다 — `oof_target_encode()`의 `overlap` 검사와
`AUDIT.md` §2.2의 실측이 이를 뒷받침한다.

**이번에 발견한 것은 그보다 한 단계 안쪽, 기존 판정이 좌표조차 잡지 않았던 경계다.**
`src.cv.run_fold()`가 early stopping을 위해 `X_train`(outer-train, 이미 위
`oof_target_encode()`를 거쳐 TE 인코딩이 끝난 상태)을 다시 80/20으로 쪼개
inner-train/inner-holdout을 만드는데, **이 inner 경계에서는 TE가 다시 계산되지
않는다** — TE는 이미 outer 경계에서 한 번 계산된 상태로 `run_fold()`에 그대로
전달된다. `AUDIT.md`의 "정상" 판정은 이 inner 경계의 존재 자체를 검토 대상으로
삼지 않았다(당연하다 — `run_fold()`의 내부 80/20 분할은 `src/cv.py`가 담당하고,
TE 계산은 `src/features.py`가 담당하며, 두 모듈이 서로 독립적으로 "옳게" 설계되어
있어도 **호출 순서**가 문제를 만든다는 것은 두 모듈 중 어느 쪽의 단위 검증으로도
드러나지 않는다). 즉 이번 발견은 기존 §2.2 판정을 **뒤집는 것이 아니라, 그 판정이
검토하지 않은 별개의 좌표에서 발생하는 별개의 누수를 추가하는 것**이다.

## 2. 메커니즘

`rerun_all_phases.run_phase()`(약 470-494행)의 순서:

```python
te = oof_target_encode(X_lab, y, tr, va, cols=spec.cat_cols, m=TE_SMOOTHING_M,
                        inner_splits=spec.inner_splits, seed=CFG.seed,
                        drop_original=spec.te_drop_original, extra_fit_rows=extra)
X_tr, y_tr, X_va = te.X_train, te.y_train, te.X_valid
...
fit = run_fold(model_factory(spw), X_tr, y_tr, X_va, CFG, fold=fold, ...)
```

1. `oof_target_encode()`는 outer-train 전체(이 fold 기준 204,001행)를
   `inner_splits`(기본 5)개의 **TE 자체 내부 K-fold**로 회전시키며 각 행의 `TE_*`를
   계산한다. 즉 outer-train의 어떤 행이 TE-내부-fold `i`에 속하면, 그 행의 `TE_*`는
   나머지 TE-내부-fold(`i`를 제외한 4개)에 속한 행들의 라벨로 계산된다.
2. `run_fold()`는 이렇게 이미 TE 인코딩이 끝난 `X_tr`(204,001행)을
   `train_test_split(idx, test_size=0.2, random_state=seed+fold, stratify=y)`로
   ES-inner-train(80%)/ES-inner-holdout(20%)으로 **다시** 쪼갠다. 이 분할은
   1단계의 TE-내부-K-fold 분할과 **완전히 독립적인 별도의 무작위 분할**이다 —
   같은 시드 계열에서 나오지만 분할 알고리즘 자체가 다르고(`StratifiedKFold` 회전
   vs 단일 `train_test_split`), 두 분할의 경계가 정렬될 이유가 없다.
3. 그 결과 ES-inner-holdout 행 대다수는, 자신이 속한 TE-내부-fold가 **ES-inner-train
   쪽으로 분류된 행들의 라벨을 상당수 포함**하고 있어서, `TE_*` 값에 **ES-inner-train
   행들의 라벨 정보가 간접적으로 반영**된다. 반대로 ES-inner-train 쪽 행들의 `TE_*`에도
   ES-inner-holdout 행들의 라벨 정보가 새어 들어간다.
4. 트리 수가 늘어날수록 모델은 이 간접 정보(사실상 "그 지역/기체/항공사의 지연 평균에
   ES-inner-holdout 자신의 답이 일부 섞여 있다"는 신호)를 점점 더 정교하게 활용해
   ES-inner-holdout의 `binary_logloss`를 실제 일반화 능력과 무관하게 계속 낮출 수
   있다. patience 기반 early stopping은 이 허위 개선을 "아직 학습 중"이라고 오판해
   훨씬 더 오래 학습을 계속한다.

## 3. 실측 — fold 2(1-indexed)에서 직접 대조

같은 inner-train/inner-holdout 행 집합(`run_fold()`가 실제로 뽑는 것과 동일한
`train_test_split(idx, test_size=0.2, random_state=43, stratify=y)`, inner-train
163,200행 / inner-holdout 40,801행)에 대해 두 가지 TE 계산 순서만 바꿔
`n_estimators=700`까지 조기종료 없이 강제 학습했다(`record_evaluation`으로 라운드별
`binary_logloss` 기록, `scale_pos_weight` 없음, LightGBM 4.7.0, `src/cv.py`·
`src/features.py`·`rerun_all_phases.py`는 임포트만 하고 수정하지 않음).

| 버전 | TE 계산 순서 | 전역 최솟값 라운드 | 그 지점 inner-holdout LogLoss | outer-valid 진짜 LogLoss | outer-valid 진짜 AUC |
| :--- | :--- | ---: | ---: | ---: | ---: |
| A. 리키지 있음(현재 `run_fold()` 구조) | outer-train 전체 TE 인코딩 → 80/20 분할 | **499** | 0.444431 | 0.4572 | 0.6246 |
| B. clean(작업 2-C 5단계 절차) | 80/20 분할 → `oof_target_encode()`를 inner 경계에서 재호출 | **54** | 0.447913 | **0.4471** | **0.6446** |
| (참고) outer-valid 오라클 최적점 | — | LogLoss 최적 50 / AUC 최적 35 | — | 0.4471 | 0.6456 |

(리키지 있음 버전의 전역 최솟값이 라운드 499라는 것은 `es_protocol_final.md` 작업
2-B의 "부수 발견"과 `argmin=499, min=0.444431`까지 정확히 일치해 재현을 확인했다.)

**해석**: TE를 outer-train 전체에 먼저 계산하면(현재 구조, A) ES-inner-holdout
LogLoss가 라운드 499까지 계속 허위로 개선되는 것처럼 보인다 — 그 지점의
진짜(outer-valid) 성능은 오라클 최적점(50그루)보다 LogLoss +0.0101, AUC −0.0210
나쁘다. TE를 80/20 분할 **이후** inner 경계에서 다시 계산하면(clean, B), 곡선의
전역 최솟값이 라운드 **54**로 이동한다 — 오라클 최적점(50)과 불과 4라운드 차이이며,
그 지점의 outer-valid 성능(LogLoss 0.4471, AUC 0.6446)은 오라클 최적점(LogLoss
0.4471, AUC 0.6450)과 사실상 동일하다. 곡선의 **형태**도 근본적으로 다르다 — A는
정점 이후 완만하게 계속 개선되는 형태(진짜 U자형이 아니라 매우 얕은 우하향)인 반면,
B는 outer-valid 오라클 곡선과 거의 같은 U자형(초반 급락 → 라운드 ~54 부근 최저점 →
이후 뚜렷한 우상향)을 그린다. 그림: `output/te_leak_check_fold2.png`.

## 4. 영향 범위

**필요 조건 — early stopping이 실제로 inner-holdout을 보는 경우에만 발생한다.**
`n_estimators`를 고정하고 조기종료를 쓰지 않는 방식(예: `es_protocol_final.md`가
채택한 조건 C, 또는 단순 고정 스윕)은 애초에 inner-holdout을 학습 중 참조하지
않으므로 이 리키지의 영향을 받지 않는다. 즉 이 리키지가 문제를 일으키는 조건은
"TE 사용 + ES(조기종료) 동시 사용"이다.

`rerun_all_phases.py`의 `PhaseSpec` 중 `te=True`인 8개 전부가 해당 조건을
만족하며(코드 직접 확인, 181-238행), 전부 표준 `run_fold()`를 거쳐 학습되므로
영향권에 있다.

| Phase key | 비고 |
| :--- | :--- |
| P3 | Target Encoded (원본 범주 제거) |
| P4 | Hybrid (TE + 원본 category, 기준 Phase) |
| P4_te0 | Phase 4, `inner_splits=0` (TE 자체의 K-fold 회전이 없어도 outer→inner 분할 불일치 문제 자체는 동일하게 존재 — TE 통계가 outer-train 전체 라벨로 계산된다는 점은 같다) |
| P4_nospw | Phase 4, `scale_pos_weight` 미적용 (진단용) |
| P5_leaky | Pseudo-Labeling(누수 재현판) — 이 Phase는 이미 §2.1의 pseudo-label 누수를 안고 있고, 이번 TE-inner 리키지가 **추가로** 얹힌다 |
| P5_honest | Pseudo-Labeling(정직판) — pseudo-label 누수는 없지만 TE-inner 리키지는 동일하게 존재 |
| P6_legacy | Advanced(Traffic 레거시) |
| P6_fixed | Advanced(Traffic 결측 제외) |

원본 `run_*.py` 스크립트 중 TE를 쓰는 5개(`AUDIT.md` §1·§2에서 이미 특정한 목록과
일치) — `run_target_encoded.py`, `run_hybrid.py`, `run_pseudo_labeling.py`,
`run_advanced_features.py`, `run_phase7_weather_model.py` — 도 전부 자체 구현의
TE를 fold 학습 데이터 전체(또는 그 이상, `run_pseudo_labeling.py`의 경우
pseudo-label까지 포함)로 먼저 계산한 뒤 `eval_set`으로 조기종료를 거는 구조이므로,
동일한 형태의 inner-holdout TE 리키지 위험을 구조적으로 안고 있다. 다만 이
원본 스크립트들은 애초에 `es_diagnosis.md` §1이 확정한 `best_iteration∈{2,3}`
붕괴(즉 `binary_logloss`+`scale_pos_weight` 조합의 목적함수-지표 불일치) 때문에
조기종료가 사실상 "학습을 하지 못하는" 상태로 멈춰 있었다 — 그 상태에서는 TE-inner
리키지가 발현될 만큼 트리가 자라지 못했으므로, 원본 스크립트가 보고한 수치 자체는
이 리키지보다 §1의 붕괴가 지배적 원인이었을 가능성이 높다. 그러나 §1의 문제를
바로잡아 조기종료가 정상적으로 수십~수백 라운드 진행되게 만드는 순간(`rerun_all_phases.py`가
한 것처럼), 이번에 실측한 TE-inner 리키지가 그 자리를 대신 채운다.

## 5. `es_protocol_final.md`의 구조 변경 권고와의 연결

`es_protocol_final.md`는 이미(작업 2-C, "최종 권장 `n_estimators` 선택 절차" 5단계)
"TE를 outer-train 전체가 아니라 inner-train/inner-holdout 분할 이후 계산해야 한다"는
구조 변경을 권고했다. 그 문서의 서술은 이 변경을 주로 **naive nested n_estimators
선택이 붕괴하는 것을 막기 위한 수정**(작업 2-C)으로 제시했다.

이번 부록의 실측은 그 권고가 naive nested 선택 하나만의 문제가 아니라, **현재
`rerun_all_phases.py`가 실제로 쓰고 있는 표준 `run_fold()`(조건 A, ES 기반) 자체에도
구조적으로 동일한 리키지가 존재한다**는 것을 보여준다. 즉 5단계 절차의 "TE를
inner 경계 이후 계산하라"는 지시는 **성능 튜닝 상의 개선 제안이 아니라, 이미
운영 중인 조기종료 메커니즘에서 실제로 발생하고 있는 데이터 누수를 원천 차단하는
수정**이다 — 3절의 실측(리키지 있음일 때 outer-valid 성능이 오라클보다 뚜렷이
나쁜 지점에서 곡선이 멈추고, clean으로 바꾸면 곡선 형태 자체가 오라클과 거의
같아짐)이 그 근거다.

## 6. 명시적으로 바뀌지 않는 것

`es_protocol_final.md`의 최종 결론 — ES 계열(조건 A/B)을 폐기하고 조건 C(clean
nested 고정)로 대체해야 한다는 것, `rerun_all_phases.py`를 현재 코드 그대로
재실행해서는 안 된다는 것 — 은 이번 발견으로 바뀌지 않는다. 이번 부록은 그
결론의 **원인 서사**(fold 1의 "정점 부근 평탄 + 순수 잡음" 설명을, 최소한
일부 fold에 대해서는 "TE-inner 경계 리키지가 만든 체계적 편향"으로 보강)와
`AUDIT.md`의 **커버리지 공백**(outer fold 경계는 검토했으나 inner ES holdout
경계는 검토되지 않았음)을 메우는 것이 목적이며, 재실행 가부 판정 자체를
번복하지 않는다.
