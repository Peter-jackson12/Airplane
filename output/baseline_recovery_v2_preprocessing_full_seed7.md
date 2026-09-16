# 기준선 복구 — Phase 1~6 통일 프로토콜 재측정 결과

> 생성 시각: 2026-09-16T02:39:45+00:00  
> 원본 데이터: `baseline_recovery_v2_preprocessing_full_seed7.csv`  
> 실험 ID: `b6db8b293186d53fe0a39aa144aea8ee5491a76d61360abf689c2da7de2e22a8`  
> P5_leaky는 의도적 누수 대조군이며 성능 순위에서 제외한다.  
> 근거 문서: [AUDIT.md](../AUDIT.md) §3, [PLAN.md](../PLAN.md) §4-0 P0-2·P0-3

---

## 1. 적용한 통일 프로토콜

| 항목 | 값 |
| :--- | :--- |
| `n_splits` | `5` |
| `shuffle` | `True` |
| `seed` | `7` |
| `stopping_rounds` | `40` |
| `threshold_grid` | `[0.1, 0.7, 0.01]` |
| `metric_aggregation` | `pooled_oof` |
| `inner_holdout_frac` | `0.2` |
| `n_thresholds` | `61` |
| `inner_early_stopping` | `False` |
| `nested_threshold` | `True` |
| `threshold_selection` | `outer_train_holdout` |
| `tree_selection` | `nested_grid` |
| `n_estimators_grid` | `[10, 25, 50, 75, 100, 150, 300, 600]` |
| `teacher_scope` | `student_inner_train` |
| `teacher_selection` | `nested_grid` |
| `stopping_rounds_used` | `False` |

**LightGBM 하이퍼파라미터 (전 Phase 동일 고정, Phase 4 설정 기준)**

```json
{
  "objective": "binary",
  "metric": "binary_logloss",
  "boosting_type": "gbdt",
  "learning_rate": 0.05,
  "num_leaves": 63,
  "max_depth": 8,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "random_state": 7,
  "verbose": -1
}
```

추가 고정 사항:

* TE 평활 계수 `m` = 20.0 — 기존 Phase 6 만 25.0 이었다.
* TE 내부 K-fold `inner_splits` = 5 — 기존 5개 스크립트는 0 에 해당한다.
* 범주 vocabulary 는 전 Phase 라벨+미라벨 합쳐 fit — 기존에는 Phase 5 만 그랬다.
* `scale_pos_weight` 는 전 Phase 미적용(es_protocol_final.md 작업 5, 기본값 1.0).
* `n_estimators` 는 ES 대신 nested grid(`[10, 25, 50, 75, 100, 150, 300, 600]`)로 fold마다 선택한다 
  (es_protocol_final.md 조건 C — TE-inner 리키지 차단, `run_fold_nested_grid()`).
* Phase 7-Ex 는 원천 데이터 결함으로 재측정 대상에서 제외.

---

## 2. 구프로토콜 대 신프로토콜 대조표

`macro_f1_nested` 가 보고용 값이다. `naive` 는 기존 6개 스크립트 방식(전체 OOF 
최댓값)으로, winner's curse 를 포함하므로 성능 수치로 인용해서는 안 된다.

| Phase | LogLoss 구 | LogLoss 신 | AUC 구 | AUC 신 | F1 구 | **F1 신(nested)** | F1 신(naive) | naive−보고 F1 | TP 구 | TP 신 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 4: Hybrid | 0.4633 | 0.4480 | 0.6107 | 0.6410 | 0.5579 | **0.5752** | 0.5754 | +0.0002 | 14,386 | 15,364 |
| Phase 6: Advanced (Traffic 결측 제외) | — | 0.4474 | — | 0.6433 | — | **0.5773** | 0.5770 | +-0.0002 | — | 15,606 |
| Phase 4: 전처리 개선 | — | 0.4481 | — | 0.6410 | — | **0.5752** | 0.5754 | +0.0002 | — | 15,555 |
| Phase 6: 전처리 개선 | — | 0.4478 | — | 0.6418 | — | **0.5751** | 0.5754 | +0.0004 | — | 14,674 |
| Phase 4: Hybrid / impute only | — | 0.4481 | — | 0.6407 | — | **0.5749** | 0.5749 | +0.0000 | — | 16,193 |
| Phase 4: Hybrid / ratio only | — | 0.4483 | — | 0.6401 | — | **0.5742** | 0.5742 | +0.0000 | — | 14,434 |
| Phase 4: Hybrid / missing only | — | 0.4479 | — | 0.6417 | — | **0.5756** | 0.5759 | +0.0002 | — | 15,767 |
| Phase 6: Advanced (Traffic 결측 제외) / impute only | — | 0.4474 | — | 0.6430 | — | **0.5770** | 0.5769 | +-0.0000 | — | 15,559 |
| Phase 6: Advanced (Traffic 결측 제외) / ratio only | — | 0.4475 | — | 0.6428 | — | **0.5764** | 0.5766 | +0.0002 | — | 15,272 |
| Phase 6: Advanced (Traffic 결측 제외) / missing only | — | 0.4476 | — | 0.6423 | — | **0.5757** | 0.5760 | +0.0003 | — | 15,896 |

> 구프로토콜 Phase 1 의 F1 은 **fold 평균** 집계이고 임계값 탐색도 없었다(0.50 고정). 
> 신프로토콜 값과는 애초에 같은 종류의 양이 아니므로 증감 해석에 쓸 수 없다.

### 2-1. 실행 세부

| Phase | 피처 수 | 배포 임계값 | fold별 임계값 | fold별 선택 n_estimators(nested grid) | pseudo 선별 | 소요(초) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: |
| Phase 4: Hybrid | 20 | 0.22 | [0.23, 0.22, 0.23, 0.22, 0.22] | [50, 50, 50, 50, 50] | — | 64.0 |
| Phase 6: Advanced (Traffic 결측 제외) | 24 | 0.23 | [0.23, 0.23, 0.23, 0.22, 0.22] | [50, 50, 50, 50, 50] | — | 68.3 |
| Phase 4: 전처리 개선 | 26 | 0.22 | [0.22, 0.22, 0.23, 0.22, 0.23] | [50, 50, 75, 50, 75] | — | 72.2 |
| Phase 6: 전처리 개선 | 30 | 0.23 | [0.23, 0.23, 0.23, 0.23, 0.24] | [75, 75, 75, 75, 50] | — | 69.3 |
| Phase 4: Hybrid / impute only | 20 | 0.22 | [0.22, 0.22, 0.22, 0.22, 0.22] | [50, 50, 50, 50, 50] | — | 62.2 |
| Phase 4: Hybrid / ratio only | 22 | 0.23 | [0.23, 0.23, 0.23, 0.22, 0.23] | [50, 50, 50, 50, 50] | — | 70.4 |
| Phase 4: Hybrid / missing only | 24 | 0.22 | [0.22, 0.22, 0.22, 0.22, 0.23] | [50, 50, 50, 50, 50] | — | 66.1 |
| Phase 6: Advanced (Traffic 결측 제외) / impute only | 24 | 0.23 | [0.23, 0.23, 0.23, 0.22, 0.22] | [50, 50, 50, 50, 50] | — | 64.9 |
| Phase 6: Advanced (Traffic 결측 제외) / ratio only | 26 | 0.23 | [0.22, 0.23, 0.23, 0.23, 0.23] | [50, 50, 75, 50, 75] | — | 68.8 |
| Phase 6: Advanced (Traffic 결측 제외) / missing only | 28 | 0.23 | [0.23, 0.22, 0.23, 0.22, 0.23] | [75, 75, 50, 50, 75] | — | 70.9 |

---

## 3. 해석 범위

트리 수와 임계값은 각 outer-train 내부 holdout에서 선택했다.
혼동행렬과 recall도 각 fold의 사전 선택 임계값으로 계산한다.
배포 임계값은 fold별 임계값의 중앙값이며 전체 데이터 재학습 시 별도 검증이 필요하다.
naive F1과 보고 F1의 차이는 진단값이며 편향의 인과 추정치가 아니다.
단일 시드 결과나 0.002 기준만으로 유의성/동률을 판정하지 않는다.
후속 비교는 동일 시드/분할의 Phase 간 차이를 짝지어 보고해야 한다.
구프로토콜 수치는 역사적 기록이며 새 수치와의 차이를 피처 효과로 귀속하지 않는다.

| Phase | Macro F1 | LogLoss | AUC |
| :--- | ---: | ---: | ---: |
| Phase 6: Advanced (Traffic 결측 제외) | 0.5773 | 0.4474 | 0.6433 |
| Phase 6: Advanced (Traffic 결측 제외) / impute only | 0.5770 | 0.4474 | 0.6430 |
| Phase 6: Advanced (Traffic 결측 제외) / ratio only | 0.5764 | 0.4475 | 0.6428 |
| Phase 6: Advanced (Traffic 결측 제외) / missing only | 0.5757 | 0.4476 | 0.6423 |
| Phase 4: Hybrid / missing only | 0.5756 | 0.4479 | 0.6417 |
| Phase 4: Hybrid | 0.5752 | 0.4480 | 0.6410 |
| Phase 4: 전처리 개선 | 0.5752 | 0.4481 | 0.6410 |
| Phase 6: 전처리 개선 | 0.5751 | 0.4478 | 0.6418 |
| Phase 4: Hybrid / impute only | 0.5749 | 0.4481 | 0.6407 |
| Phase 4: Hybrid / ratio only | 0.5742 | 0.4483 | 0.6401 |

P5_leaky는 outer-valid 라벨의 간접 유입을 의도적으로 유지한 진단 대조군이다.
P4_nospw는 현재 P4와 동일한 호환용 별칭이므로 순위에서 제외했다.