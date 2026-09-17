# 확률 보정 실험 결과 — baseline_recovery_v2_calibration_smoke_20260917

비교는 같은 조건·방식·시드·그룹 안에서 보정 없음 대비 짝지은 차이입니다.
±는 3시드 표본 표준편차이며 신뢰구간이나 유의성의 근거가 아닙니다.
음수가 좋은 지표: Brier, ECE, LogLoss.

## 전체 라벨 행

| 조건 | 방식 | 보정기 | ΔBrier | ΔECE(%p) | ΔLogLoss | ΔMacro F1 | Δ지연 재현율 |
|---|---|---|---:|---:|---:|---:|---:|
| P6_clean | 공유형 | isotonic | +0.002005 ± nan | +0.9309 ± nan | +0.332423 ± nan | +0.008428 ± nan | +5.56%p ± nan |
| P6_clean | 공유형 | platt | -0.001238 ± nan | -1.8033 ± nan | -0.003407 ± nan | -0.009975 ± nan | +5.56%p ± nan |
| P6_clean | 분리형 | isotonic | +0.001316 ± nan | +3.8026 ± nan | +0.453919 ± nan | -0.013414 ± nan | -17.78%p ± nan |
| P6_clean | 분리형 | platt | -0.001014 ± nan | -1.2582 ± nan | -0.002334 ± nan | -0.019769 ± nan | +5.56%p ± nan |
| P6_fixed | 공유형 | isotonic | +0.003850 ± nan | +1.1414 ± nan | +0.203912 ± nan | +0.013618 ± nan | +5.56%p ± nan |
| P6_fixed | 공유형 | platt | -0.001012 ± nan | -3.3501 ± nan | -0.003253 ± nan | +0.016224 ± nan | +2.22%p ± nan |
| P6_fixed | 분리형 | isotonic | +0.003331 ± nan | +2.4515 ± nan | +0.593920 ± nan | -0.000484 ± nan | -4.44%p ± nan |
| P6_fixed | 분리형 | platt | -0.000166 ± nan | -1.5024 ± nan | +0.000878 ± nan | -0.011983 ± nan | -21.11%p ± nan |

## 양쪽 시각 결측 그룹

| 조건 | 방식 | 보정기 | ΔBrier | ΔECE(%p) | ΔLogLoss | ΔMacro F1 | Δ지연 재현율 |
|---|---|---|---:|---:|---:|---:|---:|
| P6_clean | 공유형 | isotonic | +0.006617 ± nan | +4.4870 ± nan | -0.017807 ± nan | -0.089286 ± nan | +0.00%p ± nan |
| P6_clean | 공유형 | platt | -0.015848 ± nan | -16.5170 ± nan | -0.075532 ± nan | -0.089286 ± nan | +0.00%p ± nan |
| P6_clean | 분리형 | isotonic | +0.007755 ± nan | +4.9425 ± nan | -0.003966 ± nan | +0.089286 ± nan | +0.00%p ± nan |
| P6_clean | 분리형 | platt | -0.030278 ± nan | -19.2360 ± nan | -0.126103 ± nan | +0.000000 ± nan | +0.00%p ± nan |
| P6_fixed | 공유형 | isotonic | +0.013218 ± nan | +18.3652 ± nan | +0.043249 ± nan | -0.069444 ± nan | +0.00%p ± nan |
| P6_fixed | 공유형 | platt | -0.005734 ± nan | -1.0611 ± nan | -0.018194 ± nan | +0.000000 ± nan | +0.00%p ± nan |
| P6_fixed | 분리형 | isotonic | -0.007158 ± nan | -1.6149 ± nan | -0.028153 ± nan | -0.069444 ± nan | +0.00%p ± nan |
| P6_fixed | 분리형 | platt | -0.013242 ± nan | +0.8453 ± nan | -0.046482 ± nan | +0.000000 ± nan | +0.00%p ± nan |

## 양쪽 시각 관측 그룹

| 조건 | 방식 | 보정기 | ΔBrier | ΔECE(%p) | ΔLogLoss | ΔMacro F1 | Δ지연 재현율 |
|---|---|---|---:|---:|---:|---:|---:|
| P6_clean | 공유형 | isotonic | +0.001996 ± nan | +1.3982 ± nan | +0.321608 ± nan | +0.002738 ± nan | +6.85%p ± nan |
| P6_clean | 공유형 | platt | -0.000537 ± nan | -1.5543 ± nan | -0.000638 ± nan | -0.026027 ± nan | +4.11%p ± nan |
| P6_clean | 분리형 | isotonic | +0.001014 ± nan | +3.6257 ± nan | +0.480051 ± nan | -0.024254 ± nan | -19.18%p ± nan |
| P6_clean | 분리형 | platt | -0.000315 ± nan | -0.9266 ± nan | +0.000493 ± nan | -0.042640 ± nan | +4.11%p ± nan |
| P6_fixed | 공유형 | isotonic | +0.002174 ± nan | +0.4974 ± nan | +0.163745 ± nan | +0.015073 ± nan | +5.48%p ± nan |
| P6_fixed | 공유형 | platt | -0.000934 ± nan | -3.9142 ± nan | -0.003047 ± nan | +0.013491 ± nan | +4.11%p ± nan |
| P6_fixed | 분리형 | isotonic | +0.001638 ± nan | +1.3072 ± nan | +0.566155 ± nan | +0.014668 ± nan | -1.37%p ± nan |
| P6_fixed | 분리형 | platt | +0.000036 ± nan | -2.8360 ± nan | +0.001347 ± nan | -0.023261 ± nan | -21.92%p ± nan |
