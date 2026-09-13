# ✈️ Flight Delay Prediction — Project Master Plan & Roadmap

> **저장소:** `https://github.com/Peter-jackson12/Airplane.git`  
> **환경:** Windows PowerShell, Python 3.14, `uv` 패키지 매니저, LightGBM, Pandas, Scikit-learn  
> **최종 갱신일:** 2026-09-08 (Phase 6 완료 시점)

---

## 1. 프로젝트 개요 및 핵심 데이터셋 명세

* **문제 정의:** 미국 교통통계국(BTS) 항공 운항 데이터를 기반으로 항공편의 15분 이상 도착 지연(`Delay`) 여부를 사전 예측 (불균형 이진 분류)
* **평가 지표:** **Macro F1-Score**, **Log Loss** (보조: **ROC-AUC**)
* **데이터 규모 (100만 행):**
  * 원본 데이터: `data/train.csv` (1,000,000행 × 19열, 약 111.3MB, `.gitignore` 처리 완료)
  * 라벨 데이터(학습/검증용): **255,001행 (25.5%)**
    * `Not_Delayed` (0): 210,001건 (82.35%)
    * `Delayed` (1): 45,000건 (17.65%) ➔ **클래스 불균형 약 4.67 : 1**
  * 미라벨 데이터(준지도 학습용): **744,999행 (74.5%)** (`Delay` 결측치)

---

## 2. 지금까지 달성한 6단계 실험 및 성능 추이 (Phase 1 ~ Phase 6)

모든 평가는 라벨 데이터 255,001건에 대해 **Stratified 5-Fold OOF (Out-of-Fold)**로 데이터 누수 없이 공정하게 측정됨.

| 실험 단계 | 스크립트 파일명 | 핵심 엔지니어링 기법 | 최적 임계값 | Log Loss | ROC-AUC | Macro F1 | 지연 감지(TP) | 비고 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 1** | `run_baseline.py` | 기본 전처리 + Label Encoding | 0.50 | 0.4636 | 0.6366 | 0.4516 | 0건 | 임계값 0.50 All-Zero 참사 |
| **Phase 2** | `run_tuned.py` | 노이즈 Pruning + Threshold Search | **0.22** | 0.4635 | 0.6357 | **0.5724** | 13,039건 | F1 폭등 (+26.7%), TP 1.3만 건 구출 |
| **Phase 3** | `run_target_encoded.py` | 카테고리 삭제 후 OOF TE 단독 치환 | 0.22 | 0.4635 | 0.6043 | 0.5521 | 14,895건 | 카테고리 분기력 상실로 AUC 급락 |
| **Phase 4** | `run_hybrid.py` | Category Dtype + OOF TE 동시 투입 | 0.22 | 0.4633 | 0.6107 | 0.5579 | 14,386건 | AUC 반등 및 LogLoss 최저치 경신 |
| **Phase 5** | `run_pseudo_labeling.py` | **8.9만 건 증강 준지도 학습** | **0.24** | **0.4587** | **0.6416** | **0.5746** | 13,527건 | **전 지표 최고치 (LogLoss 0.45대 돌파)** |
| **Phase 6** | `run_advanced_features.py` | **시간 결측 역산 + 공항 트래픽** | **0.21** | **0.4629** | **0.6345** | 0.5721 | **15,866건** | **순수 라벨 기준 역대 최다 지연 적발** |

---

## 3. 핵심 도메인 규칙 및 검증된 트러블슈팅 인사이트

1. **임계값과 베이스 레이트 연동:** 자연 지연율이 17.65%이므로 기본값 0.50은 무조건 실패함. 최적 임계값은 항상 **`0.21 ~ 0.24`** 구간에서 형성됨.
2. **하이브리드 카테고리 보존:** 고카디널리티 변수(`Tail_Number`, `Route` 등)는 Target Encoding만 쓰면 안 되며, **원본 판다스 `category` 컬럼과 `TE_` 수치 컬럼을 반드시 함께 투입**해야 함.
3. **안전한 준지도 학습(Pseudo-Labeling):** 단순 백분위수 15:15 선별은 정상 확률 78%짜리를 지연으로 속이는 데이터 오염을 유발함. **정상은 하위 10%($p \le 0.184$), 지연은 상위 2%($p \ge 0.235$)만 엄선**해야 성능이 상승함 (89,438건 증강 성공).
4. **시간 결측 역산 공식:** 출발/도착 중 한쪽만 있는 단측 결측치(10.9%)는 동일 노선(`Route`)의 중앙값 소요시간(`median_duration`)으로 역산($Arr = Dep + Dur$)하여 `Estimated_Duration` 결측치를 0%로 제거함.
5. **판다스 `category` 병합 주의:** `pd.concat` 후 카테고리 목록 불일치로 `object`로 풀리는 버그 방지를 위해 병합 후 `df[col] = df[col].astype("category")` 명시적 재선언 필수.

---

## 4. 향후 로드맵 (Next Steps)

* [ ] **Phase 7: 그랜드 슬램 융합 모델 (`run_grand_slam.py`)**
  * Phase 5의 [8.9만 건 준지도 증강 데이터] + Phase 6의 [시간 결측 역산 & 공항 트래픽 피처]를 단일 파이프라인으로 결합.
  * 목표: LogLoss 0.455대 진입 및 ROC-AUC 0.6500 돌파.
* [ ] **Phase 8: CatBoost / XGBoost 이종 앙상블**
  * 고카디널리티에 특화된 CatBoost 교차 검증 후 LightGBM과 5:5 Soft-Voting 앙상블.
* [ ] **Phase 9: 모델 경량화 및 추론 서빙 파이프라인 (MLOps)**
  * 실시간 추론용 단일 스크립트 작성 및 Feature Pipeline 직렬화 (`joblib` / `onnx`).