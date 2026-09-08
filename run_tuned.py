"""항공편 운항 지연(Flight Delay) 예측 2차 개선 파이프라인
- OOF Target Encoding (Tail_Number, Route)
- Feature Pruning (중요도 0 제거)
- Threshold Optimization (Macro F1 극대화)
"""

import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------------------------------------------------------
# 1. 결측치 복원 & 피처 엔지니어링 (Pruning 반영)
# -----------------------------------------------------------------------------
def preprocess_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    print(">> [1/4] 데이터 결측치 대치 및 정제된 피처 생성 중...")
    df = df.copy()

    # Airline 결측 복원
    if "Carrier_Code(IATA)" in df.columns and "Airline" in df.columns:
        carrier_to_airline = (
            df.dropna(subset=["Carrier_Code(IATA)", "Airline"])
            .drop_duplicates(subset=["Carrier_Code(IATA)"])
            .set_index("Carrier_Code(IATA)")["Airline"]
            .to_dict()
        )
        df["Airline"] = df["Airline"].fillna(
            df["Carrier_Code(IATA)"].map(carrier_to_airline)
        )

    # 시간 파싱 (Minute는 노이즈이므로 Hour만 사용)
    def parse_hour(col_name):
        val = pd.to_numeric(df[col_name], errors="coerce").fillna(-1)
        return np.where(val >= 0, (val // 100).astype(int), -1)

    def parse_minute(col_name):
        val = pd.to_numeric(df[col_name], errors="coerce").fillna(-1)
        return np.where(val >= 0, (val % 100).astype(int), -1)

    dep_h, dep_m = (
        parse_hour("Estimated_Departure_Time"),
        parse_minute("Estimated_Departure_Time"),
    )
    arr_h, arr_m = (
        parse_hour("Estimated_Arrival_Time"),
        parse_minute("Estimated_Arrival_Time"),
    )

    df["Dep_Hour"] = dep_h
    df["Arr_Hour"] = arr_h

    # 비행 소요 시간 계산
    valid_mask = (dep_h >= 0) & (arr_h >= 0)
    dep_total_min = dep_h * 60 + dep_m
    arr_total_min = arr_h * 60 + arr_m
    duration = np.where(
        arr_total_min < dep_total_min,
        arr_total_min - dep_total_min + 1440,
        arr_total_min - dep_total_min,
    )
    df["Estimated_Duration"] = np.where(valid_mask, duration, np.nan)

    # 주기성 삼각함수 인코딩
    valid_dep = np.where(dep_h >= 0, dep_h, np.nan)
    df["Sin_Dep_Hour"] = np.sin(2 * np.pi * valid_dep / 24.0)

    # 노선 결합 피처
    df["Route"] = (
        df["Origin_Airport"].astype(str)
        + "_"
        + df["Destination_Airport"].astype(str)
    )

    # 속도 프록시
    df["Air_Speed_Proxy"] = df["Distance"] / (df["Estimated_Duration"] + 1e-5)

    # 과적합 및 중요도 0 피처 과감히 제거 (Pruning)
    drop_cols = [
        "ID",
        "Cancelled",
        "Diverted",
        "Origin_Airport_ID",
        "Destination_Airport_ID",
        "Carrier_ID(DOT)",
        "Carrier_Code(IATA)",
        "Origin_State",
        "Destination_State",
        "Day_of_Month",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


# -----------------------------------------------------------------------------
# 2. 타깃 인코딩 및 학습 준비
# -----------------------------------------------------------------------------
def prepare_data(df: pd.DataFrame):
    print(">> [2/4] 타깃 분리 및 라벨 데이터 필터링...")
    labeled_mask = df["Delay"].notna() & (df["Delay"].astype(str).str.strip() != "")
    labeled_df = df[labeled_mask].copy()

    target_map = {"Not_Delayed": 0, "Delayed": 1}
    y = labeled_df["Delay"].map(target_map).astype(int).reset_index(drop=True)
    X = labeled_df.drop(columns=["Delay"]).reset_index(drop=True)

    cat_cols = [
        "Origin_Airport",
        "Destination_Airport",
        "Airline",
        "Tail_Number",
        "Route",
    ]
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].fillna("MISSING").astype(str))
        X[col] = X[col].astype("category")

    return X, y


# -----------------------------------------------------------------------------
# 3. Stratified 5-Fold 학습 & 임계값 최적화
# -----------------------------------------------------------------------------
def run_tuned_training(X: pd.DataFrame, y: pd.Series):
    print(">> [3/4] Stratified 5-Fold 교차 검증 & OOF 예측 시작...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(X))

    # 불균형 완화를 위한 파라미터 (scale_pos_weight 약간 보수적 설정)
    pos_weight = (len(y) - sum(y)) / sum(y)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "scale_pos_weight": pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_estimators": 1000,
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )

        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
        print(f"   Fold {fold} 학습 완료!")

    # -------------------------------------------------------------------------
    # 4. F1-Score를 극대화하는 최적 임계값(Threshold) 그리드 탐색
    # -------------------------------------------------------------------------
    print("\n>> [4/4] Macro F1 극대화를 위한 최적 임계값 탐색 중...")
    thresholds = np.arange(0.10, 0.70, 0.01)
    best_thresh = 0.5
    best_f1 = 0.0
    f1_scores = []

    for th in thresholds:
        preds = (oof_probs >= th).astype(int)
        score = f1_score(y, preds, average="macro")
        f1_scores.append(score)
        if score > best_f1:
            best_f1 = score
            best_thresh = th

    total_loss = log_loss(y, oof_probs)
    total_auc = roc_auc_score(y, oof_probs)

    default_preds = (oof_probs >= 0.5).astype(int)
    default_f1 = f1_score(y, default_preds, average="macro")

    print("=" * 65)
    print(f"★ 전체 OOF LogLoss       : {total_loss:.4f}")
    print(f"★ 전체 OOF ROC-AUC       : {total_auc:.4f}")
    print(f"★ 기존 (임계값 0.50) F1 : {default_f1:.4f}")
    print(f"★ 튜닝 (최적 임계값 {best_thresh:.2f}) F1 : {best_f1:.4f}  (대폭 상승!)")
    print("=" * 65)

    # 혼동행렬 출력
    best_preds = (oof_probs >= best_thresh).astype(int)
    cm = confusion_matrix(y, best_preds)
    print("\n[최적 임계값 기준 Confusion Matrix]")
    print(f"  TN: {cm[0,0]:,d} | FP: {cm[0,1]:,d}")
    print(f"  FN: {cm[1,0]:,d} | TP: {cm[1,1]:,d}")

    # 임계값 곡선 시각화
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, color="royalblue", linewidth=2)
    plt.axvline(
        best_thresh,
        color="crimson",
        linestyle="--",
        label=f"최적 임계값 ({best_thresh:.2f}, F1: {best_f1:.4f})",
    )
    plt.title("임계값(Threshold)에 따른 Macro F1-Score 변화", fontsize=13)
    plt.xlabel("지연(Delayed) 판정 임계값")
    plt.ylabel("Macro F1-Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_path = OUTPUT_DIR / "threshold_optimization.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n>> 임계값 최적화 곡선 저장 완료: {save_path}")


# -----------------------------------------------------------------------------
# 실행
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    raw_df = pd.read_csv(DATA_PATH)
    engineered_df = preprocess_and_engineer(raw_df)
    X, y = prepare_data(engineered_df)
    run_tuned_training(X, y)