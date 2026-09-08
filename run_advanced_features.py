"""항공편 운항 지연(Flight Delay) 예측 6차 파이프라인
- Route 기반 결측 비행시간 역산 완벽 복원 (Duration Imputation)
- 공항별 시간대별 출발/도착 트래픽 혼잡도(Congestion) 피처 엔지니어링
- Hybrid Target Encoding + LightGBM 5-Fold OOF 교차 검증
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
# 1. 고도화된 전처리 및 결측치 완전 역산 복원
# -----------------------------------------------------------------------------
def preprocess_advanced(df: pd.DataFrame) -> pd.DataFrame:
    print(">> [1/5] 항공사 매핑 대치 및 노선 피처 생성 중...")
    df = df.copy()

    # Airline 결측 복원
    if "Carrier_Code(IATA)" in df.columns and "Airline" in df.columns:
        c_to_a = (
            df.dropna(subset=["Carrier_Code(IATA)", "Airline"])
            .drop_duplicates(subset=["Carrier_Code(IATA)"])
            .set_index("Carrier_Code(IATA)")["Airline"]
            .to_dict()
        )
        df["Airline"] = df["Airline"].fillna(
            df["Carrier_Code(IATA)"].map(c_to_a)
        )

    # 노선 결합 피처 (사전 생성)
    df["Route"] = (
        df["Origin_Airport"].astype(str)
        + "_"
        + df["Destination_Airport"].astype(str)
    )

    # 시간 분리 파싱
    def parse_time(col_name):
        val = pd.to_numeric(df[col_name], errors="coerce").fillna(-1)
        h = np.where(val >= 0, (val // 100).astype(int), -1)
        m = np.where(val >= 0, (val % 100).astype(int), -1)
        return h, m

    dep_h, dep_m = parse_time("Estimated_Departure_Time")
    arr_h, arr_m = parse_time("Estimated_Arrival_Time")

    df["Dep_Hour"] = dep_h
    df["Dep_Minute"] = dep_m
    df["Arr_Hour"] = arr_h
    df["Arr_Minute"] = arr_m

    # 유효한 출발/도착 시간 기준 초기 소요시간 계산
    valid_time_mask = (dep_h >= 0) & (arr_h >= 0)
    dep_total_m = dep_h * 60 + dep_m
    arr_total_m = arr_h * 60 + arr_m
    raw_duration = np.where(
        arr_total_m < dep_total_m,
        arr_total_m - dep_total_m + 1440,
        arr_total_m - dep_total_m,
    )
    df["Estimated_Duration"] = np.where(valid_time_mask, raw_duration, np.nan)

    print(">> [2/5] 노선(Route) 기반 비행 소요시간 결측치 역산 복원 중...")
    # 노선별 중앙값 소요시간 산출 (대체 기준)
    route_median_dur = df.groupby("Route")["Estimated_Duration"].transform(
        "median"
    )
    global_median_dur = df["Estimated_Duration"].median()

    # 결측된 Duration 채우기
    df["Estimated_Duration"] = df["Estimated_Duration"].fillna(route_median_dur)
    df["Estimated_Duration"] = df["Estimated_Duration"].fillna(
        global_median_dur
    )

    # 단측 결측 시간 복원
    # 1) 출발시간만 있고 도착시간이 없는 경우
    arr_missing = (df["Arr_Hour"] == -1) & (df["Dep_Hour"] >= 0)
    calc_arr_m = (
        df.loc[arr_missing, "Dep_Hour"] * 60
        + df.loc[arr_missing, "Dep_Minute"]
        + df.loc[arr_missing, "Estimated_Duration"]
    ) % 1440
    df.loc[arr_missing, "Arr_Hour"] = (calc_arr_m // 60).astype(int)
    df.loc[arr_missing, "Arr_Minute"] = (calc_arr_m % 60).astype(int)

    # 2) 도착시간만 있고 출발시간이 없는 경우
    dep_missing = (df["Dep_Hour"] == -1) & (df["Arr_Hour"] >= 0)
    calc_dep_m = (
        df.loc[dep_missing, "Arr_Hour"] * 60
        + df.loc[dep_missing, "Arr_Minute"]
        - df.loc[dep_missing, "Estimated_Duration"]
    ) % 1440
    df.loc[dep_missing, "Dep_Hour"] = (calc_dep_m // 60).astype(int)
    df.loc[dep_missing, "Dep_Minute"] = (calc_dep_m % 60).astype(int)

    # 공항별 시간대별 운항 편수(혼잡도) 피처 생성
    print(">> [3/5] 공항별 시간대별 출발/도착 트래픽 혼잡도(Congestion) 피처 생성...")
    df["Origin_Traffic"] = df.groupby(
        ["Month", "Day_of_Month", "Origin_Airport", "Dep_Hour"]
    )["Route"].transform("count")
    df["Dest_Traffic"] = df.groupby(
        ["Month", "Day_of_Month", "Destination_Airport", "Arr_Hour"]
    )["Route"].transform("count")

    # 주기성 삼각함수 인코딩
    valid_dep = np.where(df["Dep_Hour"] >= 0, df["Dep_Hour"], 12)
    df["Sin_Dep_Hour"] = np.sin(2 * np.pi * valid_dep / 24.0)
    df["Cos_Dep_Hour"] = np.cos(2 * np.pi * valid_dep / 24.0)

    # 속도 프록시
    df["Air_Speed_Proxy"] = df["Distance"] / (df["Estimated_Duration"] + 1e-5)

    # 노이즈 및 식별자 피처 삭제
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
        "Dep_Minute",
        "Arr_Minute",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df


def get_smoothed_target_encoding(
    train_s: pd.Series, target: pd.Series, test_s: pd.Series, m: float = 20.0
):
    global_mean = target.mean()
    stats = target.groupby(train_s).agg(["count", "mean"])
    counts = stats["count"]
    means = stats["mean"]
    smooth = (counts * means + m * global_mean) / (counts + m)

    train_encoded = train_s.map(smooth).fillna(global_mean).values
    test_encoded = test_s.map(smooth).fillna(global_mean).values
    return train_encoded, test_encoded


# -----------------------------------------------------------------------------
# 2. 모델 학습 파이프라인
# -----------------------------------------------------------------------------
def run_advanced_training(df: pd.DataFrame):
    print(">> [4/5] 데이터 분리 및 범주형 인코딩...")
    labeled_mask = df["Delay"].notna() & (df["Delay"].astype(str).str.strip() != "")
    labeled_df = df[labeled_mask].copy().reset_index(drop=True)

    target_map = {"Not_Delayed": 0, "Delayed": 1}
    y = labeled_df["Delay"].map(target_map).astype(int)
    X = labeled_df.drop(columns=["Delay"])

    cat_cols = [
        "Tail_Number",
        "Route",
        "Origin_Airport",
        "Destination_Airport",
        "Airline",
    ]

    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].fillna("MISSING").astype(str))
        X[col] = X[col].astype("category")

    print(
        f">> [5/5] Stratified 5-Fold 교차 검증 시작 (라벨 {len(X):,d}건 전수 검증)..."
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(X))
    pos_weight = (len(y) - sum(y)) / sum(y)

    # 튜닝된 GBDT 파라미터 (트리 수 확장, 분기 세밀화)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 79,
        "max_depth": 9,
        "min_child_samples": 40,
        "scale_pos_weight": pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "random_state": 42,
        "n_estimators": 1200,
        "verbose": -1,
    }

    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx]

        for col in cat_cols:
            tr_enc, val_enc = get_smoothed_target_encoding(
                X_train[col], y_train, X_val[col], m=25.0
            )
            X_train[f"TE_{col}"] = tr_enc
            X_val[f"TE_{col}"] = val_enc

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)],
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs

        fold_auc = roc_auc_score(y_val, val_probs)
        fold_aucs.append(fold_auc)
        print(f"   Fold {fold} 완료 | Fold ROC-AUC: {fold_auc:.4f}")

    # 최종 채점
    thresholds = np.arange(0.15, 0.50, 0.01)
    best_thresh = 0.5
    best_f1 = 0.0

    for th in thresholds:
        preds = (oof_probs >= th).astype(int)
        score = f1_score(y, preds, average="macro")
        if score > best_f1:
            best_f1 = score
            best_thresh = th

    total_loss = log_loss(y, oof_probs)
    total_auc = roc_auc_score(y, oof_probs)
    default_f1 = f1_score(y, (oof_probs >= 0.5).astype(int), average="macro")

    print("\n" + "=" * 65)
    print("★ [Phase 6: 결측 역산 복원 + 공항 혼잡도 피처 최종 성적표]")
    print(f"★ 전체 OOF LogLoss       : {total_loss:.4f}")
    print(f"★ 전체 OOF ROC-AUC       : {total_auc:.4f}")
    print(f"★ 기존 (임계값 0.50) F1 : {default_f1:.4f}")
    print(f"★ 튜닝 (최적 임계값 {best_thresh:.2f}) F1 : {best_f1:.4f}")
    print("=" * 65)

    best_preds = (oof_probs >= best_thresh).astype(int)
    cm = confusion_matrix(y, best_preds)
    print("\n[최적 임계값 기준 Confusion Matrix]")
    print(f"  TN: {cm[0,0]:,d} | FP: {cm[0,1]:,d}")
    print(f"  FN: {cm[1,0]:,d} | TP: {cm[1,1]:,d}")
    print(f"  지연 감지율(Recall): {cm[1,1]/(cm[1,0]+cm[1,1])*100:.2f}%")


if __name__ == "__main__":
    raw_df = pd.read_csv(DATA_PATH)
    processed_df = preprocess_advanced(raw_df)
    run_advanced_training(processed_df)