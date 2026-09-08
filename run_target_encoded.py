"""항공편 운항 지연(Flight Delay) 예측 3차 개선 파이프라인
- Bayesian Smoothed OOF Target Encoding (Tail_Number, Route, Airline, Origin_Airport)
- Threshold Optimization (Macro F1 재극대화)
- Out-of-Fold Data Leakage 완전 차단
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
# 1. 전처리 및 기본 파생 피처 생성
# -----------------------------------------------------------------------------
def preprocess_base(df: pd.DataFrame) -> pd.DataFrame:
    print(">> [1/5] 기본 결측 대치 및 파생 피처 생성 중...")
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

    # 시간 파싱
    def parse_time(col_name):
        val = pd.to_numeric(df[col_name], errors="coerce").fillna(-1)
        h = np.where(val >= 0, (val // 100).astype(int), -1)
        m = np.where(val >= 0, (val % 100).astype(int), -1)
        return h, m

    dep_h, dep_m = parse_time("Estimated_Departure_Time")
    arr_h, arr_m = parse_time("Estimated_Arrival_Time")

    df["Dep_Hour"] = dep_h
    df["Arr_Hour"] = arr_h

    # 비행 소요 시간 (자정 보정)
    valid_mask = (dep_h >= 0) & (arr_h >= 0)
    dep_total_m = dep_h * 60 + dep_m
    arr_total_m = arr_h * 60 + arr_m
    duration = np.where(
        arr_total_m < dep_total_m,
        arr_total_m - dep_total_m + 1440,
        arr_total_m - dep_total_m,
    )
    df["Estimated_Duration"] = np.where(valid_mask, duration, np.nan)

    # 주기성 삼각함수 인코딩
    valid_dep = np.where(dep_h >= 0, dep_h, np.nan)
    df["Sin_Dep_Hour"] = np.sin(2 * np.pi * valid_dep / 24.0)
    df["Cos_Dep_Hour"] = np.cos(2 * np.pi * valid_dep / 24.0)

    # 노선 결합 피처
    df["Route"] = (
        df["Origin_Airport"].astype(str)
        + "_"
        + df["Destination_Airport"].astype(str)
    )

    # 속도 프록시
    df["Air_Speed_Proxy"] = df["Distance"] / (df["Estimated_Duration"] + 1e-5)

    # 불필요 피처 정리
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
# 2. 베이지안 평활화 타깃 인코더 함수
# -----------------------------------------------------------------------------
def get_smoothed_target_encoding(
    train_s: pd.Series, target: pd.Series, test_s: pd.Series, m: float = 20.0
):
    """표본 수가 적을 때 전체 평균으로 수렴시키는 베이지안 스무딩 적용 타깃 인코딩"""
    global_mean = target.mean()
    stats = target.groupby(train_s).agg(["count", "mean"])
    counts = stats["count"]
    means = stats["mean"]

    # 평활화 수식 적용
    smooth = (counts * means + m * global_mean) / (counts + m)

    # 매핑 (새로운 카테고리는 글로벌 평균으로 대치)
    train_encoded = train_s.map(smooth).fillna(global_mean).values
    test_encoded = test_s.map(smooth).fillna(global_mean).values

    return train_encoded, test_encoded


# -----------------------------------------------------------------------------
# 3. Stratified 5-Fold OOF 타깃 인코딩 & 모델 학습
# -----------------------------------------------------------------------------
def run_target_encoded_pipeline(df: pd.DataFrame):
    print(">> [2/5] 타깃 분리 및 유효 라벨 필터링...")
    labeled_mask = df["Delay"].notna() & (df["Delay"].astype(str).str.strip() != "")
    labeled_df = df[labeled_mask].copy().reset_index(drop=True)

    target_map = {"Not_Delayed": 0, "Delayed": 1}
    y = labeled_df["Delay"].map(target_map).astype(int)
    X = labeled_df.drop(columns=["Delay"])

    te_cols = [
        "Tail_Number",
        "Route",
        "Origin_Airport",
        "Destination_Airport",
        "Airline",
    ]

    print(
        f">> [3/5] Stratified 5-Fold OOF 타깃 인코딩 및 LightGBM 교차 검증 시작..."
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(X))
    feature_importances = np.zeros(
        X.shape[1] + len(te_cols) - len(te_cols)
    )  # TE 치환 후 피처 수

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

    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx]

        # Fold 내부에서만 Target Encoding 수행 (Data Leakage 원천 차단!)
        for col in te_cols:
            tr_enc, val_enc = get_smoothed_target_encoding(
                X_train[col], y_train, X_val[col], m=20.0
            )
            X_train[f"TE_{col}"] = tr_enc
            X_val[f"TE_{col}"] = val_enc

        # 원본 고카디널리티 문자열 컬럼은 삭제 (인코딩 수치형으로 완전 대체)
        X_train = X_train.drop(columns=te_cols)
        X_val = X_val.drop(columns=te_cols)

        # 모델 학습
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs

        fold_auc = roc_auc_score(y_val, val_probs)
        fold_aucs.append(fold_auc)
        print(f"   Fold {fold} 완료 | Fold ROC-AUC: {fold_auc:.4f}")

    # -------------------------------------------------------------------------
    # 4. 최적 임계값 재탐색 및 최종 지표 산출
    # -------------------------------------------------------------------------
    print("\n>> [4/5] 타깃 인코딩 모델 최적 임계값 탐색 중...")
    thresholds = np.arange(0.10, 0.70, 0.01)
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
    print("★ [3차 OOF Target Encoding 최종 결과]")
    print(f"★ 전체 OOF LogLoss       : {total_loss:.4f}")
    print(f"★ 전체 OOF ROC-AUC       : {total_auc:.4f}  (AUC 상승 여부 확인!)")
    print(f"★ 기존 (임계값 0.50) F1 : {default_f1:.4f}")
    print(
        f"★ 튜닝 (최적 임계값 {best_thresh:.2f}) F1 : {best_f1:.4f}  (최고치 경신 목표)"
    )
    print("=" * 65)

    # -------------------------------------------------------------------------
    # 5. 혼동행렬 출력
    # -------------------------------------------------------------------------
    best_preds = (oof_probs >= best_thresh).astype(int)
    cm = confusion_matrix(y, best_preds)
    print("\n[최적 임계값 기준 Confusion Matrix]")
    print(f"  TN: {cm[0,0]:,d} | FP: {cm[0,1]:,d}")
    print(f"  FN: {cm[1,0]:,d} | TP: {cm[1,1]:,d}")
    print(f"  지연 감지율(Recall): {cm[1,1]/(cm[1,0]+cm[1,1])*100:.2f}%")


# -----------------------------------------------------------------------------
# 메인 실행
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    raw_df = pd.read_csv(DATA_PATH)
    processed_df = preprocess_base(raw_df)
    run_target_encoded_pipeline(processed_df)