"""
run_phase7_weather_model.py
Phase 7: 외부 실측 기상 데이터(풍속, 돌풍, 적설, 강수 등)를 결합한 LightGBM 5-Fold 최종 검증 파이프라인
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train_with_weather.csv"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 한글 폰트 설정 (Windows Malgun Gothic)
plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False


def preprocess_with_weather(df: pd.DataFrame) -> pd.DataFrame:
    print(">> [1/4] 데이터 전처리 및 파생 피처 + 기상 피처 정제 중...")
    df = df.copy()

    # 1. 항공사 결측치 복원
    if "Carrier_Code(IATA)" in df.columns and "Airline" in df.columns:
        c_to_a = (
            df.dropna(subset=["Carrier_Code(IATA)", "Airline"])
            .drop_duplicates(subset=["Carrier_Code(IATA)"])
            .set_index("Carrier_Code(IATA)")["Airline"]
            .to_dict()
        )
        df["Airline"] = df["Airline"].fillna(df["Carrier_Code(IATA)"].map(c_to_a))

    # 2. 노선 결합 피처
    df["Route"] = df["Origin_Airport"].astype(str) + "_" + df["Destination_Airport"].astype(str)

    # 3. 시간 파싱 및 비행 소요시간 역산 복원
    def parse_time(col_name):
        val = pd.to_numeric(df[col_name], errors="coerce").fillna(-1)
        h = np.where(val >= 0, (val // 100).astype(int), -1)
        m = np.where(val >= 0, (val % 100).astype(int), -1)
        return h, m

    dep_h, dep_m = parse_time("Estimated_Departure_Time")
    arr_h, arr_m = parse_time("Estimated_Arrival_Time")

    df["Dep_Hour"], df["Dep_Minute"] = dep_h, dep_m
    df["Arr_Hour"], df["Arr_Minute"] = arr_h, arr_m

    valid_time = (dep_h >= 0) & (arr_h >= 0)
    dep_total_m = dep_h * 60 + dep_m
    arr_total_m = arr_h * 60 + arr_m
    raw_duration = np.where(arr_total_m < dep_total_m, arr_total_m - dep_total_m + 1440, arr_total_m - dep_total_m)
    df["Estimated_Duration"] = np.where(valid_time, raw_duration, np.nan)

    # 노선별 중앙값으로 소요시간 결측 역산
    route_median = df.groupby("Route")["Estimated_Duration"].transform("median")
    df["Estimated_Duration"] = df["Estimated_Duration"].fillna(route_median).fillna(df["Estimated_Duration"].median())

    # 4. 공항별 시간대별 트래픽 혼잡도
    df["Origin_Traffic"] = df.groupby(["Month", "Day_of_Month", "Origin_Airport", "Dep_Hour"])["Route"].transform("count")
    df["Dest_Traffic"] = df.groupby(["Month", "Day_of_Month", "Destination_Airport", "Arr_Hour"])["Route"].transform("count")

    # 5. 주기성 삼각함수
    valid_dep = np.where(df["Dep_Hour"] >= 0, df["Dep_Hour"], 12)
    df["Sin_Dep_Hour"] = np.sin(2 * np.pi * valid_dep / 24.0)
    df["Cos_Dep_Hour"] = np.cos(2 * np.pi * valid_dep / 24.0)

    # 6. 속도 프록시
    df["Air_Speed_Proxy"] = df["Distance"] / (df["Estimated_Duration"] + 1e-5)

    # 7. 기상 피처 결측 여부 플래그 (30대 허브 공항 여부 대변)
    df["Weather_Available"] = df["Weather_Origin_wspd"].notna().astype(int)

    # 불필요 컬럼 제거
    drop_cols = [
        "ID", "Cancelled", "Diverted", "Origin_Airport_ID", "Destination_Airport_ID",
        "Carrier_ID(DOT)", "Carrier_Code(IATA)", "Origin_State", "Destination_State",
        "Day_of_Month", "Dep_Minute", "Arr_Minute"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df


def get_smoothed_target_encoding(train_s: pd.Series, target: pd.Series, test_s: pd.Series, m: float = 25.0):
    global_mean = target.mean()
    stats = target.groupby(train_s).agg(["count", "mean"])
    counts = stats["count"]
    means = stats["mean"]
    smooth = (counts * means + m * global_mean) / (counts + m)
    return train_s.map(smooth).fillna(global_mean).values, test_s.map(smooth).fillna(global_mean).values


def run_weather_pipeline():
    if not DATA_PATH.exists():
        print(f"[!] 데이터 파일이 없습니다: {DATA_PATH}")
        return

    raw_df = pd.read_csv(DATA_PATH)
    processed_df = preprocess_with_weather(raw_df)

    # 라벨 데이터 필터링
    print(">> [2/4] 라벨 데이터 추출 및 범주형 인코딩...")
    labeled_mask = processed_df["Delay"].notna() & (processed_df["Delay"].astype(str).str.strip() != "")
    labeled_df = processed_df[labeled_mask].copy().reset_index(drop=True)

    target_map = {"Not_Delayed": 0, "Delayed": 1}
    y = labeled_df["Delay"].map(target_map).astype(int)
    X = labeled_df.drop(columns=["Delay"])

    cat_cols = ["Tail_Number", "Route", "Origin_Airport", "Destination_Airport", "Airline"]
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].fillna("MISSING").astype(str))
        X[col] = X[col].astype("category")

    print(f">> [3/4] Stratified 5-Fold 교차 검증 시작 (라벨 {len(X):,}건 전수 검증)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(X))
    pos_weight = (len(y) - sum(y)) / sum(y)
    feature_importances = np.zeros(X.shape[1] + len(cat_cols))

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
        X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]

        # Target Encoding
        for col in cat_cols:
            tr_enc, val_enc = get_smoothed_target_encoding(X_train[col], y_train, X_val[col], m=25.0)
            X_train[f"TE_{col}"] = tr_enc
            X_val[f"TE_{col}"] = val_enc

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)],
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs
        feature_importances += model.feature_importances_ / 5.0

        fold_auc = roc_auc_score(y_val, val_probs)
        fold_aucs.append(fold_auc)
        print(f"   Fold {fold} 완료 | Fold ROC-AUC: {fold_auc:.4f}")

    # 4. 임계값 튜닝 및 최종 성적 산출
    print("\n>> [4/4] 기상 결합 모델 성능 평가 및 임계값 탐색 중...")
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
    print("★ [Phase 7: 기상 데이터(ASOS) 결합 모델 최종 성적표]")
    print(f"★ 전체 OOF LogLoss       : {total_loss:.4f} (과거 최고: 0.4587)")
    print(f"★ 전체 OOF ROC-AUC       : {total_auc:.4f}  (과거 최고: 0.6416)")
    print(f"★ 기본 (임계값 0.50) F1 : {default_f1:.4f}")
    print(f"★ 튜닝 (최적 임계값 {best_thresh:.2f}) F1 : {best_f1:.4f}  (과거 최고: 0.5746)")
    print("=" * 65)

    best_preds = (oof_probs >= best_thresh).astype(int)
    cm = confusion_matrix(y, best_preds)
    print("\n[최적 임계값 기준 Confusion Matrix]")
    print(f"  TN: {cm[0,0]:,d} | FP: {cm[0,1]:,d}")
    print(f"  FN: {cm[1,0]:,d} | TP: {cm[1,1]:,d}")
    print(f"  지연 감지율(Recall): {cm[1,1]/(cm[1,0]+cm[1,1])*100:.2f}%")

    # 5. 피처 중요도 상위 20개 차트 저장
    feature_names = list(X.columns) + [f"TE_{col}" for col in cat_cols]
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances}).sort_values(by="Importance", ascending=False)

    print("\n[★ 기상 변수 중요도 Top 순위]")
    weather_fi = fi_df[fi_df["Feature"].str.contains("Weather|wspd|wpgt|prcp|snow|temp|coco")].head(10)
    print(weather_fi.to_string(index=False))

    plt.figure(figsize=(10, 8))
    top_fi = fi_df.head(20).sort_values(by="Importance", ascending=True)
    plt.barh(top_fi["Feature"], top_fi["Importance"], color="teal")
    plt.title("Phase 7 (날씨 결합) Feature Importance Top 20", fontsize=13)
    plt.xlabel("Importance (Split 기준)")
    plt.tight_layout()
    save_path = OUTPUT_DIR / "phase7_weather_feature_importance.png"
    plt.savefig(save_path, dpi=300)
    print(f">> 피처 중요도 차트 저장 완료: {save_path}")


if __name__ == "__main__":
    run_weather_pipeline()