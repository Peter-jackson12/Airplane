"""항공편 운항 지연(Flight Delay) 예측 베이스라인 파이프라인
프로젝트 루트(최상위 디렉토리) 실행용
"""

import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 0. 경로 설정: 최상위 폴더 기준 (ROOT = 현재 파일 위치)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib 한글 폰트 설정 (Windows Malgun Gothic)
plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------------------------------------------------------
# 1. 공항 및 항공사 식별자 상호 결측치 복원
# -----------------------------------------------------------------------------
def load_and_impute_mappings(df: pd.DataFrame) -> pd.DataFrame:
    print(">> [1/5] 공항 및 항공사 식별자 상호 교차 대치 시작...")
    df = df.copy()

    # 1) Carrier_Code <-> Airline 상호 대치
    if "Carrier_Code(IATA)" in df.columns and "Airline" in df.columns:
        carrier_to_airline = (
            df.dropna(subset=["Carrier_Code(IATA)", "Airline"])
            .drop_duplicates(subset=["Carrier_Code(IATA)"])
            .set_index("Carrier_Code(IATA)")["Airline"]
            .to_dict()
        )
        airline_to_carrier = (
            df.dropna(subset=["Carrier_Code(IATA)", "Airline"])
            .drop_duplicates(subset=["Airline"])
            .set_index("Airline")["Carrier_Code(IATA)"]
            .to_dict()
        )

        df["Airline"] = df["Airline"].fillna(
            df["Carrier_Code(IATA)"].map(carrier_to_airline)
        )
        df["Carrier_Code(IATA)"] = df["Carrier_Code(IATA)"].fillna(
            df["Airline"].map(airline_to_carrier)
        )

    # 2) Origin / Destination Airport <-> State 상호 대치
    if "Origin_Airport" in df.columns and "Origin_State" in df.columns:
        origin_to_state = (
            df.dropna(subset=["Origin_Airport", "Origin_State"])
            .drop_duplicates(subset=["Origin_Airport"])
            .set_index("Origin_Airport")["Origin_State"]
            .to_dict()
        )
        df["Origin_State"] = df["Origin_State"].fillna(
            df["Origin_Airport"].map(origin_to_state)
        )

    if (
        "Destination_Airport" in df.columns
        and "Destination_State" in df.columns
    ):
        dest_to_state = (
            df.dropna(subset=["Destination_Airport", "Destination_State"])
            .drop_duplicates(subset=["Destination_Airport"])
            .set_index("Destination_Airport")["Destination_State"]
            .to_dict()
        )
        df["Destination_State"] = df["Destination_State"].fillna(
            df["Destination_Airport"].map(dest_to_state)
        )

    return df


# -----------------------------------------------------------------------------
# 2. 시간 파싱 및 비행 소요 시간 파생변수 생성
# -----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print(">> [2/5] 시간 및 비행 파생 피처 엔지니어링 수행 중...")
    df = df.copy()

    # HHMM -> Hour, Minute 안전 파싱 (소수점, NaN 대응)
    def parse_hhmm(col_name):
        val = pd.to_numeric(df[col_name], errors="coerce").fillna(-1)
        hour = np.where(val >= 0, (val // 100).astype(int), -1)
        minute = np.where(val >= 0, (val % 100).astype(int), -1)
        return hour, minute

    df["Dep_Hour"], df["Dep_Minute"] = parse_hhmm("Estimated_Departure_Time")
    df["Arr_Hour"], df["Arr_Minute"] = parse_hhmm("Estimated_Arrival_Time")

    # 소요 시간 계산 (자정 넘김: Arr < Dep 인 경우 +1440분)
    valid_mask = (df["Dep_Hour"] >= 0) & (df["Arr_Hour"] >= 0)
    dep_total_min = df["Dep_Hour"] * 60 + df["Dep_Minute"]
    arr_total_min = df["Arr_Hour"] * 60 + df["Arr_Minute"]

    duration = arr_total_min - dep_total_min
    duration = np.where(duration < 0, duration + 1440, duration)
    df["Estimated_Duration"] = np.where(valid_mask, duration, np.nan)

    # 시간대 24시간 순환 삼각함수 인코딩
    valid_hour = np.where(df["Dep_Hour"] >= 0, df["Dep_Hour"], np.nan)
    df["Sin_Dep_Hour"] = np.sin(2 * np.pi * valid_hour / 24.0)
    df["Cos_Dep_Hour"] = np.cos(2 * np.pi * valid_hour / 24.0)

    # 노선 결합 피처
    df["Route"] = (
        df["Origin_Airport"].astype(str)
        + "_"
        + df["Destination_Airport"].astype(str)
    )

    # 속도 추정 프록시 (마일 / 분)
    df["Air_Speed_Proxy"] = df["Distance"] / (df["Estimated_Duration"] + 1e-5)

    return df


# -----------------------------------------------------------------------------
# 3. 모델 입력셋 및 타깃 분리
# -----------------------------------------------------------------------------
def prepare_train_test(df: pd.DataFrame):
    print(">> [3/5] 타깃 분리 및 모델 입력 데이터셋 구성...")

    # 누수 및 불필요 컬럼 정리
    drop_cols = [
        "ID",
        "Cancelled",
        "Diverted",
        "Origin_Airport_ID",
        "Destination_Airport_ID",
        "Carrier_ID(DOT)",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # 라벨링된 행만 필터링 (결측치 제외)
    labeled_mask = df["Delay"].notna() & (df["Delay"].astype(str).str.strip() != "")
    labeled_df = df[labeled_mask].copy()
    unlabeled_df = df[~labeled_mask].copy()

    target_map = {"Not_Delayed": 0, "Delayed": 1}
    y = labeled_df["Delay"].map(target_map).astype(int)
    X = labeled_df.drop(columns=["Delay"])

    # 범주형 컬럼 인코딩
    cat_cols = [
        "Origin_Airport",
        "Origin_State",
        "Destination_Airport",
        "Destination_State",
        "Airline",
        "Carrier_Code(IATA)",
        "Tail_Number",
        "Route",
    ]

    for col in cat_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna("MISSING").astype(str))
            X[col] = X[col].astype("category")

    print(
        f"   - 학습용 라벨 데이터 수: {X.shape[0]:,}행 | 피처 수: {X.shape[1]}개"
    )
    print(f"   - 지연(Delayed) 발생 비율: {y.mean()*100:.2f}%")
    print(f"   - 미라벨(준지도 대상) 데이터 수: {unlabeled_df.shape[0]:,}행")

    return X, y, cat_cols


# -----------------------------------------------------------------------------
# 4. Stratified 5-Fold LightGBM 교차 검증
# -----------------------------------------------------------------------------
def run_cv_training(X: pd.DataFrame, y: pd.Series, cat_cols: list):
    print(">> [4/5] Stratified 5-Fold 교차 검증 시작 (LightGBM)...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))
    feature_importances = np.zeros(X.shape[1])

    pos_weight = (len(y) - sum(y)) / sum(y)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 47,
        "max_depth": 7,
        "scale_pos_weight": pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_estimators": 800,
        "verbose": -1,
    }

    fold_losses, fold_f1s, fold_aucs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs
        feature_importances += model.feature_importances_ / skf.n_splits

        val_pred_labels = (val_probs >= 0.5).astype(int)

        loss = log_loss(y_val, val_probs)
        f1 = f1_score(y_val, val_pred_labels, average="macro")
        auc = roc_auc_score(y_val, val_probs)

        fold_losses.append(loss)
        fold_f1s.append(f1)
        fold_aucs.append(auc)

        print(
            f"   Fold {fold} | LogLoss: {loss:.4f} | Macro F1: {f1:.4f} | ROC-AUC: {auc:.4f}"
        )

    print("\n" + "=" * 60)
    print(f"★ OOF 최종 평균 LogLoss : {np.mean(fold_losses):.4f}")
    print(f"★ OOF 최종 평균 Macro F1: {np.mean(fold_f1s):.4f}")
    print(f"★ OOF 최종 평균 ROC-AUC : {np.mean(fold_aucs):.4f}")
    print("=" * 60)

    # -----------------------------------------------------------------------------
    # 5. 피처 중요도 차트 생성
    # -----------------------------------------------------------------------------
    print(">> [5/5] 피처 중요도 차트 시각화 및 파일 저장...")
    fi_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(fi_df["Feature"], fi_df["Importance"], color="royalblue")
    plt.title("LightGBM 피처 중요도 (Feature Importance)", fontsize=13)
    plt.xlabel("Importance (Split 기준)")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(save_path, dpi=300)
    print(f">> 완료! 저장 위치: {save_path}")

    return oof_preds, fi_df


# -----------------------------------------------------------------------------
# 메인 실행
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if not DATA_PATH.exists():
        print(f"[!] 데이터 파일을 찾을 수 없습니다: {DATA_PATH.resolve()}")
        print(
            "    최상위 폴더 기준 'data/train.csv' 위치에 파일이 있는지 확인해 주세요."
        )
    else:
        print(f">> 데이터 로드 중: {DATA_PATH.resolve()}")
        raw_df = pd.read_csv(DATA_PATH)
        print(f">> 원본 데이터 크기: {raw_df.shape}")

        df_imputed = load_and_impute_mappings(raw_df)
        df_engineered = engineer_features(df_imputed)
        X, y, cat_cols = prepare_train_test(df_engineered)
        oof_preds, fi_df = run_cv_training(X, y, cat_cols)