"""항공편 운항 지연(Flight Delay) 예측 5차 파이프라인
- 준지도 학습 (Pseudo-Labeling): 74.5만 건 미라벨 결측치 추론 및 데이터 증강
- High-Confidence Filtering (상위 확신 데이터만 학습셋에 편입)
- Retraining & OOF Evaluation
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
# 1. 전처리 함수
# -----------------------------------------------------------------------------
def preprocess_base(df: pd.DataFrame) -> pd.DataFrame:
    print(">> [1/6] 데이터 결측치 대치 및 피처 생성 중...")
    df = df.copy()

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

    def parse_time(col_name):
        val = pd.to_numeric(df[col_name], errors="coerce").fillna(-1)
        h = np.where(val >= 0, (val // 100).astype(int), -1)
        m = np.where(val >= 0, (val % 100).astype(int), -1)
        return h, m

    dep_h, dep_m = parse_time("Estimated_Departure_Time")
    arr_h, arr_m = parse_time("Estimated_Arrival_Time")

    df["Dep_Hour"] = dep_h
    df["Arr_Hour"] = arr_h

    valid_mask = (dep_h >= 0) & (arr_h >= 0)
    dep_total_m = dep_h * 60 + dep_m
    arr_total_m = arr_h * 60 + arr_m
    duration = np.where(
        arr_total_m < dep_total_m,
        arr_total_m - dep_total_m + 1440,
        arr_total_m - dep_total_m,
    )
    df["Estimated_Duration"] = np.where(valid_mask, duration, np.nan)

    valid_dep = np.where(dep_h >= 0, dep_h, np.nan)
    df["Sin_Dep_Hour"] = np.sin(2 * np.pi * valid_dep / 24.0)
    df["Cos_Dep_Hour"] = np.cos(2 * np.pi * valid_dep / 24.0)

    df["Route"] = (
        df["Origin_Airport"].astype(str)
        + "_"
        + df["Destination_Airport"].astype(str)
    )
    df["Air_Speed_Proxy"] = df["Distance"] / (df["Estimated_Duration"] + 1e-5)

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
# 2. 메인 준지도 학습 파이프라인
# -----------------------------------------------------------------------------
def run_pseudo_labeling_pipeline(df: pd.DataFrame):
    print(">> [2/6] 라벨(25.5만)과 미라벨(74.5만) 데이터 분리 중...")
    labeled_mask = df["Delay"].notna() & (df["Delay"].astype(str).str.strip() != "")
    labeled_df = df[labeled_mask].copy().reset_index(drop=True)
    unlabeled_df = df[~labeled_mask].copy().reset_index(drop=True)

    target_map = {"Not_Delayed": 0, "Delayed": 1}
    y_labeled = labeled_df["Delay"].map(target_map).astype(int)
    X_labeled = labeled_df.drop(columns=["Delay"])
    X_unlabeled = unlabeled_df.drop(columns=["Delay"])

    cat_cols = [
        "Tail_Number",
        "Route",
        "Origin_Airport",
        "Destination_Airport",
        "Airline",
    ]

    # 범주형 인코딩 (전체 데이터 기준으로 일관성 유지)
    for col in cat_cols:
        le = LabelEncoder()
        full_series = pd.concat([X_labeled[col], X_unlabeled[col]]).astype(str)
        le.fit(full_series.fillna("MISSING"))
        X_labeled[col] = le.transform(
            X_labeled[col].fillna("MISSING").astype(str)
        )
        X_labeled[col] = X_labeled[col].astype("category")
        X_unlabeled[col] = le.transform(
            X_unlabeled[col].fillna("MISSING").astype(str)
        )
        X_unlabeled[col] = X_unlabeled[col].astype("category")

    # -------------------------------------------------------------------------
    # 1단계: 1차 선생님 모델 5-Fold 학습 & 미라벨 데이터 추론
    # -------------------------------------------------------------------------
    print(">> [3/6] 1차 선생님 모델 학습 및 74.5만 건 결측치 추론 중...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    unlabeled_preds = np.zeros(len(X_unlabeled))
    pos_weight = (len(y_labeled) - sum(y_labeled)) / sum(y_labeled)

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
        "n_estimators": 800,
        "verbose": -1,
    }

    teacher_models = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled), 1):
        X_tr = X_labeled.iloc[train_idx].copy()
        y_tr = y_labeled.iloc[train_idx]
        X_vl = X_labeled.iloc[val_idx].copy()
        y_vl = y_labeled.iloc[val_idx]

        for col in cat_cols:
            tr_enc, vl_enc = get_smoothed_target_encoding(
                X_tr[col], y_tr, X_vl[col], m=20.0
            )
            X_tr[f"TE_{col}"] = tr_enc
            X_vl[f"TE_{col}"] = vl_enc

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_vl, y_vl)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        teacher_models.append((model, X_tr, y_tr))

        # 미라벨 74.5만 건 예측 누적 (앙상블)
        X_unlab_fold = X_unlabeled.copy()
        for col in cat_cols:
            _, unlab_enc = get_smoothed_target_encoding(
                X_tr[col], y_tr, X_unlab_fold[col], m=20.0
            )
            X_unlab_fold[f"TE_{col}"] = unlab_enc

        unlabeled_preds += model.predict_proba(X_unlab_fold)[:, 1] / 5.0
        print(f"   선생님 모델 Fold {fold} 완료!")

# -------------------------------------------------------------------------
    # 2단계: 안전한 고확신 데이터 선별 (Pseudo-Labeling)
    # -------------------------------------------------------------------------
    print("\n>> [4/6] 안전한 고확신 데이터 선별 (Pseudo-Labeling)...")

    # 정상(0)은 하위 10% 추출, 지연(1)은 데이터 오염 방지를 위해 진짜 고위험인 상위 2%만 추출!
    thresh_neg = np.percentile(unlabeled_preds, 10)  # 하위 10% (매우 안전)
    thresh_pos = np.percentile(unlabeled_preds, 98)  # 상위 2% (진짜 초고위험군만)

    high_pos_idx = np.where(unlabeled_preds >= thresh_pos)[0]
    high_neg_idx = np.where(unlabeled_preds <= thresh_neg)[0]

    print(
        f"   - 진짜 위험한 지연(Delayed=1) 추론 : {len(high_pos_idx):,d} 건 (확신도 >= {thresh_pos:.3f})"
    )
    print(
        f"   - 확실한 정상(Not_Delayed=0) 추론   : {len(high_neg_idx):,d} 건 (확신도 <= {thresh_neg:.3f})"
    )

    pseudo_X = pd.concat(
        [X_unlabeled.iloc[high_pos_idx], X_unlabeled.iloc[high_neg_idx]]
    ).copy()
    pseudo_y = pd.Series(
        [1] * len(high_pos_idx) + [0] * len(high_neg_idx), dtype=int
    )

    print(
        f"   ★ 총 {len(pseudo_X):,d} 건의 순도 높은 결측치를 복원하여 훈련셋에 추가합니다!"
    )

    # -------------------------------------------------------------------------
    # 3단계: 증강 데이터셋으로 최종 학생(Student) 모델 학습 및 OOF 채점
    # -------------------------------------------------------------------------
    print(
        "\n>> [5/6] 증강된 데이터셋으로 최종 학생 모델 5-Fold 재학습 및 검증..."
    )
    final_oof_probs = np.zeros(len(X_labeled))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled), 1):
        X_tr_orig = X_labeled.iloc[train_idx].copy()
        y_tr_orig = y_labeled.iloc[train_idx]
        X_vl_orig = X_labeled.iloc[val_idx].copy()
        y_vl_orig = y_labeled.iloc[val_idx]

        X_tr_aug = pd.concat([X_tr_orig, pseudo_X], ignore_index=True)
        y_tr_aug = pd.concat([y_tr_orig, pseudo_y], ignore_index=True)

        # [핵심 수정] concat 후 깨진 category 타입을 양쪽 모두 강제로 복원!
        for col in cat_cols:
            X_tr_aug[col] = X_tr_aug[col].astype("category")
            X_vl_orig[col] = X_vl_orig[col].astype("category")

            # 타깃 인코딩
            tr_enc, vl_enc = get_smoothed_target_encoding(
                X_tr_aug[col], y_tr_aug, X_vl_orig[col], m=20.0
            )
            X_tr_aug[f"TE_{col}"] = tr_enc
            X_vl_orig[f"TE_{col}"] = vl_enc

        student_model = lgb.LGBMClassifier(**params)
        student_model.fit(
            X_tr_aug,
            y_tr_aug,
            eval_set=[(X_vl_orig, y_vl_orig)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        final_oof_probs[val_idx] = student_model.predict_proba(X_vl_orig)[:, 1]
        print(f"   학생 모델 Fold {fold} 완료!")

    # -------------------------------------------------------------------------
    # 4단계: 최종 평가 및 결과
    # -------------------------------------------------------------------------
    print("\n>> [6/6] 최종 준지도 학습 성능 채점 중...")
    thresholds = np.arange(0.15, 0.50, 0.01)
    best_thresh = 0.5
    best_f1 = 0.0

    for th in thresholds:
        preds = (final_oof_probs >= th).astype(int)
        score = f1_score(y_labeled, preds, average="macro")
        if score > best_f1:
            best_f1 = score
            best_thresh = th

    total_loss = log_loss(y_labeled, final_oof_probs)
    total_auc = roc_auc_score(y_labeled, final_oof_probs)
    default_f1 = f1_score(
        y_labeled, (final_oof_probs >= 0.5).astype(int), average="macro"
    )

    print("\n" + "=" * 65)
    print("★ [5차 준지도 학습 (Pseudo-Labeling) 최종 성적표]")
    print(f"★ 전체 OOF LogLoss       : {total_loss:.4f}")
    print(f"★ 전체 OOF ROC-AUC       : {total_auc:.4f}")
    print(f"★ 기존 (임계값 0.50) F1 : {default_f1:.4f}")
    print(f"★ 튜닝 (최적 임계값 {best_thresh:.2f}) F1 : {best_f1:.4f}")
    print("=" * 65)

    best_preds = (final_oof_probs >= best_thresh).astype(int)
    cm = confusion_matrix(y_labeled, best_preds)
    print("\n[최종 Confusion Matrix]")
    print(f"  TN: {cm[0,0]:,d} | FP: {cm[0,1]:,d}")
    print(f"  FN: {cm[1,0]:,d} | TP: {cm[1,1]:,d}")
    print(f"  지연 감지율(Recall): {cm[1,1]/(cm[1,0]+cm[1,1])*100:.2f}%")


# -----------------------------------------------------------------------------
# 실행
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    raw_df = pd.read_csv(DATA_PATH)
    processed_df = preprocess_base(raw_df)
    run_pseudo_labeling_pipeline(processed_df)