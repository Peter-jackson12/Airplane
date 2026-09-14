"""항공편 운항 지연(Flight Delay) 예측 - 그랜드 슬램 융합 파이프라인
- Phase 6: 노선별 중앙값 소요시간 역산 복원 + 공항 시간대별 트래픽 혼잡도 + 주기성 피처
- Phase 5: 비대칭 백분위수 필터링 기반 고확신 준지도 증강(Pseudo-Labeling)
- Hybrid Encoding (원본 category dtype + fold 내부 OOF Target Encoding) + LightGBM
- 증강 데이터는 각 fold의 train 파트에만 주입 (valid는 순수 라벨 데이터로만 채점)
- 기상(Weather) 계열 외부 피처는 Phase 7 부검 결과에 따라 일절 사용하지 않음
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

CAT_COLS = [
    "Tail_Number",
    "Route",
    "Origin_Airport",
    "Destination_Airport",
    "Airline",
]

BASELINE = {"logloss": 0.4587, "auc": 0.6416, "f1": 0.5746}

PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 79,
    "max_depth": 9,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "random_state": 42,
    "n_estimators": 1200,
    "verbose": -1,
}


# -----------------------------------------------------------------------------
# 1. Phase 6 전처리 (결측 시간 역산 복원 + 혼잡도 + 주기성)
# -----------------------------------------------------------------------------
def preprocess_advanced(df: pd.DataFrame) -> pd.DataFrame:
    print(">> [1/7] 항공사 매핑 대치 및 노선 피처 생성 중...")
    df = df.copy()

    if "Carrier_Code(IATA)" in df.columns and "Airline" in df.columns:
        c_to_a = (
            df.dropna(subset=["Carrier_Code(IATA)", "Airline"])
            .drop_duplicates(subset=["Carrier_Code(IATA)"])
            .set_index("Carrier_Code(IATA)")["Airline"]
            .to_dict()
        )
        df["Airline"] = df["Airline"].fillna(df["Carrier_Code(IATA)"].map(c_to_a))

    df["Route"] = (
        df["Origin_Airport"].astype(str) + "_" + df["Destination_Airport"].astype(str)
    )

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

    valid_time_mask = (dep_h >= 0) & (arr_h >= 0)
    dep_total_m = dep_h * 60 + dep_m
    arr_total_m = arr_h * 60 + arr_m
    # 자정을 넘기는 야간 비행 보정 (+1440분)
    raw_duration = np.where(
        arr_total_m < dep_total_m,
        arr_total_m - dep_total_m + 1440,
        arr_total_m - dep_total_m,
    )
    df["Estimated_Duration"] = np.where(valid_time_mask, raw_duration, np.nan)

    print(">> [2/7] 노선(Route) 기반 비행 소요시간 결측치 역산 복원 중...")
    route_median_dur = df.groupby("Route")["Estimated_Duration"].transform("median")
    global_median_dur = df["Estimated_Duration"].median()
    df["Estimated_Duration"] = df["Estimated_Duration"].fillna(route_median_dur)
    df["Estimated_Duration"] = df["Estimated_Duration"].fillna(global_median_dur)

    arr_missing = (df["Arr_Hour"] == -1) & (df["Dep_Hour"] >= 0)
    calc_arr_m = (
        df.loc[arr_missing, "Dep_Hour"] * 60
        + df.loc[arr_missing, "Dep_Minute"]
        + df.loc[arr_missing, "Estimated_Duration"]
    ) % 1440
    df.loc[arr_missing, "Arr_Hour"] = (calc_arr_m // 60).astype(int)
    df.loc[arr_missing, "Arr_Minute"] = (calc_arr_m % 60).astype(int)

    dep_missing = (df["Dep_Hour"] == -1) & (df["Arr_Hour"] >= 0)
    calc_dep_m = (
        df.loc[dep_missing, "Arr_Hour"] * 60
        + df.loc[dep_missing, "Arr_Minute"]
        - df.loc[dep_missing, "Estimated_Duration"]
    ) % 1440
    df.loc[dep_missing, "Dep_Hour"] = (calc_dep_m // 60).astype(int)
    df.loc[dep_missing, "Dep_Minute"] = (calc_dep_m % 60).astype(int)

    print(">> [3/7] 공항 혼잡도(Congestion) 피처 산출 (ablation 결과에 따라 최종 제외)...")
    df["Origin_Traffic"] = df.groupby(
        ["Month", "Day_of_Month", "Origin_Airport", "Dep_Hour"]
    )["Route"].transform("count")
    df["Dest_Traffic"] = df.groupby(
        ["Month", "Day_of_Month", "Destination_Airport", "Arr_Hour"]
    )["Route"].transform("count")

    valid_dep = np.where(df["Dep_Hour"] >= 0, df["Dep_Hour"], 12)
    df["Sin_Dep_Hour"] = np.sin(2 * np.pi * valid_dep / 24.0)
    df["Cos_Dep_Hour"] = np.cos(2 * np.pi * valid_dep / 24.0)

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
        "Dep_Minute",
        "Arr_Minute",
        # 준지도 증강과 결합 시 혼잡도/속도 계열은 colsample_bytree 하에서
        # Route/Tail_Number의 학습 기회를 잠식함 (ablation 실측: 제외 시
        # LogLoss 0.4593 -> 0.4581, AUC 0.6414 -> 0.6463, TP +1,839건)
        "Origin_Traffic",
        "Dest_Traffic",
        "Air_Speed_Proxy",
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


def get_smoothed_target_encoding(
    train_s: pd.Series, target: pd.Series, test_s: pd.Series, m: float = 25.0
):
    global_mean = target.mean()
    stats = target.groupby(train_s).agg(["count", "mean"])
    smooth = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)
    train_encoded = train_s.map(smooth).fillna(global_mean).values
    test_encoded = test_s.map(smooth).fillna(global_mean).values
    return train_encoded, test_encoded


def encode_categories(X_labeled: pd.DataFrame, X_unlabeled: pd.DataFrame):
    """라벨/미라벨 전체에 걸쳐 동일한 코드 체계를 갖는 category dtype을 부여."""
    cat_dtypes = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        full_series = (
            pd.concat([X_labeled[col], X_unlabeled[col]]).fillna("MISSING").astype(str)
        )
        le.fit(full_series)
        dtype = pd.CategoricalDtype(categories=range(len(le.classes_)))
        cat_dtypes[col] = dtype
        for frame in (X_labeled, X_unlabeled):
            codes = le.transform(frame[col].fillna("MISSING").astype(str))
            frame[col] = pd.Series(codes, index=frame.index).astype(dtype)
    return cat_dtypes


# -----------------------------------------------------------------------------
# 2. 그랜드 슬램 파이프라인
# -----------------------------------------------------------------------------
def run_grand_slam(df: pd.DataFrame):
    print(">> [4/7] 라벨/미라벨 분리 및 Hybrid 범주형 인코딩...")
    labeled_mask = df["Delay"].notna() & (df["Delay"].astype(str).str.strip() != "")
    labeled_df = df[labeled_mask].copy().reset_index(drop=True)
    unlabeled_df = df[~labeled_mask].copy().reset_index(drop=True)

    target_map = {"Not_Delayed": 0, "Delayed": 1}
    y_labeled = labeled_df["Delay"].map(target_map).astype(int)
    X_labeled = labeled_df.drop(columns=["Delay"])
    X_unlabeled = unlabeled_df.drop(columns=["Delay"])

    cat_dtypes = encode_categories(X_labeled, X_unlabeled)
    print(f"   - 라벨 데이터 {len(X_labeled):,d}건 / 미라벨 데이터 {len(X_unlabeled):,d}건")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pos_weight = (len(y_labeled) - sum(y_labeled)) / sum(y_labeled)
    params = {**PARAMS, "scale_pos_weight": pos_weight}

    # -------------------------------------------------------------------------
    # 1단계: 선생님(Teacher) 모델 5-Fold 학습 및 미라벨 74.5만 건 추론
    # -------------------------------------------------------------------------
    print(">> [5/7] 선생님 모델 5-Fold 학습 및 미라벨 데이터 확률 추론 중...")
    unlabeled_preds = np.zeros(len(X_unlabeled))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled), 1):
        X_tr, y_tr = X_labeled.iloc[train_idx].copy(), y_labeled.iloc[train_idx]
        X_vl, y_vl = X_labeled.iloc[val_idx].copy(), y_labeled.iloc[val_idx]
        X_unlab_fold = X_unlabeled.copy()

        for col in CAT_COLS:
            tr_enc, vl_enc = get_smoothed_target_encoding(X_tr[col], y_tr, X_vl[col])
            X_tr[f"TE_{col}"] = tr_enc
            X_vl[f"TE_{col}"] = vl_enc
            _, unlab_enc = get_smoothed_target_encoding(
                X_tr[col], y_tr, X_unlab_fold[col]
            )
            X_unlab_fold[f"TE_{col}"] = unlab_enc

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_vl, y_vl)],
            callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)],
        )
        unlabeled_preds += model.predict_proba(X_unlab_fold)[:, 1] / 5.0
        print(f"   선생님 Fold {fold} 완료 | Fold ROC-AUC: "
              f"{roc_auc_score(y_vl, model.predict_proba(X_vl)[:, 1]):.4f}")

    # -------------------------------------------------------------------------
    # 2단계: 비대칭 백분위수 필터링 (정상 하위 10% / 지연 상위 2%)
    # -------------------------------------------------------------------------
    print("\n>> [6/7] 비대칭 필터링으로 고확신 준지도 데이터 선별 중...")
    thresh_neg = np.percentile(unlabeled_preds, 10)
    thresh_pos = np.percentile(unlabeled_preds, 98)

    high_pos_idx = np.where(unlabeled_preds >= thresh_pos)[0]
    high_neg_idx = np.where(unlabeled_preds <= thresh_neg)[0]

    print(f"   - 초고위험 지연(1) 추론 : {len(high_pos_idx):,d} 건 (p >= {thresh_pos:.3f})")
    print(f"   - 확실한 정상(0) 추론   : {len(high_neg_idx):,d} 건 (p <= {thresh_neg:.3f})")

    pseudo_X = pd.concat(
        [X_unlabeled.iloc[high_pos_idx], X_unlabeled.iloc[high_neg_idx]]
    ).copy()
    pseudo_y = pd.Series([1] * len(high_pos_idx) + [0] * len(high_neg_idx), dtype=int)
    print(f"   ★ 총 {len(pseudo_X):,d} 건을 각 fold의 train 파트에만 주입합니다.")

    # -------------------------------------------------------------------------
    # 3단계: 학생(Student) 모델 5-Fold OOF (valid는 순수 라벨 데이터만 사용)
    # -------------------------------------------------------------------------
    print("\n>> [7/7] 증강 데이터로 학생 모델 재학습 및 OOF 채점 중...")
    oof_probs = np.zeros(len(X_labeled))
    importance_sum = None
    feature_names = None
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled), 1):
        X_tr_orig = X_labeled.iloc[train_idx].copy()
        y_tr_orig = y_labeled.iloc[train_idx]
        X_vl = X_labeled.iloc[val_idx].copy()
        y_vl = y_labeled.iloc[val_idx]

        X_tr_aug = pd.concat([X_tr_orig, pseudo_X], ignore_index=True)
        y_tr_aug = pd.concat([y_tr_orig, pseudo_y], ignore_index=True)

        # concat 이후 풀려버린 category dtype 명시적 재선언 (양쪽 동일 코드 체계 유지)
        for col in CAT_COLS:
            X_tr_aug[col] = X_tr_aug[col].astype(cat_dtypes[col])
            X_vl[col] = X_vl[col].astype(cat_dtypes[col])

            # Target Encoding은 fold 내부(train 파트)에서만 산출
            tr_enc, vl_enc = get_smoothed_target_encoding(
                X_tr_aug[col], y_tr_aug, X_vl[col]
            )
            X_tr_aug[f"TE_{col}"] = tr_enc
            X_vl[f"TE_{col}"] = vl_enc

        student = lgb.LGBMClassifier(**params)
        student.fit(
            X_tr_aug,
            y_tr_aug,
            eval_set=[(X_vl, y_vl)],
            callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)],
        )

        val_probs = student.predict_proba(X_vl)[:, 1]
        oof_probs[val_idx] = val_probs

        if importance_sum is None:
            feature_names = list(X_tr_aug.columns)
            importance_sum = np.zeros(len(feature_names))
        importance_sum += student.feature_importances_ / 5.0

        fold_auc = roc_auc_score(y_vl, val_probs)
        fold_aucs.append(fold_auc)
        print(f"   학생 Fold {fold} 완료 | Fold ROC-AUC: {fold_auc:.4f} "
              f"| 증강 학습셋 {len(X_tr_aug):,d}건")

    return evaluate(y_labeled, oof_probs, feature_names, importance_sum, fold_aucs)


# -----------------------------------------------------------------------------
# 3. 평가 및 시각화
# -----------------------------------------------------------------------------
def evaluate(y, oof_probs, feature_names, importances, fold_aucs):
    thresholds = np.round(np.arange(0.10, 0.4001, 0.01), 2)
    f1_scores = [
        f1_score(y, (oof_probs >= th).astype(int), average="macro") for th in thresholds
    ]
    best_i = int(np.argmax(f1_scores))
    best_thresh = float(thresholds[best_i])
    best_f1 = float(f1_scores[best_i])

    total_loss = log_loss(y, oof_probs)
    total_auc = roc_auc_score(y, oof_probs)
    default_f1 = f1_score(y, (oof_probs >= 0.5).astype(int), average="macro")

    cm = confusion_matrix(y, (oof_probs >= best_thresh).astype(int))
    tp = int(cm[1, 1])
    recall = tp / (cm[1, 0] + cm[1, 1])

    print("\n" + "=" * 72)
    print("★ [Grand Slam: Phase 5 준지도 증강 + Phase 6 도메인 피처 최종 성적표]")
    print(f"★ Fold별 ROC-AUC         : {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"★ 전체 OOF LogLoss       : {total_loss:.4f} (Phase 5 기준선 {BASELINE['logloss']:.4f})")
    print(f"★ 전체 OOF ROC-AUC       : {total_auc:.4f} (Phase 5 기준선 {BASELINE['auc']:.4f})")
    print(f"★ 기본 (임계값 0.50) F1  : {default_f1:.4f}")
    print(f"★ 튜닝 (최적 임계값 {best_thresh:.2f}) F1 : {best_f1:.4f} "
          f"(Phase 5 기준선 {BASELINE['f1']:.4f})")
    print("=" * 72)

    print("\n[최적 임계값 기준 Confusion Matrix]")
    print(f"  TN: {cm[0,0]:,d} | FP: {cm[0,1]:,d}")
    print(f"  FN: {cm[1,0]:,d} | TP: {tp:,d}")
    print(f"  지연 감지율(Recall): {recall*100:.2f}%")

    print("\n[기준선 대비 델타]")
    print(f"  LogLoss : {total_loss - BASELINE['logloss']:+.4f} (낮을수록 우수)")
    print(f"  ROC-AUC : {total_auc - BASELINE['auc']:+.4f}")
    print(f"  Macro F1: {best_f1 - BASELINE['f1']:+.4f}")

    fi_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)
    print("\n[피처 중요도 Top 15]")
    print(fi_df.head(15).to_string(index=False))

    plt.figure(figsize=(10, 8))
    top_fi = fi_df.head(20).sort_values(by="Importance", ascending=True)
    plt.barh(top_fi["Feature"], top_fi["Importance"], color="teal")
    plt.title("Grand Slam (준지도 증강 + 도메인 피처) Feature Importance Top 20", fontsize=13)
    plt.xlabel("Importance (Split 기준)")
    plt.tight_layout()
    fi_path = OUTPUT_DIR / "grand_slam_feature_importance.png"
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"\n>> 피처 중요도 차트 저장 완료: {fi_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, color="royalblue", linewidth=2)
    plt.axvline(
        best_thresh,
        color="crimson",
        linestyle="--",
        label=f"최적 임계값 {best_thresh:.2f} (F1 {best_f1:.4f})",
    )
    plt.axhline(
        BASELINE["f1"],
        color="gray",
        linestyle=":",
        label=f"Phase 5 기준선 F1 {BASELINE['f1']:.4f}",
    )
    plt.title("Grand Slam 임계값(Threshold)에 따른 Macro F1-Score 변화", fontsize=13)
    plt.xlabel("지연(Delayed) 판정 임계값")
    plt.ylabel("Macro F1-Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    th_path = OUTPUT_DIR / "grand_slam_threshold_curve.png"
    plt.savefig(th_path, dpi=300)
    plt.close()
    print(f">> 임계값 곡선 차트 저장 완료: {th_path}")

    return {
        "logloss": total_loss,
        "auc": total_auc,
        "f1": best_f1,
        "threshold": best_thresh,
        "tp": tp,
        "recall": recall,
    }


if __name__ == "__main__":
    raw_df = pd.read_csv(DATA_PATH)
    processed_df = preprocess_advanced(raw_df)
    run_grand_slam(processed_df)
