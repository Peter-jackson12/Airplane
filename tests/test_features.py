"""`src/features.py` · `src/cv.py` 단위 테스트.

`PLAN.md` §4-0 P0-1 / P0-4 검증. 테스트의 무게중심은 "함수가 동작하는가"가 아니라
**"누수가 구조적으로 불가능한가"** 에 있다. 누수 차단 계약은 `TestLeakageContracts`
와 `TestNoLeakageEndToEnd` 에 모여 있다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cv import (
    CVConfig,
    evaluate_oof,
    make_folds,
    run_fold,
    tune_threshold_nested,
)
from src.features import (
    CAT_COLS,
    DROP_PRESETS,
    MISSING_HOUR,
    build_cyclic_features,
    build_time_features,
    build_traffic_features,
    encode_categoricals,
    fit_fold_teacher,
    impute_cross,
    load_data,
    make_pseudo_labels,
    oof_target_encode,
    parse_hhmm,
    prune_columns,
    restore_time_missing,
    smoothed_target_encode,
    split_labeled,
)

# =============================================================================
# 픽스처
# =============================================================================


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """원본 `train.csv` 스키마를 축소 재현한 12행 프레임.

    의도적으로 포함한 것들:
      · `Airline` 결측 (Carrier_Code 로 복원 가능한 행 / 불가능한 행)
      · 출발·도착 시각 양측 결측, 단측 결측
      · 자정을 넘기는 야간 비행
      · 라벨 / 미라벨 혼재
      · 같은 (날짜, 공항) 에 출발 시각이 결측인 행 2개(행 4·8) — `-1` 버킷이
        뭉치는지 확인하기 위한 구성

    출발 슬롯 구성 (Month=1 고정):
      · 1/5 ATL 08시 → 행 0, 1, 2, 6, 7  (5편)
      · 1/5 ATL 09시 → 행 5             (1편)
      · 1/5 JFK 23시 → 행 3             (1편)
      · 1/5 JFK 결측 → 행 4, 8          (집계 제외 대상)
      · 1/6 LAX 14시 → 행 9, 10         (2편)
      · 1/6 ATL 08시 → 행 11            (1편)
    """
    return pd.DataFrame(
        {
            "ID": [f"T{i:03d}" for i in range(12)],
            "Month": [1] * 12,
            "Day_of_Month": [5] * 9 + [6] * 3,
            "Estimated_Departure_Time": [
                800.0, 830.0, 800.0, 2330.0, np.nan, 915.0,
                800.0, 830.0, np.nan, 1400.0, 1400.0, 830.0,
            ],
            "Estimated_Arrival_Time": [
                1000.0, 1030.0, 1000.0, 130.0, 1200.0, np.nan,
                1000.0, 1030.0, 1130.0, 1600.0, 1600.0, 1030.0,
            ],
            "Cancelled": [0] * 12,
            "Diverted": [0] * 12,
            "Origin_Airport": ["ATL", "ATL", "ATL", "JFK", "JFK", "ATL",
                               "ATL", "ATL", "JFK", "LAX", "LAX", "ATL"],
            "Origin_Airport_ID": [1, 1, 1, 2, 2, 1, 1, 1, 2, 3, 3, 1],
            "Origin_State": ["Georgia", None, "Georgia", "New York", None,
                             "Georgia", "Georgia", None, "New York",
                             "California", "California", "Georgia"],
            "Destination_Airport": ["LAX", "LAX", "ORD", "LAX", "ORD", "LAX",
                                    "LAX", "ORD", "LAX", "ATL", "ATL", "LAX"],
            "Destination_Airport_ID": [3, 3, 4, 3, 4, 3, 3, 4, 3, 1, 1, 3],
            "Destination_State": ["California"] * 12,
            "Distance": [1946.0, 1946.0, 606.0, 2475.0, 740.0, 1946.0,
                         1946.0, 606.0, 2475.0, 1946.0, 1946.0, 1946.0],
            "Airline": ["Delta", None, "Delta", "JetBlue", "JetBlue", "Delta",
                        "Delta", None, "JetBlue", None, "Spirit", "Delta"],
            "Carrier_Code(IATA)": ["DL", "DL", "DL", "B6", "B6", "DL",
                                   "DL", "DL", "B6", None, "NK", "DL"],
            "Carrier_ID(DOT)": [19790.0] * 12,
            "Tail_Number": ["N1", "N2", "N1", "N3", "N3", "N2",
                            "N1", "N2", "N3", "N4", "N4", "N1"],
            "Delay": ["Delayed", "Not_Delayed", "Not_Delayed", "Delayed",
                      None, "Not_Delayed", "Delayed", "Not_Delayed",
                      None, None, "Delayed", "Not_Delayed"],
        }
    )


@pytest.fixture
def te_frame() -> tuple[pd.DataFrame, pd.Series]:
    """TE 테스트용 40행 프레임. 그룹별 지연율이 뚜렷하게 다르도록 구성."""
    rng = np.random.default_rng(0)
    n = 40
    groups = np.array(["A"] * 20 + ["B"] * 20)
    # A 그룹은 지연율 0.8, B 그룹은 0.2
    y = np.concatenate([rng.binomial(1, 0.8, 20), rng.binomial(1, 0.2, 20)])
    X = pd.DataFrame({"Tail_Number": groups, "Distance": rng.normal(size=n)})
    return X, pd.Series(y, dtype=int)


# =============================================================================
# B. 타깃 비의존 전처리
# =============================================================================


class TestLoadData:
    def test_reads_csv(self, tmp_path, raw_df):
        path = tmp_path / "train.csv"
        raw_df.to_csv(path, index=False)
        assert len(load_data(path)) == len(raw_df)

    def test_with_weather_drops_precomputed_dep_hour(self, tmp_path, raw_df):
        """병합 시점 Dep_Hour 는 학습 시점 정의와 어긋날 수 있어 즉시 제거한다."""
        df = raw_df.assign(Dep_Hour=999, Weather_Origin_wspd=1.0)
        path = tmp_path / "train_with_weather.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")

        out = load_data(path, with_weather=True)
        assert "Dep_Hour" not in out.columns
        assert "Weather_Origin_wspd" in out.columns
        # utf-8-sig(BOM) 이어도 첫 컬럼이 ID 로 인식되어야 한다 (AUDIT.md §2.7-e).
        assert out.columns[0] == "ID"

    def test_without_weather_flag_keeps_dep_hour(self, tmp_path, raw_df):
        path = tmp_path / "t.csv"
        raw_df.assign(Dep_Hour=999).to_csv(path, index=False)
        assert "Dep_Hour" in load_data(path).columns


class TestImputeCross:
    def test_forward_fill_airline_from_carrier(self, raw_df):
        out = impute_cross(raw_df, pairs=[("Carrier_Code(IATA)", "Airline")])
        # DL -> Delta 매핑으로 결측이 채워진다.
        assert out.loc[1, "Airline"] == "Delta"
        assert out.loc[7, "Airline"] == "Delta"
        # Carrier_Code 자체가 결측인 행은 단방향으로는 못 채운다.
        assert pd.isna(out.loc[9, "Airline"])

    def test_bidirectional_fills_carrier_from_airline(self, raw_df):
        df = raw_df.copy()
        df.loc[9, "Airline"] = "Spirit"  # NK 로 역매핑 가능해진다
        out = impute_cross(
            df, pairs=[("Carrier_Code(IATA)", "Airline")], bidirectional=True
        )
        assert out.loc[9, "Carrier_Code(IATA)"] == "NK"

    def test_unidirectional_leaves_carrier_alone(self, raw_df):
        df = raw_df.copy()
        df.loc[9, "Airline"] = "Spirit"
        out = impute_cross(df, pairs=[("Carrier_Code(IATA)", "Airline")])
        assert pd.isna(out.loc[9, "Carrier_Code(IATA)"])

    def test_missing_columns_are_skipped(self, raw_df):
        out = impute_cross(raw_df.drop(columns=["Origin_State"]))
        assert "Origin_State" not in out.columns

    def test_does_not_mutate_input(self, raw_df):
        before = raw_df.copy()
        impute_cross(raw_df)
        pd.testing.assert_frame_equal(raw_df, before)


class TestParseHhmm:
    def test_basic_split(self):
        h, m = parse_hhmm(pd.Series([800.0, 2330.0, 5.0, 0.0]))
        assert list(h) == [8, 23, 0, 0]
        assert list(m) == [0, 30, 5, 0]

    def test_missing_becomes_sentinel(self):
        h, m = parse_hhmm(pd.Series([np.nan, None, "junk"]))
        assert list(h) == [MISSING_HOUR] * 3
        assert list(m) == [MISSING_HOUR] * 3

    def test_strict_false_preserves_original_behaviour(self):
        """원본 스크립트는 2470 같은 값을 걸러내지 않았다. 재현성을 위해 유지."""
        h, m = parse_hhmm(pd.Series([2470.0]), strict=False)
        assert (h[0], m[0]) == (24, 70)

    def test_strict_true_invalidates_impossible_clock(self):
        h, m = parse_hhmm(pd.Series([2470.0, 800.0]), strict=True)
        assert (h[0], m[0]) == (MISSING_HOUR, MISSING_HOUR)
        assert (h[1], m[1]) == (8, 0)


class TestBuildTimeFeatures:
    def test_duration_and_hours(self, raw_df):
        out = build_time_features(raw_df, keep_minute=True)
        assert out.loc[0, "Dep_Hour"] == 8
        assert out.loc[0, "Arr_Hour"] == 10
        assert out.loc[0, "Estimated_Duration"] == 120

    def test_midnight_wrap(self, raw_df):
        """23:30 출발 → 01:30 도착은 120분이어야 한다 (-1320 이 아니라)."""
        out = build_time_features(raw_df, keep_minute=True)
        assert out.loc[3, "Estimated_Duration"] == 120

    def test_midnight_wrap_disabled_gives_negative(self, raw_df):
        out = build_time_features(raw_df, keep_minute=True, midnight_wrap=False)
        assert out.loc[3, "Estimated_Duration"] == -1320

    def test_one_sided_missing_leaves_duration_nan(self, raw_df):
        out = build_time_features(raw_df, keep_minute=True)
        assert pd.isna(out.loc[4, "Estimated_Duration"])  # 출발 결측
        assert pd.isna(out.loc[5, "Estimated_Duration"])  # 도착 결측

    def test_keep_minute_false_drops_minute_columns(self, raw_df):
        out = build_time_features(raw_df, keep_minute=False)
        assert "Dep_Minute" not in out.columns
        assert "Arr_Minute" not in out.columns


class TestRestoreTimeMissing:
    @pytest.fixture
    def prepared(self, raw_df):
        df = raw_df.copy()
        df["Route"] = df["Origin_Airport"] + "_" + df["Destination_Airport"]
        return build_time_features(df, keep_minute=True)

    def test_fill_duration_uses_route_median(self, prepared):
        out = restore_time_missing(prepared, fill_hours=False)
        assert out["Estimated_Duration"].notna().all()
        # 행 5 는 ATL_LAX 노선. 같은 노선의 유효 소요시간은 모두 120분.
        assert out.loc[5, "Estimated_Duration"] == 120

    def test_fill_hours_restores_one_sided_arrival(self, prepared):
        out = restore_time_missing(prepared)
        # 행 5: 09:15 출발 + 120분 = 11:15 도착
        assert out.loc[5, "Arr_Hour"] == 11
        assert out.loc[5, "Arr_Minute"] == 15

    def test_fill_hours_restores_one_sided_departure(self, prepared):
        out = restore_time_missing(prepared)
        # 행 4: 12:00 도착 - (JFK_ORD 노선 소요시간) 로 출발 역산
        assert out.loc[4, "Dep_Hour"] >= 0

    def test_fill_hours_false_leaves_sentinel(self, prepared):
        """Phase 7-Ex 가 이 블록을 누락한 상태의 재현 (AUDIT.md §2.3-c)."""
        out = restore_time_missing(prepared, fill_hours=False)
        assert out.loc[5, "Arr_Hour"] == MISSING_HOUR
        assert out.loc[4, "Dep_Hour"] == MISSING_HOUR

    def test_requires_duration_column(self, raw_df):
        with pytest.raises(KeyError, match="build_time_features"):
            restore_time_missing(raw_df)

    def test_fill_hours_requires_minutes(self, raw_df):
        df = raw_df.copy()
        df["Route"] = "X"
        prepared = build_time_features(df, keep_minute=False)
        with pytest.raises(KeyError, match="분 단위"):
            restore_time_missing(prepared, fill_hours=True)


class TestBuildTrafficFeatures:
    @pytest.fixture
    def prepared(self, raw_df):
        df = raw_df.copy()
        df["Route"] = df["Origin_Airport"] + "_" + df["Destination_Airport"]
        return build_time_features(df, keep_minute=True)

    def test_counts_same_airport_hour_slot(self, prepared):
        out = build_traffic_features(prepared)
        # 1/5 ATL 08시 = 행 0, 1, 2, 6, 7 → 5편 (830 도 시각 8시로 파싱된다)
        for row in (0, 1, 2, 6, 7):
            assert out.loc[row, "Origin_Traffic"] == 5
        assert out.loc[5, "Origin_Traffic"] == 1   # 1/5 ATL 09시 단독
        assert out.loc[11, "Origin_Traffic"] == 1  # 1/6 ATL 08시 — 날짜가 다르면 별도 슬롯
        assert out.loc[9, "Origin_Traffic"] == 2   # 1/6 LAX 14시

    def test_missing_hour_excluded_and_flagged(self, prepared):
        """AUDIT.md §2.3-c 핵심 지적: -1 을 하나의 버킷으로 세면 안 된다."""
        out = build_traffic_features(prepared, exclude_missing_hour=True)
        # 행 4, 8 은 출발 시각 결측 → Traffic 은 NaN, 플래그는 1
        assert pd.isna(out.loc[4, "Origin_Traffic"])
        assert pd.isna(out.loc[8, "Origin_Traffic"])
        assert out.loc[4, "Origin_Hour_Missing"] == 1
        assert out.loc[8, "Origin_Hour_Missing"] == 1
        assert out.loc[0, "Origin_Hour_Missing"] == 0

    def test_missing_hour_rows_do_not_inflate_valid_counts(self, prepared):
        """결측 행이 유효 슬롯의 편수를 부풀리지 않아야 한다."""
        out = build_traffic_features(prepared, exclude_missing_hour=True)
        valid_total = out.loc[out["Origin_Hour_Missing"] == 0, "Origin_Traffic"]
        assert valid_total.notna().all()
        assert out["Origin_Traffic"].isna().sum() == 2

    def test_legacy_mode_lumps_missing_into_one_bucket(self, prepared):
        """현행 두 스크립트의 동작 재현 — 결측 밀도를 혼잡도로 오인하는 상태.

        행 4·8 은 둘 다 1/5 JFK 이고 출발 시각이 결측이다. 레거시 모드에서는 이 둘이
        `hour=-1` 이라는 하나의 버킷으로 묶여 "2편이 같은 시간대에 출발"한 것처럼
        집계된다. 실제로는 시각을 모를 뿐이다 (AUDIT.md §2.3-c).
        """
        legacy = build_traffic_features(prepared, exclude_missing_hour=False)
        assert legacy.loc[4, "Origin_Traffic"] == 2
        assert legacy.loc[8, "Origin_Traffic"] == 2
        assert legacy["Origin_Traffic"].isna().sum() == 0

        fixed = build_traffic_features(prepared, exclude_missing_hour=True)
        assert pd.isna(fixed.loc[4, "Origin_Traffic"])
        assert pd.isna(fixed.loc[8, "Origin_Traffic"])

    def test_flag_can_be_disabled(self, prepared):
        out = build_traffic_features(prepared, add_missing_flag=False)
        assert "Origin_Hour_Missing" not in out.columns

    def test_rejects_non_unique_index(self, prepared):
        dup = pd.concat([prepared, prepared])
        with pytest.raises(ValueError, match="고유하지 않"):
            build_traffic_features(dup)

    def test_missing_required_column_raises(self, prepared):
        with pytest.raises(KeyError, match="Origin_Traffic"):
            build_traffic_features(prepared.drop(columns=["Month"]))


class TestBuildCyclicFeatures:
    @pytest.fixture
    def prepared(self, raw_df):
        return build_time_features(raw_df, keep_minute=True)

    def test_sin_cos_values(self, prepared):
        out = build_cyclic_features(prepared)
        assert out.loc[0, "Sin_Dep_Hour"] == pytest.approx(np.sin(2 * np.pi * 8 / 24))
        assert out.loc[0, "Cos_Dep_Hour"] == pytest.approx(np.cos(2 * np.pi * 8 / 24))

    def test_missing_nan_mode(self, prepared):
        out = build_cyclic_features(prepared, missing="nan")
        assert pd.isna(out.loc[4, "Sin_Dep_Hour"])

    def test_missing_noon_mode(self, prepared):
        """advanced / phase7-Ex 동작. 결측을 정오로 간주해 분포가 달라진다."""
        out = build_cyclic_features(prepared, missing="noon")
        assert out.loc[4, "Sin_Dep_Hour"] == pytest.approx(np.sin(np.pi))

    def test_include_cos_false(self, prepared):
        out = build_cyclic_features(prepared, include_cos=False)
        assert "Sin_Dep_Hour" in out.columns
        assert "Cos_Dep_Hour" not in out.columns

    def test_invalid_missing_mode(self, prepared):
        with pytest.raises(ValueError, match="missing"):
            build_cyclic_features(prepared, missing="zero")


class TestPruneColumns:
    def test_presets_differ_as_audited(self, raw_df):
        df = raw_df.assign(Dep_Minute=0, Arr_Minute=0)
        base = prune_columns(df, preset="baseline")
        pruned = prune_columns(df, preset="pruned")
        advanced = prune_columns(df, preset="advanced")

        assert "Origin_State" in base.columns
        assert "Origin_State" not in pruned.columns
        assert "Day_of_Month" in base.columns
        assert "Day_of_Month" not in pruned.columns
        assert "Dep_Minute" in pruned.columns
        assert "Dep_Minute" not in advanced.columns

    def test_extra_columns(self, raw_df):
        out = prune_columns(raw_df, preset="baseline", extra=["Distance"])
        assert "Distance" not in out.columns

    def test_absent_columns_ignored(self):
        out = prune_columns(pd.DataFrame({"A": [1]}), preset="pruned")
        assert list(out.columns) == ["A"]

    def test_unknown_preset(self, raw_df):
        with pytest.raises(ValueError, match="preset"):
            prune_columns(raw_df, preset="nope")

    def test_cancelled_diverted_in_every_preset(self):
        """상수 0 컬럼이므로 어떤 프리셋에서도 제거된다 (AUDIT.md §2.3-b)."""
        for preset in DROP_PRESETS.values():
            assert "Cancelled" in preset and "Diverted" in preset


class TestSplitLabeled:
    def test_split_counts(self, raw_df):
        X_lab, y, X_unlab = split_labeled(raw_df)
        assert len(X_lab) == 9
        assert len(X_unlab) == 3
        assert len(y) == 9

    def test_target_mapped_and_dropped(self, raw_df):
        X_lab, y, X_unlab = split_labeled(raw_df)
        assert "Delay" not in X_lab.columns
        assert "Delay" not in X_unlab.columns
        assert set(y.unique()) <= {0, 1}
        assert y.sum() == 4

    def test_index_is_reset(self, raw_df):
        X_lab, y, X_unlab = split_labeled(raw_df)
        assert list(X_lab.index) == list(range(len(X_lab)))
        assert list(y.index) == list(range(len(y)))
        assert list(X_unlab.index) == list(range(len(X_unlab)))

    def test_whitespace_only_treated_as_unlabeled(self, raw_df):
        df = raw_df.copy()
        df.loc[0, "Delay"] = "   "
        X_lab, _, X_unlab = split_labeled(df)
        assert len(X_lab) == 8
        assert len(X_unlab) == 4

    def test_unknown_label_raises(self, raw_df):
        df = raw_df.copy()
        df.loc[0, "Delay"] = "Maybe"
        with pytest.raises(ValueError, match="target_map"):
            split_labeled(df)


class TestEncodeCategoricals:
    def test_produces_category_dtype(self, raw_df):
        df = raw_df.copy()
        df["Route"] = df["Origin_Airport"] + "_" + df["Destination_Airport"]
        out, encoders = encode_categoricals(df, cat_cols=["Tail_Number", "Route"])
        assert isinstance(out["Tail_Number"].dtype, pd.CategoricalDtype)
        assert set(encoders) == {"Tail_Number", "Route"}

    def test_nan_becomes_explicit_token(self, raw_df):
        """pandas 3 의 astype(str) 은 NaN 을 보존하므로 명시적 치환이 필요하다."""
        out, encoders = encode_categoricals(raw_df, cat_cols=["Airline"])
        assert "MISSING" in set(encoders["Airline"].classes_)
        assert out["Airline"].isna().sum() == 0

    def test_fit_frames_extend_vocabulary(self, raw_df):
        """run_pseudo_labeling.py 만 미라벨까지 합쳐 fit 했다 (AUDIT.md §2.1-D)."""
        X_lab, _, X_unlab = split_labeled(raw_df)
        narrow, enc_narrow = encode_categoricals(X_lab, cat_cols=["Tail_Number"])
        wide, enc_wide = encode_categoricals(
            X_lab, cat_cols=["Tail_Number"], fit_frames=(X_unlab,)
        )
        assert len(enc_wide["Tail_Number"].classes_) >= len(
            enc_narrow["Tail_Number"].classes_
        )

    def test_absent_column_skipped(self, raw_df):
        out, encoders = encode_categoricals(raw_df, cat_cols=["NotThere"])
        assert encoders == {}


# =============================================================================
# C·D. 누수 차단 계약 — 이 파일의 핵심
# =============================================================================


class TestLeakageContracts:
    """AUDIT.md §2.1 / PLAN.md §3-6 이 지적한 누수가 재발 불가능함을 검증한다."""

    def test_oof_target_encode_without_fold_indices_raises_typeerror(self, te_frame):
        """★ 명시 요구 케이스: fold 인덱스 없이 호출하면 실패해야 한다."""
        X, y = te_frame
        with pytest.raises(TypeError):
            oof_target_encode(X, y)

    def test_oof_target_encode_with_only_train_idx_raises_typeerror(self, te_frame):
        X, y = te_frame
        with pytest.raises(TypeError):
            oof_target_encode(X, y, np.arange(30))

    def test_oof_target_encode_indices_have_no_defaults(self):
        """기본값이 생기면 위 두 테스트가 조용히 무력화되므로 시그니처를 직접 본다."""
        import inspect

        params = inspect.signature(oof_target_encode).parameters
        for name in ("outer_train_idx", "outer_valid_idx"):
            assert params[name].default is inspect.Parameter.empty
            assert params[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    def test_oof_target_encode_rejects_overlapping_folds(self, te_frame):
        """시그니처를 통과해도 겹치는 인덱스는 런타임에서 거부한다."""
        X, y = te_frame
        with pytest.raises(ValueError, match="겹칩니다"):
            oof_target_encode(X, y, np.arange(30), np.arange(20, 40))

    def test_make_pseudo_labels_rejects_model_ensemble_list(self, te_frame):
        """run_pseudo_labeling.py:201 의 '5개 fold 모델 평균' 주입 시도를 막는다."""
        X, _ = te_frame
        with pytest.raises(TypeError, match="앙상블"):
            make_pseudo_labels([object(), object()], X)

    def test_make_pseudo_labels_rejects_non_callable(self, te_frame):
        X, _ = te_frame
        with pytest.raises(TypeError, match="콜러블"):
            make_pseudo_labels(np.zeros(len(X)), X)

    def test_make_pseudo_labels_signature_has_no_label_access(self):
        """함수 시그니처에 y / X_labeled 가 없어야 구조적 차단이 성립한다."""
        import inspect

        params = set(inspect.signature(make_pseudo_labels).parameters)
        assert not params & {"y", "y_labeled", "X_labeled", "target"}
        # 미리 계산한 분위수 벡터를 주입하는 우회로도 없어야 한다.
        assert "percentile_basis" not in params

    def test_make_pseudo_labels_recomputes_percentiles_per_fold(self, te_frame):
        """분위수가 콜러블 출력에서 매번 새로 계산되는지 확인."""
        X, _ = te_frame
        low = make_pseudo_labels(lambda f: np.linspace(0.0, 0.5, len(f)), X)
        high = make_pseudo_labels(lambda f: np.linspace(0.5, 1.0, len(f)), X)
        assert low.thresh_pos != high.thresh_pos
        assert low.thresh_neg != high.thresh_neg

    def test_fit_fold_teacher_rejects_full_label_set(self, te_frame):
        """teacher 가 전체 라벨을 보면 거부한다."""
        X, y = te_frame
        with pytest.raises(ValueError, match="전체 라벨"):
            fit_fold_teacher(
                X, y, np.arange(len(X)), fit_predict=lambda a, b, c: np.zeros(len(c))
            )

    def test_fit_fold_teacher_closure_sees_only_fold_train(self, te_frame):
        """클로저에 갇힌 라벨이 fold 학습 파트뿐임을 확인."""
        X, y = te_frame
        seen: dict[str, int] = {}

        def capture(X_tr, y_tr, X_apply):
            seen["n"] = len(y_tr)
            return np.zeros(len(X_apply))

        teacher = fit_fold_teacher(X, y, np.arange(30), fit_predict=capture)
        teacher(X)
        assert seen["n"] == 30  # 40행 중 30행만


class TestSmoothedTargetEncode:
    def test_shrinks_small_groups_toward_global_mean(self):
        keys = pd.Series(["A"] * 100 + ["B"])
        target = pd.Series([1] * 100 + [0])
        enc = smoothed_target_encode(keys, target, pd.Series(["A", "B"]), m=20.0)
        global_mean = target.mean()
        # 표본 100개인 A 는 자기 평균(1.0)에 가깝고,
        # 표본 1개인 B 는 전역 평균 쪽으로 강하게 끌려간다.
        assert enc[0] > 0.8
        assert abs(enc[1] - global_mean) < abs(0.0 - global_mean)

    def test_unseen_category_gets_global_mean(self):
        keys = pd.Series(["A", "A", "B", "B"])
        target = pd.Series([1, 1, 0, 0])
        enc = smoothed_target_encode(keys, target, pd.Series(["ZZZ"]))
        assert enc[0] == pytest.approx(target.mean())

    def test_m_zero_gives_raw_group_mean(self):
        keys = pd.Series(["A", "A", "B", "B"])
        target = pd.Series([1, 1, 1, 0])
        enc = smoothed_target_encode(keys, target, pd.Series(["A", "B"]), m=0.0)
        assert enc[0] == pytest.approx(1.0)
        assert enc[1] == pytest.approx(0.5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="길이"):
            smoothed_target_encode(pd.Series([1, 2]), pd.Series([1]), pd.Series([1]))

    def test_negative_m_raises(self):
        with pytest.raises(ValueError, match="평활"):
            smoothed_target_encode(
                pd.Series(["A"]), pd.Series([1]), pd.Series(["A"]), m=-1.0
            )


class TestOofTargetEncode:
    def test_adds_te_columns_and_keeps_originals(self, te_frame):
        X, y = te_frame
        res = oof_target_encode(X, y, np.arange(30), np.arange(30, 40))
        assert "TE_Tail_Number" in res.X_train.columns
        assert "TE_Tail_Number" in res.X_valid.columns
        assert "Tail_Number" in res.X_train.columns

    def test_drop_original(self, te_frame):
        X, y = te_frame
        res = oof_target_encode(
            X, y, np.arange(30), np.arange(30, 40), drop_original=True
        )
        assert "Tail_Number" not in res.X_train.columns
        assert "TE_Tail_Number" in res.X_train.columns

    def test_shapes_match_fold_sizes(self, te_frame):
        X, y = te_frame
        res = oof_target_encode(X, y, np.arange(30), np.arange(30, 40))
        assert len(res.X_train) == 30
        assert len(res.y_train) == 30
        assert len(res.X_valid) == 10

    def test_inner_kfold_breaks_self_encoding(self, te_frame):
        """학습행이 자기 라벨로 인코딩되지 않는지 확인 (AUDIT.md §2.2 말미).

        inner_splits=0 이면 학습행 인코딩이 그룹 하나당 단일 값으로 수렴하지만,
        inner K-fold 를 쓰면 같은 그룹 안에서도 fold 마다 값이 갈린다.
        """
        X, y = te_frame
        tr, va = np.arange(30), np.arange(30, 40)

        legacy = oof_target_encode(X, y, tr, va, inner_splits=0)
        nested = oof_target_encode(X, y, tr, va, inner_splits=5, seed=0)

        group_a = legacy.X_train["Tail_Number"] == "A"
        assert legacy.X_train.loc[group_a, "TE_Tail_Number"].nunique() == 1
        assert nested.X_train.loc[group_a, "TE_Tail_Number"].nunique() > 1

    def test_validation_encoding_identical_regardless_of_inner_splits(self, te_frame):
        """검증 행 인코딩은 항상 학습 측 라벨 전체로 계산된다 — inner 설정과 무관."""
        X, y = te_frame
        tr, va = np.arange(30), np.arange(30, 40)
        a = oof_target_encode(X, y, tr, va, inner_splits=0)
        b = oof_target_encode(X, y, tr, va, inner_splits=5)
        np.testing.assert_allclose(
            a.X_valid["TE_Tail_Number"], b.X_valid["TE_Tail_Number"]
        )

    def test_validation_labels_never_enter_encoding(self):
        """검증 fold 의 라벨을 뒤집어도 검증 행 인코딩이 변하지 않아야 한다."""
        X = pd.DataFrame({"Tail_Number": ["A"] * 20 + ["B"] * 20})
        y1 = pd.Series([1] * 10 + [0] * 10 + [1] * 10 + [0] * 10)
        y2 = y1.copy()
        tr = np.arange(20)
        va = np.arange(20, 40)
        y2.iloc[20:] = 1 - y2.iloc[20:]  # 검증 fold 라벨만 반전

        a = oof_target_encode(X, y1, tr, va)
        b = oof_target_encode(X, y2, tr, va)
        np.testing.assert_allclose(
            a.X_valid["TE_Tail_Number"], b.X_valid["TE_Tail_Number"]
        )

    def test_extra_fit_rows_extend_train_and_keep_y_in_lockstep(self, te_frame):
        X, y = te_frame
        extra_X = pd.DataFrame(
            {"Tail_Number": ["A"] * 5, "Distance": np.zeros(5)}
        )
        extra_y = pd.Series([1] * 5)
        res = oof_target_encode(
            X, y, np.arange(30), np.arange(30, 40),
            extra_fit_rows=(extra_X, extra_y),
        )
        assert len(res.X_train) == 35
        assert len(res.y_train) == 35

    def test_extra_fit_rows_length_mismatch_raises(self, te_frame):
        X, y = te_frame
        with pytest.raises(ValueError, match="길이가 다릅니다"):
            oof_target_encode(
                X, y, np.arange(30), np.arange(30, 40),
                extra_fit_rows=(pd.DataFrame({"Tail_Number": ["A"]}), pd.Series([1, 0])),
            )

    def test_extra_fit_rows_absent_by_default(self, te_frame):
        """명시하지 않으면 추가 행이 절대 섞이지 않는다 (AUDIT.md §2.1-B)."""
        X, y = te_frame
        res = oof_target_encode(X, y, np.arange(30), np.arange(30, 40))
        assert len(res.X_train) == 30

    def test_inner_splits_one_raises(self, te_frame):
        X, y = te_frame
        with pytest.raises(ValueError, match="inner_splits"):
            oof_target_encode(X, y, np.arange(30), np.arange(30, 40), inner_splits=1)

    def test_empty_fold_raises(self, te_frame):
        X, y = te_frame
        with pytest.raises(ValueError, match="비어 있"):
            oof_target_encode(X, y, np.array([], dtype=int), np.arange(10))


class TestMakePseudoLabels:
    @pytest.fixture
    def unlabeled(self) -> pd.DataFrame:
        return pd.DataFrame({"Tail_Number": ["A"] * 100, "Distance": np.arange(100.0)})

    def test_selects_extremes_only(self, unlabeled):
        probs = np.linspace(0.0, 1.0, 100)
        res = make_pseudo_labels(
            lambda f: probs, unlabeled, neg_percentile=10, pos_percentile=98
        )
        assert (res.y == 1).sum() == 2   # p98=0.98 이상: 0.9899, 1.0
        assert (res.y == 0).sum() == 10  # p10=0.10 이하: 0.0 ~ 0.0909
        # 중간 구간 88행은 어느 쪽으로도 라벨링되지 않는다.
        assert len(res.X) == 12

    def test_selected_positives_all_rank_above_negatives(self, unlabeled):
        """선별된 양성/음성이 확률 순서상 완전히 분리되어야 한다."""
        probs = np.linspace(0.0, 1.0, 100)
        res = make_pseudo_labels(lambda f: probs, unlabeled)
        pos_dist = res.X.loc[res.y == 1, "Distance"]
        neg_dist = res.X.loc[res.y == 0, "Distance"]
        assert pos_dist.min() > neg_dist.max()

    def test_positive_rows_come_first(self, unlabeled):
        probs = np.linspace(0.0, 1.0, 100)
        res = make_pseudo_labels(lambda f: probs, unlabeled)
        assert res.y.iloc[0] == 1
        assert res.y.iloc[-1] == 0

    def test_thresholds_reported(self, unlabeled):
        probs = np.linspace(0.0, 1.0, 100)
        res = make_pseudo_labels(lambda f: probs, unlabeled)
        assert res.thresh_neg == pytest.approx(np.percentile(probs, 10))
        assert res.thresh_pos == pytest.approx(np.percentile(probs, 98))

    def test_wrong_prediction_length_raises(self, unlabeled):
        with pytest.raises(ValueError, match="1차원 확률 배열"):
            make_pseudo_labels(lambda f: np.zeros(5), unlabeled)

    def test_percentile_order_validated(self, unlabeled):
        with pytest.raises(ValueError, match="neg_percentile"):
            make_pseudo_labels(
                lambda f: np.zeros(len(f)), unlabeled,
                neg_percentile=90, pos_percentile=10,
            )

    def test_unpacks_as_x_y(self, unlabeled):
        probs = np.linspace(0.0, 1.0, 100)
        pseudo_X, pseudo_y, *_ = make_pseudo_labels(lambda f: probs, unlabeled)
        assert isinstance(pseudo_X, pd.DataFrame)
        assert isinstance(pseudo_y, pd.Series)


# =============================================================================
# src/cv.py
# =============================================================================


class TestCVConfig:
    def test_defaults(self):
        cfg = CVConfig()
        assert cfg.n_splits == 5
        assert cfg.seed == 42
        assert cfg.metric_aggregation == "pooled_oof"

    def test_threshold_grid_includes_stop(self):
        cfg = CVConfig(threshold_grid=(0.1, 0.7, 0.01))
        th = cfg.thresholds()
        assert th[0] == pytest.approx(0.10)
        assert th[-1] == pytest.approx(0.70)
        assert th.size == 61

    def test_fold_mean_aggregation_rejected(self):
        """Phase 1 만 쓰던 fold 평균 집계는 허용하지 않는다 (AUDIT.md §3.4)."""
        with pytest.raises(ValueError, match="pooled_oof"):
            CVConfig(metric_aggregation="fold_mean")

    def test_invalid_grid_rejected(self):
        with pytest.raises(ValueError, match="start"):
            CVConfig(threshold_grid=(0.7, 0.1, 0.01))
        with pytest.raises(ValueError, match="step"):
            CVConfig(threshold_grid=(0.1, 0.7, 0.0))

    def test_invalid_n_splits_rejected(self):
        with pytest.raises(ValueError, match="n_splits"):
            CVConfig(n_splits=1)

    def test_invalid_holdout_frac_rejected(self):
        with pytest.raises(ValueError, match="inner_holdout_frac"):
            CVConfig(inner_holdout_frac=0.8)

    def test_is_frozen(self):
        cfg = CVConfig()
        with pytest.raises(Exception):
            cfg.n_splits = 10  # type: ignore[misc]

    def test_describe_records_protocol(self):
        d = CVConfig().describe()
        assert d["inner_early_stopping"] is True
        assert d["nested_threshold"] is True
        assert d["n_thresholds"] == 61


class TestMakeFolds:
    def test_deterministic_across_calls(self):
        y = pd.Series([0] * 80 + [1] * 20)
        a = make_folds(y, CVConfig())
        b = make_folds(y, CVConfig())
        for (a_tr, a_va), (b_tr, b_va) in zip(a, b):
            np.testing.assert_array_equal(a_tr, b_tr)
            np.testing.assert_array_equal(a_va, b_va)

    def test_folds_partition_all_rows_exactly_once(self):
        y = pd.Series([0] * 80 + [1] * 20)
        folds = make_folds(y, CVConfig())
        covered = np.concatenate([va for _, va in folds])
        np.testing.assert_array_equal(np.sort(covered), np.arange(100))

    def test_train_valid_disjoint(self):
        y = pd.Series([0] * 80 + [1] * 20)
        for tr, va in make_folds(y, CVConfig()):
            assert np.intersect1d(tr, va).size == 0

    def test_stratified(self):
        y = pd.Series([0] * 80 + [1] * 20)
        for _, va in make_folds(y, CVConfig()):
            assert y.iloc[va].mean() == pytest.approx(0.2, abs=0.05)


class TestRunFold:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(1)
        n = 400
        X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
        y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
        return X, y

    def _factory(self):
        import lightgbm as lgb

        return lambda: lgb.LGBMClassifier(
            n_estimators=50, learning_rate=0.1, num_leaves=7, verbose=-1, random_state=0
        )

    def test_returns_probs_for_every_valid_row(self, data):
        X, y = data
        tr, va = np.arange(300), np.arange(300, 400)
        fit = run_fold(
            self._factory(), X.iloc[tr], y.iloc[tr], X.iloc[va], CVConfig()
        )
        assert fit.valid_probs.shape == (100,)
        assert ((fit.valid_probs >= 0) & (fit.valid_probs <= 1)).all()

    def test_holdout_carved_from_train_only(self, data):
        """★ early stopping holdout 은 학습 fold 에서만 나온다 (AUDIT.md §2.5)."""
        X, y = data
        tr, va = np.arange(300), np.arange(300, 400)
        fit = run_fold(
            self._factory(), X.iloc[tr], y.iloc[tr], X.iloc[va], CVConfig()
        )
        assert fit.n_inner_train + fit.n_inner_holdout == 300
        assert fit.n_inner_holdout == pytest.approx(60, abs=2)

    def test_valid_rows_never_reach_eval_set(self):
        """★ AUDIT.md §2.5 의 핵심 계약 검증.

        대역 추정기로 `fit()` 이 실제로 받은 평가셋을 가로채, 그 행들이
          (a) 전부 `X_train` 에서 왔고,
          (b) `X_valid` 와는 한 행도 겹치지 않는지
        를 **인덱스가 아니라 행 내용으로** 대조한다. 인덱스 라벨은 `reset_index()` 를
        거치면 행을 식별하지 못하므로 계약 검증의 근거가 될 수 없다.
        """
        n = 200
        X = pd.DataFrame({"row_id": np.arange(n), "feat": np.arange(n, dtype=float)})
        y = pd.Series(([0, 1] * (n // 2)))

        captured: dict[str, pd.DataFrame] = {}

        class SpyModel:
            def fit(self, X_fit, y_fit, eval_X=None, eval_y=None, callbacks=None, **kw):
                captured["train"] = X_fit.copy()
                captured["eval"] = None if eval_X is None else eval_X.copy()
                return self

            def predict_proba(self, X_pred):
                return np.column_stack([np.full(len(X_pred), 0.5)] * 2)

        # 두 프레임 모두 reset_index 를 거쳐 인덱스 라벨이 0부터 겹치게 만든다.
        X_train = X.iloc[:150].reset_index(drop=True)
        y_train = y.iloc[:150].reset_index(drop=True)
        X_valid = X.iloc[150:].reset_index(drop=True)
        assert set(X_train.index) & set(X_valid.index), "인덱스가 겹치는 상황을 만든다"

        run_fold(SpyModel, X_train, y_train, X_valid, CVConfig())

        eval_ids = set(captured["eval"]["row_id"])
        assert eval_ids, "early stopping 평가셋이 전달되지 않았다"
        assert eval_ids <= set(X_train["row_id"]), "평가셋이 학습 fold 밖에서 왔다"
        assert not (eval_ids & set(X_valid["row_id"])), "채점 fold 행이 평가셋에 들어갔다"
        # 내부 학습셋과 평가셋도 서로 겹치지 않아야 한다.
        assert not (eval_ids & set(captured["train"]["row_id"]))

    def test_pseudo_rows_never_reach_eval_set(self):
        """★ 준지도 2차 오염 차단 검증.

        `X_train` 에 pseudo-label 행이 섞인 상태에서 `holdout_eligible` 로 진짜 라벨
        행만 후보로 지정하면, early stopping 평가셋에 pseudo 행이 한 줄도 들어가지
        않아야 한다. 그렇지 않으면 트리 개수가 teacher 가 만든 가짜 라벨에 대해
        최적화되어 준지도 효과 측정 자체가 오염된다.

        동시에 pseudo 행은 **내부 학습셋에는 계속 남아 있어야** 한다. 증강 효과까지
        사라지면 실험의 의미가 없다.
        """
        n_real, n_pseudo = 200, 80
        X_train = pd.DataFrame(
            {
                "row_id": np.arange(n_real + n_pseudo),
                "is_pseudo": [0] * n_real + [1] * n_pseudo,
                "feat": np.arange(n_real + n_pseudo, dtype=float),
            }
        )
        y_train = pd.Series([0, 1] * ((n_real + n_pseudo) // 2))
        X_valid = pd.DataFrame(
            {"row_id": np.arange(900, 950), "is_pseudo": 0, "feat": 1.0}
        )
        real_positions = np.arange(n_real)

        captured: dict[str, pd.DataFrame] = {}

        class SpyModel:
            def fit(self, X_fit, y_fit, eval_X=None, eval_y=None, callbacks=None, **kw):
                captured["train"] = X_fit.copy()
                captured["eval"] = eval_X.copy()
                return self

            def predict_proba(self, X_pred):
                return np.column_stack([np.full(len(X_pred), 0.5)] * 2)

        fit = run_fold(
            SpyModel, X_train, y_train, X_valid, CVConfig(),
            holdout_eligible=real_positions,
        )

        eval_rows = captured["eval"]
        assert len(eval_rows) > 0
        # (a) 평가셋에 pseudo 행이 한 줄도 없다 — 행 내용으로 대조.
        assert eval_rows["is_pseudo"].sum() == 0
        assert set(eval_rows["row_id"]) <= set(X_train.loc[: n_real - 1, "row_id"])
        # (b) 그러나 pseudo 행은 내부 학습셋에 전부 남아 있다.
        assert captured["train"]["is_pseudo"].sum() == n_pseudo
        # (c) 학습셋과 평가셋은 서로 겹치지 않고, 합치면 X_train 전체다.
        assert not (set(eval_rows["row_id"]) & set(captured["train"]["row_id"]))
        assert fit.n_inner_train + fit.n_inner_holdout == n_real + n_pseudo

    def test_holdout_eligible_default_matches_previous_behaviour(self, data):
        """인자를 주지 않으면 기존 동작과 완전히 동일해야 한다."""
        X, y = data
        tr, va = np.arange(300), np.arange(300, 400)
        a = run_fold(self._factory(), X.iloc[tr], y.iloc[tr], X.iloc[va], CVConfig())
        b = run_fold(
            self._factory(), X.iloc[tr], y.iloc[tr], X.iloc[va], CVConfig(),
            holdout_eligible=None,
        )
        np.testing.assert_allclose(a.valid_probs, b.valid_probs)
        assert (a.n_inner_train, a.n_inner_holdout) == (b.n_inner_train, b.n_inner_holdout)

    def test_holdout_eligible_out_of_range_rejected(self, data):
        X, y = data
        tr, va = np.arange(300), np.arange(300, 400)
        with pytest.raises(ValueError, match="범위"):
            run_fold(
                self._factory(), X.iloc[tr], y.iloc[tr], X.iloc[va], CVConfig(),
                holdout_eligible=np.array([0, 1, 9999]),
            )

    def test_holdout_eligible_empty_rejected(self, data):
        X, y = data
        tr, va = np.arange(300), np.arange(300, 400)
        with pytest.raises(ValueError, match="비어 있"):
            run_fold(
                self._factory(), X.iloc[tr], y.iloc[tr], X.iloc[va], CVConfig(),
                holdout_eligible=np.array([], dtype=int),
            )

    def test_length_mismatch_rejected(self, data):
        X, y = data
        with pytest.raises(ValueError, match="길이가 다릅니다"):
            run_fold(self._factory(), X.iloc[:100], y.iloc[:50], X.iloc[300:], CVConfig())

    def test_model_factory_called_per_fold(self, data):
        X, y = data
        calls = {"n": 0}
        base = self._factory()

        def factory():
            calls["n"] += 1
            return base()

        for fold, (tr, va) in enumerate(make_folds(y, CVConfig())):
            run_fold(factory, X.iloc[tr], y.iloc[tr], X.iloc[va], CVConfig(), fold=fold)
        assert calls["n"] == 5


class TestTuneThresholdNested:
    @pytest.fixture
    def oof(self):
        rng = np.random.default_rng(7)
        n = 600
        y = pd.Series(rng.binomial(1, 0.2, n))
        probs = np.clip(rng.normal(0.2 + 0.3 * y, 0.2), 0.01, 0.99)
        folds = make_folds(y, CVConfig())
        return y, probs, folds

    def test_nested_f1_not_greater_than_naive(self, oof):
        """★ winner's curse 검증: nested 값이 naive 최댓값을 넘을 수 없다."""
        y, probs, folds = oof
        res = tune_threshold_nested(y, probs, folds, CVConfig())
        assert res.nested_f1 <= res.naive_f1 + 1e-12
        assert res.optimism >= -1e-12

    def test_one_threshold_per_fold(self, oof):
        y, probs, folds = oof
        res = tune_threshold_nested(y, probs, folds, CVConfig())
        assert len(res.per_fold_thresholds) == len(folds)

    def test_thresholds_within_grid(self, oof):
        y, probs, folds = oof
        cfg = CVConfig()
        res = tune_threshold_nested(y, probs, folds, cfg)
        grid = set(np.round(cfg.thresholds(), 10))
        assert all(round(t, 10) in grid for t in res.per_fold_thresholds)
        assert round(res.naive_threshold, 10) in grid

    def test_fold_threshold_ignores_own_rows(self, oof):
        """fold k 의 임계값은 fold k 의 행을 보지 않고 결정된다."""
        y, probs, folds = oof
        cfg = CVConfig()
        base = tune_threshold_nested(y, probs, folds, cfg)

        # fold 0 의 확률만 망가뜨려도 fold 0 의 임계값은 바뀌지 않아야 한다.
        tampered = probs.copy()
        tampered[folds[0][1]] = 0.99
        after = tune_threshold_nested(y, tampered, folds, cfg)
        assert after.per_fold_thresholds[0] == base.per_fold_thresholds[0]

    def test_deployment_threshold_is_median_of_folds(self, oof):
        y, probs, folds = oof
        res = tune_threshold_nested(y, probs, folds, CVConfig())
        assert res.deployment_threshold == pytest.approx(
            float(np.median(res.per_fold_thresholds))
        )

    def test_shape_mismatch_raises(self, oof):
        y, probs, folds = oof
        with pytest.raises(ValueError, match="형태가 다릅니다"):
            tune_threshold_nested(y, probs[:-1], folds, CVConfig())

    def test_single_fold_rejected(self, oof):
        y, probs, folds = oof
        with pytest.raises(ValueError, match="2개 이상"):
            tune_threshold_nested(y, probs, folds[:1], CVConfig())


class TestEvaluateOof:
    @pytest.fixture
    def oof(self):
        rng = np.random.default_rng(11)
        n = 600
        y = pd.Series(rng.binomial(1, 0.2, n))
        probs = np.clip(rng.normal(0.2 + 0.3 * y, 0.2), 0.01, 0.99)
        return y, probs, make_folds(y, CVConfig())

    def test_reports_nested_f1_as_macro_f1(self, oof):
        y, probs, folds = oof
        res = evaluate_oof(y, probs, folds, CVConfig())
        nested = tune_threshold_nested(y, probs, folds, CVConfig())
        assert res["macro_f1"] == pytest.approx(nested.nested_f1)
        assert res["naive_macro_f1"] == pytest.approx(nested.naive_f1)

    def test_optimism_is_difference(self, oof):
        y, probs, folds = oof
        res = evaluate_oof(y, probs, folds, CVConfig())
        assert res["threshold_optimism"] == pytest.approx(
            res["naive_macro_f1"] - res["macro_f1"]
        )

    def test_confusion_matrix_totals(self, oof):
        y, probs, folds = oof
        res = evaluate_oof(y, probs, folds, CVConfig())
        cm = res["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y)
        assert cm["tp"] + cm["fn"] == int(y.sum())

    def test_protocol_is_embedded(self, oof):
        y, probs, folds = oof
        res = evaluate_oof(y, probs, folds, CVConfig(), label="phase-test")
        assert res["label"] == "phase-test"
        assert res["protocol"]["seed"] == 42
        assert res["protocol"]["nested_threshold"] is True

    def test_feature_hash_recorded(self, oof):
        y, probs, folds = oof
        res = evaluate_oof(y, probs, folds, CVConfig(), feature_names=["a", "b"])
        assert res["n_features"] == 2
        assert len(res["feature_hash"]) == 8


# =============================================================================
# 통합 — 누수 없는 준지도 파이프라인 한 바퀴
# =============================================================================


class TestNoLeakageEndToEnd:
    """Phase 8 이 밟아야 할 경로를 축소 재현한다 (PLAN.md §4-1).

    `run_pseudo_labeling.py` / `run_grand_slam_v0_leaky.py` 와의 차이는 단 하나 —
    teacher 가 fold 루프 **안에서** 그 fold 의 학습 파트만으로 적합된다는 것이다.
    """

    @pytest.fixture
    def parts(self):
        rng = np.random.default_rng(3)
        n_lab, n_unlab = 200, 120
        groups = rng.choice(["A", "B", "C"], size=n_lab)
        y = pd.Series(
            (rng.random(n_lab) < np.where(groups == "A", 0.7, 0.2)).astype(int)
        )
        X_lab = pd.DataFrame(
            {"Tail_Number": groups, "Distance": rng.normal(size=n_lab)}
        )
        X_unlab = pd.DataFrame(
            {
                "Tail_Number": rng.choice(["A", "B", "C"], size=n_unlab),
                "Distance": rng.normal(size=n_unlab),
            }
        )
        return X_lab, y, X_unlab

    def test_pseudo_labels_differ_across_folds(self, parts):
        """★ fold 마다 teacher 가 다르므로 pseudo 집합도 달라야 한다.

        기존 구현은 fold 루프 밖에서 만든 **단 하나의 동결된** pseudo 집합을 모든
        fold 에 주입했다. 그것이 누수의 전달 경로였다 (AUDIT.md §2.1-③).
        """
        X_lab, y, X_unlab = parts
        cfg = CVConfig(n_splits=5)

        def fit_predict(X_tr, y_tr, X_apply):
            rate = smoothed_target_encode(
                X_tr["Tail_Number"], y_tr, X_apply["Tail_Number"], m=5.0
            )
            return rate

        signatures = []
        for tr, _ in make_folds(y, cfg):
            teacher = fit_fold_teacher(X_lab, y, tr, fit_predict=fit_predict)
            res = make_pseudo_labels(teacher, X_unlab)
            signatures.append((res.thresh_neg, res.thresh_pos, len(res.X)))

        assert len(set(signatures)) > 1, "fold 간 pseudo 집합이 동일하다 = 동결된 누수"

    def test_full_fold_loop_produces_valid_oof(self, parts):
        """teacher → pseudo → TE → 학습 → 채점 전 과정이 맞물려 돌아가는지."""
        import lightgbm as lgb

        X_lab, y, X_unlab = parts
        cfg = CVConfig(n_splits=5, stopping_rounds=10)
        folds = make_folds(y, cfg)
        oof = np.full(len(y), np.nan)

        def fit_predict(X_tr, y_tr, X_apply):
            return smoothed_target_encode(
                X_tr["Tail_Number"], y_tr, X_apply["Tail_Number"], m=5.0
            )

        for fold, (tr, va) in enumerate(folds):
            teacher = fit_fold_teacher(X_lab, y, tr, fit_predict=fit_predict)
            pseudo = make_pseudo_labels(teacher, X_unlab)

            te = oof_target_encode(
                X_lab, y, tr, va,
                cols=["Tail_Number"],
                inner_splits=3,
                drop_original=True,
                extra_fit_rows=(pseudo.X, pseudo.y),
            )
            fit = run_fold(
                lambda: lgb.LGBMClassifier(
                    n_estimators=40, num_leaves=7, verbose=-1, random_state=0
                ),
                te.X_train.drop(columns=["Tail_Number"], errors="ignore"),
                te.y_train,
                te.X_valid.drop(columns=["Tail_Number"], errors="ignore"),
                cfg,
                fold=fold,
            )
            oof[va] = fit.valid_probs

        assert not np.isnan(oof).any(), "모든 행이 정확히 한 번 채점되어야 한다"
        res = evaluate_oof(y, oof, folds, cfg, label="e2e")
        assert 0.0 <= res["roc_auc"] <= 1.0
        assert res["macro_f1"] <= res["naive_macro_f1"] + 1e-12
