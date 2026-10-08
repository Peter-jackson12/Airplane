"""Synthetic contracts for per-classifier probability calibration
(src/classifier_calibration.py, notebooks/run_classifier_calibration.py)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import rerun_all_phases as runner
from notebooks import run_classifier_calibration as cal_run
from src import classifier_calibration as ccal
from src import classifier_compare as cc
from src.cv import _make_inner_split, make_folds
from tests.test_classifier_comparison import CFG, TE_COLS, synthetic

SPEC = SimpleNamespace(cat_cols=TE_COLS, inner_splits=5, te_drop_original=False)
TE_KW = dict(te_cols=TE_COLS, te_m=20.0, te_inner_splits=5)
LGBM_PARAMS = {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 20,
               "subsample": 0.8, "subsample_freq": 0, "n_estimators": 30}


def builders():
    base = dict(runner.LGBM_PARAMS)
    return {
        "logistic_regression": ccal.selected_model_builder("logistic_regression", {"C": 0.1}),
        "random_forest_tuned": ccal.selected_model_builder(
            "random_forest_tuned", {"min_samples_leaf": 10, "max_features": 0.5},
            rf_n_jobs=1, rf_n_estimators=20),
        "lightgbm_tuned": ccal.selected_model_builder(
            "lightgbm_tuned", LGBM_PARAMS, lgbm_base_params=base, lgbm_n_jobs=1),
    }


# ---------------------------------------------------------------------------
# ECE / reliability bins
# ---------------------------------------------------------------------------


def test_ece_hand_computed_equal_frequency_and_width():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 1, 1, 1])
    ef = ccal.equal_frequency_bins(y, p, n_bins=2)
    assert ef.n.tolist() == [2, 2]
    assert ccal.ece_from_bins(ef) == pytest.approx((2 * 0.4 + 2 * 0.1) / 4)
    ew = ccal.equal_width_bins(y, p, n_bins=10)
    assert ew.n.sum() == 4 and ew.loc[1, "n"] == 2 and ew.loc[9, "n"] == 2
    assert ccal.ece_from_bins(ew) == pytest.approx(0.25)


def test_ece_equal_width_matches_repository_ece_10():
    from src.oof import metrics
    rng = np.random.default_rng(3)
    p = rng.beta(2, 8, 5000)
    y = (rng.random(5000) < p * 1.2).astype(int)
    pred = (p >= 0.2).astype(int)
    ref = metrics(pd.DataFrame({"y_true": y, "probability": p, "prediction": pred}))["ece_10"]
    assert ccal.ece_from_bins(ccal.equal_width_bins(y, p, 10)) == pytest.approx(ref, abs=1e-12)


def test_ece_near_zero_for_calibrated_and_large_for_shifted():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.6, 200_000)
    y = (rng.random(p.size) < p).astype(int)
    assert ccal.ece_from_bins(ccal.equal_frequency_bins(y, p, 15)) < 0.005
    shifted = np.clip(p + 0.1, 0, 1)
    assert ccal.ece_from_bins(ccal.equal_frequency_bins(y, shifted, 15)) == pytest.approx(0.1, abs=0.005)


def test_equal_frequency_bins_keep_ties_together():
    p = np.array([0.2] * 6 + [0.5, 0.6, 0.7, 0.8])
    y = np.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 1])
    idx, _ = ccal.equal_frequency_assign(p, n_bins=5)
    assert len(set(idx[:6])) == 1  # tied probabilities never split across bins
    bins = ccal.equal_frequency_bins(y, p, n_bins=5)
    assert bins.n.sum() == 10
    manual = sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum()
                 for b in np.unique(idx)) / 10
    assert ccal.ece_from_bins(bins) == pytest.approx(manual)
    # distinct values -> exactly equal-size bins
    q = np.linspace(0.01, 0.99, 100)
    assert ccal.equal_frequency_bins(np.r_[np.zeros(50), np.ones(50)].astype(int), q, 10).n.tolist() == [10] * 10


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------


def test_calibrators_are_monotone_and_fix_a_miscalibrated_score():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.05, 0.5, 20_000)
    y = (rng.random(true_p.size) < true_p).astype(int)
    raw = true_p ** 0.5  # overconfident towards 1
    for kind in ("platt", "isotonic"):
        cal = ccal.fit_calibrator(kind, y, raw)
        grid = np.linspace(0.0, 1.0, 101)
        out = cal.transform(grid)
        assert np.all(np.diff(out) >= -1e-12)
        q = cal.transform(raw)
        before = ccal.ece_from_bins(ccal.equal_frequency_bins(y, raw))
        after = ccal.ece_from_bins(ccal.equal_frequency_bins(y, q))
        assert after < before / 5
    iso = ccal.fit_calibrator("isotonic", y, raw).transform(np.array([0.0, 1.0]))
    assert iso.min() >= ccal.ISO_CLIP and iso.max() <= 1 - ccal.ISO_CLIP
    with pytest.raises(ValueError, match="both classes"):
        ccal.fit_calibrator("platt", np.zeros(5), np.full(5, 0.3))


# ---------------------------------------------------------------------------
# Information boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_key", ["logistic_regression", "random_forest_tuned", "lightgbm_tuned"])
def test_calibrator_and_threshold_only_see_inner_data(monkeypatch, model_key):
    X, y = synthetic(n=900, seed=11)
    tr, va = make_folds(y, CFG)[0]
    X_tr, y_tr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
    X_va = X.iloc[va].reset_index(drop=True)
    inner_tr, inner_ho = _make_inner_split(len(X_tr), y_tr, CFG, 0, None)
    y_inner, y_hold = y_tr.to_numpy()[inner_tr], y_tr.to_numpy()[inner_ho]

    seen_cal, seen_thr = [], []
    real_fit, real_sel = ccal.fit_calibrator, ccal.select_threshold

    def spy_fit(kind, yy, pp):
        seen_cal.append((kind, np.asarray(yy).copy(), np.asarray(pp).copy()))
        return real_fit(kind, yy, pp)

    def spy_sel(yy, pp, cfg):
        seen_thr.append(np.asarray(yy).copy())
        return real_sel(yy, pp, cfg)

    monkeypatch.setattr(ccal, "fit_calibrator", spy_fit)
    monkeypatch.setattr(ccal, "select_threshold", spy_sel)
    res = ccal.calibrate_fold(builders()[model_key], X_tr, y_tr, X_va, CFG, fold=0, **TE_KW)

    assert set(res["arms"]) == set(ccal.ARM_NAMES)
    assert len(seen_thr) == len(ccal.ARMS)
    for yy in seen_thr:  # every threshold is chosen on inner-holdout labels only
        np.testing.assert_array_equal(yy, y_hold)
    kinds = [(k, len(yy)) for k, yy, _ in seen_cal]
    assert ("platt", len(y_hold)) in kinds and ("isotonic", len(y_inner)) in kinds
    for kind, yy, pp in seen_cal:
        assert len(yy) in (len(y_hold), len(y_inner)) and len(yy) != len(va)
        if len(yy) == len(y_inner):
            np.testing.assert_array_equal(yy, y_inner)
        else:
            np.testing.assert_array_equal(yy, y_hold)
    for arm in res["arms"].values():
        assert len(arm["valid_probs"]) == len(va)


def test_calibrate_fold_has_no_outer_valid_label_argument_and_flip_is_inert():
    import inspect
    params = inspect.signature(ccal.calibrate_fold).parameters
    assert not any(name.startswith("y_va") or name == "y_valid" for name in params)

    X, y = synthetic(n=900, seed=5)
    folds = make_folds(y, CFG)
    tr, va = folds[0]
    y_flip = y.copy()
    y_flip.iloc[va] = 1 - y_flip.iloc[va]
    build = builders()["lightgbm_tuned"]
    out = []
    for labels in (y, y_flip):
        res = cal_run.run_fold("lightgbm_tuned", X, labels, tr, va, CFG, 0, SPEC,
                               params=LGBM_PARAMS,
                               build_kwargs={"lgbm_base_params": dict(runner.LGBM_PARAMS),
                                             "lgbm_n_jobs": 1})
        out.append(res)
    for arm in ccal.ARM_NAMES:
        np.testing.assert_array_equal(out[0]["arms"][arm]["valid_probs"], out[1]["arms"][arm]["valid_probs"])
        assert out[0]["arms"][arm]["threshold"] == out[1]["arms"][arm]["threshold"]
        assert out[0]["arms"][arm]["calibrator_info"] == out[1]["arms"][arm]["calibrator_info"]
    del build


def test_none_arm_reproduces_generic_nested_harness():
    """The refit with a reused config equals run_fold_nested_params with a 1-point grid."""
    X, y = synthetic(n=900, seed=2)
    tr, va = make_folds(y, CFG)[1]
    X_tr, y_tr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
    X_va = X.iloc[va].reset_index(drop=True)
    build = builders()["logistic_regression"]
    ref = cc.run_fold_nested_params(lambda p, cols: build(cols), [{"C": 0.1}], X_tr, y_tr, X_va,
                                    CFG, fold=1, **TE_KW)
    res = ccal.calibrate_fold(build, X_tr, y_tr, X_va, CFG, fold=1, **TE_KW)
    np.testing.assert_allclose(res["raw_valid_probs"], ref.valid_probs, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(res["arms"]["none"]["valid_probs"], res["raw_valid_probs"])
    assert res["arms"]["none"]["threshold"] == ref.threshold
    assert res["raw_inner_holdout_log_loss"] == pytest.approx(
        ref.grid_scores[0]["inner_holdout_log_loss"], abs=1e-12)


def test_crossfit_scores_are_out_of_fold_and_te_is_recomputed(monkeypatch):
    """Each cross-fit model never fits on the part it scores."""
    X, y = synthetic(n=600, seed=9)
    fitted, scored = [], []

    class Spy:
        def __init__(self, cols):
            self.cols = cols

        def fit(self, Xf, yf):
            fitted.append(set(Xf["row_id"].astype(int)))
            assert "TE_Route" in Xf.columns
            self.rate = float(np.mean(yf))
            return self

        def predict_proba(self, Xp):
            scored.append(set(Xp["row_id"].astype(int)))
            p = np.full(len(Xp), self.rate)
            return np.column_stack([1 - p, p])

    oof = ccal.crossfit_scores(lambda cols: Spy(cols), X, y, k=4, seed=3, te_cols=TE_COLS,
                               te_m=20.0, te_inner_splits=5, te_seed=7, te_drop_original=False)
    assert len(fitted) == len(scored) == 4
    for f, s in zip(fitted, scored):
        assert not (f & s)
        assert f | s == set(range(600))
    assert np.isfinite(oof).all()


# ---------------------------------------------------------------------------
# Scoring and paired deltas
# ---------------------------------------------------------------------------


def test_score_pooled_uses_per_fold_thresholds_and_monotone_auc():
    X, y = synthetic(n=600, seed=4)
    folds = make_folds(y, CFG)
    rng = np.random.default_rng(0)
    p = np.clip(0.2 + 0.1 * rng.standard_normal(len(y)) + 0.1 * y.to_numpy(), 0.01, 0.99)
    s1 = ccal.score_pooled(y, p, folds, [0.2, 0.25, 0.3])
    pred = np.zeros(len(y), dtype=int)
    for (_, va), th in zip(folds, [0.2, 0.25, 0.3]):
        pred[va] = p[va] >= th
    from sklearn.metrics import f1_score
    assert s1["macro_f1_nested"] == pytest.approx(f1_score(y, pred, average="macro"))
    platt = ccal.fit_calibrator("platt", y, p).transform(p)
    assert ccal.score_pooled(y, platt, folds, [0.2] * 3)["roc_auc"] == pytest.approx(s1["roc_auc"], abs=1e-12)
    # different monotone maps per fold: pooled AUC may move, within-fold mean does not
    per_fold = p.copy()
    for i, (_, va) in enumerate(folds):
        per_fold[va] = p[va] ** (1 + i)
    s2 = ccal.score_pooled(y, per_fold, folds, [0.2] * 3)
    assert s2["roc_auc_within_fold_mean"] == pytest.approx(s1["roc_auc_within_fold_mean"], abs=1e-12)


def test_paired_vs_none_and_summary_sign_consistency():
    rows = []
    for seed, base in ((42, 0.44), (1, 0.45), (7, 0.43)):
        for arm, kind, src in ccal.ARMS:
            shift = {"none": 0.0, "platt_inner_holdout": -0.001, "isotonic_inner_holdout": 0.002,
                     "platt_crossfit": -0.0005, "isotonic_crossfit": 0.001}[arm]
            rows.append({"seed": seed, "model": "lightgbm_tuned", "arm": arm, "calibrator": kind,
                         "calibration_data": src, "fold_fingerprint": f"fp{seed}",
                         **{m: base + shift for m in ccal.RUN_METRICS}})
    paired = ccal.paired_vs_none(pd.DataFrame(rows))
    assert len(paired) == 3 * 4 * len(ccal.RUN_METRICS)
    summ = ccal.summarize_deltas(paired)
    d = summ["lightgbm_tuned"]["platt_inner_holdout"]["log_loss"]
    assert d["mean"] == pytest.approx(-0.001) and d["sign_consistent_across_seeds"]
    bad = pd.DataFrame(rows)
    bad.loc[bad.arm.eq("platt_crossfit") & bad.seed.eq(1), "fold_fingerprint"] = "other"
    with pytest.raises(ValueError, match="different outer folds"):
        ccal.paired_vs_none(bad)


def test_selected_model_builder_validates_params():
    with pytest.raises(ValueError, match="unexpected LR params"):
        ccal.selected_model_builder("logistic_regression", {"C": 1.0, "x": 1})
    with pytest.raises(ValueError, match="unexpected LightGBM params"):
        ccal.selected_model_builder("lightgbm_tuned", {"n_estimators": 5},
                                    lgbm_base_params=dict(runner.LGBM_PARAMS))
    model = ccal.selected_model_builder("lightgbm_tuned", LGBM_PARAMS,
                                        lgbm_base_params=dict(runner.LGBM_PARAMS), lgbm_n_jobs=3)([])
    assert model.get_params()["n_jobs"] == 3 and model.get_params()["n_estimators"] == 30
    assert model.get_params()["random_state"] == 42


def test_runner_identity_includes_code_hashes():
    assert "src/classifier_calibration.py" in cal_run.IDENTITY_CODE_FILES
    assert set(cal_run.IDENTITY_CODE_FILES) <= set(cal_run.CODE_FILES)
