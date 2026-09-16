"""Synthetic unit fixtures only; actual performance comes from data/train.csv."""
import json

import numpy as np
import pandas as pd
import pytest

import rerun_all_phases as runner
from src.oof import OOF_SCHEMA_VERSION, build_oof_rows, calibration_bins, diagnose, labeled_identity, metrics, validate_oof_rows
from src.run_store import digest, read_oof, save_oof


@pytest.fixture
def raw():
    return pd.DataFrame({"ID": [f"r{i}" for i in range(7)],
                         "Delay": ["Not_Delayed", None, "Delayed", " ", "Not_Delayed", "Delayed", "Not_Delayed"],
                         "Estimated_Departure_Time": [100, None, None, 200, 300, None, 400],
                         "Estimated_Arrival_Time": [200, None, 200, 300, None, None, 500],
                         "Airline": ["A", None, None, "A", "B", "A", "B"]})


@pytest.fixture
def rows(raw):
    folds = [(np.array([2, 3, 4]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3, 4]))]
    return build_oof_rows(raw, [0, 1, 0, 1, 0], [.2, .7, .4, .3, .8], folds,
                          [.3, .6], [10, 25], run_id="test", phase_key="P6_clean", seed=42)


def test_identity_survives_unlabeled_rows_and_index_reset(raw, rows):
    raw.index = [70, 60, 50, 40, 30, 20, 10]
    assert labeled_identity(raw).source_row.tolist() == [0, 2, 4, 5, 6]
    assert rows.ID.tolist() == ["r0", "r2", "r4", "r5", "r6"]
    assert rows.prediction.tolist() == [0, 1, 0, 0, 1]
    assert rows["raw_missing__Airline"].tolist() == [0, 1, 0, 0, 0]


@pytest.mark.parametrize("valid", [[0, 1, 1, 2, 3, 4], [0, 1, 2, 3]])
def test_repeated_or_unscored_rows_rejected(raw, valid):
    with pytest.raises(ValueError, match="exactly once"):
        build_oof_rows(raw, [0, 1, 0, 1, 0], [.2] * 5,
                        [(np.array([], dtype=int), np.array(valid))], [.3], [10],
                        run_id="test", phase_key="P6_clean", seed=42)


@pytest.mark.parametrize("column,value", [("probability", np.inf), ("probability", -.1),
                                          ("prediction", 1), ("threshold", .9), ("fold", 8)])
def test_corrupt_rows_rejected(rows, column, value):
    rows.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_oof_rows(rows, run_id="test", phase_key="P6_clean", seed=42, n_rows=5, n_splits=2)


def test_metrics_independent_arithmetic(rows):
    m = metrics(rows)
    assert [m[k] for k in ("tn", "fp", "fn", "tp")] == [2, 1, 1, 1]
    assert m["macro_f1"] == pytest.approx((4 / 6 + 2 / 4) / 2)
    assert m["brier"] == pytest.approx((.04 + .09 + .16 + .49 + .64) / 5)
    assert m["log_loss"] == pytest.approx(-np.log([.8, .7, .6, .3, .2]).mean())


def test_empty_single_class_groups_and_time_partition(rows):
    scores, bins = diagnose(rows)
    time = scores[scores.dimension.eq("raw_time_pattern")]
    assert time.n.sum() == len(rows)
    single = scores[(scores.dimension == "raw_missing__Airline") & (scores.group == "missing")].iloc[0]
    assert single.n == 1 and np.isnan(single.roc_auc)
    assert bins[bins.dimension.eq("overall")].n.sum() == len(rows)


def test_calibration_endpoints_and_empty_bins():
    bins = calibration_bins([0, 1, 0, 1], [0, .1, .99, 1])
    assert bins.n.tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 2]
    assert np.isnan(bins.loc[2, "observed_rate"])
    assert bins.loc[9, "observed_rate"] == .5


def checkpoint(tmp_path, rows):
    summary = tmp_path / "baseline_recovery_v2_test.csv"
    meta = {"oof_schema_version": OOF_SCHEMA_VERSION, "cv": {"seed": 42, "n_splits": 2}}
    rows = rows.copy()
    rows["run_id"] = run_id = digest(meta)
    path, sha = save_oof(rows, tmp_path / f"{summary.stem}_oof", "P6_clean")
    record = dict(run_id=run_id, run_metadata=json.dumps(meta), phase_key="P6_clean",
                  n_rows=5, oof_path=path.relative_to(tmp_path).as_posix(), oof_sha256=sha,
                  oof_columns=json.dumps(list(rows.columns)),
                  per_fold_thresholds="[0.3, 0.6]", selected_n_estimators="[10, 25]")
    return summary, record, path, rows


def test_compressed_roundtrip_and_content_addressing(tmp_path, rows):
    summary, record, path, expected = checkpoint(tmp_path, rows)
    actual = read_oof(summary, record)
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False)
    same, same_sha = save_oof(expected, path.parent, "P6_clean")
    assert same == path and same_sha == record["oof_sha256"]
    expected.loc[0, "probability"] = .25
    other, _ = save_oof(expected, path.parent, "P6_clean")
    assert other != path and path.exists()
    assert read_oof(summary, record).probability.iloc[0] == .2


@pytest.mark.parametrize("damage", ["missing", "hash", "columns", "fold", "path"])
def test_checkpoint_rejects_damaged_artifacts(tmp_path, rows, damage):
    summary, record, path, _ = checkpoint(tmp_path, rows)
    if damage == "missing":
        path.unlink()
    elif damage == "hash":
        path.write_bytes(b"damaged")
    elif damage == "columns":
        record["oof_columns"] = "[]"
    elif damage == "fold":
        record["per_fold_thresholds"] = "[0.5, 0.5]"
    else:
        record["oof_path"] = "../wrong.csv.gz"
    with pytest.raises(ValueError):
        read_oof(summary, record)


def test_resume_never_skips_missing_oof_even_with_force(tmp_path, rows, monkeypatch):
    summary, record, path, _ = checkpoint(tmp_path, rows)
    monkeypatch.setattr(runner, "CSV_PATH", summary)
    monkeypatch.setattr(runner, "SAVE_OOF", True)
    monkeypatch.setattr(runner, "RUN_ID", record["run_id"])
    monkeypatch.setattr(runner, "read_rows", lambda *a: {"P6_clean": record})
    assert "P6_clean" in runner.load_done(False)
    path.unlink()
    for force in (False, True):
        with pytest.raises(ValueError, match="missing"):
            runner.load_done(force)


@pytest.mark.parametrize("damage", ["duplicate", "unknown_target", "alignment"])
def test_bad_source_identity_rejected(raw, rows, damage):
    if damage == "duplicate":
        raw.loc[2, "ID"] = raw.loc[0, "ID"]
    elif damage == "unknown_target":
        raw.loc[2, "Delay"] = "invalid"
    else:
        raw.loc[2, "Delay"] = "Not_Delayed"
    with pytest.raises(ValueError):
        build_oof_rows(raw, [0, 1, 0, 1, 0], [.2] * 5,
                        [(np.array([2, 3, 4]), np.array([0, 1])),
                         (np.array([0, 1]), np.array([2, 3, 4]))], [.3, .6], [10, 25],
                        run_id="test", phase_key="P6_clean", seed=42)


def test_all_observed_has_explicit_empty_missing_group(rows):
    rows["raw_missing__Airline"] = 0
    scores, bins = diagnose(rows)
    empty = scores[(scores.dimension == "raw_missing__Airline") & (scores.group == "missing")].iloc[0]
    assert empty.n == 0 and np.isnan(empty.log_loss)
    assert not ((bins.dimension == "raw_missing__Airline") & (bins.group == "missing")).any()


def test_actual_runner_export_preserves_predictions(tmp_path, monkeypatch):
    from src.cv import CVConfig
    from tests.test_pipeline_regressions import ProbeModel

    n = 120
    raw = pd.DataFrame({"ID": [f"row{i}" for i in range(n)],
                        "Delay": ["Not_Delayed", "Delayed"] * (n // 2),
                        "cat": pd.Categorical(np.arange(n) % 3)})
    X = raw[["cat"]].copy()
    y = pd.Series(np.arange(n) % 2)
    monkeypatch.setattr(runner, "build_features", lambda *a: (X, y, X.iloc[:0]))
    monkeypatch.setattr(runner, "CFG", CVConfig(n_splits=3))
    monkeypatch.setattr(runner, "N_ESTIMATORS_GRID", [10, 25])
    monkeypatch.setattr(runner, "model_factory", ProbeModel)
    monkeypatch.setattr(runner, "CSV_PATH", tmp_path / "baseline_recovery_v2_runner.csv")
    meta = {"oof_schema_version": OOF_SCHEMA_VERSION, "cv": {"seed": 42, "n_splits": 3}}
    monkeypatch.setattr(runner, "RUN_METADATA", meta)
    monkeypatch.setattr(runner, "RUN_ID", digest(meta))
    monkeypatch.setattr(runner, "SAVE_OOF", False)
    spec = runner.PhaseSpec("test", "test", te=True, cat_cols=("cat",))
    original = runner.run_phase(spec, raw, {})
    monkeypatch.setattr(runner, "SAVE_OOF", True)
    saved = runner.run_phase(spec, raw, {})
    runner.append_row(saved)
    restored = read_oof(runner.CSV_PATH, runner.load_done(False)["test"])
    assert restored.ID.tolist() == raw.ID.tolist()
    for metric in ("macro_f1_nested", "log_loss", "roc_auc", "tn", "fp", "fn", "tp"):
        assert original[metric] == saved[metric]


def test_encoded_airline_missingness_uses_encoder_metadata(raw):
    from src.features import encode_categoricals

    selected = raw.iloc[[0, 2, 4, 5, 6]][["Airline"]].reset_index(drop=True)
    encoded, encoders = encode_categoricals(selected, cat_cols=["Airline"])
    encoded.attrs["missing_category_codes"] = {"Airline": int(encoders["Airline"].transform(["MISSING"])[0])}
    folds = [(np.array([2, 3, 4]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3, 4]))]
    def build():
        return build_oof_rows(raw, [0, 1, 0, 1, 0], [.2] * 5, folds, [.3, .6], [10, 25],
                              run_id="test", phase_key="P6_clean", seed=42, features=encoded)
    assert build()["feature_missing__Airline"].tolist() == [0, 1, 0, 0, 0]
    encoded.attrs.clear()
    with pytest.raises(ValueError, match="category-code"):
        build()
