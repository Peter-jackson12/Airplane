"""Structural leakage and checkpoint regressions. Synthetic rows are unit tests only."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, f1_score

import rerun_all_phases as runner
import src.cv as cv
from src.run_store import read_rows, upsert_row


class ProbeModel:
    fits = []

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

    def fit(self, X, y):
        self.fits.append((X.copy(), y.copy()))
        self.feature_name_ = list(X.columns)
        self.mean = float(y.mean())
        return self

    def predict_proba(self, X):
        p = np.clip(np.asarray(X.get("TE_cat", np.full(len(X), self.mean))), .01, .99)
        return np.column_stack([1 - p, p])


@pytest.fixture
def data():
    ProbeModel.fits = []
    X = pd.DataFrame({"row_id": np.arange(120), "cat": pd.Categorical(np.arange(120) % 3)})
    y = pd.Series(np.arange(120) % 2)
    return X, y, X.iloc[:8].assign(row_id=lambda x: x.row_id + 1000)


def test_pseudo_factory_never_receives_student_holdout(data, monkeypatch):
    X, y, valid = data
    train, hold = np.arange(90), np.arange(90, 120)
    monkeypatch.setattr(cv, "_make_inner_split", lambda *a: (train, hold))
    observed = []

    def factory(Xi, yi):
        observed.append((Xi.copy(), yi.copy()))
        return Xi.iloc[:2].copy(), yi.iloc[:2].copy()

    def run(labels):
        return cv.run_fold_nested_grid(ProbeModel, X, labels, valid, cv.CVConfig(),
                                      n_estimators_grid=[10, 25], te_cols=["cat"],
                                      extra_fit_factory=factory)

    run(y)
    original_fits = ProbeModel.fits.copy()
    changed = y.copy()
    changed.iloc[hold] = 1 - changed.iloc[hold]
    ProbeModel.fits = []
    run(changed)
    pd.testing.assert_frame_equal(observed[0][0], observed[1][0])
    pd.testing.assert_series_equal(observed[0][1], observed[1][1])
    assert set(observed[0][0].row_id).isdisjoint(hold)
    for (Xa, ya), (Xb, yb) in zip(original_fits, ProbeModel.fits):
        pd.testing.assert_frame_equal(Xa, Xb)
        pd.testing.assert_series_equal(ya, yb)


def test_outer_valid_changes_cannot_change_selection_or_threshold(data):
    X, y, valid = data
    def run(V):
        metadata = {}
        result = cv.run_fold_nested_grid(ProbeModel, X, y, V, cv.CVConfig(),
                                        n_estimators_grid=[10, 25], te_cols=["cat"],
                                        selection_metadata=metadata)
        return result, metadata
    first, m1 = run(valid)
    second, m2 = run(valid.assign(cat=pd.Categorical([900] * len(valid))))
    assert first.grid_scores == second.grid_scores
    assert first.selected_n_estimators == second.selected_n_estimators
    assert m1 == m2
    assert all(set(Xi.row_id).isdisjoint(valid.row_id) for Xi, _ in ProbeModel.fits)


def test_honest_p5_actual_runner_scopes_teacher_to_student_train(data, monkeypatch):
    X, y, unlab = data
    monkeypatch.setattr(runner, "build_features", lambda *a: (X, y, unlab))
    monkeypatch.setattr(runner, "CFG", cv.CVConfig(n_splits=3))
    monkeypatch.setattr(runner, "N_ESTIMATORS_GRID", [10, 25])
    monkeypatch.setattr(runner, "model_factory", ProbeModel)
    spec = runner.PhaseSpec("test", "test", te=True, pseudo="honest", cat_cols=("cat",))
    folds = cv.make_folds(y, runner.CFG)
    seen = []
    actual_teacher = runner._teacher_fit_predict

    def teacher(Xt, yt, apply, spec, fold):
        tr, va = folds[fold]
        it, ih = cv._make_inner_split(len(tr), y.iloc[tr].reset_index(drop=True), runner.CFG, fold, None)
        assert set(Xt.row_id) == set(X.iloc[tr[it]].row_id)
        assert set(Xt.row_id).isdisjoint(X.iloc[np.r_[va, tr[ih]]].row_id)
        seen.append(fold)
        return actual_teacher(Xt, yt, apply, spec, fold)

    monkeypatch.setattr(runner, "_teacher_fit_predict", teacher)
    result = runner.run_phase(spec, pd.DataFrame(), {})
    assert seen == [0, 1, 2]
    assert result["pseudo_total"] > 0
    assert 'outer_train_holdout' in result["protocol"]


def test_fixed_thresholds_are_used_for_f1_and_confusion_matrix():
    y = pd.Series([0, 1, 0, 1, 1, 0])
    probs = np.array([.2, .7, .4, .3, .6, .8])
    folds = [(np.arange(3, 6), np.arange(3)), (np.arange(3), np.arange(3, 6))]
    thresholds = [.3, .7]
    result = cv.evaluate_oof(y, probs, folds, cv.CVConfig(n_splits=2),
                             per_fold_thresholds=thresholds)
    expected = np.r_[probs[:3] >= .3, probs[3:] >= .7]
    assert result["macro_f1"] == f1_score(y, expected, average="macro")
    assert list(result["confusion_matrix"].values()) == list(confusion_matrix(y, expected).ravel())
    changed = cv.evaluate_oof(1-y, probs, folds, cv.CVConfig(n_splits=2),
                              per_fold_thresholds=thresholds)
    assert changed["per_fold_thresholds"] == thresholds


def test_external_eval_set_rejected(data):
    X, y, valid = data
    with pytest.raises(ValueError, match="eval_set"):
        cv.run_fold_nested_grid(ProbeModel, X, y, valid, cv.CVConfig(),
                                n_estimators_grid=[10], te_cols=["cat"],
                                fit_kwargs={"eval_set": [(valid, y.iloc[:len(valid)])]})


@pytest.mark.parametrize("grid", [[0], [-1], [1.5], [True]])
def test_invalid_grid_rejected(data, grid):
    X, y, valid = data
    with pytest.raises(ValueError, match="grid"):
        cv.run_fold_nested_grid(ProbeModel, X, y, valid, cv.CVConfig(), n_estimators_grid=grid)


def test_preencoded_input_rejected(data):
    X, y, valid = data
    with pytest.raises(ValueError, match="TE"):
        cv.run_fold_nested_grid(ProbeModel, X.assign(TE_cat=y), y, valid,
                                cv.CVConfig(), n_estimators_grid=[10])


def test_checkpoint_rejects_old_schema_without_changing_bytes(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("phase_key,best_iterations\nP4,[2]\n")
    before = path.read_bytes()
    fields = ["run_id", "phase_key", "value"]
    with pytest.raises(ValueError, match="schema"):
        upsert_row(path, fields, "new", dict(run_id="new", phase_key="P4", value="50"))
    assert path.read_bytes() == before


def test_checkpoint_force_replaces_phase_and_preserves_other_phases(tmp_path):
    path = tmp_path / "new.csv"
    fields = ["run_id", "phase_key", "value"]
    for key, value in [("P4", "50"), ("P3", "25"), ("P4", "75")]:
        upsert_row(path, fields, "id", dict(run_id="id", phase_key=key, value=value))
    rows = read_rows(path, fields, "id")
    assert len(rows) == 2 and rows["P4"]["value"] == "75" and rows["P3"]["value"] == "25"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Experiment"):
        upsert_row(path, fields, "other", dict(run_id="other", phase_key="P4", value="10"))
    assert before == path.read_bytes()


@pytest.mark.parametrize("suffix", ["extra,cell", ""])
def test_malformed_csv_row_rejected(tmp_path, suffix):
    path = tmp_path / "bad.csv"
    path.write_text("run_id,phase_key,value\nid,P4," + suffix + "\n" if suffix else
                    "run_id,phase_key,value\nid,P4\n")
    with pytest.raises(ValueError, match="Malformed"):
        read_rows(path, ["run_id", "phase_key", "value"], "id")


def test_resume_identity_changes_with_data_sample_config_and_code(tmp_path, monkeypatch):
    for name in ("src/cv.py", "src/features.py", "src/run_store.py", "rerun_all_phases.py", "data/train.csv"):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("initial")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "DATA_PATH", tmp_path / "data/train.csv")
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path / "output")
    # Restore module-level mutable execution identity after this test.
    for key in ("CSV_PATH", "MD_PATH", "RUN_METADATA", "RUN_ID", "CFG"):
        monkeypatch.setattr(runner, key, getattr(runner, key))
    runner.configure_run(None, None)
    initial = runner.RUN_ID
    runner.configure_run(None, None)
    assert initial == runner.RUN_ID
    runner.configure_run(20, None)
    assert initial != runner.RUN_ID and "sample_20" in str(runner.CSV_PATH)
    runner.configure_run(None, None)
    runner.DATA_PATH.write_text("different")
    runner.configure_run(None, None)
    assert initial != runner.RUN_ID
    data_id = runner.RUN_ID
    runner.CFG = replace(runner.CFG, seed=7)
    runner.configure_run(None, None)
    assert data_id != runner.RUN_ID
    config_id = runner.RUN_ID
    (tmp_path / "src/cv.py").write_text("changed")
    runner.configure_run(None, None)
    assert config_id != runner.RUN_ID
    with pytest.raises(ValueError):
        runner.configure_run(None, "baseline_recovery")
    with pytest.raises(ValueError):
        runner.configure_run(None, "es_diagnosis")


def test_force_does_not_bypass_checkpoint_validation(tmp_path, monkeypatch):
    path = tmp_path / "old.csv"
    path.write_text("phase_key,best_iterations\nP4,[2]\n")
    monkeypatch.setattr(runner, "CSV_PATH", path)
    monkeypatch.setattr(runner, "RUN_ID", "new")
    with pytest.raises(ValueError, match="schema"):
        runner.load_done(True)
