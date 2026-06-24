"""
Regression tests for the model-discovery layer: AlgModelFinder, sequentialThLin,
and the selection/comparison helpers stols and compare_models_.

Core idea: build data with a KNOWN exact algebraic relation (z = 2x + 3y) and
assert the discovery routines recover the coefficients and a perfect fit.
"""
import numpy as np
import pandas as pd
import pytest

from daeFinder import dae_finder as df


# --------------------------------------------------------------------------- AlgModelFinder
def test_alg_model_finder_recovers_exact_relation(linear_relation_data):
    amf = df.AlgModelFinder(model_id="LR", fit_intercept=False)
    amf.fit(linear_relation_data, features_to_fit=["z"])
    coefs = amf.get_fitted_models()["z"]
    assert coefs["x"] == pytest.approx(2.0, abs=1e-6)
    assert coefs["y"] == pytest.approx(3.0, abs=1e-6)
    assert amf.r2_score_dict["z"] == pytest.approx(1.0, abs=1e-9)


def test_alg_model_finder_scaling_roundtrip(linear_relation_data):
    # With column scaling the fit happens in standardized space; get_fitted_models
    # must scale the coefficients back to the original units (still 2 and 3).
    amf = df.AlgModelFinder(model_id="LR", fit_intercept=False)
    amf.fit(linear_relation_data, features_to_fit=["z"], scale_columns=True)
    coefs = amf.get_fitted_models(scale_coef=True)["z"]
    assert coefs["x"] == pytest.approx(2.0, abs=1e-6)
    assert coefs["y"] == pytest.approx(3.0, abs=1e-6)


def test_alg_model_finder_predict_features(linear_relation_data):
    amf = df.AlgModelFinder(model_id="LR", fit_intercept=False)
    amf.fit(linear_relation_data, features_to_fit=["z"])
    pred = amf.predict_features(linear_relation_data, ["z"])
    np.testing.assert_allclose(pred["z"], linear_relation_data["z"], atol=1e-6)


def test_alg_model_finder_feature_to_library_map():
    # 'w' is a spurious column NOT in the true relation. The map must restrict the
    # library for 'z' to {x, y}, excluding w (whereas the default library is {x,y,w}).
    rng = np.random.RandomState(4)
    x, y, w = rng.rand(80), rng.rand(80), rng.rand(80)
    data = pd.DataFrame({"x": x, "y": y, "w": w, "z": 2 * x + 3 * y})
    amf = df.AlgModelFinder(model_id="LR", fit_intercept=False)
    amf.fit(data, features_to_fit=["z"], feature_to_library_map_={"z": ["x", "y"]})
    coefs = amf.get_fitted_models()["z"]
    assert set(coefs.keys()) == {"x", "y"}          # w excluded by the map
    assert coefs["x"] == pytest.approx(2.0, abs=1e-6)
    assert coefs["y"] == pytest.approx(3.0, abs=1e-6)


def test_alg_model_finder_best_models_and_compare(linear_relation_data):
    amf = df.AlgModelFinder(model_id="LR", fit_intercept=False)
    amf.fit(linear_relation_data, features_to_fit=["z"])
    best = amf.best_models(num=1)
    assert list(best.columns) == ["z"]
    assert "r2- metric" in best.index
    assert best.loc["x", "z"] == pytest.approx(2.0, abs=1e-6)

    true_model = pd.DataFrame({"z": [2.0, 3.0]}, index=["x", "y"])
    true_model.loc["r2- metric"] = [1.0]
    diff = amf.compare_models(true_model)
    # perfect structural match -> all zeros and zero inconsistent terms
    assert diff.loc["# incosistent terms", "z"] == 0.0


def test_alg_model_finder_intercept(linear_relation_data):
    data = linear_relation_data.copy()
    data["z"] = data["z"] + 1.5
    amf = df.AlgModelFinder(model_id="LR", fit_intercept=True)
    amf.fit(data, features_to_fit=["z"])
    assert amf.get_fitted_intercepts()["z"] == pytest.approx(1.5, abs=1e-6)


# --------------------------------------------------------------------------- sequentialThLin
def test_sequential_thlin_recovers_sparse_support():
    rng = np.random.RandomState(0)
    a = rng.rand(60)
    b = rng.rand(60)
    c = rng.rand(60)
    X = pd.DataFrame({"a": a, "b": b, "c": c})
    y = 2.0 * a  # only 'a' is relevant
    model = df.sequentialThLin(model_id="LR", coef_threshold=0.1,
                               fit_intercept=False, silent=True)
    model.fit(X, y=y)
    coefs = dict(zip(model.feature_names_in_, model.coef_))
    assert coefs["a"] == pytest.approx(2.0, abs=1e-6)
    assert coefs["b"] == 0.0 and coefs["c"] == 0.0
    assert model.score(X, y) == pytest.approx(1.0, abs=1e-9)


def test_sequential_thlin_warns_when_all_below_threshold():
    rng = np.random.RandomState(3)
    X = pd.DataFrame({"a": rng.rand(40) * 1e-3, "b": rng.rand(40) * 1e-3})
    y = rng.rand(40) * 1e-3
    model = df.sequentialThLin(model_id="LR", coef_threshold=10.0,
                               fit_intercept=False, silent=True)
    with pytest.warns(UserWarning):
        model.fit(X, y=y)


# --------------------------------------------------------------------------- stols
def test_stols_thresholds_by_magnitude():
    selected = df.stols({"a": 0.5, "b": 0.01, "c": 0.9}, threshold=0.1, pd_dict=False)
    assert selected == {"a": 0.5, "c": 0.9}


def test_stols_boundary_is_strict():
    # selection uses abs(coef) > threshold (strict): a coef exactly at the threshold drops.
    selected = df.stols({"a": 0.1, "b": 0.2}, threshold=0.1, pd_dict=False)
    assert selected == {"b": 0.2}


def test_stols_dataframe_output():
    out = df.stols({"a": 0.5, "b": 0.01}, threshold=0.1)
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["a"]
    assert out.loc["a", "Coefficient"] == 0.5


def test_stols_dominant_balance_uses_relative_threshold():
    # threshold becomes threshold * ||coef||_2; small terms relative to the vector drop
    coefs = {"a": 3.0, "b": 0.2, "c": 4.0}
    selected = df.stols(coefs, threshold=0.1, dominant_balance=True, pd_dict=False)
    assert selected == {"a": 3.0, "c": 4.0}


# --------------------------------------------------------------------------- compare_models_
def test_compare_models_structure_diff():
    m1 = pd.DataFrame({"z": [1.0, 0.0]}, index=["x", "y"])  # z uses x only
    m2 = pd.DataFrame({"z": [1.0, 1.0]}, index=["x", "y"])  # z uses x and y
    diff = df.compare_models_(m1.copy(), m2.copy())
    assert diff.loc["x", "z"] == 0.0      # both present -> match
    assert diff.loc["y", "z"] == -1.0     # absent in m1, present in m2
    assert diff.loc["# incosistent terms", "z"] == 1.0


def test_compare_models_identical_models():
    m = pd.DataFrame({"z": [1.0, 0.0]}, index=["x", "y"])
    diff = df.compare_models_(m.copy(), m.copy())
    assert diff.loc["# incosistent terms", "z"] == 0.0
