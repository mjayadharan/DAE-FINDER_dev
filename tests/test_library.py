"""
Regression tests for candidate-library construction and refinement:
PolyFeatureMatrix, FeatureCouplingTransformer, get_factor_feat, and the
get_refined_lib / get_refined_lib_stable family.
"""
import numpy as np
import pandas as pd
import sympy
import pytest
from scipy.sparse import coo_array

from daeFinder import dae_finder as df


# --------------------------------------------------------------------------- PolyFeatureMatrix
def test_poly_feature_matrix_names_and_values():
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    lib = df.PolyFeatureMatrix(degree=2, include_bias=True).fit_transform(data)
    assert list(lib.columns) == ["1", "x", "y", "x^2", "x y", "y^2"]
    np.testing.assert_array_equal(lib["1"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(lib["x^2"], data["x"] ** 2)
    np.testing.assert_allclose(lib["x y"], data["x"] * data["y"])
    np.testing.assert_allclose(lib["y^2"], data["y"] ** 2)


def test_poly_feature_matrix_interaction_only_no_bias():
    data = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    lib = df.PolyFeatureMatrix(degree=2, include_bias=False,
                               interaction_only=True).fit_transform(data)
    assert list(lib.columns) == ["x", "y", "x y"]  # no x^2 / y^2 with interaction_only


def test_poly_feature_matrix_array_output():
    data = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    out = df.PolyFeatureMatrix(degree=1, output_df=False).fit_transform(data)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)  # bias + x + y


# --------------------------------------------------------------------------- FeatureCouplingTransformer
def _sparsity():
    return coo_array((np.array([4, 5, 7, 5]),
                      (np.array([0, 0, 1, 1]), np.array([0, 2, 2, 1]))))


def test_feature_coupling_default_interaction():
    data = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], columns=["t", "x", "y"])
    ct = df.FeatureCouplingTransformer(_sparsity(), return_df=True)
    out = ct.fit_transform(data)
    assert list(ct.get_feature_names_out()) == ["t*t", "t*y", "x*y", "x*x"]
    # values are the products of the coupled columns
    np.testing.assert_allclose(out["t*t"], data["t"] * data["t"])
    np.testing.assert_allclose(out["t*y"], data["t"] * data["y"])
    np.testing.assert_allclose(out["x*y"], data["x"] * data["y"])
    np.testing.assert_allclose(out["x*x"], data["x"] * data["x"])


def test_feature_coupling_custom_function_and_namer():
    data = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], columns=["t", "x", "y"])
    ct = df.FeatureCouplingTransformer(
        _sparsity(),
        coupling_func=lambda a, b, i, j, k: a - b - k,
        coupling_namer=lambda a, b, i, j, k: "{}-{}-{}".format(a, b, k),
        coupling_func_args={"k": 2},
        return_df=True,
    )
    out = ct.fit_transform(data)
    assert list(ct.get_feature_names_out()) == ["t-t-2", "t-y-2", "x-y-2", "x-x-2"]
    # row 0: t-t-2 = 1-1-2 = -2 ; t-y-2 = 1-3-2 = -4 ; x-y-2 = 2-3-2 = -3 ; x-x-2 = 2-2-2 = -2
    np.testing.assert_allclose(out.iloc[0].to_numpy(), [-2.0, -4.0, -3.0, -2.0])


def test_feature_coupling_requires_coo_when_no_indices():
    with pytest.raises(AssertionError):
        df.FeatureCouplingTransformer(sparsity_matrix=np.eye(3))  # not a coo_array


# --------------------------------------------------------------------------- get_factor_feat
def test_get_factor_feat_selects_divisible_terms():
    S, E = sympy.symbols("S E")
    feat_dict = {"S*E": S * E, "S^2": S ** 2, "E": E, "1": sympy.Integer(1)}
    # features divisible by S: S*E and S^2 (and 1 is not)
    assert set(df.get_factor_feat(S, feat_dict)) == {"S*E", "S^2"}
    assert set(df.get_factor_feat(S * E, feat_dict)) == {"S*E"}


# --------------------------------------------------------------------------- refinement
def test_get_refined_lib_stable_drops_factor_terms():
    rng = np.random.RandomState(0)
    data = pd.DataFrame(rng.rand(20, 2), columns=["S", "E"])
    lib = df.PolyFeatureMatrix(degree=2).fit_transform(data)
    S, E = sympy.symbols("S E")
    dropped, refined = df.get_refined_lib_stable(S * E, data, lib, get_dropped_feat=True)
    assert dropped == {"S E"}
    assert "S E" not in refined.columns
    assert set(refined.columns) == set(lib.columns) - {"S E"}


def test_get_refined_lib_stable_list_of_factors():
    rng = np.random.RandomState(1)
    data = pd.DataFrame(rng.rand(20, 3), columns=["S", "E", "P"])
    lib = df.PolyFeatureMatrix(degree=2).fit_transform(data)
    S, E, P = sympy.symbols("S E P")
    dropped = df.get_refined_lib_stable([S * E, P ** 2], data, lib, get_dropped_feat=True)[0]
    assert {"S E", "P^2"} <= dropped


def test_get_refined_lib_and_stable_agree_on_plain_names():
    rng = np.random.RandomState(2)
    data = pd.DataFrame(rng.rand(20, 2), columns=["S", "E"])
    lib = df.PolyFeatureMatrix(degree=2).fit_transform(data)
    S, E = sympy.symbols("S E")
    a = df.get_refined_lib(S * E, data, lib)
    b = df.get_refined_lib_stable(S * E, data, lib)
    assert list(a.columns) == list(b.columns)
