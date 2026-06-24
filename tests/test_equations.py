"""
Regression tests for the symbolic post-processing layer: turning fitted
coefficient tables into simplified symbolic relations and mapping symbolic
monomials back to library feature names.

(Version-compatibility specifics of get_simplified_equation/get_refined_lib live
in test_py314_compat.py; here we test the feature behaviour.)
"""
import sympy
import pandas as pd
import pytest

from daeFinder import dae_finder as df


# --------------------------------------------------------------------------- get_simplified_equation
def test_get_simplified_equation_unsimplified_form():
    S, E = sympy.symbols("S E")
    # The 0.001 coefficient on 'S' is below coef_threshold=0.1 and must be eliminated;
    # this pins the coefficient-thresholding step (not just the symbol construction).
    bm = pd.DataFrame({"S E": [2.0, 0.5, 0.0, 0.001]}, index=["S^2", "E", "S E", "S"])
    res = df.get_simplified_equation(bm, "S E", ["S", "E", "ES", "P"],
                                     coef_threshold=0.1, simplified=False)
    assert sympy.expand(res["lhs"] - S * E) == 0
    assert sympy.expand(res["rhs"] - (2.0 * S ** 2 + 0.5 * E)) == 0  # 0.001*S dropped


def test_get_simplified_equation_simplified_preserves_ratio():
    S, E = sympy.symbols("S E")
    bm = pd.DataFrame({"S E": [2.0, 0.5, 0.0]}, index=["S^2", "E", "S E"])
    res = df.get_simplified_equation(bm, "S E", ["S", "E", "ES", "P"],
                                     coef_threshold=0.1, simplified=True)
    # simplification rewrites lhs/rhs but must preserve rhs/lhs as a rational function
    expected = sympy.cancel((2.0 * S ** 2 + 0.5 * E) / (S * E))
    assert sympy.cancel(res["rhs"] / res["lhs"] - expected) == 0


def test_get_simplified_equation_intercept_thresholding():
    S = sympy.symbols("S")
    bm = pd.DataFrame({"S^2": [3.0]}, index=["S^2"])
    # global_feature_list needs >=2 base variables (sympy.symbols returns a bare symbol
    # only for multi-element input); 'E' is an unused second base variable here.
    res = df.get_simplified_equation(bm, "S^2", ["S", "E"], coef_threshold=0.1,
                                     intercept=0.005, intercept_threshold=0.01,
                                     simplified=False)
    assert sympy.expand(res["rhs"] - 3.0 * S ** 2) == 0  # intercept 0.005 < 0.01 -> dropped
    # intercept above threshold is kept
    res2 = df.get_simplified_equation(bm, "S^2", ["S", "E"], coef_threshold=0.1,
                                      intercept=0.5, intercept_threshold=0.01,
                                      simplified=False)
    assert sympy.expand(res2["rhs"] - (3.0 * S ** 2 + 0.5)) == 0


def test_get_simplified_equation_list_covers_requested_features():
    bm = pd.DataFrame({"S E": [2.0, 0.5, 0.0], "S P": [1.0, 0.0, 0.0]},
                      index=["S^2", "E", "P"])
    res = df.get_simplified_equation_list(bm, ["S", "E", "P"], coef_threshold=0.1)
    assert set(res.keys()) == {"S E", "S P"}
    res_subset = df.get_simplified_equation_list(bm, ["S", "E", "P"],
                                                 coef_threshold=0.1,
                                                 feature_list_=["S E"])
    assert set(res_subset.keys()) == {"S E"}


# --------------------------------------------------------------------------- sympy_symb_to_feature_name
def test_sympy_symb_to_feature_name_matches_permutation():
    x, y = sympy.symbols("x y")
    # 'x*y' matches library entry 'x y' (order-insensitive via permutations)
    assert df.sympy_symb_to_feature_name(x * y, ["x y", "z"]) == "x y"
    assert df.sympy_symb_to_feature_name(x * y, ["y x"]) == "y x"


def test_sympy_symb_to_feature_name_constant_returns_none():
    assert df.sympy_symb_to_feature_name(sympy.Integer(1), ["x y"]) is None


def test_sympy_symb_to_feature_name_missing_raises():
    z = sympy.symbols("z")
    with pytest.raises(Exception):
        df.sympy_symb_to_feature_name(z, ["x y", "a b"])


# --------------------------------------------------------------------------- construct_reduced_fit_list
def test_construct_reduced_fit_list_sympy_format():
    S, E = sympy.symbols("S E")
    eqs = {"S E": {"lhs": S * E, "rhs": 2 * S ** 2 + sympy.Rational(1, 2) * E}}
    relations = df.construct_reduced_fit_list(["S E", "S^2", "E"], eqs, sympy_format=True)
    assert len(relations) == 1
    assert set(relations[0]) == {S * E, S ** 2, E}


def test_construct_reduced_fit_list_feature_name_format():
    S, E = sympy.symbols("S E")
    eqs = {"S E": {"lhs": S * E, "rhs": 2 * S ** 2 + sympy.Rational(1, 2) * E}}
    relations = df.construct_reduced_fit_list(["S E", "S^2", "E"], eqs, sympy_format=False)
    assert set(relations[0]) == {"S E", "S^2", "E"}
