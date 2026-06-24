"""
Regression + equivalence tests for the Python 3.13+/NumPy 2/pandas 3/scikit-learn 1.9
compatibility fixes in daeFinder.dae_finder.

Two complementary guarantees:
  * "works"      : the fixed implementation runs on the *current* interpreter
                   (the whole point of the fix -- exercises the exec/CoW/sklearn paths).
  * "equivalence": the fixed implementation returns results *identical* to the
                   original implementation. The original relies on pre-PEP-667
                   exec() semantics, so it can only run on Python < 3.13; those
                   tests are skipped on newer interpreters (where the "works"
                   tests carry the load instead).

Run:  pytest tests/test_py314_compat.py -v
"""
import sys
import numpy as np
import pandas as pd
import sympy
import pytest

from daeFinder import dae_finder as df

PRE_PEP667 = sys.version_info < (3, 13)
skip_old = pytest.mark.skipif(
    not PRE_PEP667,
    reason="original exec()-based implementation cannot run on Python >= 3.13 (PEP 667)",
)


# --------------------------------------------------------------------------------------
# Verbatim copies of the ORIGINAL (pre-fix) implementations, used only as a golden
# reference for the equivalence tests. They use the legacy bare-exec pattern and only
# run on Python < 3.13.
# --------------------------------------------------------------------------------------
def old_get_refined_lib(factor_exp, data_matrix_df_, candidate_library_, get_dropped_feat=False):
    feat_list = list(data_matrix_df_.columns)
    feat_list_str = ", ".join(df.remove_paranth_from_feat(data_matrix_df_.columns))
    exec(feat_list_str + "= sympy.symbols(" + str(feat_list) + ")")
    candid_features = df.remove_paranth_from_feat(df.poly_to_scipy(candidate_library_.columns))
    candid_feat_dict = {}
    for feat1, feat2 in zip(candidate_library_.columns, candid_features):
        exec("candid_feat_dict['{}'] = {}".format(feat1, feat2))
    dropped_feats = set()
    if isinstance(factor_exp, list) or isinstance(factor_exp, set):
        for factor_ in factor_exp:
            dropped_feats = dropped_feats.union(set(df.get_factor_feat(factor_, candid_feat_dict)))
    else:
        dropped_feats = dropped_feats.union(set(df.get_factor_feat(factor_exp, candid_feat_dict)))
    if get_dropped_feat:
        return (dropped_feats, candidate_library_.drop(dropped_feats, axis=1))
    else:
        return candidate_library_.drop(dropped_feats, axis=1)


def old_get_simplified_equation(best_model_df, feature, global_feature_list, coef_threshold,
                                intercept_threshold=0.01, intercept=0, simplified=True):
    global_feature_list = list(global_feature_list)
    global_feature_list_string = ", ".join(df.remove_paranth_from_feat(global_feature_list))
    exec(global_feature_list_string + "= sympy.symbols(" + str(global_feature_list) + ")")
    model_lhs = feature
    model_lhs_sp_string = df.remove_paranth_from_feat(df.poly_to_scipy([model_lhs]))[0]
    intercept = 0 if abs(intercept) < intercept_threshold else intercept
    model_coefs = best_model_df[model_lhs].values  # verbatim from main (mutates input in place)
    model_coefs[abs(model_coefs) < coef_threshold] = 0
    model_rhs_features = df.remove_paranth_from_feat(df.poly_to_scipy(best_model_df[model_lhs].keys()))
    rhs_string_sp_string = [str(coef) + "*" + feat for coef, feat in zip(model_coefs, model_rhs_features)]
    rhs_string_sp_string = "+".join(rhs_string_sp_string) + "+" + str(intercept)
    result_dict = {}
    exec("result_dict['lhs'] = {}".format(model_lhs_sp_string))
    exec("result_dict['rhs'] = {}".format(rhs_string_sp_string))
    if not simplified:
        return result_dict
    n, d = sympy.fraction(sympy.cancel(result_dict['rhs'] / result_dict['lhs']))
    result_dict['lhs'] = d
    result_dict['rhs'] = n
    return result_dict


# --------------------------------------------------------------------------------------
# Fixtures
# (mm_data and poly_lib are provided by tests/conftest.py)
# --------------------------------------------------------------------------------------
@pytest.fixture
def best_model_df():
    """Clean best-model frame (no NaN) with non-trivial coefficients to threshold."""
    idx = ["S^2", "E", "S E", "S"]
    return pd.DataFrame({"S E": [2.0, 0.5, 0.0, 0.001]}, index=idx)


# --------------------------------------------------------------------------------------
# get_refined_lib  (Fix I1: exec into explicit namespace)
# --------------------------------------------------------------------------------------
def test_get_refined_lib_drops_expected_terms(mm_data, poly_lib):
    """exec-based path must run and drop S*E-containing terms on every interpreter."""
    factor = sympy.symbols("S") * sympy.symbols("E")
    dropped, refined = df.get_refined_lib(factor, mm_data, poly_lib, get_dropped_feat=True)
    assert dropped, "expected at least one dropped feature"
    # every dropped column must literally contain both S and E
    for col in dropped:
        assert "S" in col and "E" in col
    assert set(refined.columns) == set(poly_lib.columns) - set(dropped)


@skip_old
def test_get_refined_lib_equivalence(mm_data, poly_lib):
    for factor in [sympy.symbols("S") * sympy.symbols("E"),
                   sympy.symbols("S") ** 2,
                   [sympy.symbols("S") * sympy.symbols("E"), sympy.symbols("E") * sympy.symbols("ES")]]:
        old_drop, old_ref = old_get_refined_lib(factor, mm_data, poly_lib, get_dropped_feat=True)
        new_drop, new_ref = df.get_refined_lib(factor, mm_data, poly_lib, get_dropped_feat=True)
        assert old_drop == new_drop
        assert list(old_ref.columns) == list(new_ref.columns)
        pd.testing.assert_frame_equal(old_ref, new_ref)


def test_get_refined_lib_does_not_mutate_input(mm_data, poly_lib):
    cols_before = list(poly_lib.columns)
    df.get_refined_lib(sympy.symbols("S") * sympy.symbols("E"), mm_data, poly_lib)
    assert list(poly_lib.columns) == cols_before


def _bracket_fixture():
    """Bracketed feature names ([S], [E], ...) -- this is the actual exec() hazard the
    fix targets, and the path get_refined_lib_stable delegates back to get_refined_lib for."""
    rng = np.random.RandomState(0)
    data = pd.DataFrame(rng.rand(20, 3), columns=["[S]", "[E]", "[P]"])
    lib = df.PolyFeatureMatrix(degree=2, include_bias=True).fit_transform(data)
    factor = sympy.Symbol("[S]") * sympy.Symbol("[E]")
    return data, lib, factor


def test_get_refined_lib_bracketed_names_works():
    data, lib, factor = _bracket_fixture()
    dropped, refined = df.get_refined_lib(factor, data, lib, get_dropped_feat=True)
    assert dropped == {"[S] [E]"}
    assert "[S] [E]" not in refined.columns


@skip_old
def test_get_refined_lib_bracketed_names_equivalence():
    data, lib, factor = _bracket_fixture()
    old_drop, old_ref = old_get_refined_lib(factor, data, lib, get_dropped_feat=True)
    new_drop, new_ref = df.get_refined_lib(factor, data, lib, get_dropped_feat=True)
    assert old_drop == new_drop
    pd.testing.assert_frame_equal(old_ref, new_ref)


# --------------------------------------------------------------------------------------
# get_simplified_equation  (Fix I1 exec + Fix I2 read-only .values)
# --------------------------------------------------------------------------------------
def test_get_simplified_equation_runs(best_model_df):
    S, E = sympy.symbols("S E")
    # simplified=False exposes the thresholded RHS directly (no fraction cancellation).
    res = df.get_simplified_equation(best_model_df, "S E",
                                     global_feature_list=["S", "E", "ES", "P"],
                                     coef_threshold=0.1, simplified=False)
    assert set(res.keys()) == {"lhs", "rhs"}
    # lhs is the target feature 'S E' -> S*E
    assert sympy.expand(res["lhs"] - S * E) == 0
    # coefs 0.0 ('S E') and 0.001 ('S') are below 0.1 -> eliminated;
    # 2.0*S**2 and 0.5*E survive.
    assert sympy.expand(res["rhs"] - (2.0 * S ** 2 + 0.5 * E)) == 0


def test_get_simplified_equation_does_not_mutate_input(best_model_df):
    before = best_model_df["S E"].tolist()
    df.get_simplified_equation(best_model_df, "S E", ["S", "E", "ES", "P"], coef_threshold=0.1)
    assert best_model_df["S E"].tolist() == before


@skip_old
def test_get_simplified_equation_equivalence(best_model_df):
    # Each call gets its own fresh copy: the verbatim old impl mutates its input in
    # place, so a shared frame would let one call pollute the next (and make the result
    # order-dependent). Fresh copies keep the comparison honest and order-independent.
    for thr in (1.0, 0.1, 0.0):
        old = old_get_simplified_equation(best_model_df.copy(), "S E", ["S", "E", "ES", "P"], coef_threshold=thr)
        new = df.get_simplified_equation(best_model_df.copy(), "S E", ["S", "E", "ES", "P"], coef_threshold=thr)
        assert sympy.simplify(old["lhs"] - new["lhs"]) == 0
        assert sympy.simplify(old["rhs"] - new["rhs"]) == 0


def test_get_simplified_equation_list_runs(best_model_df):
    res = df.get_simplified_equation_list(best_model_df, ["S", "E", "ES", "P"], coef_threshold=0.1)
    assert "S E" in res


def test_get_simplified_equation_preserves_integer_coef_dtype():
    """Integer-dtype coefficient columns must keep producing integer symbolic terms
    (regression guard: forcing dtype=float would emit 2.0*S instead of 2*S)."""
    S, P = sympy.symbols("S P")
    bm = pd.DataFrame({"S E": [2, 0, 4, 0]}, index=["S^2", "E", "S P", "S"])
    assert bm["S E"].dtype == np.int64
    res = df.get_simplified_equation(bm, "S E", ["S", "E", "ES", "P"],
                                     coef_threshold=0.5, simplified=False)
    assert sympy.expand(res["rhs"] - (2 * S ** 2 + 4 * S * P)) == 0
    assert "Float" not in sympy.srepr(res["rhs"])  # coefficients stay sympy Integers


@skip_old
def test_get_simplified_equation_integer_coef_equivalence():
    bm = pd.DataFrame({"S E": [2, 0, 4, 0]}, index=["S^2", "E", "S P", "S"])
    old = old_get_simplified_equation(bm.copy(), "S E", ["S", "E", "ES", "P"],
                                      coef_threshold=0.5, simplified=False)
    new = df.get_simplified_equation(bm.copy(), "S E", ["S", "E", "ES", "P"],
                                     coef_threshold=0.5, simplified=False)
    # srepr equality is stricter than simplify()==0: it also pins Integer-vs-Float.
    assert sympy.srepr(old["rhs"]) == sympy.srepr(new["rhs"])


def test_get_simplified_equation_handles_nan_coefficients():
    """Union-indexed best-model frames (the real best_models() output) carry NaN for
    features absent from a given model. The NaN must not leak into the expression."""
    S, E = sympy.symbols("S E")
    bm = pd.DataFrame({"S E": [2.0, 0.5, np.nan]}, index=["S^2", "E", "P"])
    res = df.get_simplified_equation(bm, "S E", ["S", "E", "ES", "P"],
                                     coef_threshold=0.1, simplified=False)
    # NaN coefficient on 'P' is dropped; result is the same as if 'P' were absent.
    assert sympy.expand(res["rhs"] - (2.0 * S ** 2 + 0.5 * E)) == 0


# --------------------------------------------------------------------------------------
# FeatureCouplingTransformer  (Fix I3: sklearn validate_data shim)
# --------------------------------------------------------------------------------------
def test_feature_coupling_transformer_transform():
    from scipy.sparse import coo_array
    data = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], columns=["t", "x", "y"])
    sm = coo_array((np.array([4, 5, 7, 5]),
                    (np.array([0, 0, 1, 1]), np.array([0, 2, 2, 1]))))
    ct = df.FeatureCouplingTransformer(sm)
    out = ct.fit_transform(data)
    assert out.shape == (2, 4)               # 4 coupled indices
    names = ct.get_feature_names_out()
    assert list(names) == ["t*t", "t*y", "x*y", "x*x"]
    # numeric correctness of one coupling (t*y on row 0 = 1*3 = 3)
    assert out[0, 1] == pytest.approx(3.0)


# --------------------------------------------------------------------------------------
# Minor fixes
# --------------------------------------------------------------------------------------
def test_smooth_data_bad_method_raises_valueerror(mm_data):
    data = mm_data.copy()
    data.insert(0, "t", np.linspace(0, 5, len(data)))
    with pytest.raises(ValueError):
        df.smooth_data(data, smooth_method="not-a-method")


def test_sequentialthlin_params_are_scalars():
    # The trailing-comma bug previously stored these constructor args as 1-tuples,
    # e.g. self.alpha == (0.3,). They must round-trip as the scalar values passed in.
    model = df.sequentialThLin(model_id="RR", alpha=0.3, l1_ratio=0.4,
                               tol=1e-3, selection="random", silent=True)
    assert model.alpha == 0.3 and not isinstance(model.alpha, tuple)
    assert model.l1_ratio == 0.4 and not isinstance(model.l1_ratio, tuple)
    assert model.tol == 1e-3 and not isinstance(model.tol, tuple)
    assert model.fit_intercept is False and not isinstance(model.fit_intercept, tuple)
    assert model.selection == "random" and not isinstance(model.selection, tuple)


def test_sequentialthlin_fit_and_score(poly_lib):
    y = poly_lib["S E"]
    model = df.sequentialThLin(model_id="RR", alpha=0.1, coef_threshold=0.05, silent=True)
    model.fit(poly_lib, y=y)
    score = model.score(poly_lib, y)
    assert np.isfinite(score)


# --------------------------------------------------------------------------------------
# End-to-end smoke (the full discovery pipeline touched by the fixes)
# --------------------------------------------------------------------------------------
def test_full_pipeline_smoke(mm_data, poly_lib):
    amf = df.AlgModelFinder(model_id="lasso", alpha=0.001)
    amf.fit(poly_lib, scale_columns=True)
    best = amf.best_models(num=2)
    assert best.shape[1] == 2
    preds = amf.predict_features(poly_lib, list(poly_lib.columns)[:3])
    assert preds.shape[0] == poly_lib.shape[0]
    # Simplify equations directly from the (NaN-bearing) best_models frame -- this is
    # the realistic call path and must not raise on the latest stack. global_feature_list
    # is the set of base state variables (monomials are expressed in these symbols).
    best_no_metric = best.drop("r2- metric", errors="ignore")
    eqs = df.get_simplified_equation_list(best_no_metric,
                                          global_feature_list=list(mm_data.columns),
                                          coef_threshold=0.0)
    assert len(eqs) == best_no_metric.shape[1]
