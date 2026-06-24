"""
Regression tests for the preprocessing utilities: derivative-name helpers,
finite-difference and smoothing-based derivative estimation, and noise injection.
"""
import numpy as np
import pandas as pd
import pytest

from daeFinder import dae_finder as df


# --------------------------------------------------------------------------- names
def test_get_der_names_dict_and_list():
    assert df.get_der_names(["A", "B"]) == {"A": "d(A) /dt", "B": "d(B) /dt"}
    assert df.get_der_names(["A", "B"], get_list=True) == ["d(A) /dt", "d(B) /dt"]


def test_der_label_orders():
    assert df.der_label("x", der=0) == "x"
    assert df.der_label("x", der=1) == "d(x) /dt"
    assert df.der_label("x", der=2) == "d^2(x) /dt^2"


def test_remove_paranth_from_feat():
    assert df.remove_paranth_from_feat(["[E]", "[ES]", "S"]) == ["E", "ES", "S"]
    # strings without both brackets are returned unchanged
    assert df.remove_paranth_from_feat(["E", "x*y"]) == ["E", "x*y"]


def test_poly_to_scipy():
    assert df.poly_to_scipy(["A^2", "A B^3"]) == ["A**2", "A*B**3"]
    assert df.poly_to_scipy(["1"]) == ["1"]


# --------------------------------------------------------------------------- derivatives
def test_der_matrix_calculator_is_exact_for_linear():
    t = np.arange(0, 1, 0.1)
    data = pd.DataFrame({"f": 3.0 * t, "g": -2.0 * t})
    der = df.der_matrix_calculator(data, delta_t=0.1)
    assert list(der.columns) == ["d(f) /dt", "d(g) /dt"]
    assert len(der) == len(data) - 1                 # one row lost to differencing
    np.testing.assert_allclose(der["d(f) /dt"], 3.0)  # forward diff exact on a line
    np.testing.assert_allclose(der["d(g) /dt"], -2.0)


def test_der_matrix_calculator_rejects_tiny_delta():
    with pytest.raises(AssertionError):
        df.der_matrix_calculator(pd.DataFrame({"f": [1.0, 2.0]}), delta_t=0.0)


def test_der_matrix_calculator_no_rename():
    data = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    der = df.der_matrix_calculator(data, delta_t=1.0, rename_feat=False)
    assert list(der.columns) == ["f"]
    np.testing.assert_allclose(der["f"], 1.0)


# --------------------------------------------------------------------------- smoothing
def test_smooth_data_spline_recovers_derivative_of_sin():
    t = np.linspace(0, 2 * np.pi, 200)
    data = pd.DataFrame({"t": t, "f": np.sin(t)})
    out = df.smooth_data(data, smooth_method="spline", s_param_=0.0, derr_order=2)
    assert list(out.columns) == ["t", "f", "d(f) /dt", "d^2(f) /dt^2"]
    interior = (out["t"] > 0.5) & (out["t"] < 2 * np.pi - 0.5)
    # f' ~ cos(t), f'' ~ -sin(t)
    assert np.max(np.abs(out["d(f) /dt"][interior] - np.cos(out["t"][interior]))) < 0.05
    assert np.max(np.abs(out["d^2(f) /dt^2"][interior] + np.sin(out["t"][interior]))) < 0.05


def test_smooth_data_SG_recovers_derivative_trend():
    t = np.linspace(0, 2 * np.pi, 300)
    data = pd.DataFrame({"t": t, "f": np.sin(t)})
    out = df.smooth_data(data, smooth_method="SG", derr_order=1, polyorder=3)
    assert "d(f) /dt" in out.columns
    interior = slice(20, -20)
    deriv = out["d(f) /dt"].to_numpy()[interior]
    true = np.cos(t[interior])
    # Correlation pins the shape; the absolute bound pins the magnitude/scaling
    # (e.g. that the sample spacing delta is applied) -- a scale-only bug passes
    # the correlation check but fails this one. Measured interior error ~0.011.
    assert np.corrcoef(deriv, true)[0, 1] > 0.99
    assert np.max(np.abs(deriv - true)) < 0.03


def test_smooth_data_requires_domain_var():
    data = pd.DataFrame({"f": np.arange(5.0)})  # no 't' column
    with pytest.raises(AssertionError):
        df.smooth_data(data, domain_var="t")


def test_smooth_data_rejects_unknown_method():
    t = np.linspace(0, 1, 10)
    data = pd.DataFrame({"t": t, "f": t ** 2})
    with pytest.raises(ValueError):
        df.smooth_data(data, smooth_method="nope")


# --------------------------------------------------------------------------- noise
def test_add_noise_to_df_level_and_copy_semantics():
    rng = np.random.RandomState(1)
    base = pd.DataFrame({"a": rng.rand(5000) * 10.0})
    original = base["a"].copy()
    std0 = base["a"].std()

    noisy = df.add_noise_to_df(base, noise_perc=10, random_seed=42, make_copy=True)
    # original untouched when make_copy=True
    np.testing.assert_array_equal(base["a"], original)
    # residual std ~ 10% of feature std (statistical, generous tolerance)
    resid_std = (noisy["a"] - base["a"]).std()
    assert resid_std == pytest.approx(0.10 * std0, rel=0.15)


def test_add_noise_to_df_in_place_mutates():
    base = pd.DataFrame({"a": np.linspace(0, 10, 1000)})
    before = base["a"].copy()
    # non-zero seed: add_noise_to_df only seeds when random_seed is truthy.
    out = df.add_noise_to_df(base, noise_perc=5, random_seed=7, make_copy=False)
    assert out is base                                  # same object returned
    assert not np.array_equal(base["a"], before)        # mutated in place


def test_add_noise_zero_percent_is_noop():
    base = pd.DataFrame({"a": np.linspace(0, 1, 50)})
    out = df.add_noise_to_df(base, noise_perc=0, random_seed=7)
    np.testing.assert_array_equal(out["a"], base["a"])
