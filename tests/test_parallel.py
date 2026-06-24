"""
Regression tests for AlgModelFinder's parallel fitting (``fit(parallelize=True)``).

Background
----------
``fit_and_score`` used to fit the *shared* ``self.model`` estimator. With joblib's
``require='sharedmem'`` (threading backend) every parallel task mutated that one
estimator concurrently, so one task's ``score()`` validated its data against another
task's ``feature_names_in_`` -> nondeterministic ``ValueError: feature names should
match`` (or silently scrambled coefficients). The fix clones the estimator per task
(``clone(self.model, safe=False)``) so each task is independent.

The core oracle in every test below is **parallel == serial**: for the same finder
configuration and data, ``fit(parallelize=True)`` must produce results numerically
identical to ``fit(parallelize=False)`` and must not raise. Because the underlying
estimators (LR/Lasso/Ridge with fixed params) are deterministic, the two paths must
agree bit-for-bit (compared at atol=1e-9). The race is intermittent, so several tests
repeat the parallel fit many times.

The reliable source-level guards against a regression of the fix (reverting to a shared
``self.model``) are the *repeated* parity tests -- ``test_parallel_race_repeated_wide``,
``test_feature_to_library_map_restriction_parity`` -- and ``test_model_unmutated_after_
parallel_fit``; mutation testing shows these fail every time the fix is reverted.
``test_regression_guard_old_shared_model_breaks_parity`` is a separate check that the
parity *oracle* is non-vacuous (see its docstring).
"""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn import linear_model

from daeFinder import dae_finder as df

NUM_CPU = 8        # threads for the standard parallel fits
WIDE_CPU = 16      # heavier concurrency for the dedicated race test
RACE_REPS = 12     # repetitions for intermittent-race detection (old code broke ~100%)


# --------------------------------------------------------------------------- data
def make_relational_data(seed=0, n=200):
    """Independent base columns a,b,c,d plus exact linear-combination targets.

    Every column is a linear combination of the others (rank 4 in a 7-d space), so
    each fitted model attains R^2 = 1 -- a strong correctness oracle on top of parity.
    """
    rng = np.random.RandomState(seed)
    a, b, c, d = (rng.normal(size=n) for _ in range(4))
    return pd.DataFrame({
        "a": a, "b": b, "c": c, "d": d,
        "e": 2 * a - 3 * b,
        "f": a + 0.5 * c,
        "g": -b + 2 * d,
    })


def make_wide_data(seed=1, n=240, n_base=12):
    """A wide library: n_base independent columns + n_base exact-combo targets."""
    rng = np.random.RandomState(seed)
    base = {f"u{i}": rng.normal(size=n) for i in range(n_base)}
    data = dict(base)
    for i in range(n_base):
        # each target depends on two distinct base columns -> uniquely in the span
        data[f"t{i}"] = (i + 1) * base[f"u{i}"] - (i + 2) * base[f"u{(i + 1) % n_base}"]
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- helpers
def make_finder(model_id="LR", fit_intercept=False, alpha=0.001, **kw):
    if model_id == "LR":
        return df.AlgModelFinder(model_id="LR", fit_intercept=fit_intercept, **kw)
    return df.AlgModelFinder(model_id=model_id, alpha=alpha, fit_intercept=fit_intercept, **kw)


def assert_finders_equal(parallel_finder, serial_finder, atol=1e-9):
    """Assert two fitted AlgModelFinders agree on every public output."""
    # Guard against a vacuous pass: if both finders fitted zero features the loops
    # below would assert nothing. Every parity test here fits >= 3 features.
    assert serial_finder.get_fitted_models(), "oracle ran on an empty model set"
    for scale in (True, False):
        mp = parallel_finder.get_fitted_models(scale_coef=scale)
        ms = serial_finder.get_fitted_models(scale_coef=scale)
        assert set(mp) == set(ms), "fitted feature sets differ"
        for feat in ms:
            assert set(mp[feat]) == set(ms[feat]), f"library terms differ for {feat}"
            for term in ms[feat]:
                assert abs(mp[feat][term] - ms[feat][term]) <= atol, (feat, term)
        ip = parallel_finder.get_fitted_intercepts(scale_coef=scale)
        is_ = serial_finder.get_fitted_intercepts(scale_coef=scale)
        assert set(ip) == set(is_)
        for feat in is_:
            assert abs(ip[feat] - is_[feat]) <= atol, feat
    assert set(parallel_finder.r2_score_dict) == set(serial_finder.r2_score_dict)
    for feat in serial_finder.r2_score_dict:
        assert abs(parallel_finder.r2_score_dict[feat]
                   - serial_finder.r2_score_dict[feat]) <= atol
    pd.testing.assert_frame_equal(parallel_finder.best_models(),
                                  serial_finder.best_models(), atol=atol, rtol=0)


def finders_match(parallel_finder, serial_finder, atol=1e-9):
    try:
        assert_finders_equal(parallel_finder, serial_finder, atol=atol)
        return True
    except AssertionError:
        return False


# --------------------------------------------------------------------------- core parity
@pytest.mark.parametrize("model_id", ["LR", "lasso", "RR"])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("scale_columns", [False, True])
def test_parallel_equals_serial_matrix(model_id, fit_intercept, scale_columns):
    data = make_relational_data()
    par = make_finder(model_id, fit_intercept)
    ser = make_finder(model_id, fit_intercept)
    par.fit(data, scale_columns=scale_columns, parallelize=True, num_cpu=NUM_CPU)
    ser.fit(data, scale_columns=scale_columns, parallelize=False)
    assert_finders_equal(par, ser)


def test_parallel_recovers_exact_relations():
    # correctness oracle: LR on exact-combo data attains R^2 = 1 for every target.
    data = make_relational_data()
    m = make_finder("LR", fit_intercept=False)
    m.fit(data, parallelize=True, num_cpu=NUM_CPU)
    for feat, r2 in m.r2_score_dict.items():
        assert r2 == pytest.approx(1.0, abs=1e-9), feat


# --------------------------------------------------------------------------- race detection
def test_parallel_race_repeated_wide():
    data = make_wide_data()
    ser = make_finder("LR", fit_intercept=False)
    ser.fit(data, parallelize=False)
    for rep in range(RACE_REPS):
        par = make_finder("LR", fit_intercept=False)
        par.fit(data, parallelize=True, num_cpu=WIDE_CPU)
        assert_finders_equal(par, ser)
        # each fitted model's coefficients must reference exactly its own library terms
        for feat, coefs in par.get_fitted_models().items():
            assert set(coefs) == set(data.columns.drop(feat)), (rep, feat)


# --------------------------------------------------------------------------- non-vacuous guard
def _old_buggy_fit_and_score(self, feature_, X_scaled_, feature_to_library_map_):
    """The original implementation: fits the SHARED self.model (the race)."""
    possible_library_terms = feature_to_library_map_[feature_]
    X_features = X_scaled_[possible_library_terms]
    y_target = X_scaled_[feature_]
    self.model.fit(X=X_features, y=y_target)
    coefficients = dict(zip(self.model.feature_names_in_, self.model.coef_))
    intercept = self.model.intercept_
    score = self.model.score(X_features, y_target)
    return coefficients, intercept, score


def test_regression_guard_old_shared_model_breaks_parity(monkeypatch):
    """Proves the parity oracle is NON-VACUOUS: it injects the original shared-model
    ``fit_and_score`` and shows that under it the parallel fit breaks parity (raises or
    diverges from serial). NOTE: because it monkeypatches its own buggy implementation,
    this test is independent of the real source -- it does NOT by itself guard against a
    revert of the clone fix. The reliable source-level guards are the repeated parity
    tests and test_model_unmutated_after_parallel_fit (which fail when the fix is
    reverted). This test guarantees those parity assertions can actually detect the bug."""
    data = make_wide_data()
    ser = make_finder("LR", fit_intercept=False)
    ser.fit(data, parallelize=False)

    monkeypatch.setattr(df.AlgModelFinder, "fit_and_score", _old_buggy_fit_and_score)
    broke = False
    for _ in range(RACE_REPS):
        try:
            par = make_finder("LR", fit_intercept=False)
            par.fit(data, parallelize=True, num_cpu=WIDE_CPU)
            if not finders_match(par, ser):
                broke = True
                break
        except Exception:
            broke = True
            break
    assert broke, "old shared-self.model code should break parallel/serial parity"


# --------------------------------------------------------------------------- no state leak
@pytest.mark.parametrize("custom", [False, True])
def test_model_unmutated_after_parallel_fit(custom):
    data = make_relational_data()
    if custom:
        finder = df.AlgModelFinder(custom_model=True,
                                   custom_model_ob=linear_model.LinearRegression(fit_intercept=False))
    else:
        finder = make_finder("LR", fit_intercept=False)
    finder.fit(data, parallelize=True, num_cpu=NUM_CPU)
    # the shared template estimator must never be fitted by any task (clone isolates it)
    assert not hasattr(finder.model, "coef_")
    assert not hasattr(finder.model, "feature_names_in_")


# --------------------------------------------------------------------------- custom models
class _MinimalEstimator:
    """A non-sklearn estimator (no get_params/set_params) -> exercises the
    clone(safe=False) deep-copy fallback. Delegates to LinearRegression internally."""

    def __init__(self):
        self._lr = linear_model.LinearRegression(fit_intercept=False)

    def fit(self, X, y):
        self._lr.fit(X, y)
        self.coef_ = self._lr.coef_
        self.intercept_ = self._lr.intercept_
        self.feature_names_in_ = self._lr.feature_names_in_
        return self

    def score(self, X, y):
        return self._lr.score(X, y)


@pytest.mark.parametrize("custom_ob_factory", [
    lambda: linear_model.LinearRegression(fit_intercept=False),  # clone() path
    _MinimalEstimator,                                           # deep-copy fallback path
])
def test_custom_model_parallel_equals_serial(custom_ob_factory):
    data = make_relational_data()
    par = df.AlgModelFinder(custom_model=True, custom_model_ob=custom_ob_factory())
    ser = df.AlgModelFinder(custom_model=True, custom_model_ob=custom_ob_factory())
    par.fit(data, parallelize=True, num_cpu=WIDE_CPU)
    ser.fit(data, parallelize=False)
    assert_finders_equal(par, ser)


# --------------------------------------------------------------------------- library map
def test_feature_to_library_map_restriction_parity():
    data = make_relational_data()
    # heterogeneous map: each target restricts to a distinct, differently-ordered subset;
    # 'g' is omitted and falls back to the full library.
    lib_map = {
        "e": ["b", "a", "c"],      # ordering differs from column order
        "f": ["c", "a"],
        # 'g' omitted on purpose -> default (all other columns)
    }
    features = ["g", "e", "f"]     # shuffled order stresses joblib zip alignment
    ser = make_finder("LR", fit_intercept=False)
    ser.fit(data, features_to_fit=features, feature_to_library_map_=lib_map,
            parallelize=False)
    # repeat: the race here is subtle (few features), so repetition makes this a
    # reliable catcher rather than a best-effort single shot.
    for _ in range(RACE_REPS):
        par = make_finder("LR", fit_intercept=False)
        par.fit(data, features_to_fit=features, feature_to_library_map_=lib_map,
                parallelize=True, num_cpu=WIDE_CPU)
        assert_finders_equal(par, ser)
        # the restriction is honored: e and f only see their mapped terms
        assert set(par.get_fitted_models()["e"]) == {"a", "b", "c"}
        assert set(par.get_fitted_models()["f"]) == {"a", "c"}


# --------------------------------------------------------------------------- downstream parity
@pytest.mark.parametrize("scale_columns", [False, True])
def test_downstream_methods_parity(scale_columns):
    data = make_relational_data()
    par = make_finder("LR", fit_intercept=False)
    ser = make_finder("LR", fit_intercept=False)
    par.fit(data, scale_columns=scale_columns, parallelize=True, num_cpu=NUM_CPU)
    ser.fit(data, scale_columns=scale_columns, parallelize=False)

    # best_models for both metrics
    pd.testing.assert_frame_equal(par.best_models(metric="r2"),
                                  ser.best_models(metric="r2"), atol=1e-9, rtol=0)
    pd.testing.assert_frame_equal(par.best_models(num=0, X_test=data, metric="mse"),
                                  ser.best_models(num=0, X_test=data, metric="mse"),
                                  atol=1e-9, rtol=0)
    # predict_features (raw and scaled-coef reconstruction)
    feats = ["e", "f", "g"]
    for scale_coef in (True, False):
        pd.testing.assert_frame_equal(
            par.predict_features(data, feats, scale_coef=scale_coef),
            ser.predict_features(data, feats, scale_coef=scale_coef),
            atol=1e-9, rtol=0)


# --------------------------------------------------------------------------- features_to_fit input types
def test_features_to_fit_accepts_list_index_ndarray():
    # The guard is `if features_to_fit is None:` so list / Index / ndarray of the same
    # names all behave identically (a pandas Index previously raised "ambiguous truth").
    data = make_relational_data()
    names = ["e", "f", "g"]
    results = []
    for ftf in (list(names), pd.Index(names), np.array(names)):
        m = make_finder("LR", fit_intercept=False)
        m.fit(data, features_to_fit=ftf, parallelize=True, num_cpu=NUM_CPU)
        results.append(m)
    base = results[0]
    for other in results[1:]:
        assert_finders_equal(other, base)


def test_features_to_fit_empty_list_fits_nothing():
    # `[]` is no longer silently treated as "fit everything"; it fits nothing.
    data = make_relational_data()
    m = make_finder("LR", fit_intercept=False)
    m.fit(data, features_to_fit=[], parallelize=False)
    assert m.get_fitted_models() == {}
