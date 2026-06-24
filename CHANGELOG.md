# Changelog

## [1.0.1] - 2026-06-24

First stable (1.0) release. It brings DaeFinder up to date with the modern
scientific-Python stack — **Python 3.9–3.14, NumPy 2, pandas 3, scikit-learn 1.9** —
fixes the parallel fitting feature, and adds a comprehensive test suite and CI.
(Supersedes the never-published `v0.3.0` development version; everything below is new
relative to the last PyPI release, `0.2.1`.)

### Fixed
- **Parallel fitting (`AlgModelFinder.fit(parallelize=True)`) data race.** `fit_and_score`
  fitted the shared `self.model` estimator, so with joblib's `require='sharedmem'`
  (threading backend) parallel tasks mutated one estimator concurrently — raising
  nondeterministic `ValueError: feature names should match` (or silently scrambling
  coefficients). Each task now fits its own `clone(self.model, safe=False)`, so parallel
  results are numerically identical to serial. Custom models are supported via the
  deep-copy fallback of `clone(safe=False)`.
- `AlgModelFinder.fit` now guards `features_to_fit` with `is None` instead of truthiness,
  so passing a pandas `Index`/NumPy array (e.g. `df.columns`) no longer raises "ambiguous
  truth value" and an empty list fits nothing instead of everything.
- **Python 3.13+ compatibility (PEP 667).** `get_refined_lib` and `get_simplified_equation`
  created SymPy symbols with one `exec()` call and read them back in a later
  `exec()`/`eval()`; since Python 3.13 names no longer leak into function locals, which
  raised `NameError`. These now use an explicit namespace dictionary, fixing Python
  3.13/3.14 while remaining identical on older versions.
- **pandas 3.0 compatibility.** `get_simplified_equation` mutated `Series.values` in
  place; under pandas 3.0 copy-on-write that array is read-only
  (`ValueError: assignment destination is read-only`). It now operates on a writable copy
  (`to_numpy(copy=True)`) that preserves the coefficient dtype.
- **scikit-learn ≥ 1.7 compatibility.** `FeatureCouplingTransformer.transform` called the
  removed `BaseEstimator._validate_data` method; it now uses a shim that prefers
  `sklearn.utils.validation.validate_data` and falls back to the legacy method.
- `get_simplified_equation` now skips `NaN` coefficients (present in the union-indexed
  frames returned by `AlgModelFinder.best_models`) instead of emitting `"nan"` into the
  symbolic expression.
- `smooth_data` raises `ValueError` instead of a bare string for unsupported methods.
- `sequentialThLin.__init__` stored several constructor arguments as 1-tuples due to
  stray trailing commas; they are now stored as the scalar values passed in.

### Added
- Comprehensive `pytest` test suite under `tests/` (data generation, preprocessing,
  library construction, model discovery, equations, parallel fitting, and
  cross-version compatibility), with a runner (`tests/run_tests.sh`) and docs.
- GitHub Actions CI running the suite across Python 3.9–3.14.
- `docs/py314_compatibility_report.pdf`: full compatibility audit and release plan.

### Changed
- Minimum supported Python raised to **3.9**; trove classifiers updated for 3.9–3.14.
- README refreshed with a quick start, compatibility matrix, and an updated overview.

## [v0.2.1] - 2025-01-08
### Added
- Updated readme.md to provide more information

## [v0.2.0] - 2025-01-08
### Added
- New release testing
- Updated readme.md to provide more information

### Fixed
- NIL

## [0.1.0] - 2025-01-08
### Added
- Initial release
