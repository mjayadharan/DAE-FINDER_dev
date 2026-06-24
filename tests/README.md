# daeFinder test suite

Regression and compatibility tests for the `daeFinder` package, written with
[pytest](https://docs.pytest.org/). The suite exercises every public function and
class with deterministic data and asserts **verified expected values** (numbers,
symbolic forms, conservation laws), not merely "it didn't crash".

## Quick start

From the repository root:

```bash
# 1. install the package (editable) + test deps, once
python -m pip install -e .
python -m pip install -r tests/requirements-test.txt

# 2. run the whole suite
pytest
```

`pytest` with no arguments runs everything under `tests/` (configured in
`pytest.ini`). You do **not** need to install the package for the imports to
resolve — `tests/conftest.py` puts the repo root on `sys.path` — but installing it
is recommended so you test what users get.

## Run everything and save a report

Use the helper script to run the full suite and capture a timestamped report:

```bash
./tests/run_tests.sh                       # default python3
PYTHON=python3.11 ./tests/run_tests.sh     # choose the interpreter
./tests/run_tests.sh -k coupling           # forward any args to pytest
./tests/run_tests.sh tests/test_library.py # run a single file
```

Each run writes to `tests/reports/` (git-ignored):

- `test_run_<timestamp>.txt` — full human-readable log + PASS/FAIL summary
- `test_run_<timestamp>.xml` — JUnit XML (for CI dashboards / tooling)

If `pytest-cov` is installed, a coverage summary (`--cov=daeFinder`) is added
automatically.

## Running subsets

```bash
pytest tests/test_model_discovery.py            # one file
pytest tests/test_model_discovery.py::test_stols_thresholds_by_magnitude  # one test
pytest -k "smooth or noise"                     # by keyword
pytest -x                                       # stop at first failure
pytest -q                                        # quiet
```

## Running across Python versions

The package must work from Python 3.9 through the latest release. The compatibility
tests in `test_py314_compat.py` are version-aware (some equivalence checks only run
on Python < 3.13; see below). To test another interpreter:

```bash
PYTHON=python3.14 ./tests/run_tests.sh
```

CI runs the suite on Python 3.9–3.14 (`.github/workflows/tests.yml`).

## Layout

| File | Covers |
|------|--------|
| `conftest.py` | Shared fixtures (`mm_data`, `enz_data`, `poly_lib`, `linear_relation_data`, reaction params) and the `sys.path` / headless-matplotlib bootstrap. |
| `test_data_generation.py` | ODE right-hand sides and integrators (`toyEnzRHS`, `solveToyEnz`, `toyMM_RHS`, `solveMM`) + conservation laws; plotting smoke tests. |
| `test_preprocessing.py` | Name helpers, `der_matrix_calculator`, `smooth_data` (spline + Savitzky–Golay), `add_noise_to_df`, `remove_paranth_from_feat`, `poly_to_scipy`. |
| `test_library.py` | `PolyFeatureMatrix`, `FeatureCouplingTransformer`, `get_factor_feat`, `get_refined_lib` / `get_refined_lib_stable`. |
| `test_model_discovery.py` | `AlgModelFinder` (recovery, scaling, predict, compare, library map, intercepts), `sequentialThLin`, `stols`, `compare_models_`. |
| `test_equations.py` | `get_simplified_equation`(`_list`), `sympy_symb_to_feature_name`, `construct_reduced_fit_list`. |
| `test_py314_compat.py` | Python 3.13+/NumPy 2/pandas 3/scikit-learn 1.9 compatibility, with equivalence tests proving the fixed code matches the original. |

## Adding new tests after changing the code

1. **Pick the right file** by feature area (table above), or add a new
   `test_<area>.py`. pytest auto-discovers files named `test_*.py` and functions
   named `test_*`.

2. **Reuse fixtures** from `conftest.py` by adding the fixture name as a function
   argument, e.g.:

   ```python
   def test_my_feature(poly_lib):
       result = some_function(poly_lib)
       assert ...
   ```

   Add a new fixture to `conftest.py` if several tests need the same data. Keep
   fixtures **deterministic** (fixed seeds / fixed parameters) so failures are
   reproducible.

3. **Assert verified expected values, not just "no exception".** Make the test
   *truthful*:
   - Compute the expected result independently (by hand, from a closed-form, or
     from a conservation/invariance law) and assert against it.
   - Use `pytest.approx(...)` or `numpy.testing.assert_allclose(...)` with an
     explicit, justified tolerance for floating-point comparisons.
   - For symbolic results, compare with `sympy.expand(a - b) == 0` or
     `sympy.srepr(a) == sympy.srepr(b)` (the latter also pins Integer-vs-Float).
   - Prefer strong oracles: exact algebraic relations (e.g. `z = 2x + 3y`),
     conservation laws, or known derivatives (e.g. `d/dt sin = cos`).

4. **Seed randomness.** Use `np.random.RandomState(seed)` for any random data so
   the test is reproducible. (Note: `add_noise_to_df(random_seed=0)` does *not*
   seed because `0` is falsy — use a non-zero seed when you need determinism.)

5. **Handle version-specific behavior with markers.** If a behavior only holds on
   certain Python/library versions, guard it:

   ```python
   import sys, pytest
   @pytest.mark.skipif(sys.version_info >= (3, 13), reason="...")
   def test_legacy_only(): ...
   ```

   When fixing a bug, add a regression test that **fails on the old code and passes
   on the new** — verify by temporarily reverting your fix and confirming the test
   goes red.

6. **Run before committing:**

   ```bash
   ./tests/run_tests.sh                    # latest interpreter
   PYTHON=python3.11 ./tests/run_tests.sh  # an older one too
   ```

7. **Update this README** if you add a new test file or a new shared fixture.

## Dependencies

Runtime deps come from the package itself (`pip install -e .`). Test-only deps are
in `tests/requirements-test.txt` (`pytest`, optional `pytest-cov`).
