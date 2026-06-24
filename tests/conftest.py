"""
Shared pytest configuration and fixtures for the daeFinder test suite.

This file is auto-discovered by pytest. It (1) makes the in-repo ``daeFinder``
package importable even when it has not been pip-installed, and (2) provides the
deterministic data fixtures used across the regression tests.
"""
import os
import sys

# --- make the in-repo package importable without an install -----------------
# Insert the repository root (the parent of this tests/ directory) at the front
# of sys.path so `import daeFinder` resolves to the working-tree package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# --- headless matplotlib (daeFinder imports pyplot at module load) ----------
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from daeFinder import dae_finder as df


# --- canonical reaction parameters ------------------------------------------
K_RATES = {"k": 1.0, "kr": 0.5, "kcat": 0.3}
INITIAL_CONDITIONS = {"S": 10.0, "E": 1.0, "ES": 0.0, "P": 0.0}


@pytest.fixture
def time_grid():
    """Evenly spaced time grid used by the ODE integrators."""
    return np.linspace(0, 5, 80)


@pytest.fixture
def k_rates():
    return dict(K_RATES)


@pytest.fixture
def initial_conditions():
    return dict(INITIAL_CONDITIONS)


@pytest.fixture
def enz_data(time_grid):
    """Full enzyme-kinetics solution (S, E, ES, P) from the toy ODE system."""
    sol = df.solveToyEnz(dict(INITIAL_CONDITIONS), dict(K_RATES), time_grid, "ts0")
    return pd.DataFrame(sol, columns=["S", "E", "ES", "P"])


@pytest.fixture
def mm_data(time_grid):
    """Reduced Michaelis-Menten solution (S, E, ES, P). Deterministic."""
    sol = df.solveMM(dict(INITIAL_CONDITIONS), dict(K_RATES), time_grid, "ts0")
    return pd.DataFrame(sol, columns=["S", "E", "ES", "P"])


@pytest.fixture
def poly_lib(mm_data):
    """Degree-2 polynomial library (with bias) built from mm_data."""
    pf = df.PolyFeatureMatrix(degree=2, include_bias=True)
    return pf.fit_transform(mm_data)


@pytest.fixture
def linear_relation_data():
    """Data with an exact algebraic relation z = 2*x + 3*y for recovery tests."""
    rng = np.random.RandomState(0)
    x = rng.rand(80)
    y = rng.rand(80)
    return pd.DataFrame({"x": x, "y": y, "z": 2 * x + 3 * y})
