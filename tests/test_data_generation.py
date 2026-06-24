"""
Regression tests for the data-generation layer: the toy enzyme-kinetics ODE
right-hand sides and their integrators, plus the plotting helpers.

The strongest checks here are physical conservation laws, which must hold
regardless of solver/library versions:
  * enzyme is conserved:    E + ES = E0
  * substrate is conserved: S + ES + P = S0  (full model)
                            S + P  = S0       (reduced Michaelis-Menten model)
"""
import numpy as np
import pandas as pd
import pytest

from daeFinder import dae_finder as df


# --------------------------------------------------------------------------- RHS
def test_toyEnzRHS_conserves_enzyme_and_substrate(k_rates):
    # At an arbitrary state the instantaneous rates must conserve enzyme and substrate.
    state = [7.3, 0.6, 0.4, 1.1]  # S, E, ES, P
    dS, dE, dES, dP = df.toyEnzRHS(state, 0.0, k_rates)
    assert dE + dES == pytest.approx(0.0, abs=1e-12)          # d/dt (E + ES) = 0
    assert dS + dES + dP == pytest.approx(0.0, abs=1e-12)     # d/dt (S + ES + P) = 0


def test_toyEnzRHS_known_values(k_rates):
    # Hand-computed from the rate laws: k=1, kr=0.5, kcat=0.3, state S=2,E=1,ES=0,P=0
    dS, dE, dES, dP = df.toyEnzRHS([2.0, 1.0, 0.0, 0.0], 0.0, k_rates)
    # dS = kr*ES - k*E*S = 0 - 1*1*2 = -2 ; dES = k*E*S - (kr+kcat)*ES = 2
    assert dS == pytest.approx(-2.0)
    assert dE == pytest.approx(-2.0)         # (kr+kcat)*ES - k*S*E = -2
    assert dES == pytest.approx(2.0)
    assert dP == pytest.approx(0.0)          # kcat*ES = 0

    # Second state with ES != 0 pins the kcat-dependent terms (dP = kcat*ES).
    dS2, dE2, dES2, dP2 = df.toyEnzRHS([2.0, 1.0, 3.0, 0.0], 0.0, k_rates)
    assert dP2 == pytest.approx(0.3 * 3.0)             # kcat*ES = 0.9
    assert dE2 == pytest.approx(0.8 * 3.0 - 1 * 2 * 1)  # (kr+kcat)*ES - k*S*E = 0.4
    assert dS2 == pytest.approx(0.5 * 3.0 - 1 * 1 * 2)  # kr*ES - k*E*S = -0.5


def test_toyMM_RHS_substrate_to_product(k_rates, initial_conditions):
    # In the reduced model substrate converts to product: dS/dt = -dP/dt.
    dS, dP = df.toyMM_RHS([5.0, 1.0], 0.0, k_rates, initial_conditions)
    assert dS == pytest.approx(-dP)
    assert dP > 0  # product is being produced


# --------------------------------------------------------------------------- solvers
def test_solveToyEnz_shape_ic_and_conservation(time_grid, k_rates, initial_conditions):
    sol = df.solveToyEnz(initial_conditions, k_rates, time_grid, "ts")
    assert sol.shape == (len(time_grid), 4)
    np.testing.assert_allclose(sol[0], [10.0, 1.0, 0.0, 0.0], atol=1e-9)
    S, E, ES, P = sol.T
    np.testing.assert_allclose(E + ES, 1.0, atol=1e-4)        # enzyme conserved (E0=1)
    np.testing.assert_allclose(S + ES + P, 10.0, atol=1e-3)   # substrate conserved (S0=10)
    assert np.all(np.diff(P) >= -1e-9)                        # product monotonically increases


def test_solveMM_shape_and_conservation(time_grid, k_rates, initial_conditions):
    sol = df.solveMM(initial_conditions, k_rates, time_grid, "ts")
    assert sol.shape == (len(time_grid), 4)
    S, E, ES, P = sol.T
    np.testing.assert_allclose(S + P, 10.0, atol=1e-3)        # S + P conserved
    np.testing.assert_allclose(E + ES, 1.0, atol=1e-9)        # algebraic enzyme balance
    # ES follows the quasi-steady-state expression ES = k*E0*S/(kr+kcat+k*S)
    k, kr, kcat = k_rates["k"], k_rates["kr"], k_rates["kcat"]
    expected_ES = k * 1.0 * S / (kr + kcat + k * S)
    np.testing.assert_allclose(ES, expected_ES, atol=1e-9)


def test_solveMM_matches_full_model_late_time(time_grid, k_rates, initial_conditions):
    # The reduced model should track the full model's product reasonably at late time.
    full = df.solveToyEnz(initial_conditions, k_rates, time_grid, "ts")
    reduced = df.solveMM(initial_conditions, k_rates, time_grid, "ts")
    assert reduced[-1, 3] == pytest.approx(full[-1, 3], rel=0.15)


# --------------------------------------------------------------------------- plotting
@pytest.mark.parametrize("plotter", ["plotToyEnz", "plotToy_MM"])
def test_plotters_run_headless(plotter, time_grid, k_rates, initial_conditions):
    import matplotlib.pyplot as plt
    sol = df.solveToyEnz(initial_conditions, k_rates, time_grid, "ts")
    getattr(df, plotter)(time_grid, sol, title="smoke")  # Agg backend -> show() is a no-op
    plt.close("all")
