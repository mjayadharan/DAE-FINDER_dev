import numpy as np
from scipy.integrate import solve_ivp

# Physical constant
k_B = 8.617e-5  # eV/K

def compute_emission_rates(E_C, E_V, E_t, T, c_n, c_p, N_C, N_V):
    """Compute emission rates based on detailed balance."""
    n1 = N_C * np.exp(-(E_C - E_t) / (k_B * T))
    p1 = N_V * np.exp(-(E_t - E_V) / (k_B * T))
    e_n = c_n * n1
    e_p = c_p * p1
    return e_n, e_p

def srh_system_nd(t, y, params):
    """Non-dimensional SRH ODE system."""
    n, p, n_t = y
    eps_n, eps_p = params["eps_n"], params["eps_p"]
    p_t = 1.0 - n_t  # Total trap occupancy = 1

    denom = n + p + eps_n + eps_p
    f_t = (n + eps_p) / denom if denom > 0 else 0.0

    dn_dt = -n * p_t + eps_n * n_t
    dp_dt = -p * n_t + eps_p * p_t
    dn_t_dt = n * p_t - eps_n * n_t - p * n_t + eps_p * p_t
    return [dn_dt, dp_dt, dn_t_dt]

def solve_nd_srh_traps(IC, param_overrides=None, t_span=(0, 1), t_eval=None, atol=1e-8, rtol=1e-6):
    """
    Solve the non-dimensional SRH trap model.
    
    Parameters:
        IC (dict): Initial conditions with keys 'n', 'p', 'n_t' [in physical units].
        param_overrides (dict): Optional parameters to override defaults.
        t_span (tuple): Start and end time in seconds.
        t_eval (array): Time points (in seconds) at which to return the solution.
        atol, rtol: Solver tolerances.
    
    Returns:
        Tuple: (t, n, p, n_t, p_t) in physical units.
    """
    # --- Default parameters ---
    defaults = {
        "T": 300,         # Kelvin
        "c_n": 1e-8,      # cm^3/s
        "c_p": 1e-8,      # cm^3/s
        "N_t": 1e12,      # cm^-3
        "E_C": 1.12,      # eV
        "E_V": 0.0,       # eV
        "E_t": 0.56,      # eV
        "N_C": 2.8e19,    # cm^-3
        "N_V": 1.04e19,   # cm^-3
        "n_scale": 1e12,
        "p_scale": 1e12,
    }

    # Merge with user-provided parameters
    if param_overrides:
        defaults.update(param_overrides)
    params = defaults

    # Time scaling
    t_scale = 1 / (params["c_n"] * params["N_t"])

    # Emission rates and scaled parameters
    e_n, e_p = compute_emission_rates(
        params["E_C"], params["E_V"], params["E_t"],
        params["T"], params["c_n"], params["c_p"],
        params["N_C"], params["N_V"]
    )
    eps_n = e_n * t_scale / params["n_scale"]
    eps_p = e_p * t_scale / params["p_scale"]
    ode_params = {"eps_n": eps_n, "eps_p": eps_p}

    # Initial conditions (non-dimensional)
    y0 = [
        IC["n"] / params["n_scale"],
        IC["p"] / params["p_scale"],
        IC["n_t"] / params["N_t"]
    ]

    # Time handling
    t_span_nd = (t_span[0] / t_scale, t_span[1] / t_scale)
    t_eval_nd = np.array(t_eval) / t_scale if t_eval is not None else None

    # Solve the system
    sol = solve_ivp(
        fun=lambda t, y: srh_system_nd(t, y, ode_params),
        t_span=t_span_nd,
        y0=y0,
        t_eval=t_eval_nd,
        method="RK45",
        rtol=rtol,
        atol=atol
    )

    # Rescale outputs to physical units
    t_out = sol.t * t_scale
    n = sol.y[0] * params["n_scale"]
    p = sol.y[1] * params["p_scale"]
    n_t = sol.y[2] * params["N_t"]
    p_t = params["N_t"] - n_t

    return t_out, n, p, n_t, p_t
