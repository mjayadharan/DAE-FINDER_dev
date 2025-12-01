import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Physical constants
q = 1  # C
k_b = 8.617e-5  # eV/K

def fermi_trap_occupancy(E_t, E_f, T):
    return 1 / (1 + np.exp((E_t - E_f) / (k_b * T)))

def dae_system(t, y, params):
    n, p, n_t, p_t = y
    V, T, tau_n, tau_p, E_t, E_f = params["V"], params["T"], params["tau_n"], params["tau_p"], params["E_t"], params["E_f"]
    f_t = fermi_trap_occupancy(E_t, E_f, T)
    beta = np.exp(q * V / (k_b * T))

    dn_dt = (1 / tau_n) * n_t * f_t - (1 / tau_n) * n * (1 - f_t)
    dp_dt = (1 / tau_p) * p_t * (1 - f_t) - (1 / tau_p) * p * f_t
    dnt_dt = f_t * (p / tau_p - n_t / tau_n) + (1 - f_t) * (n / tau_n - p_t / tau_p)

    if n_t <= 1e-30:
        dpt_dt = 0.0
    else:
        dpt_dt = (1 / (beta * n_t)) * (
            (1 / tau_n * n_t * f_t - 1 / tau_n * n * (1 - f_t)) * (p + beta * p_t) +
            (1 / tau_p * p_t * (1 - f_t) - 1 / tau_p * p * f_t) * (n + beta * p_t)
        )

    return [dn_dt, dp_dt, dnt_dt, dpt_dt]

def solve_srh_traps(init_cond, params=None, t_span=(0, 1), atol=1e-6, rtol=1e-6):
    # Default physical and model parameters
    defaults = {
        "V": 0.7,         # Volts
        "T": 300,         # Kelvin
        "B_n": 1e-8,      # cm^3/s
        "B_p": 1e-8,      # cm^3/s
        "N_t": 1e9,       # cm^-3
        "E_t": 0.56,      # eV
        "E_f": 0.51       # eV
    }

    # Allow user to override
    if params is not None:
        defaults.update(params)
    params = defaults

    # Compute derived quantities
    params["tau_n"] = 1 / (params["B_n"] * params["N_t"])
    params["tau_p"] = 1 / (params["B_p"] * params["N_t"])

    y0 = [init_cond["n"], init_cond["p"], init_cond["n_t"], init_cond["p_t"]]

    sol = solve_ivp(
        fun=lambda t, y: dae_system(t, y, params),
        t_span=t_span,
        y0=y0,
        method='Radau',
        atol=atol,
        rtol=rtol
    )
    return sol


def plot_srh_solution(sol, init_cond):
    time = sol.t
    n, p, n_t, p_t = sol.y
    n_i = 1e10
    delta_n = n - n_i

    ic_text = (
        f"Initial condition: n₀ = {init_cond['n']:.2e}, "
        f"p₀ = {init_cond['p']:.2e}, "
        f"nₜ₀ = {init_cond['n_t']:.2e}, "
        f"pₜ₀ = {init_cond['p_t']:.2e}"
    )

    plt.figure(figsize=(10, 6))
    plt.plot(time, n, label="n (electrons)", linewidth=2)
    plt.plot(time, p, label="p (holes)", linewidth=2)
    plt.plot(time, n_t, label="n_t (trapped electrons)", linewidth=2)
    plt.plot(time, p_t, label="p_t (trapped holes)", linewidth=2)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Concentration (cm⁻³)", fontsize=14)
    plt.title("Time Evolution of SRH Trap Dynamics", fontsize=16)
    plt.suptitle(ic_text, fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(time, delta_n, '--', label="Δn (excess carriers)", color='purple', linewidth=2)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Excess Carrier Density (cm⁻³)", fontsize=14)
    plt.title("Excess Carrier Concentration Δn(t)", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# # Example use
# if __name__ == "__main__":
#     B_n = B_p = 1e-8  # cm^3/s
#     N_t = 1e9  # cm^-3

#     tau_n = 1 / (B_n * N_t)
#     tau_p = 1 / (B_p * N_t)

#     init_cond = {"n": 1e10, "p": 1e10, "n_t": 1e8, "p_t": 1e8}
#     params = {
#         "V": 0.7,  # Volts
#         "T": 300,  # Kelvin
#         "tau_n": tau_n,
#         "tau_p": tau_p,
#         "E_t": 0.56,  # eV
#         "E_f": 0.51   # eV
#     }

#     sol = solve_srh_traps(init_cond, params, t_span=(0, 1))
#     plot_srh_solution(sol, init_cond)
