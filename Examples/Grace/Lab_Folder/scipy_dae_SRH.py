import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae

def F(t, y, yp):
    """Residual function for the DAE system."""
    n, p, nt, pt = y
    dn, dp, dnt, dpt = yp  # dpt unused since pt is algebraic

    # Constants
    q = 1       # C
    k_b = 8.617e-5  # eV/K
    T = 300         # K
    V = 0.7         # Volts

    B_p = 1e-8      # cm^3/s
    B_n = 1e-8      # cm^3/s
    N_t = 1e9       # cm^-3

    E_t = 0.56      # eV
    E_f = 0.51      # eV

    tau_n = 1 / (B_n * N_t)
    tau_p = 1 / (B_p * N_t)

    # Trap occupancy (constant)
    ft = 1 / (1 + np.exp((E_t - E_f) / (k_b * T)))
    exp_term = np.exp(q * V / (k_b * T))

    # Residuals
    F = np.zeros(4, dtype=np.common_type(y, yp))
    F[0] = dn - (1 / tau_n) * (nt * ft - n * (1 - ft))
    F[1] = dp - (1 / tau_p) * (pt * (1 - ft) - p * ft)
    F[2] = dnt + dn + dp
    F[3] = n * p - nt * pt * exp_term  # algebraic constraint
    return F

# Initial conditions
n0 = 1e10
p0 = 1e10
nt0 = 2e5


exp_term = np.exp(1 * 0.7 / (8.617e-5 * 300))
pt0 = (n0 * p0) / (nt0 * exp_term)
# Use trap occupancy again
ft = 1 / (1 + np.exp((0.56 - 0.51) / (8.617e-5 * 300)))
tau_n = 1 / (1e-8 * 1e9)
tau_p = 1 / (1e-8 * 1e9)

dn0 = (1 / tau_n) * (nt0 * ft - n0 * (1 - ft))
dp0 = (1 / tau_p) * (pt0 * (1 - ft) - p0 * ft)
dnt0 = -dn0 - dp0  # from the continuity equation
dpt0 = 0.0  # algebraic

yp0 = [dn0, dp0, dnt0, dpt0]

y0 = np.array([n0, p0, nt0, pt0], dtype=float)

# Time grid
t_span = (0.0, 1.0)
t_eval = np.linspace(*t_span, 500)

# Solver parameters
atol = rtol = 1e-6
method = "Radau"  # or "BDF"

# Solve
sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval)

# Extract results
t = sol.t
y = sol.y
print("Solver success:", sol.success)
print("Message:", sol.message)

n_vals, p_vals, nt_vals, pt_vals = y

print(p_vals - pt_vals)
# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, n_vals, label='n(t)')
ax.plot(t, p_vals, label='p(t)')
ax.plot(t, nt_vals, label='n_t(t)')
ax.plot(t, pt_vals, label='p_t(t)')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Concentration (cm⁻³)")
ax.set_title("DAE System: Carrier and Trap Concentrations")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
