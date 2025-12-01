import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Physical constants ---
q = 1
k_B = 8.617e-5     # eV/K
T = 300            # K

# --- Parameters ---
N_t = 1e12         # Trap density (cm^-3)
n_init = 1e12      # Initial electron density
p_init = 1e12      # Initial hole density
n_t_init = 0       # Initially no filled traps

c_n = 1e-8         # Electron capture coefficient
c_p = 1e-8         # Hole capture coefficient

E_C = 1.12
E_V = 0.0
E_t = 0.56
n_i = 1.5e10
N_C = 2.8e19
N_V = 1.04e19

# --- Emission rates from detailed balance ---
n1 = N_C * np.exp(-(E_C - E_t) / (k_B * T))
p1 = N_V * np.exp(-(E_t - E_V) / (k_B * T))
e_n = c_n * n1
e_p = c_p * p1

# --- Scaling factors ---
n_scale = 1e12
p_scale = 1e12
nt_scale = N_t
t_scale = 1 / (c_n * N_t)  # characteristic recombination time

eps_n = e_n * t_scale / n_scale
eps_p = e_p * t_scale / p_scale

# --- Non-dimensional ODE system ---
def srh_system_nd(t, y):
    n, p, n_t = y
    p_t = 1.0 - n_t  # since total trap density is scaled to 1
    denom = n + p + eps_n + eps_p
    f_t = (n + eps_p) / denom if denom > 0 else 0.0

    dn_dt = -n * p_t + eps_n * n_t
    dp_dt = -p * n_t + eps_p * p_t
    dn_t_dt = n * p_t - eps_n * n_t - p * n_t + eps_p * p_t
    return [dn_dt, dp_dt, dn_t_dt]

# --- Non-dimensional initial conditions ---
n0_nd = n_init / n_scale
p0_nd = p_init / p_scale
nt0_nd = n_t_init / nt_scale
y0_nd = [n0_nd, p0_nd, nt0_nd]

# --- Time setup in non-dimensional units ---
t_span_nd = (0, 1e-3 / t_scale)
t_eval_nd = np.linspace(*t_span_nd, 1000)

# --- Solve in dimensionless space ---
sol = solve_ivp(srh_system_nd, t_span_nd, y0_nd, method='RK45', t_eval=t_eval_nd, max_step=1e-2)

# --- Rescale back to physical units ---
t = sol.t
n = sol.y[0] 
p = sol.y[1] 
n_t = sol.y[2] 
p_t = N_t - n_t
R_srh = c_p * p * n_t - e_p * p_t

# --- Plotting ---
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

ax[0].plot(t * 1e3, n, label='n (electrons)')
ax[0].plot(t * 1e3, p, label='p (holes)')
ax[0].plot(t * 1e3, n_t, label='n_t (trapped electrons)')
ax[0].set_ylabel('Concentration (cm$^{-3}$)')
ax[0].set_title('Carrier and Trap Dynamics (Non-dimensionalized Solver)')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(t * 1e3, R_srh, label='R$_{SRH}$', color='darkred')
ax[1].set_xlabel('Time (ms)')
ax[1].set_ylabel('R$_{SRH}$ (cm$^{-3}$/s)')
ax[1].set_title('SRH Recombination Rate')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()
