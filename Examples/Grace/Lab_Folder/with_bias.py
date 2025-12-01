import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Physical constants ---
q = 1
k_B = 8.617e-5     # eV/K
T = 300            # K

# --- Parameters ---
N_t = 1e12         # Trap density (cm^-3)
n_init = 1e11
p_init = 1e10

n_t_init = 0

c_n = 1e-8
c_p = 1e-8

E_C = 1.12
E_V = 0.0
E_t = 0.56
n_i = 1.5e10
N_C = 2.8e19
N_V = 1.04e19

n1 = N_C * np.exp(-(E_C - E_t) / (k_B * T))
p1 = N_V * np.exp(-(E_t - E_V) / (k_B * T))
e_n = c_n * n1
e_p = c_p * p1

# --- Updated ODE system with dynamic f_t ---
def srh_system(t, y):
    n, p, n_t = y
    p_t = N_t - n_t
    denom = c_n * n + c_p * p + e_n + e_p
    f_t = (c_n * n + e_p) / denom if denom > 0 else 0.0

    dn_dt = -c_n * n * p_t + e_n * n_t
    dp_dt = -c_p * p * n_t + e_p * p_t
    dn_t_dt = c_n * n * p_t - e_n * n_t - c_p * p * n_t + e_p * p_t
    return [dn_dt, dp_dt, dn_t_dt]

# --- Solve ---
t_span = (0, 1e-2)
y0 = [n_init, p_init, n_t_init]
t_eval = np.linspace(*t_span, 1000)

sol = solve_ivp(srh_system, t_span, y0, method='RK45', t_eval=t_eval, max_step=1e-6)
n, p, n_t = sol.y
p_t = N_t - n_t
R_srh = c_p * p * n_t - e_p * p_t

# --- Plot ---
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

ax[0].plot(sol.t * 1e3, n, label='n (electrons)')
ax[0].plot(sol.t * 1e3, p, label='p (holes)')
ax[0].plot(sol.t * 1e3, n_t, label='n_t (trapped electrons)')
ax[0].set_ylabel('Concentration (cm$^{-3}$)')
ax[0].set_title('Carrier and Trap Dynamics with Non-Equilibrium f$_t$')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(sol.t * 1e3, R_srh, label='R$_{SRH}$', color='darkred')
ax[1].set_xlabel('Time (ms)')
ax[1].set_ylabel('R$_{SRH}$ (cm$^{-3}$/s)')
ax[1].set_title('SRH Recombination Rate')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()
