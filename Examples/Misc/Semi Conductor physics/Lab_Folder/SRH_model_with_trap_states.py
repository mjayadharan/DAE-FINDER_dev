import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

q = 1 # C
k_b = 8.617e-5 # eV/K
T = 300 # Kelvin
V = 0.7 # Volts

B_p = 1e-8 # cm^3/s
B_n = 1e-8 # cm^3/s
N_t = 1e9 # cm^-3

E_t = 0.56 # eV (trap level)
E_f = 0.51 # eV (Fermi level)
#means more likely to capture electrons (more likely to be empty) when E_t > E_f

tau_n = 1 / (B_n * N_t)
tau_p = 1 / (B_p * N_t)
print(tau_n)
print(tau_p)

def fermi_trap_occupancy(E_t, E_f, T):
    return 1 / (1 + np.exp((E_t - E_f) / (k_b * T)))

def dae_system(t, y, V, T):
    n, p, n_t, p_t = y
    f_t = fermi_trap_occupancy(E_t, E_f, T)
    beta = np.exp(q * V / (k_b * T))
    beta

    # ODEs
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

# y0 = [1.0, 0.5, 0.2, 0.3]  # [n, p, n_t, p_t]

# y0 = [1e12, 1e5, 2e11,1e11]  # n, p, n_t
n0 = 1e12    # high electron concentration (cm⁻³)
p0 = 1e5     # low hole concentration (cm⁻³)
nt0 = 5e3   # trap occupancy (cm⁻³)
exp_term = np.exp(1 * 0.2 / (8.617e-5 * 300))
pt0 = (n0 * p0) / (nt0 * exp_term)


y0 = [n0, p0, nt0, pt0]

# y0 = [1e10, 1e10, 1e8, 1e8]


sol = solve_ivp(
    fun=lambda t, y: dae_system(t, y, V, T),
    t_span=(0, 1),
    y0=y0,
    method='Radau',
    atol=1e-6,
    rtol=1e-6
)

time = sol.t
n, p, n_t, p_t = sol.y

n_i = 1e10
delta_n = n - n_i

ic_text = (
    f"Initial condition: n₀ = {y0[0]:.2e}, "
    f"p₀ = {y0[1]:.2e}, "
    f"nₜ₀ = {y0[2]:.2e}, "
    f"pₜ₀ = {y0[3]:.2e}"
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
