import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
q = 1  # C
k_b = 8.617e-5  # eV/K
T = 300  # Kelvin
n_i = 1e10  # Intrinsic carrier concentration (cm^-3)

# Material Parameters
B_p = 1e-8  # cm^3/s
B_n = 1e-8  # cm^3/s
N_t = 1e15   # trap density (cm^-3)

E_t = 0.56  # eV (trap level)
E_f = 0.51  # eV (Fermi level)

# Lifetimes
tau_n = 1 / (B_n * N_t)
tau_p = 1 / (B_p * N_t)

print("tau_n =", tau_n)
print("tau_p =", tau_p)

# Time-varying voltage (square pulse as example)
def V_time(t):
    return 0.8 if 1e-7 <= t <= 3e-7 else 0.6  # 0.2V pulse between 100ns and 300ns

# Fermi occupancy (assumed fixed)
def fermi_trap_occupancy(E_t, E_f, T):
    return 1 / (1 + np.exp((E_t - E_f) / (k_b * T)))

# SRH DAE system with time-varying V(t)
def dae_system(t, y, T):
    n, p, n_t, p_t = y
    Vt = V_time(t)
    f_t = fermi_trap_occupancy(E_t, E_f, T)

    # Recombination terms
    R_n = (1 / tau_n) * (n_t * f_t - n * (1 - f_t))
    R_p = (1 / tau_p) * (p_t * (1 - f_t) - p * f_t)

    # Carrier ODEs (decay from recombination)
    dn_dt = -R_n
    dp_dt = -R_p

    # Trap state dynamics (gain what carriers lose)
    dn_t_dt = -dn_dt
    dp_t_dt = -dp_dt

    return [dn_dt, dp_dt, dn_t_dt, dp_t_dt]

# Initial carrier densities from equilibrium at V(0)
V0 = V_time(0)
n0 = n_i * np.exp(q * V0 / (k_b * T))
p0 = n_i * np.exp(-q * V0 / (k_b * T))

# Initial trap occupancy: assume n_t = 0.5 * N_t, enforce p_t = N_t - n_t
nt0 = 0.5 * N_t
pt0 = N_t - nt0
y0 = [n0, p0, nt0, pt0]

# Solve the system
sol = solve_ivp(
    fun=lambda t, y: dae_system(t, y, T),
    t_span=(0, 1e-6),
    y0=y0,
    method='Radau',
    atol=1e-6,
    rtol=1e-6
)

# Unpack results
time = sol.t
n, p, n_t, p_t = sol.y
delta_n = n - n_i

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time, n, label="n (electrons)", linewidth=2)
plt.plot(time, p, label="p (holes)", linewidth=2)
plt.plot(time, n_t, label="n_t (trapped electrons)", linewidth=2)
plt.plot(time, p_t, label="p_t (trapped holes)", linewidth=2)
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Concentration (cm⁻³)", fontsize=14)
plt.title("Time Evolution of SRH Dynamics with V(t)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Δn
plt.figure(figsize=(8, 5))
plt.plot(time, delta_n, '--', label="Δn (excess carriers)", color='purple', linewidth=2)
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Excess Carrier Density (cm⁻³)", fontsize=14)
plt.title("Excess Carrier Concentration Δn(t)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot applied voltage
V_vals = np.array([V_time(ti) for ti in time])
plt.figure(figsize=(8, 4))
plt.plot(time, V_vals, label="V(t)", color='black')
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Applied Voltage (V)", fontsize=14)
plt.title("Time-Varying Applied Voltage", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()
