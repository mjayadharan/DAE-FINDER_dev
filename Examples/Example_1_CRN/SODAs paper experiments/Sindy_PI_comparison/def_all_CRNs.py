import numpy as np
from scipy.integrate import solve_ivp

###################################################
###################### CRN 1 ######################
###################################################

def make_CRN1(n_conds, n_points, t_final=4.):
    S_0_min, S_0_max = 10.0, 20.0
    S_0s = np.linspace(S_0_min, S_0_max, n_conds, endpoint=True)

    clean_data_lst = [make_one_CRN1(S_0, n_points, t_final) for S_0 in S_0s]
    return np.vstack(clean_data_lst)

def make_one_CRN1(S_0, n_points, t_final=4.):
    # Parameters for first reaction (A + E1 <-> AE1 -> B + E1)
    k_on1 = 1.0   # Rate constant for binding (E1 and A)
    k_off1 = 0.5  # Rate constant for unbinding (E1 and A)
    k_cat1 = 5.0 # Catalytic rate constant (AE1 -> B + E1)
    E1_tot = 2.0 # Total concentration of enzyme E1

    # Michaelis constants
    K_M1 = (k_off1 + k_cat1) / k_on1

    # Stoichiometric matrix for reduced system
    S = np.array([
        [-1],  # Change in A
        [ 1],  # Change in B
    ])

    # Reaction rates as a function of slow variables
    def reaction_rates(x):
        A, B = x  # Concentrations of A, B, C
        v1 = (k_cat1 * E1_tot * A) / (K_M1 + A)
        return np.array([v1])

    # Quasi-steady-state approximations for fast species
    def fast_species(A, B):
        # Solve for AE1 and BE2 under QSSA
        AE1 = (E1_tot * A) / (K_M1 + A)
        E1 = E1_tot - AE1
        return E1, AE1

    # Reduced ODE system: dx/dt = S @ v(x)
    def reduced_crn_ode(t, x):
        v = reaction_rates(x)
        dx_dt = S @ v  # Matrix-vector multiplication
        return dx_dt

    # Time span for simulation
    t_span = (0, t_final)  # Simulate from t=0 to t=50
    t_eval = np.linspace(*t_span, n_points)  # Points to evaluate

    # Solve the ODEs
    P_0 = 0
    x0 = np.array([S_0, P_0])
    solution = solve_ivp(reduced_crn_ode, t_span, x0, t_eval=t_eval)

    # Extract results
    t = solution.t
    A, B = solution.y  # Concentrations of A, B, C
    E1, AE1 = np.array([fast_species(Ai, Bi) for Ai, Bi in zip(A, B)]).T

    data = np.array([A, B, E1, AE1]).T
    
    return np.concatenate((t[:, np.newaxis], data), axis=1)

###################################################
###################### CRN 2 ######################
###################################################

def make_CRN2(n_conds, n_points):
    x0_min, x0_max = [20.0, 0.0, 0.0], [40.0, 0.0, 0.0]
    x0s = np.linspace(x0_min, x0_max, n_conds, endpoint=True)

    clean_data_lst =  [make_one_CRN2(x0, n_points) for x0 in x0s]
    return np.vstack(clean_data_lst)

def make_one_CRN2(x0, n_points):
    # Parameters for first reaction (A + E1 <-> AE1 -> B + E1)
    k_on1 = 1.0   # Rate constant for binding (E1 and A)
    k_off1 = 0.5  # Rate constant for unbinding (E1 and A)
    k_cat1 = 5.0 # Catalytic rate constant (AE1 -> B + E1)
    E1_tot = 2.0 # Total concentration of enzyme E1

    # Parameters for second reaction (A + E2 <-> AE2 -> C + E2)
    k_on2 = 2.0   # Rate constant for binding (E2 and A)
    k_off2 = 0.25  # Rate constant for unbinding (E2 and A)
    k_cat2 = 3.0  # Catalytic rate constant (AE2 -> C + E2)
    E2_tot = 4.0 # Total concentration of enzyme E2

    # Michaelis constants
    K_M1 = (k_off1 + k_cat1) / k_on1
    K_M2 = (k_off2 + k_cat2) / k_on2

    # Stoichiometric matrix for reduced system
    S = np.array([
        [-1, -1],  # Change in A
        [ 1,  0],  # Change in B
        [ 0,  1]   # Change in C
    ])

    # Reaction rates as a function of slow variables
    def reaction_rates(x):
        A, B, C = x  # Concentrations of A, B, C
        v1 = (k_cat1 * E1_tot * A) / (K_M1 + A)
        v2 = (k_cat2 * E2_tot * A) / (K_M2 + A)
        return np.array([v1, v2])

    # Quasi-steady-state approximations for fast species
    def fast_species(A, B):
        # Solve for AE1 and BE2 under QSSA
        AE1 = (E1_tot * A) / (K_M1 + A)
        E1 = E1_tot - AE1
        AE2 = (E2_tot * A) / (K_M2 + A)
        E2 = E2_tot - AE2
        return E1, AE1, E2, AE2

    # Reduced ODE system: dx/dt = S @ v(x)
    def reduced_crn_ode(t, x):
        v = reaction_rates(x)
        dx_dt = S @ v  # Matrix-vector multiplication
        return dx_dt

    # Time span for simulation
    t_span = (0, 4)  # Simulate from t=0 to t=50
    t_eval = np.linspace(*t_span, n_points)  # Points to evaluate

    # Solve the ODEs
    solution = solve_ivp(reduced_crn_ode, t_span, x0, t_eval=t_eval)

    # Extract results
    t = solution.t
    A, B, C = solution.y  # Concentrations of A, B, C
    E1, AE1, E2, AE2 = np.array([fast_species(Ai, Bi) for Ai, Bi in zip(A, B)]).T

    data = np.array([A, B, C, E1, AE1, E2, AE2]).T
    
    return np.concatenate((t[:, np.newaxis], data), axis=1)

###################################################
###################### CRN 3 ######################
###################################################

def make_CRN3(n_conds, n_points):
    x0_min, x0_max = [20.0, 0.0, 0.0, 0.0, 0.0], [70.0, 0.0, 0.0, 0.0, 0.0]
    x0s = np.linspace(x0_min, x0_max, n_conds, endpoint=True)

    clean_data_lst =  [make_one_CRN3(x0, n_points) for x0 in x0s]
    return np.vstack(clean_data_lst)

def make_one_CRN3(x0, n_points):
    # PduCDE
    Km_1 = 0.5
    k_cat1 = 300.0  # Catalytic rate constant (AE1 -> B + E1)
    E1_tot = 0.462  # Total concentration of enzyme E1

    # PduQ
    Km_for2 = 15.0
    Km_rev2 = 95.0
    k_cat2 = 55.0  # Catalytic rate constant (BE2 -> C + E2)
    k_rev2 = 6.0 # Rate constant for reverse reaction (C + E2 <-> BE2)
    E2_tot = 0.52  # Total concentration of enzyme E2

    # PduP
    Km_3 = 15.0
    k_cat3 = 55.0  # Catalytic rate constant (BE3 -> D + E3)
    E3_tot = 0.694  # Total concentration of enzyme E3

    # PduLW
    Km_4 = 20.0
    k_cat4 = 100.0  # Catalytic rate constant (BE3 -> D + E3)
    E4_tot = 0.1  # Total concentration of enzyme E3

    # Effective rates for v1, v2, and v3
    def v1(A):
        return (k_cat1 * E1_tot * A) / (Km_1 + A)

    def v2(B, C):
        Vmax_2 = k_cat2*E2_tot
        numer = Vmax_2 * B
        denom = Km_for2 + B + (Km_for2 / Km_rev2) * C
        return numer / denom

    def v3(B, C):
        Vmax_2r = k_rev2*E2_tot
        numer = Vmax_2r * C
        denom = Km_rev2 + C + (Km_rev2 / Km_for2) * B
        return numer / denom

    def v4(B):
        return (k_cat3 * E3_tot * B) / (Km_3 + B)
    
    def v5(D):
        return (k_cat4 * E4_tot * D) / (Km_4 + D)

    # Stoichiometric matrix for reduced system
    S = np.array([
        [-1,  0,  0,  0,  0],  # Change in A
        [ 1, -1,  1, -1,  0],  # Change in B
        [ 0,  1, -1,  0,  0],  # Change in C
        [ 0,  0,  0,  1, -1],  # Change in D
        [ 0,  0,  0,  0,  1]   # Change in F
    ])

    # Reaction rates as a function of slow variables
    def reaction_rates(x):
        A, B, C, D, F = x  # Concentrations of A, B, C, D
        return np.array([v1(A), v2(B, C), v3(B, C), v4(B), v5(D)])

    # Quasi-steady-state approximations for fast species
    def fast_species(A, B, C, D):
        # Solve for complexes under QSSA
        AE1 = (E1_tot * A) / (Km_1 + A)
        E1 = E1_tot - AE1
        BE2 = (E2_tot * B) / (Km_for2 + B + (k_rev2/k_cat2) * C)
        CE2 = (E2_tot * C) / (Km_rev2 + C + (k_cat2/k_rev2) * B)
        E2 = E2_tot - BE2 - CE2
        BE3 = (E3_tot * B) / (Km_3 + B)
        E3 = E3_tot - BE3
        DE4 = (E4_tot * D) / (Km_4 + D)
        E4 = E4_tot - DE4
        return E1, AE1, E2, BE2, CE2, E3, BE3, E4, DE4

    # Reduced ODE system: dx/dt = S @ v(x)
    def reduced_crn_ode(t, x):
        v = reaction_rates(x)  # Get reaction rates for A, B, C, D
        dx_dt = S @ v  # Matrix-vector multiplication for reduced system
        return dx_dt

    # Time span for simulation
    t_span = (0, 10.0)  # Simulate from t=0 to t=50
    t_eval = np.linspace(*t_span, n_points)  # Points to evaluate

    # Solve the ODEs
    solution = solve_ivp(reduced_crn_ode, t_span, x0, t_eval=t_eval, method='Radau')

    # Extract results
    t = solution.t
    A, B, C, D, F = solution.y  # Concentrations of A, B, C, D, F

    # Calculate fast species concentrations at each timepoint
    E1, AE1, E2, BE2, CE2, E3, BE3, E4, DE4 = np.array([fast_species(Ai, Bi, Ci, Di) for Ai, Bi, Ci, Di in zip(A, B,C, D)]).T
    data = np.array([A, B, C, D, F, E1, AE1, E2, BE2, CE2, E3, BE3, E4, DE4]).T
    
    return np.concatenate((t[:, np.newaxis], data), axis=1)
