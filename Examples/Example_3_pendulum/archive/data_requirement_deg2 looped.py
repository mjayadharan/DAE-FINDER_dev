import os, sys

path_to_add = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(path_to_add)
print(os.path.join(path_to_add, "daeFinder"))
sys.path.append(os.path.join(path_to_add, "daeFinder"))


import numpy as np
from scipy.integrate import odeint
import pandas as pd
import warnings
pd.set_option('display.float_format', '{:0.8f}'.format)
import operator
import sympy
from dae_finder import construct_reduced_fit_list

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp




from matplotlib import pyplot as plt
from dae_finder import smooth_data
from dae_finder import add_noise_to_df
from sklearn import decomposition
from sklearn.linear_model import LinearRegression
from dae_finder import get_simplified_equation_list
from dae_finder import get_refined_lib, remove_paranth_from_feat
from dae_finder import AlgModelFinder
from dae_finder import sequentialThLin, AlgModelFinder
from dae_finder import PolyFeatureMatrix
from copy import deepcopy

# ---------------------------------------------------------
# Example pendulum RHS
# ---------------------------------------------------------
def pendulum_rhs(t, y, gamma, L=1):
    """ 
    Simple pendulum with damping 
    y = [theta, omega], 
    alpha = - (g/L) * sin(theta) - gamma*omega
    """
    g = 9.81
    theta, omega = y
    alpha = - (g / L) * np.sin(theta) - gamma * omega
    return [omega, alpha]

# ---------------------------------------------------------
# 1. Generate full data (2000 points) once
# ---------------------------------------------------------
def generate_data_at_max_resolution(num_time_points, params_df, IC_df, noise_perc=10, random_seed=111, L=5.0):
    """
    Simulates the pendulum ODE for multiple initial conditions (and optionally parameters)
    but here simplified to a single param set for illustration. 
    Returns:
        data_matrix_df_list_full (list of DataFrames): Each DF has the columns [t, x, y].
        t_eval_full (numpy array): The time points used (length = num_time_points).
    """
    L = 5.0

    num_time_points = 400
    # Time span
    t_span = (0.0, 10)  # from 0 to 10 seconds
    #Valuation points
    t_eval_ = np.linspace(t_span[0], t_span[1], num_time_points)
    data_matrix_df_list = []


    for param_index in params_df.index:
        params = params_df.loc[param_index]
        # Define parameters
        m_c = params['m_c']  # Mass of the cart (kg)
        m_p = params['m_p']  # Mass of the pendulum (kg)
        l = params['l']    # Length of the pendulum (m)
        for IC_index in IC_df.index:
            IC = IC_df.loc[IC_index]
            y0 = IC.values
                    # Parameters
            theta0 = IC["theta"]  # Initial angle (radians)
            omega0 = IC["omega"]        # Initial angular velocity (radians per second)
            gamma = 0.0         # Damping coefficient
            # Solve the ODEs
            # sol = solve_ivp(lambda t, y: pendulum_rhs(t, y, gamma, L), t_span, [theta0, omega0], method='RK45', t_eval=t_eval_)
            sol = solve_ivp(lambda t, y: pendulum_rhs(t, y, gamma, L), t_span, [theta0, omega0], t_eval=t_eval_, rtol=1e-8)
            
            sol_df = pd.DataFrame(sol.y.T, columns=["theta", "omega"])
            sol_df["x"] = L*np.sin(sol_df["theta"])
            sol_df["y"] = -L*np.cos(sol_df["theta"])
            sol_df["t"] = t_eval_
            data_matrix_df_list.append(sol_df[["t", "x", "y"]])


    data_matrix_df = pd.concat(data_matrix_df_list, ignore_index=True)

    return data_matrix_df_list, t_eval_

# ---------------------------------------------------------
# 2. Downsample to 'mid' points
# ---------------------------------------------------------
def downsample_data(data_matrix_df_list_full, new_num_time_points):
    """
    Given the full data (each DF has 2000 points),
    downsample to new_num_time_points points using uniform spacing.
    """
    data_matrix_df_list_downsampled = []
    for df_full in data_matrix_df_list_full:
        n_full = len(df_full)
        # Indices for downsampling
        indices = np.round(np.linspace(0, n_full - 1, new_num_time_points)).astype(int)
        df_down = df_full.iloc[indices].reset_index(drop=True)
        data_matrix_df_list_downsampled.append(df_down)
    return data_matrix_df_list_downsampled

# ---------------------------------------------------------
# 3. Check if relationships are discovered
#    (Wrap your existing relationship discovery code in a function)
# ---------------------------------------------------------
def check_relationship_discovery(data_matrix_df_list, poly_degree=2, noise_perc=10):
    """
    Returns True if the two algebraic relationships are discovered, otherwise False.
    """
    # ----------- Smooth data -----------
    # Suppose we want to produce derivatives at higher resolution
    # but we only have 'num_time_points' in each data_frame
    # We'll define the same approach as in your snippet:
    df_first = data_matrix_df_list[0]
    t_min, t_max = df_first["t"].iloc[0], df_first["t"].iloc[-1]
    num_time_points = len(df_first)

    # Possibly amplify the time steps for derivative calc
    data_amplify_fact = 1
    num_smoothed_points = num_time_points * data_amplify_fact
    t_eval_new = np.linspace(t_min, t_max, num_smoothed_points)

    data_matrix_smooth_df_list = [
        smooth_data(df_, derr_order=1, noise_perc=noise_perc, eval_points=t_eval_new) 
        for df_ in data_matrix_df_list
    ]

    # Concatenate if multiple
    if len(data_matrix_smooth_df_list) > 1:
        data_matrix_df_smooth_appended = pd.concat(data_matrix_smooth_df_list, ignore_index=True)
    else:
        data_matrix_df_smooth_appended = data_matrix_smooth_df_list[0]

    # Keep only relevant columns, drop some boundary points, etc.
    data_matrix_df_smooth = data_matrix_df_smooth_appended[["x","y","d(x) /dt","d(y) /dt"]].iloc[5:-5]
    data_matrix_df_smooth.columns = ["x","y","x_dot","y_dot"]

    # -------------- Build polynomial library --------------
    poly_feature_ob = PolyFeatureMatrix(poly_degree)
    candidate_lib_full = poly_feature_ob.fit_transform(data_matrix_df_smooth)
    if "1" in candidate_lib_full.columns:
        candidate_lib_full = candidate_lib_full.drop(["1"], axis=1)

    # -------------- True relationships for check --------------
    true_relationship_dict_2degree = {
        1: {'y^2', 'x^2'},
        2: {'y', 'x_dot^2', 'y_dot^2'}
    }
    relationship_refinement_2degree = {
        1: {'x^2','x*x_dot'},
        2: {'x_dot^2'}
    }

    # -------------- Fit first relationship --------------
    seq_th_model = sequentialThLin(fit_intercept=True, alpha=0.3, coef_threshold=0.05, silent=True)
    algebraic_model_th = AlgModelFinder(custom_model=True, custom_model_ob=seq_th_model)
    algebraic_model_th.fit(candidate_lib_full, scale_columns=False)
    best_models_full = algebraic_model_th.best_models(1)
    intercept_dictionary = algebraic_model_th.get_fitted_intercepts()

    simplified_equations = get_simplified_equation_list(
        best_model_df=best_models_full.fillna(0)[:-1],
        coef_threshold=0.05,
        intercept_threshold=0.01,
        global_feature_list=data_matrix_df_smooth.columns,
        intercept_dict=intercept_dictionary,
        simplified=True
    )
    reduced_relationship_features = construct_reduced_fit_list(
        candidate_lib_full.columns, 
        simplified_eqs=simplified_equations
    )
    best_relationship = reduced_relationship_features[0]
    if None in best_relationship:
        best_relationship.remove(None)
    best_relationship_set = set(best_relationship)

    discovered_true = False
    relationship_index = None
    for k, v in true_relationship_dict_2degree.items():
        if best_relationship_set == v:
            discovered_true = True
            relationship_index = k
            break

    if not discovered_true:
        return False  # didn't discover the first relationship

    # -------------- Fit second relationship --------------
    # remove some features, refine library, then fit again
    feats_to_remove = [sympy.sympify(feat) for feat in relationship_refinement_2degree[relationship_index]]
    _, refined_candid_lib = get_refined_lib(
        feats_to_remove, 
        data_matrix_df_smooth, 
        candidate_lib_full, 
        get_dropped_feat=True
    )
    seq_th_model = sequentialThLin(fit_intercept=True, alpha=0.3, coef_threshold=0.05, silent=True)
    algebraic_model_th = AlgModelFinder(custom_model=True, custom_model_ob=seq_th_model)
    algebraic_model_th.fit(refined_candid_lib, scale_columns=True)
    best_models_full = algebraic_model_th.best_models(1)
    intercept_dictionary = algebraic_model_th.get_fitted_intercepts()

    simplified_equations = get_simplified_equation_list(
        best_model_df=best_models_full.fillna(0)[:-1],
        coef_threshold=0.05,
        intercept_threshold=0.01,
        global_feature_list=data_matrix_df_smooth.columns,
        intercept_dict=intercept_dictionary,
        simplified=True
    )
    reduced_relationship_features = construct_reduced_fit_list(
        candidate_lib_full.columns, 
        simplified_eqs=simplified_equations
    )
    best_relationship = reduced_relationship_features[0]
    if None in best_relationship:
        best_relationship.remove(None)
    best_relationship_set = set(best_relationship)

    discovered_second_true = False
    for k, v in true_relationship_dict_2degree.items():
        if best_relationship_set == v:
            discovered_second_true = True
            break

    return (discovered_true and discovered_second_true)

# ---------------------------------------------------------
# 4. Main routine to do binary search
# ---------------------------------------------------------
# if __name__ == "__main__":

IC_df = pd.read_csv(os.path.join(path_to_add, "parameters/init_cond_simp_pend.csv"))
# IC_df = IC_df.iloc[[0,3]]
params_df = pd.read_csv(os.path.join(path_to_add, "parameters/pend_param.csv"))
g = 9.81   # Acceleration due to gravity (m/s^2)
# First, generate your full, noise-added dataset at 2000 points
max_points = 2000


data_matrix_df_list_full, t_eval_full = generate_data_at_max_resolution(
    max_points, 
    params_df,
    IC_df,
    noise_perc=0, 
    random_seed=111
)
print(len(data_matrix_df_list_full))
# Set your binary-search bounds
low = 10
high = max_points
best_valid = high  # store best known feasible

while low <= high:
    mid = (low + high) // 2
    print(f"\n[Binary Search] Trying mid = {mid} points")
    # Downsample to mid points and run the discovery check
    df_list_mid = downsample_data(data_matrix_df_list_full, mid)
    discovered = check_relationship_discovery(df_list_mid, poly_degree=2, noise_perc=10)

    if discovered:
        # If discovered, we can try fewer points
        best_valid = mid
        high = mid - 1
        print(f"Relationships discovered with {mid} points.")
    else:
        # If not discovered, we need more points
        low = mid + 1
        print(f"Relationships NOT discovered with {mid} points.")

print("\nBinary search complete.")
print(f"Minimum number of time points required: {best_valid}")
    
