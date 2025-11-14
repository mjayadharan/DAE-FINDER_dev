import os, sys

# path_to_add = os.path.abspath(os.path.join(os.getcwd()))
path_to_add = os.path.abspath(os.path.join(os.getcwd(), "../../"))
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


"""
----------------------------------------------------------------------------------------------------------------
### Main parameters for the code

----------------------------------------------------------------------------------------------------------------
"""
poly_degree = 2
noise_perc = 10
"""
----------------------------------------------------------------------------------------------------------------
"""




# Function to compute derivatives
def pendulum_rhs(t, y, gamma, L=1):
    """
    Function to compute derivatives for simple pendulum with damping
    
    Parameters:
        t : float
            Time
        y : array_like
            Vector containing [theta, omega], where
            theta is the angle and omega is the angular velocity
        gamma : float
            Damping coefficient
        L : float
            Length of the pendulum
        
    Returns:
        dydt : array_like
            Vector containing [omega, alpha], where
            omega is the angular velocity and alpha is the angular acceleration
    """
    theta, omega = y
    alpha = - (9.81 / L) * np.sin(theta) - gamma * omega
    return [omega, alpha]

# Parameters
theta0 = np.pi / 4  # Initial angle (radians)
omega0 = 0.0        # Initial angular velocity (radians per second)
gamma = 0.0       # Damping coefficient
L = 1.0             # Length of the pendulum (meters)
t_span = (0, 10)    # Time span for the simulation

# Function to integrate the system of ODEs
def integrate_pendulum(t_span, y0, gamma, L):

    sol = solve_ivp(lambda t, y: pendulum_rhs(t, y, gamma, L), t_span, y0, method='RK45', t_eval=np.linspace(*t_span, 1000))
    return sol

# Integrate the pendulum system
sol = integrate_pendulum(t_span, [theta0, omega0], gamma, L)

# Plot the results
# plt.figure(figsize=(10, 5))
# plt.plot(sol.t, sol.y[0], label='Angle (radians)')
# plt.plot(sol.t, sol.y[1], label='Angular velocity (rad/s)')
# plt.title('Damped Simple Pendulum Simulation using scipy.solve_ivp')
# plt.xlabel('Time (s)')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True)
# plt.show()

wrong_relation_dict = {}

IC_df = pd.read_csv(os.path.join(path_to_add, "parameters/init_cond_simp_pend.csv"))
# IC_df = IC_df.iloc[[0,3]]
params_df = pd.read_csv(os.path.join(path_to_add, "parameters/pend_param.csv"))
g = 9.81   # Acceleration due to gravity (m/s^2)



"""
----------------------------------------------------------------------------------------------------------------
### Synthesizing data from different ICs

----------------------------------------------------------------------------------------------------------------
"""
L = 5.0
# y_shift = 0.9 * L
# y_shift = 0
wrong_relation_dict[noise_perc] = []

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
# print("# data points per IC: {}".format(data_matrix_df.shape[0]/len(IC_df)))


# Adding noise to time-series
sys.path.append(os.path.join(path_to_add, "daeFinder"))

data_matrix_features = data_matrix_df_list[0].columns
for ind, data_matrix_ in enumerate(data_matrix_df_list):
    t_exact = data_matrix_["t"]
    noisy_data_df = add_noise_to_df(data_matrix_, noise_perc=noise_perc, random_seed=111)
    noisy_data_df["t"] = t_exact
    data_matrix_df_list[ind] = noisy_data_df



"""
----------------------------------------------------------------------------------------------------------------
### Smoothing data and finding derivatives
----------------------------------------------------------------------------------------------------------------
"""


data_amplify_fact = 1
num_smoothed_points = num_time_points*data_amplify_fact


t_eval_new = np.linspace(data_matrix_df_list[0]["t"].iloc[0], data_matrix_df_list[0]["t"].iloc[-1], num_smoothed_points)

#Calling the smoothening function
data_matrix_smooth_df_list = [smooth_data(data_matrix,derr_order=1, noise_perc=noise_perc, eval_points=t_eval_new) for data_matrix in data_matrix_df_list]

if len(data_matrix_smooth_df_list) == 1:
    data_matrix_df_smooth_appended = data_matrix_smooth_df_list[0]
else:
    data_matrix_df_smooth_appended = pd.concat(data_matrix_smooth_df_list[:-1], ignore_index=True)

data_matrix_df_smooth = data_matrix_df_smooth_appended[["x","y", "d(x) /dt", "d(y) /dt"]]
data_matrix_df_smooth = data_matrix_df_smooth.iloc[5:-5]

#Removing big bumps in the time series due to noise
new_df = deepcopy(data_matrix_df_smooth)
new_df["energy"] = 0.5*((new_df["d(x) /dt"])**2 + (new_df["d(y) /dt"])**2) +  9.81*new_df["y"]
data_matrix_df_smooth = data_matrix_df_smooth[abs(new_df["energy"]-new_df["energy"].mean()) < 0.5*new_df["energy"].std()]
data_matrix_df_smooth = data_matrix_df_smooth.rename(columns= dict(zip(data_matrix_df_smooth, ['x', 'y', 'x_dot', 'y_dot'])))


"""
----------------------------------------------------------------------------------------------------------------
## Forming candiate library
----------------------------------------------------------------------------------------------------------------
"""




poly_feature_ob = PolyFeatureMatrix(poly_degree)
candidate_lib_full = poly_feature_ob.fit_transform(data_matrix_df_smooth)

candidate_lib_full = candidate_lib_full.drop(["1"], axis=1)
# print("Degree of library: {}".format(poly_degree))
# print("# terms in the library: {}".format(candidate_lib_full.shape[1]))

#Optionally removing features from the library
terms_to_drop_corr = set()
candidate_lib_full = candidate_lib_full.drop(terms_to_drop_corr, axis=1)
# print("Full candidate library has the following features: {}".format(candidate_lib_full.columns))


"""
----------------------------------------------------------------------------------------------------------------
## Finding first algebraic relationship
----------------------------------------------------------------------------------------------------------------
"""
true_relationship_dict_2degree = {
    1: {'y^2', 'x^2'},
    2: {'y', 'x_dot^2', 'y_dot^2'}
}

relationship_refinement_2degree = {
    1: {'x^2','x*x_dot'},
    2: {'x_dot^2'}
}



# Adding the state variables as scipy symbols
feat_list = list(data_matrix_df_smooth.columns)
feat_list_str = ", ".join(remove_paranth_from_feat(data_matrix_df_smooth.columns))
exec(feat_list_str+ "= sympy.symbols("+str(feat_list)+")")




seq_th_model = sequentialThLin(fit_intercept=True, alpha=0.3, coef_threshold= 0.05, silent=True)
# seq_th_model = sequentialThLin(model_id="LR", alhp coef_threshold= 0.1)

algebraic_model_th = AlgModelFinder(custom_model=True, custom_model_ob= seq_th_model)

algebraic_model_th.fit(candidate_lib_full, scale_columns= False)
best_models_full = algebraic_model_th.best_models(1)


intercept_dictionary = algebraic_model_th.get_fitted_intercepts()

simplified_equations = get_simplified_equation_list(best_model_df=best_models_full.fillna(0)[:-1],
                            coef_threshold=0.05,
                            intercept_threshold= 0.01,
                             global_feature_list=data_matrix_df_smooth.columns,
                             intercept_dict= intercept_dictionary,
                             simplified = True)


reduced_relationship_features = construct_reduced_fit_list(candidate_lib_full.columns, simplified_eqs=simplified_equations)

reduced_relationship_features
best_relationship = reduced_relationship_features[0]
if None in best_relationship:
    best_relationship.remove(None)

best_relationship_set = set(best_relationship)

discovered_true = False
relationship_index = None

for key_,value_ in true_relationship_dict_2degree.items():
    if best_relationship_set == value_:
        discovered_true = True
        relationship_index = key_
if not discovered_true:
    wrong_relation_dict[noise_perc].append(simplified_equations)


"""
----------------------------------------------------------------------------------------------------------------
## Finding first algebraic relationship
----------------------------------------------------------------------------------------------------------------
"""

features_to_remove =[sympy.sympify(feature) for feature in relationship_refinement_2degree[relationship_index]]

features_to_remove, refined_candid_lib = get_refined_lib(features_to_remove, data_matrix_df_smooth,
                                                  candidate_lib_full, get_dropped_feat=True)

seq_th_model = sequentialThLin(fit_intercept=True, alpha=0.3, coef_threshold= 0.05, silent=True)
algebraic_model_th = AlgModelFinder(custom_model=True, custom_model_ob= seq_th_model)
algebraic_model_th.fit(refined_candid_lib, scale_columns= True)


best_models_full = algebraic_model_th.best_models(1)


intercept_dictionary = algebraic_model_th.get_fitted_intercepts()

simplified_equations = get_simplified_equation_list(best_model_df=best_models_full.fillna(0)[:-1],
                            coef_threshold=0.05,
                            intercept_threshold= 0.01,
                             global_feature_list=data_matrix_df_smooth.columns,
                             intercept_dict= intercept_dictionary,
                             simplified = True)

reduced_relationship_features = construct_reduced_fit_list(candidate_lib_full.columns, simplified_eqs=simplified_equations)

reduced_relationship_features
best_relationship = reduced_relationship_features[0]
if None in best_relationship:
    best_relationship.remove(None)

best_relationship_set = set(best_relationship)

discovered_second_true = False
relationship_index = None

for key_,value_ in true_relationship_dict_2degree.items():
    if best_relationship_set == value_:
        discovered_second_true = True
        relationship_index = key_

if not discovered_second_true:
    wrong_relation_dict[noise_perc].append(simplified_equations)



if discovered_second_true and discovered_true:
    print("--------Found all the relationships with {} datapoints--------".format(num_time_points))



