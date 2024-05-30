1:
import numpy as np
from scipy.integrate import odeint
import pandas as pd
import warnings

pd.set_option('display.float_format', '{:0.8f}'.format)
import operator

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import coo_array

2: gamma_df = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_gamma.csv")
3: gamma_df
4: data_matrix_df_orig = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_timeseries.csv")
5:
skip_n_rows_btw = 100
rows_to_keep = np.arange(0, len(data_matrix_df_orig), skip_n_rows_btw)
6: data_matrix_df = data_matrix_df_orig.iloc[rows_to_keep].reset_index(drop=True)
7:
new_column_names = ["time", "Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                    "om_0", "om_1", "P_0", "P_1", "P_2", "P_3", "P_4", "P_5",
                    "Q_0", "Q_1", "Q_2", "Q_3", "Q_4", "Q_5"]
data_matrix_df.rename(columns=dict((zip(data_matrix_df.columns, new_column_names))),
                      inplace=True)
8:
data_matrix_df = data_matrix_df[["time", "Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                                 "om_0", "om_1", "P_0", "P_1", "P_2", "P_3", "P_4", "P_5"]]

data_matrix_df
9: data_matrix_df.columns
10:
admittance_Y_df = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_Y.csv")
for column in admittance_Y_df.columns:
    admittance_Y_df[column] = admittance_Y_df[column].apply(lambda x: x.replace('i', 'j'))
11: admittance_Y_df
12: static_param_df = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_staticparams.csv")
13: static_param_df
14: coupling_K_df = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_K.csv")
15:
coupling_K_df_labeled = coupling_K_df.set_index(coupling_K_df.columns)
coupling_K_df_labeled
16: gamma_df
17:
gamma_matrix = gamma_df.to_numpy()
admittance_Y_matrix = admittance_Y_df.to_numpy()

gamma_matrix
18:
coupling_matrix_init = np.ones(admittance_Y_matrix.shape)
# coupling_matrix_init = np.zeros(admittance_Y_matrix.shape)
# coupling_matrix_init[3,:] = 1

coupling_matrix_init = np.triu(coupling_matrix_init, 0)
coupling_matrix_init
sparse_coupling_matrix_init = coo_array(coupling_matrix_init)
sparse_coupling_matrix_init.toarray()
19:
from dae_finder import FeatureCouplingTransformer


def coup_fun(x, y, i, j, gam_matrix):
    # return np.sin(x-y)
    return np.sin(x - y - gam_matrix[i, j])


def coup_namer(x, y, i, j, gam_matrix):
    return "sin( {}-{} -gamma_{},{} )".format(x, y, i, j)


dummy_tr_sin_diff = FeatureCouplingTransformer(sparse_coupling_matrix_init,
                                               coupling_func=coup_fun,
                                               coupling_namer=coup_namer,
                                               coupling_func_args={"gam_matrix": gamma_matrix},
                                               return_df=True)
20:
sin_diff_library = dummy_tr_sin_diff.fit_transform(data_matrix_df.drop(["time"], axis=1))
cop_ind = dummy_tr_sin_diff.coupled_indices_list

# cop_ind
21: cop_ind
22: sin_diff_library
23:
candidate_lib = pd.concat([data_matrix_df.drop("time", axis=1),
                           sin_diff_library], axis=1)
24:
non_zero_column_series = (candidate_lib ** 2).sum() > 0.00001
non_zero_column_series
non_columns = [column for column in candidate_lib if non_zero_column_series[column]]

candidate_lib = candidate_lib[non_columns]
25: candidate_lib
26:
from dae_finder import add_noise_to_df

noise_perc = 0
data_matrix_df_list = [data_matrix_df]
num_time_points = len(data_matrix_df)
data_matrix_features = data_matrix_df_list[0].columns
for ind, data_matrix_ in enumerate(data_matrix_df_list):
    t_exact = data_matrix_["time"]
    noisy_data_df = add_noise_to_df(data_matrix_, noise_perc=noise_perc, random_seed=111)
    noisy_data_df["time"] = t_exact
    data_matrix_df_list[ind] = noisy_data_df
27:
from dae_finder import smooth_data

# Calling the smoothening function
data_matrix_smooth_df_list = [smooth_data(data_matrix, domain_var="time", derr_order=1, noise_perc=noise_perc) for
                              data_matrix in data_matrix_df_list]

if len(data_matrix_df_list) > 1:
    data_matrix_df_smooth_appended = pd.concat(data_matrix_smooth_df_list, ignore_index=True)
else:
    data_matrix_df_smooth_appended = data_matrix_smooth_df_list[0]

data_matrix_df_smooth = data_matrix_df_smooth_appended[data_matrix_features]
# if "time" in data_matrix_df_smooth:
#     data_matrix_df_smooth = data_matrix_df_smooth.drop("time", axis=1)
28: data_matrix_df_smooth - data_matrix_df
29:

ind = 0
feature_ = "Phi_5"

plt.figure()
# plt.plot(data_matrix_df_list[1]["t"], data_matrix_df_list[1]["x"], "x", t_eval_new, x_new,
#         data_matrix_df[50:100]["t"], data_matrix_df[50:100]["x"], "o")

plt.plot(data_matrix_df_list[ind]["time"], data_matrix_df_list[ind][feature_], ".",
         data_matrix_smooth_df_list[ind]["time"],
         data_matrix_smooth_df_list[ind][feature_], ".",
         data_matrix_df[ind * num_time_points:(ind + 1) * num_time_points]["time"],
         data_matrix_df[ind * num_time_points:(ind + 1) * num_time_points][feature_], ".")
plt.legend(['Noisy', 'Cubic Spline', 'True'])
# plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.title('Cubic-spline interpolation of {} - Noise: {}%'.format(feature_, noise_perc))
plt.show()
30:
# Removing some of the outliers coming from sudden jump during perturbations

new_df = data_matrix_df_smooth_appended[abs(data_matrix_df_smooth_appended) <= 20]

plt.plot(new_df[["time"]], new_df[["d(Phi_0) /dt"]], ".",
         new_df[["time"]], new_df[["om_0"]], ".",
         new_df[["time"]], new_df[["d(om_0) /dt"]], ".")

new_df.plot()
31:
import sympy

from dae_finder import get_refined_lib, remove_paranth_from_feat

# Adding the state variables as scipy symbols
feat_list = list(data_matrix_df.columns)
feat_list_str = ", ".join(remove_paranth_from_feat(data_matrix_df.columns))
exec(feat_list_str + "= sympy.symbols(" + str(feat_list) + ")")
32:
from dae_finder import sequentialThLin, AlgModelFinder

algebraic_model_lasso = AlgModelFinder(model_id='lasso',
                                       alpha=0.3,
                                       fit_intercept=True)
33:
features_to_fit_ = ["Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                    "P_0", "P_1", "P_2", "P_3", "P_4", "P_5"]
# features_to_fit_ = ["P_0", "P_1", "P_2", "P_3", "P_4", "P_5"]
num_nodes = 6
power_features = ["P_{}".format(ind) for ind in range(num_nodes)]
# Mapping each power feature to possible expressions in the algebraic relationship
feature_to_libr_map = {power_feat: candidate_lib.columns.drop(power_features) for power_feat in power_features}

algebraic_model_lasso.fit(candidate_lib, scale_columns=True,
                          features_to_fit=features_to_fit_,
                          feature_to_library_map=feature_to_libr_map)
34: algebraic_model_lasso.best_models()
35: algebraic_model_lasso.best_models()["P_0"][abs(algebraic_model_lasso.best_models()["P_1"]) > 0.01]
36:
feat = "P_2"
algebraic_model_lasso.best_models()[feat][abs(algebraic_model_lasso.best_models()[feat]) > 0.1]
37:
features_to_fit_ = ["Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                    "P_0", "P_1", "P_2", "P_3", "P_4", "P_5"]
# features_to_fit_ = ["P_0", "P_1", "P_2", "P_3", "P_4", "P_5"]
num_nodes = 6
power_features = ["P_{}".format(ind) for ind in range(num_nodes)]
# Mapping each power feature to possible expressions in the algebraic relationship
feature_to_libr_map = {power_feat: candidate_lib.columns.drop(power_features) for power_feat in power_features}

algebraic_model_lasso.fit(candidate_lib, scale_columns=True,
                          features_to_fit=features_to_fit_,
                          feature_to_library_map=feature_to_libr_map)
38: algebraic_model_lasso.best_models()
39: algebraic_model_lasso.best_models()["P_0"][abs(algebraic_model_lasso.best_models()["P_1"]) > 0.01]
40:
feat = "P_2"
algebraic_model_lasso.best_models()[feat][abs(algebraic_model_lasso.best_models()[feat]) > 0.1]
41:
# from dae_finder import sequentialThLin

# seq_th_model = sequentialThLin(fit_intercept=False)

# seq_th_model.fit(X=candidate_lib_full,  y=data_matrix_df_smooth_appended['d([P]) /dt'])
# features_to_remove = {E, S*ES}

# features_to_remove, refined_candid_lib = get_refined_lib(features_to_remove, data_matrix_df,
#                                                   candidate_lib_full, get_dropped_feat=True)

# refined_candid_lib = candidate_lib[['Phi_0', 'Phi_1', 'Phi_2', 'Phi_3', 'Phi_4', 'Phi_5', 'om_0',
#        'om_1', 'P_0', 'P_1', 'P_2', 'P_3', 'P_4', 'P_5']]
refined_candid_lib = data_matrix_df_smooth_appended[['Phi_0', 'Phi_1', 'Phi_2', 'Phi_3', 'Phi_4', 'Phi_5', 'om_0',
                                                     'om_1', 'P_0', 'P_1', 'P_2', 'P_3', 'P_4', 'P_5']]

from sklearn.preprocessing import StandardScaler

s_scaler = StandardScaler(with_std=True, with_mean=False)
scaled_refined_lib = pd.DataFrame(s_scaler.fit_transform(refined_candid_lib), columns=s_scaler.feature_names_in_)
scaled_cand_lib = pd.DataFrame(s_scaler.fit_transform(candidate_lib), columns=s_scaler.feature_names_in_)
42:
from sklearn.linear_model import Lasso

alg_lasso = Lasso(fit_intercept=True, alpha=0.3)
alg_lasso.fit(X=scaled_refined_lib, y=data_matrix_df_smooth_appended['d(om_0) /dt'])
alg_lasso.score(X=scaled_refined_lib, y=data_matrix_df_smooth_appended['d(om_0) /dt'])
43: dict(zip(alg_lasso.feature_names_in_, alg_lasso.coef_))
44:
from sklearn.linear_model import LinearRegression

lin_model = LinearRegression()
lin_model.fit(X=scaled_refined_lib[["[ES]"]], y=data_matrix_df_smooth_appended['d([P]) /dt'])
45: lin_model.intercept_
46: alg_lasso.fit(X=scaled_cand_lib, y=data_matrix_df_smooth_appended['d(om_0) /dt'])
47: dict(zip(alg_lasso.feature_names_in_, alg_lasso.coef_))
48:
from dae_finder import sequentialThLin, AlgModelFinder
from sklearn.linear_model import LinearRegression

# lin_reg_model = LinearRegression
# lin_reg_model_arg = {"fit_intercept": True}
# seq_th_model = sequentialThLin(custom_model=True,
#                                custom_model_ob = lin_reg_model,
#                                custom_model_arg= lin_reg_model_arg,
#                               coef_threshold=0.1)
seq_th_model = sequentialThLin(coef_threshold=0.1, fit_intercept=True)

algebraic_model_th = AlgModelFinder(custom_model=True, custom_model_ob=seq_th_model)
49:
algebraic_model_th.fit(candidate_lib, scale_columns=True,
                       features_to_fit=features_to_fit_,
                       feature_to_library_map=feature_to_libr_map)
50:
# Best 10 models using R2 metrix
algebraic_model_th.best_models()
51: algebraic_model_th.get_fitted_intercepts()
52:
feat = "P_3"
algebraic_model_th.best_models()[feat][abs(algebraic_model_th.best_models()[feat]) > 0.1]
53:
feat = "P_2"
algebraic_model_th.best_models()[feat][abs(algebraic_model_th.best_models()[feat]) > 0.1]
54:
from dae_finder import sequentialThLin, AlgModelFinder
from sklearn.linear_model import LinearRegression

# lin_reg_model = LinearRegression
# lin_reg_model_arg = {"fit_intercept": True}
# seq_th_model = sequentialThLin(custom_model=True,
#                                custom_model_ob = lin_reg_model,
#                                custom_model_arg= lin_reg_model_arg,
#                               coef_threshold=0.1)
seq_th_model = sequentialThLin(model_id="lasso", coef_threshold=0.1, fit_intercept=True)

seq_th_model.fit(X=scaled_refined_lib, y=data_matrix_df_smooth_appended['d(om_1) /dt'])
seq_th_model.score(X=scaled_refined_lib, y=data_matrix_df_smooth_appended['d(om_1) /dt'])
55: dict(zip(seq_th_model.feature_names_in_, seq_th_model.coef_))
56:
from dae_finder import sequentialThLin, AlgModelFinder
from sklearn.linear_model import LinearRegression

# lin_reg_model = LinearRegression
# lin_reg_model_arg = {"fit_intercept": True}
# seq_th_model = sequentialThLin(custom_model=True,
#                                custom_model_ob = lin_reg_model,
#                                custom_model_arg= lin_reg_model_arg,
#                               coef_threshold=0.1)
seq_th_model = sequentialThLin(model_id="lasso", coef_threshold=0.1, fit_intercept=True)

seq_th_model.fit(X=scaled_refined_lib, y=data_matrix_df_smooth_appended['d(om_0) /dt'])
seq_th_model.score(X=scaled_refined_lib, y=data_matrix_df_smooth_appended['d(om_0) /dt'])
57: dict(zip(seq_th_model.feature_names_in_, seq_th_model.coef_))
58:
from dae_finder import sequentialThLin, AlgModelFinder
from sklearn.linear_model import LinearRegression

# lin_reg_model = LinearRegression
# lin_reg_model_arg = {"fit_intercept": True}
# seq_th_model = sequentialThLin(custom_model=True,
#                                custom_model_ob = lin_reg_model,
#                                custom_model_arg= lin_reg_model_arg,
#                               coef_threshold=0.1)
seq_th_model = sequentialThLin(model_id="lasso", coef_threshold=0.5, fit_intercept=True)

seq_th_model.fit(X=scaled_refined_lib, y=data_matrix_df_smooth_appended['d(om_0) /dt'])
seq_th_model.score(X=scaled_refined_lib, y=data_matrix_df_smooth_appended['d(om_0) /dt'])
59: dict(zip(seq_th_model.feature_names_in_, seq_th_model.coef_))
60:
from dae_finder import sequentialThLin, AlgModelFinder
from sklearn.linear_model import LinearRegression

# lin_reg_model = LinearRegression
# lin_reg_model_arg = {"fit_intercept": True}
# seq_th_model = sequentialThLin(custom_model=True,
#                                custom_model_ob = lin_reg_model,
#                                custom_model_arg= lin_reg_model_arg,
#                               coef_threshold=0.1)
seq_th_model = sequentialThLin(model_id="lasso", coef_threshold=0.1, fit_intercept=True)

seq_th_model.fit(X=scaled_refined_lib, y=data_matrix_df_smooth_appended['d(om_0) /dt'])
seq_th_model.score(X=scaled_refined_lib, y=data_matrix_df_smooth_appended['d(om_0) /dt'])
61: dict(zip(seq_th_model.feature_names_in_, seq_th_model.coef_))
62: seq_th_model.intercept_
63:
coef_dict = dict(zip(seq_th_model.feature_names_in_, seq_th_model.coef_))
coef_dict
64: non_zero_feat = [feat for feat, coef in coef_dict.items() if abs(coef) > 0.01]
65: non_zero_feat
66:
from sklearn.linear_model import LinearRegression

lin_model = LinearRegression(fit_intercept=True)
lin_model.fit(X=X = scaled_refined_lib[non_zero_feat], y = data_matrix_df_smooth_appended['d(om_0) /dt'])
67:
from sklearn.linear_model import LinearRegression

lin_model = LinearRegression(fit_intercept=True)
lin_model.fit(X=scaled_refined_lib[non_zero_feat], y=data_matrix_df_smooth_appended['d(om_0) /dt'])
68:
from sklearn.linear_model import LinearRegression

lin_model = LinearRegression(fit_intercept=True)
lin_model.fit(X=scaled_refined_lib[non_zero_feat], y=data_matrix_df_smooth_appended['d(om_0) /dt'])
lin_model.score(X=scaled_refined_lib[non_zero_feat], y=data_matrix_df_smooth_appended['d(om_0) /dt'])
69: dict(zip(lin_model.feature_names_in_, lin_model.coef_))
70:
seq_th_model.fit(X=scaled_cand_lib, y=data_matrix_df_smooth_appended['d(om_0) /dt'])
seq_th_model.score(X=scaled_cand_lib, y=data_matrix_df_smooth_appended['d(om_0) /dt'])
71: dict(zip(seq_th_model.feature_names_in_, seq_th_model.coef_))
72:
skip_n_rows_btw = 10
rows_to_keep = np.arange(0, len(data_matrix_df_orig), skip_n_rows_btw)
73: data_matrix_df = data_matrix_df_orig.iloc[rows_to_keep].reset_index(drop=True)
74:
new_column_names = ["time", "Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                    "om_0", "om_1", "P_0", "P_1", "P_2", "P_3", "P_4", "P_5",
                    "Q_0", "Q_1", "Q_2", "Q_3", "Q_4", "Q_5"]
data_matrix_df.rename(columns=dict((zip(data_matrix_df.columns, new_column_names))),
                      inplace=True)
75:
data_matrix_df = data_matrix_df[["time", "Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                                 "om_0", "om_1", "P_0", "P_1", "P_2", "P_3", "P_4", "P_5"]]

data_matrix_df
76: data_matrix_df.columns
77:
admittance_Y_df = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_Y.csv")
for column in admittance_Y_df.columns:
    admittance_Y_df[column] = admittance_Y_df[column].apply(lambda x: x.replace('i', 'j'))
78: admittance_Y_df
79: static_param_df = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_staticparams.csv")
80: static_param_df
81: coupling_K_df = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_K.csv")
82:
coupling_K_df_labeled = coupling_K_df.set_index(coupling_K_df.columns)
coupling_K_df_labeled
83: gamma_df
84:
gamma_matrix = gamma_df.to_numpy()
admittance_Y_matrix = admittance_Y_df.to_numpy()

gamma_matrix
85:
coupling_matrix_init = np.ones(admittance_Y_matrix.shape)
# coupling_matrix_init = np.zeros(admittance_Y_matrix.shape)
# coupling_matrix_init[3,:] = 1

coupling_matrix_init = np.triu(coupling_matrix_init, 0)
coupling_matrix_init
sparse_coupling_matrix_init = coo_array(coupling_matrix_init)
sparse_coupling_matrix_init.toarray()
86:
from dae_finder import FeatureCouplingTransformer


def coup_fun(x, y, i, j, gam_matrix):
    # return np.sin(x-y)
    return np.sin(x - y - gam_matrix[i, j])


def coup_namer(x, y, i, j, gam_matrix):
    return "sin( {}-{} -gamma_{},{} )".format(x, y, i, j)


dummy_tr_sin_diff = FeatureCouplingTransformer(sparse_coupling_matrix_init,
                                               coupling_func=coup_fun,
                                               coupling_namer=coup_namer,
                                               coupling_func_args={"gam_matrix": gamma_matrix},
                                               return_df=True)
87:
sin_diff_library = dummy_tr_sin_diff.fit_transform(data_matrix_df.drop(["time"], axis=1))
cop_ind = dummy_tr_sin_diff.coupled_indices_list

# cop_ind
88: cop_ind
89: sin_diff_library
90:
candidate_lib = pd.concat([data_matrix_df.drop("time", axis=1),
                           sin_diff_library], axis=1)
91:
non_zero_column_series = (candidate_lib ** 2).sum() > 0.00001
non_zero_column_series
non_columns = [column for column in candidate_lib if non_zero_column_series[column]]

candidate_lib = candidate_lib[non_columns]
92: candidate_lib
93:
from dae_finder import add_noise_to_df

noise_perc = 0
data_matrix_df_list = [data_matrix_df]
num_time_points = len(data_matrix_df)
data_matrix_features = data_matrix_df_list[0].columns
for ind, data_matrix_ in enumerate(data_matrix_df_list):
    t_exact = data_matrix_["time"]
    noisy_data_df = add_noise_to_df(data_matrix_, noise_perc=noise_perc, random_seed=111)
    noisy_data_df["time"] = t_exact
    data_matrix_df_list[ind] = noisy_data_df
94:
from dae_finder import smooth_data

# Calling the smoothening function
data_matrix_smooth_df_list = [smooth_data(data_matrix, domain_var="time", derr_order=1, noise_perc=noise_perc) for
                              data_matrix in data_matrix_df_list]

if len(data_matrix_df_list) > 1:
    data_matrix_df_smooth_appended = pd.concat(data_matrix_smooth_df_list, ignore_index=True)
else:
    data_matrix_df_smooth_appended = data_matrix_smooth_df_list[0]

data_matrix_df_smooth = data_matrix_df_smooth_appended[data_matrix_features]
# if "time" in data_matrix_df_smooth:
#     data_matrix_df_smooth = data_matrix_df_smooth.drop("time", axis=1)
95: data_matrix_df_smooth - data_matrix_df
96:

ind = 0
feature_ = "Phi_5"

plt.figure()
# plt.plot(data_matrix_df_list[1]["t"], data_matrix_df_list[1]["x"], "x", t_eval_new, x_new,
#         data_matrix_df[50:100]["t"], data_matrix_df[50:100]["x"], "o")

plt.plot(data_matrix_df_list[ind]["time"], data_matrix_df_list[ind][feature_], ".",
         data_matrix_smooth_df_list[ind]["time"],
         data_matrix_smooth_df_list[ind][feature_], ".",
         data_matrix_df[ind * num_time_points:(ind + 1) * num_time_points]["time"],
         data_matrix_df[ind * num_time_points:(ind + 1) * num_time_points][feature_], ".")
plt.legend(['Noisy', 'Cubic Spline', 'True'])
# plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.title('Cubic-spline interpolation of {} - Noise: {}%'.format(feature_, noise_perc))
plt.show()
97:
# Removing some of the outliers coming from sudden jump during perturbations

new_df = data_matrix_df_smooth_appended[abs(data_matrix_df_smooth_appended) <= 20]

plt.plot(new_df[["time"]], new_df[["d(Phi_0) /dt"]], ".",
         new_df[["time"]], new_df[["om_0"]], ".",
         new_df[["time"]], new_df[["d(om_0) /dt"]], ".")

new_df.plot()
98:
import sympy

from dae_finder import get_refined_lib, remove_paranth_from_feat

# Adding the state variables as scipy symbols
feat_list = list(data_matrix_df.columns)
feat_list_str = ", ".join(remove_paranth_from_feat(data_matrix_df.columns))
exec(feat_list_str + "= sympy.symbols(" + str(feat_list) + ")")
99:
from dae_finder import sequentialThLin, AlgModelFinder

algebraic_model_lasso = AlgModelFinder(model_id='lasso',
                                       alpha=0.3,
                                       fit_intercept=True)
100:
features_to_fit_ = ["Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                    "P_0", "P_1", "P_2", "P_3", "P_4", "P_5"]
# features_to_fit_ = ["P_0", "P_1", "P_2", "P_3", "P_4", "P_5"]
num_nodes = 6
power_features = ["P_{}".format(ind) for ind in range(num_nodes)]
# Mapping each power feature to possible expressions in the algebraic relationship
feature_to_libr_map = {power_feat: candidate_lib.columns.drop(power_features) for power_feat in power_features}

algebraic_model_lasso.fit(candidate_lib, scale_columns=True,
                          features_to_fit=features_to_fit_,
                          feature_to_library_map=feature_to_libr_map)
101: algebraic_model_lasso.best_models()
102: algebraic_model_lasso.best_models()["P_0"][abs(algebraic_model_lasso.best_models()["P_1"]) > 0.01]
103:
feat = "P_2"
algebraic_model_lasso.best_models()[feat][abs(algebraic_model_lasso.best_models()[feat]) > 0.1]
104:
feat = "P_2"
algebraic_model_lasso.best_models()[feat][abs(algebraic_model_lasso.best_models()[feat]) > 0.1]
105:
feat = "P_2"
algebraic_model_lasso.best_models()[feat][abs(algebraic_model_lasso.best_models()[feat]) > 0.1]
106:
feat = "P_2"
algebraic_model_lasso.best_models()[feat][abs(algebraic_model_lasso.best_models()[feat]) > 0.1]
107:
skip_n_rows_btw = 100
rows_to_keep = np.arange(0, len(data_matrix_df_orig), skip_n_rows_btw)
108: data_matrix_df = data_matrix_df_orig.iloc[rows_to_keep].reset_index(drop=True)
109:
new_column_names = ["time", "Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                    "om_0", "om_1", "P_0", "P_1", "P_2", "P_3", "P_4", "P_5",
                    "Q_0", "Q_1", "Q_2", "Q_3", "Q_4", "Q_5"]
data_matrix_df.rename(columns=dict((zip(data_matrix_df.columns, new_column_names))),
                      inplace=True)
110:
data_matrix_df = data_matrix_df[["time", "Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                                 "om_0", "om_1", "P_0", "P_1", "P_2", "P_3", "P_4", "P_5"]]

data_matrix_df
111: data_matrix_df.columns
112:
admittance_Y_df = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_Y.csv")
for column in admittance_Y_df.columns:
    admittance_Y_df[column] = admittance_Y_df[column].apply(lambda x: x.replace('i', 'j'))
113: admittance_Y_df
114: static_param_df = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_staticparams.csv")
115: static_param_df
116: coupling_K_df = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_K.csv")
117:
coupling_K_df_labeled = coupling_K_df.set_index(coupling_K_df.columns)
coupling_K_df_labeled
118: gamma_df
119:
gamma_matrix = gamma_df.to_numpy()
admittance_Y_matrix = admittance_Y_df.to_numpy()

gamma_matrix
120:
coupling_matrix_init = np.ones(admittance_Y_matrix.shape)
# coupling_matrix_init = np.zeros(admittance_Y_matrix.shape)
# coupling_matrix_init[3,:] = 1

coupling_matrix_init = np.triu(coupling_matrix_init, 0)
coupling_matrix_init
sparse_coupling_matrix_init = coo_array(coupling_matrix_init)
sparse_coupling_matrix_init.toarray()
121:
from dae_finder import FeatureCouplingTransformer


def coup_fun(x, y, i, j, gam_matrix):
    # return np.sin(x-y)
    return np.sin(x - y - gam_matrix[i, j])


def coup_namer(x, y, i, j, gam_matrix):
    return "sin( {}-{} -gamma_{},{} )".format(x, y, i, j)


dummy_tr_sin_diff = FeatureCouplingTransformer(sparse_coupling_matrix_init,
                                               coupling_func=coup_fun,
                                               coupling_namer=coup_namer,
                                               coupling_func_args={"gam_matrix": gamma_matrix},
                                               return_df=True)
122:
sin_diff_library = dummy_tr_sin_diff.fit_transform(data_matrix_df.drop(["time"], axis=1))
cop_ind = dummy_tr_sin_diff.coupled_indices_list

# cop_ind
123: cop_ind
124: sin_diff_library
125:
candidate_lib = pd.concat([data_matrix_df.drop("time", axis=1),
                           sin_diff_library], axis=1)
126:
non_zero_column_series = (candidate_lib ** 2).sum() > 0.00001
non_zero_column_series
non_columns = [column for column in candidate_lib if non_zero_column_series[column]]

candidate_lib = candidate_lib[non_columns]
127: candidate_lib
128:
from dae_finder import add_noise_to_df

noise_perc = 0
data_matrix_df_list = [data_matrix_df]
num_time_points = len(data_matrix_df)
data_matrix_features = data_matrix_df_list[0].columns
for ind, data_matrix_ in enumerate(data_matrix_df_list):
    t_exact = data_matrix_["time"]
    noisy_data_df = add_noise_to_df(data_matrix_, noise_perc=noise_perc, random_seed=111)
    noisy_data_df["time"] = t_exact
    data_matrix_df_list[ind] = noisy_data_df
129:
from dae_finder import smooth_data

# Calling the smoothening function
data_matrix_smooth_df_list = [smooth_data(data_matrix, domain_var="time", derr_order=1, noise_perc=noise_perc) for
                              data_matrix in data_matrix_df_list]

if len(data_matrix_df_list) > 1:
    data_matrix_df_smooth_appended = pd.concat(data_matrix_smooth_df_list, ignore_index=True)
else:
    data_matrix_df_smooth_appended = data_matrix_smooth_df_list[0]

data_matrix_df_smooth = data_matrix_df_smooth_appended[data_matrix_features]
# if "time" in data_matrix_df_smooth:
#     data_matrix_df_smooth = data_matrix_df_smooth.drop("time", axis=1)
130: data_matrix_df_smooth - data_matrix_df
131:

ind = 0
feature_ = "Phi_5"

plt.figure()
# plt.plot(data_matrix_df_list[1]["t"], data_matrix_df_list[1]["x"], "x", t_eval_new, x_new,
#         data_matrix_df[50:100]["t"], data_matrix_df[50:100]["x"], "o")

plt.plot(data_matrix_df_list[ind]["time"], data_matrix_df_list[ind][feature_], ".",
         data_matrix_smooth_df_list[ind]["time"],
         data_matrix_smooth_df_list[ind][feature_], ".",
         data_matrix_df[ind * num_time_points:(ind + 1) * num_time_points]["time"],
         data_matrix_df[ind * num_time_points:(ind + 1) * num_time_points][feature_], ".")
plt.legend(['Noisy', 'Cubic Spline', 'True'])
# plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.title('Cubic-spline interpolation of {} - Noise: {}%'.format(feature_, noise_perc))
plt.show()
132:
# Removing some of the outliers coming from sudden jump during perturbations

new_df = data_matrix_df_smooth_appended[abs(data_matrix_df_smooth_appended) <= 20]

plt.plot(new_df[["time"]], new_df[["d(Phi_0) /dt"]], ".",
         new_df[["time"]], new_df[["om_0"]], ".",
         new_df[["time"]], new_df[["d(om_0) /dt"]], ".")

new_df.plot()
133:
import sympy

from dae_finder import get_refined_lib, remove_paranth_from_feat

# Adding the state variables as scipy symbols
feat_list = list(data_matrix_df.columns)
feat_list_str = ", ".join(remove_paranth_from_feat(data_matrix_df.columns))
exec(feat_list_str + "= sympy.symbols(" + str(feat_list) + ")")
134:
from dae_finder import sequentialThLin, AlgModelFinder

algebraic_model_lasso = AlgModelFinder(model_id='lasso',
                                       alpha=0.3,
                                       fit_intercept=True)
135:
features_to_fit_ = ["Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                    "P_0", "P_1", "P_2", "P_3", "P_4", "P_5"]
# features_to_fit_ = ["P_0", "P_1", "P_2", "P_3", "P_4", "P_5"]
num_nodes = 6
power_features = ["P_{}".format(ind) for ind in range(num_nodes)]
# Mapping each power feature to possible expressions in the algebraic relationship
feature_to_libr_map = {power_feat: candidate_lib.columns.drop(power_features) for power_feat in power_features}

algebraic_model_lasso.fit(candidate_lib, scale_columns=True,
                          features_to_fit=features_to_fit_,
                          feature_to_library_map=feature_to_libr_map)
136: algebraic_model_lasso.best_models()
137:
feat = "P_2"
algebraic_model_lasso.best_models()[feat][abs(algebraic_model_lasso.best_models()[feat]) > 0.1]
138:
import numpy as np
from scipy.integrate import odeint
import pandas as pd
import warnings

pd.set_option('display.float_format', '{:0.8f}'.format)
import operator

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import coo_array

139: gamma_df = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_gamma.csv")
140: gamma_df
141: data_matrix_df_orig = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_timeseries.csv")
142:
skip_n_rows_btw = 100
rows_to_keep = np.arange(0, len(data_matrix_df_orig), skip_n_rows_btw)
143: data_matrix_df = data_matrix_df_orig.iloc[rows_to_keep].reset_index(drop=True)
144:
new_column_names = ["time", "Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                    "om_0", "om_1", "P_0", "P_1", "P_2", "P_3", "P_4", "P_5",
                    "Q_0", "Q_1", "Q_2", "Q_3", "Q_4", "Q_5"]
data_matrix_df.rename(columns=dict((zip(data_matrix_df.columns, new_column_names))),
                      inplace=True)
145:
data_matrix_df = data_matrix_df[["time", "Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                                 "om_0", "om_1", "P_0", "P_1", "P_2", "P_3", "P_4", "P_5"]]

data_matrix_df
146: data_matrix_df.columns
147:
admittance_Y_df = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_Y.csv")
for column in admittance_Y_df.columns:
    admittance_Y_df[column] = admittance_Y_df[column].apply(lambda x: x.replace('i', 'j'))
148: admittance_Y_df
149: static_param_df = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_staticparams.csv")
150: static_param_df
151: coupling_K_df = pd.read_csv("powergrid/Datasets/case_4bus2gen_largeperturb/case_4bus2gen_K.csv")
152:
coupling_K_df_labeled = coupling_K_df.set_index(coupling_K_df.columns)
coupling_K_df_labeled
153: gamma_df
154:
gamma_matrix = gamma_df.to_numpy()
admittance_Y_matrix = admittance_Y_df.to_numpy()

gamma_matrix
155:
coupling_matrix_init = np.ones(admittance_Y_matrix.shape)
# coupling_matrix_init = np.zeros(admittance_Y_matrix.shape)
# coupling_matrix_init[3,:] = 1

coupling_matrix_init = np.triu(coupling_matrix_init, 0)
coupling_matrix_init
sparse_coupling_matrix_init = coo_array(coupling_matrix_init)
sparse_coupling_matrix_init.toarray()
156:
from dae_finder import FeatureCouplingTransformer


def coup_fun(x, y, i, j, gam_matrix):
    return np.sin(x - y - gam_matrix[i, j])


def coup_namer(x, y, i, j, gam_matrix):
    return "sin( {}-{} -gamma_{},{} )".format(x, y, i, j)


dummy_tr_sin_diff = FeatureCouplingTransformer(sparse_coupling_matrix_init,
                                               coupling_func=coup_fun,
                                               coupling_namer=coup_namer,
                                               coupling_func_args={"gam_matrix": gamma_matrix},
                                               return_df=True)
157:
sin_diff_library = dummy_tr_sin_diff.fit_transform(data_matrix_df.drop(["time"], axis=1))
cop_ind = dummy_tr_sin_diff.coupled_indices_list

# cop_ind
158: sin_diff_library
159: candidate_lib = pd.concat([data_matrix_df.drop("time", axis=1), sin_diff_library], axis=1)
160: candidate_lib
161:
import sympy

from dae_finder import get_refined_lib, remove_paranth_from_feat

# Adding the state variables as scipy symbols
feat_list = list(data_matrix_df.columns)
feat_list_str = ", ".join(remove_paranth_from_feat(data_matrix_df.columns))
exec(feat_list_str + "= sympy.symbols(" + str(feat_list) + ")")
162:
from dae_finder import sequentialThLin, AlgModelFinder

algebraic_model_lasso = AlgModelFinder(model_id='lasso',
                                       alpha=0.3,
                                       fit_intercept=True)
163:
algebraic_model_lasso.fit(candidate_lib, scale_columns=True,
                          features_to_fit=["Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                                           "P_0", "P_1", "P_2", "P_3", "P_4", "P_5"])
164: algebraic_model_lasso.best_models()
165: algebraic_model_lasso.best_models()["P_1"][abs(algebraic_model_lasso.best_models()["P_1"]) > 0.01]
166:
feat = "P_5"
algebraic_model_lasso.best_models()[feat][abs(algebraic_model_lasso.best_models()[feat]) > 0.1]
167:
from dae_finder import sequentialThLin, AlgModelFinder
from sklearn.linear_model import LinearRegression

lin_reg_model = LinearRegression
lin_reg_model_arg = {"fit_intercept": False}
seq_th_model = sequentialThLin(custom_model=True,
                               custom_model_ob=lin_reg_model,
                               custom_model_arg=lin_reg_model_arg,
                               coef_threshold=0.5)
# seq_th_model = sequentialThLin(coef_threshold=0.1, fit_intercept=True)

algebraic_model_th = AlgModelFinder(custom_model=True, custom_model_ob=seq_th_model)
168:
algebraic_model_th.fit(candidate_lib, scale_columns=False,
                       features_to_fit=["Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                                        "P_0", "P_1", "P_2", "P_3", "P_4", "P_5"])
169:
# Best 10 models using R2 metrix
algebraic_model_th.best_models()
170: algebraic_model_th.get_fitted_intercepts()
171:
feat = "P_2"
algebraic_model_th.best_models()[feat][abs(algebraic_model_th.best_models()[feat]) > 0.1]
172:
from dae_finder import sequentialThLin, AlgModelFinder

algebraic_model_lasso = AlgModelFinder(model_id='lasso',
                                       alpha=0.3,
                                       fit_intercept=True)
173:
algebraic_model_lasso.fit(candidate_lib, scale_columns=True,
                          features_to_fit=["Phi_0", "Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5",
                                           "P_0", "P_1", "P_2", "P_3", "P_4", "P_5"])
174: algebraic_model_lasso.best_models()
175: algebraic_model_lasso.best_models()["P_1"][abs(algebraic_model_lasso.best_models()["P_1"]) > 0.01]
176:
feat = "P_5"
algebraic_model_lasso.best_models()[feat][abs(algebraic_model_lasso.best_models()[feat]) > 0.1]
177:
from sklearn.linear_model import LinearRegression, Lasso, Ridge

178:
lin_model = LinearRegression()
lass_model = Lasso(alpha=0.01)
ridge_model = Ridge()
179:
lin_model.fit(candidate_lib[['sin( Phi_3-Phi_0 -gamma_3,0 )', 'sin( Phi_3-Phi_1 -gamma_3,1 )',
                             'sin( Phi_3-Phi_2 -gamma_3,2 )', 'sin( Phi_3-Phi_3 -gamma_3,3 )',
                             'sin( Phi_3-Phi_4 -gamma_3,4 )', 'sin( Phi_3-Phi_5 -gamma_3,5 )']], candidate_lib["P_3"])
180: gamma_df
181: %history - g

Click
to
add
a
cell.