import numpy as np
import def_all_CRNs as CRNs
import pandas as pd
from sklearn.linear_model import LinearRegression
from dae_finder import PolyFeatureMatrix
from sklearn.preprocessing import StandardScaler
import sympy
import argparse
import itertools
import matplotlib.pyplot as plt
from dae_finder import get_refined_lib
from dae_finder import AlgModelFinder
from dae_finder import get_simplified_equation_list
from dae_finder import construct_reduced_fit_list
from dae_finder import smooth_data
from dae_finder import add_noise_to_df
from dae_finder import solveMM

import matplotlib.pyplot as plt

num_points = 50
noise_perc = 5
t_final = 4

clean_data = CRNs.make_CRN1(5, num_points, t_final)
clean_df = pd.DataFrame(clean_data, columns=['[t]', '[A]', '[B]', '[E1]', '[AE1]'])

data_matrix_df_list = [clean_df.iloc[i:i+num_points].reset_index(drop=True) for i in range(0, len(clean_df), num_points)]

tSolve = list(data_matrix_df_list[0]['[t]'])
num_time_points = len(tSolve)

for ind, data_matrix_ in enumerate(data_matrix_df_list):
        t_exact = data_matrix_["[t]"]
        noisy_data_df = add_noise_to_df(data_matrix_, noise_perc=noise_perc, random_seed=8) # Use trial as RNG seed
        #noisy_data_df = noisy_data_df/noisy_data_df['[A]'].iloc[0]
        noisy_data_df["[t]"] = t_exact
        data_matrix_df_list[ind] = noisy_data_df

data_matrix_features = data_matrix_df_list[0].columns
# 3) Smooth noisy data
num_smoothed_points = num_time_points
t_eval_new = np.linspace(data_matrix_df_list[0]["[t]"].iloc[0], data_matrix_df_list[0]["[t]"].iloc[-1], num_smoothed_points)
data_matrix_smooth_df_list = [smooth_data(data_matrix,domain_var="[t]",derr_order=1, noise_perc=noise_perc,
                                        eval_points=t_eval_new) for data_matrix in data_matrix_df_list]

if len(data_matrix_df_list) > 1:
        data_matrix_df_smooth_appended = pd.concat(data_matrix_smooth_df_list, ignore_index=True)
else:
        data_matrix_df_smooth_appended = data_matrix_smooth_df_list[0]

data_matrix_df_smooth = data_matrix_df_smooth_appended[data_matrix_features]
if "[t]" in data_matrix_df_smooth:
        data_matrix_df_smooth = data_matrix_df_smooth.drop("[t]", axis=1)

data = data_matrix_df_list[0]

colors = ["#332288", "#332288","#CC6677", "#88CCEB", "#18793D"]
labels = ['t', "[S]", "[P]", '[E]', '[ES]']
fig, ax = plt.subplots(figsize=(7, 5))
for i, xi in enumerate(data):
    if xi != '[t]':
        ax.plot(data['[t]'], data[xi], 'o', color=colors[i], label=labels[i])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("time", fontsize=16)
plt.ylabel("concentration", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

handles, labels = plt.gca().get_legend_handles_labels()
order = [0,2,3,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=14, loc ='center right', bbox_to_anchor=(0.5, 0., 0.5, 0.6))
plt.savefig("m.svg")
plt.show()