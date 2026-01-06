import numpy as np
import def_all_CRNs as CRNs
import pandas as pd
from sklearn.linear_model import LinearRegression
from dae_finder import PolyFeatureMatrix
from sklearn.preprocessing import StandardScaler
import sympy
import argparse
import itertools
from dae_finder import get_refined_lib
from dae_finder import AlgModelFinder
from dae_finder import get_simplified_equation_list
from dae_finder import construct_reduced_fit_list
from dae_finder import smooth_data
from dae_finder import add_noise_to_df
from dae_finder import solveMM
from matplotlib import colors

options = ['CRN1', 'CRN2', 'CRN3']
colors = ["#343084", "#18793D", "#882255", "#44AA99", "#DDCC77", "#AA469A", "#CC6677", "#999933", "#FF9933", "#6600CC", "#FFD700", "#797C2A", "#3399CC", "#A60000", "#0066CC"]

num_points = 100
noise_perc = 5

for option in options:
        if option == 'CRN1':
                clean_data = CRNs.make_CRN1(5, num_points)
                clean_df = pd.DataFrame(clean_data, columns=['[t]', '[A]', '[B]', '[C]', '[E1]'])
                color = "#343084"
        elif option == 'CRN2':
                clean_data = CRNs.make_CRN2(5, num_points)
                clean_df = pd.DataFrame(clean_data, columns=['[t]', '[A]', '[B]', '[C]', '[E1]', '[AE1]', '[E2]', '[AE2]'])
                color = "#AA469A"
        else:
                clean_data = CRNs.make_CRN3(5, num_points)
                clean_df = pd.DataFrame(clean_data, columns=['[t]', 'A', 'B', 'C', 'D', 'F', 'E1', 'AE1', 'E2', 'BE2', 'CE2', 'E3', 'BE3', 'E4', 'DE4'])
                color = "#18793D"


        data_matrix_df_list = [clean_df.iloc[i:i+num_points].reset_index(drop=True) for i in range(0, len(clean_df), num_points)]

        tSolve = list(data_matrix_df_list[0]['[t]'])
        num_time_points = len(tSolve)

        for ind, data_matrix_ in enumerate(data_matrix_df_list):
                t_exact = data_matrix_["[t]"]
                noisy_data_df = add_noise_to_df(data_matrix_, noise_perc=noise_perc, random_seed=9) # Use trial as RNG seed
                noisy_data_df["[t]"] = t_exact
                data_matrix_df_list[ind] = noisy_data_df

        data_matrix_features = data_matrix_df_list[0].columns

        # 3) Smooth noisy data
        num_smoothed_points = num_time_points
        t_eval_new = np.linspace(data_matrix_df_list[0]["[t]"].iloc[0], data_matrix_df_list[0]["[t]"].iloc[-1], num_smoothed_points)
        data_matrix_smooth_df_list = [smooth_data(data_matrix,domain_var="[t]",derr_order=1, noise_perc=noise_perc,
                                                eval_points=t_eval_new) for data_matrix in data_matrix_df_list]

        if len(data_matrix_df_list) >1:
                data_matrix_df_smooth_appended = pd.concat(data_matrix_smooth_df_list, ignore_index=True)
        else:
                data_matrix_df_smooth_appended = data_matrix_smooth_df_list[0]

        data_matrix_df_smooth = data_matrix_df_smooth_appended[data_matrix_features]
        if "[t]" in data_matrix_df_smooth:
                data_matrix_df_smooth = data_matrix_df_smooth.drop("[t]", axis=1)


        import matplotlib.pyplot as plt

        #data_matrix_df_full = pd.concat(data_matrix_df_list, ignore_index=True)

        fig, ax = plt.subplots()
        for i, xi in enumerate(data_matrix_df_list):
                for j, xj in enumerate(xi.drop('[t]', axis=1)):
                        print(j)
                        ax.plot(xi['[t]'], xi[xj],'o', ms=4, color=colors[j])
                plt.xlabel('time', fontsize=20)
                plt.ylabel('concentration', fontsize=20)

        if option == 'CRN3':
                plt.xticks(fontsize=16)
        else:
                plt.xticks(fontsize=16)

        plt.xlabel('time', fontsize=20)
        plt.ylabel('concentration', fontsize=20)
                
        plt.yticks(fontsize=16)

        plt.savefig(f"figs/sim_figs/diagramCompressed_{option}.svg", bbox_inches='tight')
        plt.savefig(f"figs/sim_figs/diagramCompressed_{option}.png", bbox_inches='tight')