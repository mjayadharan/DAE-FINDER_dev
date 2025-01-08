import numpy as np
import pandas as pd
import warnings
pd.set_option('display.float_format', '{:0.8f}'.format)
import matplotlib.pyplot as plt

folder_names = {"case_4bus2gen_onetenthperturb",
                "case_9bus3gen_onetenthperturb",
                "case39bus10gen_onetenthperturb"}

case_names = ["Case 4", "Case 9", "Case 39"]

#Loading data frames
num_permutations = 4
# folder_name_1 = "case39bus9gen_halfperturb"
folder_name_1 = "case39bus10gen_onetenthperturb"
# noise_percent_list = [0, 0.0001, 0.001, 0.01]
noise_perc_value = 0.001
snr = "30dB"

# snr_list = ["No noise", "40dB", "30dB", "20dB"]
result_df_dict = {}
for folder_name_, case_name_ in zip(folder_names, case_names):
    result_df_dict[case_name_] = pd.read_csv(
        "/Users/manu_jay/git_repos/DAE-FINDER_dev/Numerical_Exa"
        "mple/power_grid/{}/{}-{}_noise_{}_permutation.csv".format(folder_name_, folder_name_, noise_perc_value,
                                                                   num_permutations))

#Plotting each of the dataframes
total_relations_list = [6, 12, 49]




# """
# Combined 30dB plots across Case 4, 9, and 39
# """
#
# #Plotting combined image of large and small perturbations
# if folder_name_1 == "case39bus10gen_onetenthperturb":
#     result_df_small = result_dict_up
# elif folder_name_1 == "case39bus9gen_halfperturb":
#     result_df_large = result_dict_up
#
# snr_ = "30dB"
# col = 'green'
# plt.plot(result_df_small[snr_]["#Perturbations"],
#              (total_relations-result_df_small[snr_]["#Incorrect relationship"])*100/total_relations, '.-', label = "Small Perturbation")

# plt.plot(result_df_large[snr_]["#Perturbations"],
#              (total_relations-result_df_large[snr_]["#Incorrect relationship"])*100/total_relations, '.-', label = "Large Perturbation")
for case_name_, total_relations in zip(case_names, total_relations_list):
    print(case_name_)
    result_df = result_df_dict[case_name_]
    gen_full_recovery_pert_small = \
    result_df[result_df["#Incorrect gen relationship mean"] == 0]["#Perturbations"].iloc[0]
    gen_full_recovery_incorr_small = (total_relations -
                                      result_df[result_df["#Incorrect gen relationship mean"] == 0][
                                          "#Incorrect relationship mean"].iloc[0]) * 100 / total_relations

    label_ = '100% Generator Recovery' if case_name_ == "Case 39" else ''
    # if case_name_ == "Case 319":
    #     col = 'green'
    #     plt.errorbar(result_df["#Perturbations"],
    #                  (total_relations - result_df["#Incorrect relationship mean"]) * 100 / total_relations,
    #                  (result_df["#Incorrect relationship std"]) * 100 / total_relations,
    #                  fmt='-o', label=case_name_, capsize=5, color=col)
    #     plt.plot([gen_full_recovery_pert_small], [gen_full_recovery_incorr_small], marker='o', markersize=15,
    #              linestyle='',
    #              markerfacecolor='none', label='100% Generator Recovery', color='black')  # Square markers, no line
    # else:
    plt.errorbar(result_df["#Perturbations"],
                 (total_relations - result_df["#Incorrect relationship mean"]) * 100 / total_relations,
                 (result_df["#Incorrect relationship std"]) * 100 / total_relations,
                 fmt='-o', label=case_name_, capsize=5)
    plt.plot([gen_full_recovery_pert_small], [gen_full_recovery_incorr_small], marker='o', markersize=15, linestyle='',
             markerfacecolor='none', label=label_, color='black')  # Square markers, no line





# gen_full_recovery_pert_large = \
# result_df_large[snr_][result_df_large[snr_]["#Incorrect gen relationship mean"] == 0]["#Perturbations"].iloc[0]
# gen_full_recovery_incorr_large = (total_relations -
#                                   result_df_large[snr_][result_df_large[snr_]["#Incorrect gen relationship mean"] == 0][
#                                       "#Incorrect relationship mean"].iloc[0]) * 100 / total_relations
#
# plt.errorbar(result_df_small[snr_]["#Perturbations"],
#              (total_relations - result_df_small[snr_]["#Incorrect relationship mean"]) * 100 / total_relations,
#              (result_df_small[snr_]["#Incorrect relationship std"]) * 100 / total_relations,
#              fmt='-o', label="Small Perturbation", capsize=5, color=col)
#
# plt.errorbar(result_df_large[snr_]["#Perturbations"],
#              (total_relations - result_df_large[snr_]["#Incorrect relationship mean"]) * 100 / total_relations,
#              (result_df_large[snr_]["#Incorrect relationship std"]) * 100 / total_relations,
#              fmt='--^', label="Large Perturbation", capsize=5, color=col)
#
# plt.plot([gen_full_recovery_pert_small], [gen_full_recovery_incorr_small], marker='o', markersize=15, linestyle='',
#          markerfacecolor='none', label='100% Generator Recovery', color='black')  # Square markers, no line
# plt.plot([gen_full_recovery_pert_large], [gen_full_recovery_incorr_large], marker='o', markersize=15, linestyle='',
#          markerfacecolor='none', label='', color='black')  # Square markers, no line

plt.axhline(y=80, color='r', linestyle='--', linewidth=1, label='80% Recovery')

plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 14
# Set labels for X and Y axes
plt.xlabel('Number of Perturbations')
plt.ylabel('% of Algebraic Relationships correctly identified')
plt.xticks([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

plt.legend(
    # loc='lower right',
    prop={'size': 8},
    frameon=True,  # Ensure the legend box is visible
    borderpad=1,  # Padding between the border and the legend content
    borderaxespad=1,  # Padding between the legend and the axes
    fancybox=True,  # Rounded border corners (set to False for square corners)
    edgecolor='black'
)

plt.title("Recovery of model for different cases with {} SNR signal".format(snr))

plt.savefig('/Users/manu_jay/git_repos/DAE-FINDER_dev/Numerical_Example/power_grid/diff_cases_{}.svg'.format(snr),
            format='svg', bbox_inches='tight')

