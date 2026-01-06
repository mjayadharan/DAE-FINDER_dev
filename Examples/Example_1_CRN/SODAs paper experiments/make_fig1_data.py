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

def test_code(noise_perc, num_points, trial_id, poly=2, tfinal=4, alpha=0.2):
    clean_data = CRNs.make_CRN1(5, num_points, tfinal)
    clean_df = pd.DataFrame(clean_data, columns=['[t]', '[A]', '[B]', '[E1]', '[AE1]'])

    data_matrix_df_list = [clean_df.iloc[i:i+num_points].reset_index(drop=True) for i in range(0, len(clean_df), num_points)]
    tSolve = list(data_matrix_df_list[0]['[t]'])
    num_time_points = len(tSolve)

    for ind, data_matrix_ in enumerate(data_matrix_df_list):
            t_exact = data_matrix_["[t]"]
            noisy_data_df = add_noise_to_df(data_matrix_, noise_perc=noise_perc, random_seed=trial_id) # Use trial as RNG seed
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

    # 4) Make feature matrix, scale
    poly_degree = poly
    poly_feature_ob = PolyFeatureMatrix(poly_degree)
    candidate_lib_full = poly_feature_ob.fit_transform(data_matrix_df_smooth)
    candidate_lib_full = candidate_lib_full.drop(["1"], axis=1)

    s_scaler = StandardScaler(with_std=True, with_mean=False)
    scaled_cand_lib = pd.DataFrame(s_scaler.fit_transform(candidate_lib_full), columns=s_scaler.feature_names_in_)
    if '1' in scaled_cand_lib.columns:
            scaled_cand_lib['1'] = 1

    feat_list = list(clean_df.columns)
    t, A, B, E1, AE1 = sympy.symbols(feat_list)

    # 5) Discover conservation law 1
    algebraic_model_lasso = AlgModelFinder(model_id='lasso',
                                        fit_intercept=True,
                                        alpha=0.2)
    algebraic_model_lasso.fit(data_matrix_df_smooth, scale_columns= True)
    intercept_dictionary = algebraic_model_lasso.get_fitted_intercepts()
    best_models_full = algebraic_model_lasso.best_models()

    simplified_equations = get_simplified_equation_list(best_model_df=best_models_full.fillna(0)[:-1],
                                coef_threshold=0.05,
                                intercept_threshold= 0.01,
                                global_feature_list=data_matrix_features,
                                intercept_dict= intercept_dictionary,
                                simplified = True)
    reduced_relationship_features = construct_reduced_fit_list(best_models_full.fillna(0)[:-1], simplified_eqs=simplified_equations)
    best_relationship = reduced_relationship_features[0]

    if None in best_relationship:
        best_relationship.remove(None)

    # 6) Perform first check: did we find the first law?
    if set(best_relationship) != set(['[E1]', '[AE1]']):
        return False

    # 7) Discover conservation law 2
    _, refined_candid_lib = get_refined_lib({E1}, clean_df,
                                                candidate_lib_full, get_dropped_feat=True)
    
    # 8) Discover first QSSA 
    algebraic_model_lasso = AlgModelFinder(model_id='lasso',
                                        fit_intercept=False, alpha=0.2)
    algebraic_model_lasso.fit(refined_candid_lib, scale_columns= True)
    best_models_full = algebraic_model_lasso.best_models()
    intercept_dictionary = algebraic_model_lasso.get_fitted_intercepts()

    simplified_equations = get_simplified_equation_list(best_model_df=best_models_full.fillna(0)[:-1],
                                coef_threshold=0.05,
                                intercept_threshold= 0.01,
                                global_feature_list=clean_df.columns,
                                intercept_dict= intercept_dictionary,
                                simplified = True)
    reduced_relationship_features = construct_reduced_fit_list(best_models_full.fillna(0)[:-1], simplified_eqs=simplified_equations)
    best_relationship = reduced_relationship_features[0]
    if None in best_relationship:
        best_relationship.remove(None)
        
    lin_model = LinearRegression()

    best_relation_lhs = best_relationship[0]
    best_relation_rhs = best_relationship[1:]
    lin_model.fit(candidate_lib_full[best_relation_rhs], candidate_lib_full[best_relation_lhs])
    d = dict(zip(lin_model.feature_names_in_, lin_model.coef_))
    d['intercept'] = lin_model.intercept_
    d[best_relationship[0]] = np.inf # Placehold for LHS

    d_filt = {k: v for k, v in d.items() if np.abs(v) >= 0.05} # keep large elements
    if d['intercept'] >= 0.01: # keep large intercept
        d_filt['intercept'] = d['intercept']

    print(d_filt)
    # 9) Did we get it right?
    if set(d_filt) != set({'[A]', '[A] [AE1]', '[AE1]'}):
        return False
    else:
        return True
    

"""
Two versions of sim code included here: main_loop allows plotting of the true/false progression,
while main is faster, and does not save all data for all values tested
"""

def main_loop(output_file, poly):
    # 1) Run experiment

    # Grid of noise levels to test
    noise_levels = np.linspace(0, 15, 16)
    min_points = 1
    max_points = 1200
    step = 1
    frequencies = np.array(range(min_points, max_points, step))
    trials = 10 # Number of trials per noise level

    results = []
    for noise, frequency in itertools.product(noise_levels, frequencies):
        for trial_id in range(1, trials+1):
            try:
                success = test_code(noise, frequency, trial_id, poly)  # Run your code
            except:
                success = False # Error handling

            results.append({
                "Noise": noise,
                "Frequency": frequency,
                "Trial": trial_id,
                "Success": success
            })

    # Convert to DF, save to csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"unprocessed_{output_file}.csv", index=False)
    print(f"\nResults saved to unprocessed_{output_file}.csv")

    # 2) Process results to find fail threshold
    processed_results = []
    grouped = results_df.groupby(["Noise", "Trial"])

    for (noise, trial), group in grouped:
        false_rows = group[group["Success"] == False]
    
        if not false_rows.empty:
            # Find the maximum frequency where Success is False
            max_frequency = false_rows["Frequency"].max() + step
        else:
            # If no False values, set max_frequency to None
            max_frequency = frequencies[1]

        processed_results.append({
            "Noise": noise,
            "Trial": trial,
            "Frequency": max_frequency
        })

    # Convert to DF, csv
    processed_df = pd.DataFrame(processed_results)
    processed_df.to_csv(f"processed_{output_file}.csv", index=False)

    print(f"Processed results saved to 'processed_{output_file}.csv'.")

"""def main(output_file, poly):
    # 1) Run experiment

    # Grid of noise levels to test
    noise_levels = np.linspace(0, 15, 16)
    min_points = 1
    max_points = 1200
    step = 1
    frequencies = np.array(range(max_points, min_points, -step))
    trials = 10 # Number of trials per noise level

    results = []
    for noise in noise_levels:
        for trial in range(1, trials + 1):
            last_success_frequency = None  # Track the last successful frequency
            for frequency in frequencies:
                try:
                    success = test_code(noise, frequency, trial, poly)
                except: 
                    success = False # Error handling

                if success:
                    last_success_frequency = frequency
                else:
                    if last_success_frequency is not None:
                        results.append({
                            "Noise": noise,
                            "Frequency": last_success_frequency,
                            "Trial": trial
                        })
                    break  # Stop testing after the first failure
        
    # Convert to DF, csv
    processed_df = pd.DataFrame(results)
    processed_df.to_csv(f"processed_{output_file}.csv", index=False)

    print(f"Processed results saved to 'processed_{output_file}.csv'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", type=str)
    parser.add_argument("poly", type=int)
    args = parser.parse_args()
    main_loop(args.output_file, args.poly)"""
