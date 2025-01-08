import itertools

from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    _check_feature_names_in,
    check_is_fitted,
)

import pandas as pd
import numpy as np
import warnings
import operator
from copy import deepcopy
from itertools import permutations

from scipy.integrate import odeint
from scipy import interpolate
from scipy.sparse import coo_array


import sympy
from sympy import prod, Poly

import matplotlib.pyplot as plt

from joblib import delayed, Parallel


"""Data Generation functions
"""


def toyEnzRHS(y, t, k_rates):
    # Unpack states, params
    S, E, ES, P = y
    k, kr, kcat = k_rates['k'], k_rates['kr'], k_rates['kcat']

    dydt = [kr * ES - k * E * S,
            (kr + kcat) * ES - k * S * E,
            k * E * S - (kr + kcat) * ES,
            kcat * ES]
    return dydt


def solveToyEnz(init_cond, k_rates, solvedT, tsID, print_to_file=False):
    y0 = [init_cond["S"], init_cond["E"],
          init_cond["ES"], init_cond["P"]]
    sol = odeint(lambda y, t: toyEnzRHS(y, t, k_rates), y0, solvedT)

    paramID = "".join(str(k_rates.values).strip("()").split())
    if print_to_file:
        np.savetxt('data/toyEnzData_' + paramID + '_' + tsID + '.txt', sol)
    return sol


def toyMM_RHS(y, t, k_rates, IC):
    # Unpack states, params
    # S, E, ES, P = y
    S, P = y
    E_0 = IC["E"]

    k, kr, kcat = k_rates['k'], k_rates['kr'], k_rates['kcat']

    # dydt = [S*(-k + kr/(kr+kcat))*(E_0 - (k*E_0*S)/(kr+kcat+k*S)),
    #         (k*kcat*E_0*S)/(kr+kcat+k*S)]

    dydt = [-(k * kcat * E_0 * S) / (kr + kcat + k * S),
            (k * kcat * E_0 * S) / (kr + kcat + k * S)]
    return dydt


def solveMM(init_cond, k_rates, solvedT, tsID, print_to_scr = False, print_to_file=False):
    if print_to_scr:
     print("Solving for Initial Conditions: {} \n and k_rates: {}".format(init_cond, k_rates))
    y0 = [init_cond["S"], init_cond["P"]]
    E_0 = init_cond["E"]
    k, kr, kcat = k_rates['k'], k_rates['kr'], k_rates['kcat']
    sol = odeint(lambda y, t: toyMM_RHS(y, t, k_rates, init_cond), y0, solvedT)
    # print(sol[-5:,1])
    ES_sol = k * E_0 * sol[:, 0] / (kr + kcat + k * sol[:, 0])
    E_sol = E_0 - ES_sol

    final_sol = np.column_stack((sol[:, 0], E_sol, ES_sol, sol[:, 1]))

    # paramID = "".join(str(k_rates.values).strip("()").split())
    # print(paramID)
    if print_to_file:
        np.savetxt('data/MM_Data_' + 'k_' + str(k_rates.values) + '__' + str(init_cond.values) + '_' + tsID + '.txt',
                   final_sol)
    return final_sol

def plotToyEnz(solT, sol, title = ""):
    plt.plot(solT, sol[:, 0], '-ob', label='S(t)', ms=3)
    plt.plot(solT, sol[:, 1], '-og', label='E(t)', ms=3)
    plt.plot(solT, sol[:, 2], '-or', label='ES(t)', ms=3)
    plt.plot(solT, sol[:, 3], '-ok', label='P(t)', ms=3)
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.title(title)
    plt.show()
    return

def plotToy_MM(solT, sol, title =""):
    plt.plot(solT, sol[:, 0], '-ob', label='S(t)', ms=3)
    # plt.plot(solT, sol[:, 1], '-og', label='E(t)', ms=3)
    # plt.plot(solT, sol[:, 2], '-or', label='ES(t)', ms=3)
    plt.plot(solT, sol[:, 1], '-ok', label='P(t)', ms=3)
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.title(title)
    plt.grid()
    plt.show()
    return

def add_noise_to_df(data_df, noise_perc, make_copy=True,
                    random_seed = None, method= "std"):
    """
    data_df: pandas df with columns representing features.
    Add noise to each feature column in the data matrix using a Gaussian distribution with mean zero and standard deviation equal to
    noise_percentage/100 * std of the feature.
    """

    if random_seed:
        np.random.seed(random_seed)
    if make_copy:
        data_df_new = deepcopy(data_df)
    else:
        data_df_new = data_df
    if method == "std":
        std_features = data_df_new.std()
        for feature in data_df_new:
            noise_level = std_features[feature] * noise_perc/100
            data_df_new[feature] += np.random.normal(loc=0.0, scale=noise_level, size=data_df_new[feature].shape)

    return data_df_new


def get_der_names(feature_list, get_list=False):
    """
    Utility function to get a strings denoting the derivatives of the features in the feature_list
    :param feature_list: ['A', 'B', 'C'] or any iterable of strings
    :param get_list: If True, a list of strings are returned, else a dictionary is returned.
    :return: dictionary of the form {'A': 'd(A) /dt'}.
    """
    if get_list:
        return ["d(" + feature + ") /dt" for feature in feature_list]
    return {feature: "d(" + feature + ") /dt" for feature in feature_list}


def der_matrix_calculator(data_matrix, delta_t, rename_feat=True):
    """
    Utility function to calculate the derivative matrix from a data matrix.
    The data is assumed to be evenly spaced with a time interval delta_t in between.
    Frist order forward difference is then used to find the derivative using (f(t+delta_t)-f(t))/delta_t
    :param data_matrix: pd.DataFrame with features.
    :param delta_t: time difference between subsequent data points.
    :param rename_feat: if True, the features are renamed to reflected the derivative notation in the output.
    :return: pd.DataFrame with len = len(data_matrix)-1.
    """
    assert delta_t > 1.e-10, "delta_t cannot be too small or negative"
    derr_matrix = (data_matrix.iloc[1:].reset_index(drop=True) -
                   data_matrix.iloc[:-1].reset_index(drop=True)) / delta_t
    if rename_feat:
        derr_names = get_der_names(data_matrix.columns)
        derr_matrix.rename(columns=derr_names, inplace=True)

    return derr_matrix


def der_label(feature, der=1):
    if der == 0:
        return feature
    elif der == 1:
        return "d({}) /dt".format(feature)
    else:
        return "d^{}({}) /dt^{}".format(der, feature, der)


def smooth_data(data_matrix,
                domain_var="t",
                smooth_method ="spline",
                s_param_=None,
                noise_perc=0,
                derr_order=1,
                eval_points=[],
                num_time_points=0,
                silent =True):
    """
    :param data_matrix: Data matrix to smoothen. nxp data frame structure is assumed where n is the number of
                        data points and p is the number of features (predictors).
    :param domain_var: Domain variable with respect to which the data needs to be smoothened. Default is assumed to be
                        "t" (time).
    :param smooth_method: Numerical method used for smoothening.
    :param s_param: smoothening parameter.
    :param noise_perc: optional estimate of noise to signal ratio %
    :param derr_order: Number of derivatives need to be calculated, wrt the domain variable, after smoothening the data.
    :param eval_points: option list of points at which the smoothened data and derivatives will be evaluated for output
    :return: pd.DataFrame of size len(eval_points) x k where k is the number of features and their derivatives.
    """
    assert domain_var in data_matrix, "domain variable not found in the data matrix"
    s_param = deepcopy(s_param_)
    data_t = data_matrix[domain_var]
    if num_time_points == 0:
        num_time_points = len(data_matrix)
    if len(eval_points) == 0:
        eval_points = np.linspace(data_t.iloc[0], data_t.iloc[-1], num_time_points)
    t_eval_new = eval_points

    data_matrix_ = data_matrix.drop(domain_var, axis=1)
    data_matrix_std = data_matrix_.std()

    data_matrix_smooth = pd.DataFrame(t_eval_new, columns=[domain_var])

    if smooth_method == "spline":
        if s_param:
            s_param_list = [s_param for feature in data_matrix_]
        else:
            s_param_list = [num_time_points * (0.01 * noise_perc * data_matrix_std[feature]) ** 2 for
                            feature in data_matrix_]
        smoothened_values_list = [np.hstack([interpolate.splev(t_eval_new, interpolate.splrep(data_t,
                                                                                              data_matrix_[feature],
                                                                                              s=s_param_val), der=der_ind) [:, None]
                                             for der_ind in range(derr_order + 1)])
                                  for feature, s_param_val in zip(data_matrix_, s_param_list)]
        smoothened_values = np.hstack(smoothened_values_list)
        column_label_list = [[der_label(feature, der_ind) for der_ind in range(derr_order + 1)]
                             for feature in data_matrix_]
        column_label_list = list(itertools.chain.from_iterable(column_label_list))
        smoothened_df = pd.DataFrame(smoothened_values, columns=column_label_list)
        data_matrix_smooth = pd.concat([data_matrix_smooth, smoothened_df], axis=1)

        # for feature in data_matrix_:
        #     if not s_param:
        #         # smoothing parameter: when equal weightage: num_data_points * std of data
        #         s_param = num_time_points * (0.01 * noise_perc * data_matrix_std[feature]) ** 2
        #     tck = interpolate.splrep(data_t, data_matrix_[feature], s=s_param)
        #     for der_ind in range(derr_order + 1):
        #         smoothed_data = interpolate.splev(t_eval_new, tck, der=der_ind)
        #         data_matrix_smooth[der_label(feature, der_ind)] = smoothed_data
    else:
        raise "Smoothening type not supported"

    if not silent:
        print("Returning the smoothened data")
    return data_matrix_smooth

def remove_paranth_from_feat(feature_list):
    """
    Utility function to remove the parenthesis from the name of the feature if they exists.
    If either "[", or "]" are not present, the feature string is returned unchanged.
    :param feature_list: ["[E]", "[ES]"]
    :return: ["E", "ES"]
    """
    result_list = list(feature_list)
    for ind, feat in enumerate(result_list):
        if "[" in feat and "]" in feat:
            result_list[ind] = feat.replace("[", "").replace("]", "")

    return result_list


def poly_to_scipy(exp_list):
    """
    Utility function to convert the power symbol "^" from monomial strings to scipy compatible "**"
     symbol for power.
    :param exp_list: ["A^2", "A*B^3"]
    :return: ["A**2", "A*B**3"]
    """
    return [exp.replace(" ", "*").replace("^", "**") for exp in exp_list]


def get_factor_feat(factor_exp, feat_dict):
    """
    Utility function to return the list of expressions from expr_list which has factor_exp as a factor
    factor_exp: sympy expression eg: [ES]**2
    feat_dict : {'[ES]*[S]^2': [ES]*[S]**2}
    """
    return [feat for feat, feat_sym in feat_dict.items() if sympy.fraction(feat_sym / factor_exp)[1] == 1]


def get_refined_lib(factor_exp, data_matrix_df_, candidate_library_, get_dropped_feat=False):
    """
    Utility function to get the refined library by removing all features in the candidate library which
    has factor_exp as a factor in it.
    :param factor_exp: sympy expression eg. S*ES
    :param data_matrix_df_ (pd.DataFrame): data matrix containing all the state variables as column labels
    :param candidate_library_ (pd.DataFrame): candidate library that needs to be refined.
    :param get_dropped_feat: if True, both the dropped features and the refined library is returned,
    else only the refined library is returned
    :return:
    """
    # Adding the state variables as scipy symbols
    feat_list = list(data_matrix_df_.columns)
    feat_list_str = ", ".join(remove_paranth_from_feat(data_matrix_df_.columns))
    exec(feat_list_str + "= sympy.symbols(" + str(feat_list) + ")")

    # Converting the monomials in the candidate library to scipy expressions
    candid_features = remove_paranth_from_feat(poly_to_scipy(candidate_library_.columns))
    candid_feat_dict = {}
    for feat1, feat2 in zip(candidate_library_.columns, candid_features):
        exec("candid_feat_dict['{}'] = {}".format(feat1, feat2))

    dropped_feats = set()
    if (isinstance(factor_exp, list) or isinstance(factor_exp, set)):
        for factor_ in factor_exp:
            dropped_feats = dropped_feats.union(set(get_factor_feat(factor_, candid_feat_dict)))
    else:
        dropped_feats = dropped_feats.union(set(get_factor_feat(factor_exp, candid_feat_dict)))

    if get_dropped_feat:
        return (dropped_feats, candidate_library_.drop(dropped_feats, axis=1))
    else:
        return candidate_library_.drop(dropped_feats, axis=1)


def get_simplified_equation(best_model_df, feature,
                            global_feature_list, coef_threshold,
                            intercept_threshold= 0.01,
                            intercept=0, simplified=True):

    # Adding the state variables as scipy symbols
    global_feature_list = list(global_feature_list)
    global_feature_list_string = ", ".join(remove_paranth_from_feat(global_feature_list))
    exec(global_feature_list_string + "= sympy.symbols(" + str(global_feature_list) + ")")


    model_lhs = feature
    model_lhs_sp_string = remove_paranth_from_feat(poly_to_scipy([model_lhs]))[0]

    #Intercept below the threshold is assigned to zero
    intercept = 0 if abs(intercept) < intercept_threshold else intercept

    model_coefs = best_model_df[model_lhs].values
    #Coefficients of features in the model below threshold is eliminated
    model_coefs[abs(model_coefs) < coef_threshold] = 0

    model_rhs_features = remove_paranth_from_feat(poly_to_scipy(best_model_df[model_lhs].keys()))


    rhs_string_sp_string = [str(coef) + "*" + feature for coef, feature in zip(model_coefs, model_rhs_features) ]
    rhs_string_sp_string = "+".join(rhs_string_sp_string) + "+" + str(intercept)

    result_dict = {}
    exec("result_dict['lhs'] = {}".format(model_lhs_sp_string))
    exec("result_dict['rhs'] = {}".format(rhs_string_sp_string))

    if not simplified:
        return result_dict
    else:
        n, d = sympy.fraction(sympy.cancel(result_dict['rhs'] / result_dict['lhs']))
        result_dict['lhs'] = d
        result_dict['rhs'] = n

    return result_dict


def get_simplified_equation_list(best_model_df, global_feature_list,
                                 coef_threshold, intercept_threshold= 0.01,
                                 intercept_dict={}, simplified=True,
                                 feature_list_=[]):

    if len(feature_list_) > 0:
        feature_list = deepcopy(feature_list_)
        assert set(feature_list) <= set(best_model_df.columns), \
            ("fit for some features missing from the best_model_df")
    else:
        feature_list = best_model_df.columns

    result_dict = {feature: get_simplified_equation(best_model_df, feature,
                                                    global_feature_list=global_feature_list,
                                                    coef_threshold=coef_threshold,
                                                    intercept_threshold=intercept_threshold,
                                                    intercept=intercept_dict.get(feature, 0),
                                                    simplified=simplified)
                   for feature in feature_list}

    return result_dict

def sympy_symb_to_feature_name(sympy_symb, library_feat_names):
    """

    @param sympy_symb: sympy symbol string in format
    @param library_feat_names:
    @return:
    """

    symb_str = str(sympy_symb).strip()
    if symb_str == "1":
        return
    symb_str = symb_str.replace("**", "^")
    symb_list = symb_str.split("*")
    possible_permutations = permutations(symb_list)
    for symb_perm in possible_permutations:
        feat = " ".join(symb_perm)
        if feat in library_feat_names:
            return feat

    raise Exception("No feature corresponding to {} exist in the given library_df".format(sympy_symb))


def construct_reduced_fit_list(full_feature_name_list, simplified_eqs,
                               sympy_format=False):
    relation_list = []
    for simpl_eq in simplified_eqs.values():
        lhs = simpl_eq["lhs"]
        rhs = simpl_eq["rhs"]
        lhs_list = []
        rhs_list = []
        try:
            lhs_poly = Poly(lhs)
            lhs_list = [prod(x ** k for x, k in zip(lhs_poly.gens, mon)) for mon in lhs_poly.monoms()]
        except Exception as e:
            print("***Warning: exception occured while trying to find the monomials of {}:  {}".format(lhs, e))

        try:
            rhs_poly = Poly(rhs)
            rhs_list = [prod(x ** k for x, k in zip(rhs_poly.gens, mon)) for mon in rhs_poly.monoms()]
        except Exception as e:
            print("***Warning: exception occured while trying to find the monomials of {}:  {}".format(rhs, e))

        relation_list.append(lhs_list + rhs_list)

    if sympy_format:
        return relation_list
    else:
        relation_in_lib_feat = [
            [sympy_symb_to_feature_name(sympy_symb, full_feature_name_list) for sympy_symb in relations]
            for relations in relation_list]
        return relation_in_lib_feat


def compare_models_(models_df_1, models_df_2, tol=1.e-5):
    """
    Utility function to compare the structure of two models. Note that model_df_1 and model_df_2
    should have the same column labels, index labels, and shape. Returns a data frame with the same
    shape as the model data frames being compared. 0 will appear whenever the term strcture matches
    between two model df, +1 appears when a term is present in models_df_1, and absent in models_df_2.
    Similarly, -1 appears when a term is absent in models_df_1, and present in models_df_2.
    @param models_df_1: pd.DataFrame with columns = [LHS of model] index = terms in the RHS of model.
    @param models_df_2: pd.DataFrame with columns = [LHS of model] index = terms in the RHS of model.
    @param tol: tolerance that will be used for comparing model structure.
    @return: pd.DataFrame of the same shape as models_df_1 and models_df_2. 0 will appear whenever the term strcture matches
    between two model df, +1 appears when a term is present in models_df_1, and absent in models_df_2.
    Similarly, -1 appears when a term is absent in models_df_1, and present in models_df_2.
    """
    assert models_df_1.shape == models_df_2.shape, "both model dataframes should be of the same shape"
    assert all(models_df_1.columns == models_df_2.columns) and all(models_df_1.index == models_df_2.index)

    models_df_1[abs(models_df_1) > tol] = 1
    models_df_1[abs(models_df_1) <= tol] = 0

    models_df_2[abs(models_df_2) > tol] = 1
    models_df_2[abs(models_df_2) <= tol] = 0

    model_diff_df = models_df_1 - models_df_2

    model_diff_df.loc["# incosistent terms"] = abs(model_diff_df).sum()

    return model_diff_df

"""
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
"""
class PolyFeatureMatrix(BaseEstimator, TransformerMixin):
    """
    Generic class to create polynomial library terms. This class is a wrapper around
     sklearn's preprocessing.PolynomialFeatures class with support for pandas data frame.
    """
    def __init__(self, degree=2, interaction_only=False, include_bias=True, output_df=True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.output_df = output_df
        self.poly_feature = PolynomialFeatures(degree=self.degree,
                                               interaction_only=self.interaction_only,
                                               include_bias=self.include_bias)

    def fit(self, X, y=None):
        self.poly_feature.fit(X)
        return self

    def transform(self, X, y=None):
        poly_data_matrix = self.poly_feature.transform(X)
        if self.output_df:
            poly_df = pd.DataFrame(poly_data_matrix, columns=self.poly_feature.get_feature_names_out())
            return poly_df
        else:
            return poly_data_matrix

"""
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
"""
class FeatureCouplingTransformer(TransformerMixin, BaseEstimator):
    """
    Transformer class for generating features (candidate library functions) derived from coupling between features.
    The coupling between features can either be implied from a sparsity matrix (preferred), or can be explicitly
    provided to the constructor.

    Coupling behavior of the features can be explicitly fed to the constructor using hte coupling_func argument. If no
    coupling_func is provided, a second order interaction of the form feature_1*feature_2 is assumed.

    Examples
    ---------
    Case 1: No coupling_func is provided (so default interaction coupling is assumed)

    data_matrix_ = pd.DataFrame([[1,2,3], [4,5,6]], columns = ["t", "x", "y"])
    row  = np.array([0, 0, 1, 1])
    col  = np.array([0, 2, 2, 1])
    data = np.array([4, 5, 7, 5])
    sparsity_matrix = coo_array((data, (row, col)))
    coupling_transf = FeatureCouplingTransformer(sparsity_matrix)
    transformed_features = coupling_transf.fit_transform(data_matrix_)
    print(coupling_transf.get_get_feature_names_out())
    output: array(['t*t', 't*y', 'x*y', 'x*x'], dtype=object)

    Case 1: Coupling function is provided.

    data_matrix_ = pd.DataFrame([[1,2,3], [4,5,6]], columns = ["t", "x", "y"])
    row  = np.array([0, 0, 1, 1])
    col  = np.array([0, 2, 2, 1])
    data = np.array([4, 5, 7, 5])
    sparsity_matrix = coo_array((data, (row, col)))
    def coup_fun(x,y,i,j,k=0):
        return x-y-k
    coupling_transf = FeatureCouplingTransformer(sp_array_2,
                                           coupling_func= coup_fun,
                                           coupling_namer= lambda x,y,i,j,k : "{}-{}-{}".format(x,y,k),
                                           coupling_func_args={"k":2})
    transformed_features = coupling_transf.fit_transform(data_matrix_)
    print(coupling_transf.get_get_feature_names_out())
    array(['t-t-2', 't-y-2', 'x-y-2', 'x-x-2'], dtype=object)

    """

    def __init__(self, sparsity_matrix=None, coupled_indices_list=None,
                 coupling_func=None, coupling_namer=None,
                 coupling_func_args={}, return_df=False):
        """
        Note that if coupled indices list is not explicitly given to the constructor, a valid sparsity matrix
        from which the coupled indices can be implied should be provided.

        @param sparsity_matrix: Sparsity matrix in the scipy.sparse.coo_array format (preferred over directly
                                providing coupled_indices_list
        @param coupled_indices_list: List of tuples [(i,j)] which shows coupling between factors with indices i and j
        @param coupling_func: Custom function to define coupling between features.  Note that the coupling_func function
                              should have arguments (feature_1_value,feature_2_value, i, j) as the first four arguments.
        @param coupling_namer: Custom function to name the feature corresponding to each coupling. Note that the
                               coupling_namer function should have arguments (feature_1_value,feature_2_value, i, j)
                                as the first four arguments.
        @param coupling_func_args: optional keyword arguments for the coupling_function and coupling_namer functions
        @param return_df: bool flag to output pandas DataFrame instead of numpy array. False by default
        """

        if not coupled_indices_list:
            assert isinstance(sparsity_matrix, coo_array), "FeatureDiffTransformer only support sparsity matrix\
            in the scipy.sparse.coo_array format"
            self.sparsity_matrix = sparsity_matrix
        self.coupled_indices_list = coupled_indices_list
        if not coupling_func:
            self.coupling_func = lambda x, y, i, j: x * y
        else:
            # If coupling function is not given, it is defined as the interaction term feature_1*feature_2
            self.coupling_func = coupling_func

        if not coupling_namer:
            self.coupling_namer = lambda feature_1, feature_2, i, j: "{}*{}".format(feature_1, feature_2)
        else:
            self.coupling_namer = coupling_namer

        self.coupling_func_args = coupling_func_args
        self.return_df = return_df


        self.n_features_in_ = 0
        self.feature_names_in_ = None

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        @param input_features: - If `input_features is None`, then `feature_names_in_` is
                                 used as feature names in. If `feature_names_in_` is not defined,
                                  then the following input feature names are generated:
                                  `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
                                 - If `input_features` is an array-like, then `input_features` must
                                  match `feature_names_in_` if `feature_names_in_` is defined.
            It is recommended that the coupling between features are given using a sparsity matrix
            instead of coupling indices.
        @return: feature_names_out : ndarray of str objects
            Transformed feature names.
        """

        check_is_fitted(self)
        input_features = _check_feature_names_in(self, input_features)

        feature_names = [self.coupling_namer(input_features[i], input_features[j], i, j, **self.coupling_func_args) for
                         i, j in self.coupled_indices_list]

        return np.asarray(feature_names, dtype=object)

    def fit(self, X, y=None):

        self.n_features_in_ = X.shape[1]
        if len(X.columns) > 0:
            self.feature_names_in_ = X.columns
        if not self.coupled_indices_list:  # sparsity matrix gives the coupling indices
            assert max(self.sparsity_matrix.col.max(), self.sparsity_matrix.row.max()) <= self.n_features_in_ - 1, \
                "sparsity matrix has indices out of bound of the number of features"
            # Extracting the indices that has coupling with each other.
            self.coupled_indices_list = list(zip(self.sparsity_matrix.row, self.sparsity_matrix.col))

        return self

    def transform(self, X):
        """Transform data to output the coupled features

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        XP : {ndarray, sparse matrix} of shape (n_samples, NS)
            The matrix of features, where `NS` is the number of non-zero
            connections implied from the sparsity matrix. NS = len(self.get_features_names_out())
        """
        check_is_fitted(self)

        X = self._validate_data(
            X, order="F", dtype=FLOAT_DTYPES, reset=False, accept_sparse=("csr", "csc")
        )
        X_transpose = X.T
        X_coupled = np.vstack([self.coupling_func(X_transpose[i], X_transpose[j], i, j, **self.coupling_func_args)
                               for i, j in self.coupled_indices_list]).T
        if self.return_df:
            return pd.DataFrame(X_coupled, columns=self.get_feature_names_out())

        return X_coupled

"""
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
"""

class AlgModelFinder(BaseEstimator):

    """"
Class that helps with finding algrebraic relationship between features (columns)
 of a data matrix.
 - Several prebuilt model choices like lasso, ridge, elastic net etc.
- Can work with custom models that suppport .fit(), .coef_ methods.
 Simply need to pass the custom model to the constructor.
- Choice to scale columns and scale back the fitted coefficients accordingly.
- Selection of best 'n' models using different metrics. "R2" and "mse" on test data are
prebuilt. Option to pass custom metric object. Can be extended to include other relevant
 metrics as pre-built.

    """
    def __init__(self, model_id="lasso",
                 custom_model=False,
                 custom_model_ob=None,
                 alpha=0.1,
                 fit_intercept=False
                 ):
        self.model_id_dict = {"lasso": linear_model.Lasso,
                              "RR": linear_model.Ridge,
                              "LR": linear_model.LinearRegression}
        if custom_model:
            assert custom_model_ob
        else:
            assert (model_id in self.model_id_dict)

        self.custom_model = custom_model
        self.model_id = model_id
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.is_fit = False
        # {feature: R2_score} obtained from fitting each feature against rest
        self.r2_score_dict = {}
        self.__fitted_models = {}
        self.fitted_models_unscaled = {}
        self.column_scaled = False

        no_contraint_models = {"LR"}
        self.custom_model_ob = custom_model_ob
        if custom_model:
            self.model = custom_model_ob
        elif self.model_id in no_contraint_models:
            # Instantiating the model object
            self.model = self.model_id_dict[self.model_id](fit_intercept=self.fit_intercept)
        else:
            self.model = self.model_id_dict[self.model_id](alpha=self.alpha,
                                                           fit_intercept=self.fit_intercept)
        self.column_scales = None

    def fit_and_score(self, feature_, X_scaled_, feature_to_library_map_):
        possible_library_terms = feature_to_library_map_[feature_]
        X_features = X_scaled_[possible_library_terms]
        y_target = X_scaled_[feature_]

        self.model.fit(X=X_features, y=y_target)
        coefficients = dict(zip(self.model.feature_names_in_, self.model.coef_))
        intercept = self.model.intercept_
        score = self.model.score(X_features, y_target)
        return coefficients, intercept, score

    def fit(self,
            X,
            y=None,
            scale_columns=False,
            center_mean=False,
            features_to_fit = None,
            feature_to_library_map_ ={},
            coupling_matrix = None,
            parallelize = False,
            num_cpu = 4
            ):

        """
        X -> Data matrix (either (n,m) numpy array or pandas DF), where each column represents
             one feature from the candidate library.
        scale_columns -> divide the columns by std to get a unit variance for columns.
        features_to_fit -> List of features to fit against the rest of the library terms
        """
        if self.fit_intercept:
            assert "1" not in X, ("Constant column should not be part of the data set if fit_intercept "
                                  "is set to True")
        self.is_fit = True
        feature_to_library_map = deepcopy(feature_to_library_map_)

        r_2_dict_unsorted = {}
        self.__fitted_models = {}
        self.__fitted_model_intercepts = {}
        self.r2_score_dict = {}
        if scale_columns:
            s_scaler = StandardScaler(with_std=scale_columns, with_mean=center_mean)
            X_scaled = pd.DataFrame(s_scaler.fit_transform(X), columns=s_scaler.feature_names_in_)
            # Making sure constant term is not removed after mean centering to zero
            if center_mean and '1' in X_scaled:
                X_scaled['1'] = 1
            if scale_columns:
                self.column_scaled = True
                self.column_scales = X.std()
                # To avoid division by zero during the scaling step.
                self.column_scales['1'] = 1
        else:
            X_scaled = X
        if not features_to_fit:
            features_to_fit = X_scaled.columns


        for feature in features_to_fit:
            #If feature to library map is not given, all the members of the universal
            # candidate library will be fit against the feature.
            if feature not in feature_to_library_map:
                possible_library_terms = X_scaled.columns.drop(feature, errors='ignore')
            else:
                # print(feature_to_library_map)
                # print(feature, "-reached here")
                possible_library_terms = feature_to_library_map[feature]
                assert set(possible_library_terms) <= set(X_scaled.columns), \
                    ("library terms for feature {} from feature_to_library_map is not found"
                     "in the universal X library")
            feature_to_library_map[feature] = possible_library_terms


            # self.model.fit(X=X_scaled[possible_library_terms], y=X_scaled[feature])
            # self.__fitted_models[feature] = dict(zip(self.model.feature_names_in_, self.model.coef_))
            # self.__fitted_model_intercepts[feature] = self.model.intercept_
            # # self.model.score(X=X_scaled[possible_library_terms],
            # #                                               y=X_scaled[feature])
            # r_2_dict_unsorted[feature] = self.model.score(X=X_scaled[possible_library_terms],
            #                                               y=X_scaled[feature])

            #r_2_dict_unsorted = {feature: self.model.fit_score(X=X_scaled.drop([feature], axis=1),
                                                          # y=X_scaled[feature]) for feature in X_scaled}

        # Using dictionary comprehensions to store model details and R² scores

        # res = Parallel(n_jobs=20)(delayed(dummy)(x) for x in range(100))
        if parallelize:
            combined_fit_results_list = Parallel(n_jobs=num_cpu,require='sharedmem')(delayed(self.fit_and_score)
                                                                 (feature, X_scaled, feature_to_library_map)
                                                                 for feature in features_to_fit )
            combined_fit_results = dict(zip(features_to_fit, combined_fit_results_list))
        else:
            combined_fit_results = {feature: self.fit_and_score(feature, X_scaled, feature_to_library_map) for feature in features_to_fit}

        # Extracting separate dictionaries for coefficients, intercepts, and R² scores
        self.__fitted_models = {feature: result[0] for feature, result in combined_fit_results.items()}
        self.__fitted_model_intercepts = {feature: result[1] for feature, result in combined_fit_results.items()}
        r_2_dict_unsorted = {feature: result[2] for feature, result in combined_fit_results.items()}
        self.r2_score_dict = dict(sorted(r_2_dict_unsorted.items(), key=operator.itemgetter(1)))

        # feature_to_library_map = {}
        return self

    def best_models(self, num=0, X_test=None, metric="r2",
                    scale_coef=True):
        """
        If X_test == None, the r_2 scores already stored from the underlying model will be used for
        selection.
        Best models are selected according to best metric value (eg. high R2 or low mse)
        """
        assert self.is_fit, "Models need to be fit to data first"
        sorted_metric_series = []
        if num < 1:  # Output all possible models
            num = len(self.__fitted_models)
        metric_set = {"r2", "mse"}
        assert metric in metric_set, "metric {} is not supported. Only {} is supported".format(metric, metric_set)
        if metric == "r2":  # Use the already computed r_2 scores for selection
            r_2_list = list(zip(list(self.r2_score_dict.keys()),
                                list(self.r2_score_dict.values())))
            sorted_r2_dict = dict(sorted(self.r2_score_dict.items(), key=operator.itemgetter(1), reverse=True))
            sorted_metric_series = pd.Series(sorted_r2_dict)
            # sorted_metric_list = sorted(r_2_list, key = lambda x: x[1], reverse=True)

        if metric == "mse":
            assert type(X_test) == pd.DataFrame and len(X_test) > 0, "Test data test needed for calculating mse"
            predicted_df = self.predict_features(X_test=X_test,
                                                 feature_list=self.__fitted_models.keys(),
                                                 scale_coef=scale_coef)
            mse_series = ((predicted_df - X_test) ** 2).mean()
            sorted_metric_series = mse_series.sort_values(na_position='last')

        fitted_models = self.get_fitted_models(scale_coef=scale_coef)
        best_model_dict = {feature: fitted_models[feature]
                           for feature in sorted_metric_series[:num].index}
        best_model_df = pd.DataFrame(best_model_dict)
        metric_label = metric + "- metric"
        best_model_df.loc[metric_label] = {feature: metric_value
                                           for feature, metric_value in sorted_metric_series[:num].items()}
        return best_model_df

    def get_fitted_models(self, scale_coef=True):
        """
        for column scaled data matrix, the scaled coefficients for lhs = Summatiion(coef * term)  is
        calculated as coef * (std_of_lhs/std_term).
        """
        assert self.is_fit, "Models need to be fit to data first"
        if scale_coef and self.column_scaled:
            unscaled_fitted_models = self.__fitted_models
            scaled_fitted_model_coef = {
                lib_term: {term: coef * (self.column_scales[lib_term] / self.column_scales[term])
                           for term, coef in model_coefs.items()}
                for lib_term, model_coefs in unscaled_fitted_models.items()
                }
            return scaled_fitted_model_coef
        else:
            return self.__fitted_models

    def get_fitted_intercepts(self, scale_coef=True):
        """
        for column scaled data matrix, the intercept is also scaled as std_of_lhs * intercept
        """
        assert self.is_fit, "Models need to be fit to data first"
        if scale_coef and self.column_scaled:
            unscaled_intercepts = self.__fitted_model_intercepts
            scaled_fitted_model_intercepts = { lib_term: intercept_ * (self.column_scales[lib_term])
                for lib_term, intercept_ in unscaled_intercepts.items()}
            return scaled_fitted_model_intercepts
        else:
            return self.__fitted_model_intercepts

    def predict_features(self, X_test, feature_list, scale_coef=True):
        """
        Function to predict the value of each feature in feature_list, where each feature is a
         linear function of columns of X_test.
        :param X_test: Data matrix, preferably in pd.DataFrame format.
        param feature_list: list of features to be predicted. eg. ["E", "ES"]
        :param scale_coef: if True, coefficients are scaled back to reflect the
         initial column scaling of data during fitting.
        :return: pd.Dataframe of the same size as X_test
        """
        assert self.is_fit, "Models need to be fit to data first"
        assert set(feature_list) <= set(self.__fitted_models.keys()), ("Feature list should be a subset"
                                                                       " of features initially fitted")
        fitted_models = self.get_fitted_models(scale_coef=scale_coef)
        fitted_intercepts = self.get_fitted_intercepts()
        prediction_df = pd.DataFrame(columns=feature_list)
        for feature in feature_list:
            coef_features = fitted_models[feature]
            assert set(coef_features.keys()) <= set(X_test.columns), (
                "Data matrix X_test doesnot have all the feature columns"
                "required for fitting feature {}".format(feature))
            prediction_df[feature] = sum(coef_value * X_test[coef_feat] for coef_feat,
            coef_value in coef_features.items()) + fitted_intercepts[feature]

        return prediction_df

    def compare_models(self, true_model_df):
        """
        Method to compare the accuray of fitted models with true model structure. This method calls the
        utility function compare_models_(self.best_models(), true_model_df) to compare the best models
        after fitting with the true model structure. The true model dataframe should have the same column labels,
        index labels, and shape as the models from self.best_models() .

        @param true_model_df:  pd.DataFrame with columns = [LHS of model] index = terms in the RHS of model.
        """
        assert self.is_fit, "Models need to be fit to data first"
        return compare_models_(self.best_models(), true_model_df)



"""
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
"""
class sequentialThLin(MultiOutputMixin, RegressorMixin):
    """
    Model-agnostic implementation of sequential thresholdng to impose l0 sparsity.
    Current support for popular models like linear model with l1 and l2 regularizers, and their combination (ElasticNet). Also has the feature to pass in  custom models from the user.
    """

    def __init__(
            self,
            model_id="RR",
            custom_model=False,
            custom_model_ob=None,
            custom_model_arg=None,
            alpha=1.0,
            l1_ratio=0.5,
            coef_threshold=0.1,
            fit_intercept=False,
            precompute=False,
            max_iter_thresh=500,
            max_iter_optimizer=1000,
            copy_X=True,
            tol=1e-4,
            warm_start=False,
            positive=False,
            random_state=None,
            selection="cyclic",
    ):
        self.model_id = model_id
        self.custom_model = custom_model
        self.custom_model_arg = custom_model_arg
        if custom_model:
            assert custom_model_ob
            assert custom_model_arg
        self.coef_threshold = coef_threshold
        self.max_iter_thresh = max_iter_thresh
        self.max_iter_optimizer = max_iter_optimizer
        self.alpha = alpha,
        self.l1_ratio = l1_ratio,
        self.fit_intercept = fit_intercept,
        self.precompute = precompute,
        self.copy_X = copy_X,
        self.tol = tol,
        self.warm_start = warm_start,
        self.positive = positive,
        self.random_state = random_state,
        self.selection = selection,

        self.input_arg_dict = {"alpha": alpha,
                               "l1_ratio": l1_ratio,
                               "fit_intercept": fit_intercept,
                               "precompute": precompute,
                               "max_iter": max_iter_optimizer,
                               "copy_X": copy_X,
                               "tol": 1e-4,
                               "warm_start": False,
                               "positive": False,
                               "random_state": None,
                               "selection": "cyclic"}

        self.model_id_dict = {"lasso": linear_model.Lasso,
                              "RR": linear_model.Ridge,
                              "LR": linear_model.LinearRegression,
                              "EN": linear_model.ElasticNet}
        assert (model_id in self.model_id_dict)

        no_constrain_model = {"LR"}
        elastic_models = {"EN"}

        # Instantiating model objects. Note that currently only the basic arguments are passed to the constructor (init), but more flexibility cn be achieved by passing more arguments from the self.__init__ to the __init__ of the appropriate models.
        if self.custom_model:
            self.model = custom_model_ob(**self.custom_model_arg)
            self.model_for_score = custom_model_ob(**self.custom_model_arg)
        elif self.model_id in no_constrain_model:
            if fit_intercept:
                self.model = self.model_id_dict[self.model_id](fit_intercept=True)
                self.model_for_score = self.model_id_dict[self.model_id](fit_intercept=True)
            else:
                self.model = self.model_id_dict[self.model_id](fit_intercept=False)
                self.model_for_score = self.model_id_dict[self.model_id](fit_intercept=False)
        elif self.model_id in elastic_models:
            arg_input = self.input_arg_dict
            self.model = self.model_id_dict[self.model_id](**arg_input)
            self.model_for_score = self.model_id_dict[self.model_id](**arg_input)
        else:
            arg_input = self.input_arg_dict
            del arg_input["l1_ratio"], arg_input["precompute"], arg_input["warm_start"], arg_input["selection"]
            self.model = self.model_id_dict[self.model_id](**arg_input)
            self.model_for_score = self.model_id_dict[self.model_id](**arg_input)

        self.is_fit = False

        self.coef_history_df = pd.DataFrame()
        self.coef_history_df_pre_thesh = pd.DataFrame()
        self.intercept_history_df = pd.DataFrame()

        self.coef_ = None
        self.feature_names_in_ = None
        self.intercept_ = 0.0

    def fit(self, X, y=None, solver="auto"):

        # num_features = X.columns.shape[0]
        # coef_ind = np.zeros(num_features)
        self.is_fit = True
        self.coef_history_df = pd.DataFrame(columns=X.columns)
        self.coef_history_df_pre_thesh = pd.DataFrame(columns=X.columns)
        self.intercept_history_df = pd.DataFrame(columns=["1"])

        # old_sparse_index = [False] * num_features
        non_sparse_columns = X.columns
        X_ind = X[non_sparse_columns]
        for ind in range(self.max_iter_thresh):
            self.model.fit(X=X_ind, y=y)
            coef_ind = self.model.coef_
            self.coef_history_df_pre_thesh.loc[ind] = dict(zip(self.model.feature_names_in_, self.model.coef_))
            # non_sparse_index = np.ones(coef_ind.shape)
            sparse_index = abs(coef_ind) < self.coef_threshold
            coef_ind[sparse_index] = 0.0
            self.coef_history_df.loc[ind] = dict(zip(self.model.feature_names_in_, coef_ind))
            self.intercept_history_df.loc[ind]= {"1": self.model.intercept_}

            non_sparse_columns = non_sparse_columns[~sparse_index]
            if all(sparse_index):  # If all the coef go to zero after thresholding
                warnings.warn("All coefficients fell below threshold {}, please"
                              " lower threshold".format(self.coef_threshold))
                break

            if set(X_ind.columns) == set(non_sparse_columns):
                print("Sequential threshold converged in {} iterations".format(ind))
                break
            else:
                X_ind = X[non_sparse_columns]

        final_coefs = self.coef_history_df.iloc[-1].fillna(0.0)
        self.coef_ = final_coefs.values
        self.intercept_ = self.intercept_history_df.iloc[-1]["1"]
        # self.score = self.model.score
        self.feature_names_in_ = np.array(X.columns)

        return self

    def score(self, X, y, sample_weight=None):
        assert self.is_fit
        final_features = self.coef_history_df.iloc[-1].dropna().index
        if len(final_features) > 0:
            self.model_for_score.fit(X=X[final_features], y=y)
            score_ = self.model_for_score.score(X=X[final_features], y=y)
            return score_
        else:
            return 0