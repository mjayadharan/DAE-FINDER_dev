from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import warnings
import operator
from copy import deepcopy

from scipy.integrate import odeint
from scipy import interpolate

import sympy

import matplotlib.pyplot as plt


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


def smooth_data(data_matrix, domain_var="t", s_param=None, noise_perc=0, derr_order=1, eval_points=[]):
    assert domain_var in data_matrix, "domain variable not found in the data matrix"

    data_t = data_matrix[domain_var]
    num_time_points = len(data_matrix)
    find_s_param = s_param is None

    if len(eval_points) == 0:
        eval_points = np.linspace(data_t.iloc[0], data_t.iloc[-1], 10 * num_time_points)
    t_eval_new = eval_points

    data_matrix_ = data_matrix.drop(domain_var, axis=1)
    data_matrix_std = data_matrix_.std()

    data_matrix_smooth = pd.DataFrame(t_eval_new, columns=[domain_var])
    for feature in data_matrix_:
        if find_s_param:
            # smoothing parameter: when equal weightage: num_data_points * std of data
            s_param_ = num_time_points * (0.01 * noise_perc * data_matrix_std[feature]) ** 2
            # print(feature, s_param)
        else:
            s_param_ = s_param

        tck = interpolate.splrep(data_t, data_matrix_[feature], s=s_param_)
        # print(s_param)
        for der_ind in range(derr_order + 1):
            smoothed_data = interpolate.splev(t_eval_new, tck, der=der_ind)
            data_matrix_smooth[der_label(feature, der_ind)] = smoothed_data

    return data_matrix_smooth


def remove_paranth_from_feat(feature_list):
    """
    Utility function to remove the paranthesis from the name of the feature.
    :param feature_list: ["[E]", "[ES]"]
    :return: ["E", "ES"]
    """
    return [feat.replace("[", "").replace("]", "") for feat in feature_list]


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

    def fit_and_score(self, feature, X):
        X_features = X.drop([feature], axis=1)
        y_target = X[feature]

        self.model.fit(X=X_features, y=y_target)
        coefficients = dict(zip(self.model.feature_names_in_, self.model.coef_))
        intercept = self.model.intercept_
        score = self.model.score(X_features, y_target)

        return (coefficients, intercept, score)

    def fit(self,
            X,
            y=None,
            scale_columns=False,
            center_mean=False
            ):
        """
        X -> Data matrix (either (n,m) numpy array or pandas DF), where each column represents
             one feature from the candidate library.
        scale_columns -> divide the columns by std to get a unit variance for columns.
        """
        if self.fit_intercept:
            assert "1" not in X, ("Constant column should not be part of the data set if fit_intercept "
                                  "is set to True")
        self.is_fit = True
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

        # Using dictionary comprehensions to store model details and R² scores
        results = {feature: self.fit_and_score(feature, X_scaled) for feature in X_scaled}

        # Extracting separate dictionaries for coefficients, intercepts, and R² scores
        self.__fitted_models = {feature: result[0] for feature, result in results.items()}
        self.__fitted_model_intercepts = {feature: result[1] for feature, result in results.items()}
        r_2_dict_unsorted = {feature: result[2] for feature, result in results.items()}

        # Sorting the R² scores dictionary
        self.r2_score_dict = dict(sorted(r_2_dict_unsorted.items(), key=operator.itemgetter(1), reverse=True))

        # for feature in X_scaled:
        #     self.model.fit(X=X_scaled.drop([feature], axis=1), y=X_scaled[feature])
        #     self.__fitted_models[feature] = dict(zip(self.model.feature_names_in_, self.model.coef_))
        #     self.__fitted_model_intercepts[feature] = self.model.intercept_
        #     self.model.score(X=X_scaled.drop([feature], axis=1),
        #                                                   y=X_scaled[feature])
        #     r_2_dict_unsorted[feature] = self.model.score(X=X_scaled.drop([feature], axis=1),
        #                                                   y=X_scaled[feature])
        #
        # self.r2_score_dict = dict(sorted(r_2_dict_unsorted.items(), key=operator.itemgetter(1)))
        return self

    def fit_feature(feature):
        return((coef, intercepts, r2_score))


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
            self.model = self.model_id_dict[self.model_id](fit_intercept=self.fit_intercept)
            self.model_for_score = self.model_id_dict[self.model_id](fit_intercept=self.fit_intercept)
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