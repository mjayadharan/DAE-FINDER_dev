from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import warnings
import operator


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
        self.is_fit = True
        r_2_dict_unsorted = {}
        self.__fitted_models = {}
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

        for feature in X_scaled:
            self.model.fit(X=X_scaled.drop([feature], axis=1), y=X_scaled[feature])
            self.__fitted_models[feature] = dict(zip(self.model.feature_names_in_, self.model.coef_))
            r_2_dict_unsorted[feature] = self.model.score(X=X_scaled.drop([feature], axis=1),
                                                          y=X_scaled[feature])
        self.r2_score_dict = dict(sorted(r_2_dict_unsorted.items(), key=operator.itemgetter(1)))

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
        prediction_df = pd.DataFrame(columns=feature_list)
        for feature in feature_list:
            coef_features = fitted_models[feature]
            assert set(coef_features.keys()) <= set(X_test.columns), (
                "Data matrix X_test doesnot have all the feature columns"
                "required for fitting feature {}".format(feature))
            prediction_df[feature] = sum(coef_value * X_test[coef_feat] for coef_feat,
            coef_value in coef_features.items())

        return prediction_df


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
        elif self.model_id in no_constrain_model:
            self.model = self.model_id_dict[self.model_id](fit_intercept=self.fit_intercept)
        elif self.model_id in elastic_models:
            arg_input = self.input_arg_dict
            self.model = self.model_id_dict[self.model_id](**arg_input)
        else:
            arg_input = self.input_arg_dict
            del arg_input["l1_ratio"], arg_input["precompute"], arg_input["warm_start"], arg_input["selection"]

            self.model = self.model_id_dict[self.model_id](**arg_input)
        self.is_fit = False

        self.coef_history_df = pd.DataFrame()
        self.coef_history_df_pre_thesh = pd.DataFrame()

        self.coef_ = None
        self.feature_names_in_ = None
        self.score = None

    def fit(self, X, y=None, solver="auto"):

        # num_features = X.columns.shape[0]
        # coef_ind = np.zeros(num_features)
        self.coef_history_df = pd.DataFrame(columns=X.columns)
        self.coef_history_df_pre_thesh = pd.DataFrame(columns=X.columns)

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
        self.score = self.model.score
        self.feature_names_in_ = np.array(X.columns)

        return self