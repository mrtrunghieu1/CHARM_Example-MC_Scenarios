import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import TweedieRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR, NuSVR
from sklearn.multioutput import RegressorChain
from sklearn.tree import DecisionTreeRegressor


def learning_algorithm(X_train, Y_train, X_test, flag_algo, flag_combine):
    clf, algo_name = training(X_train, Y_train, flag_algo, flag_combine)
    multiple_predictors = testing(clf, X_test)
    return multiple_predictors, algo_name + "_" + flag_combine


def training(X_train, Y_train, flag, combine_type="multi_output"):
    """
    :param combine_type: "multi_output" or "chain"
    """

    algo_clf = None
    if flag == 0:
        algo_clf = KernelRidge()
    elif flag == 1:
        algo_clf = LinearSVR()
    elif flag == 2:
        algo_clf = SVR()
    elif flag == 3:
        algo_clf = NuSVR()
    elif flag == 4:
        algo_clf = LinearRegression()
    elif flag == 5:
        algo_clf = Ridge()
    elif flag == 6:
        algo_clf = Lasso()
    elif flag == 7:
        algo_clf = ElasticNet()
    elif flag == 8:
        algo_clf = Lars()
    elif flag == 9:
        algo_clf = LassoLars()
    elif flag == 10:
        algo_clf = BayesianRidge()
    elif flag == 11:
        algo_clf = SGDRegressor(loss="squared_loss")
    elif flag == 12:
        algo_clf = SGDRegressor(loss="huber")
    elif flag == 13:
        algo_clf = SGDRegressor(loss="epsilon_insensitive")
    elif flag == 14:
        algo_clf = KNeighborsRegressor()
    elif flag == 15:
        algo_clf = GaussianProcessRegressor()
    elif flag == 16:
        algo_clf = DecisionTreeRegressor()
    elif flag == 17:
        algo_clf = RandomForestRegressor(n_estimators=500)
    elif flag == 18:
        algo_clf = ExtraTreesRegressor(n_estimators=500)
    elif flag == 19:
        algo_clf = BaggingRegressor(n_estimators=500)
    elif flag == 20:
        algo_clf = AdaBoostRegressor(n_estimators=500)
    elif flag == 21:
        algo_clf = GradientBoostingRegressor(n_estimators=500)
    elif flag == 22:
        algo_clf = HistGradientBoostingRegressor()

    if combine_type == "multi_output":
        clf = MultiOutputRegressor(algo_clf).fit(X_train, Y_train)
    elif combine_type == "chain":
        clf = RegressorChain(algo_clf).fit(X_train, Y_train)
    else:
        raise Exception("Unimplemented!!!")

    return clf, algo_clf.__class__.__name__


def testing(clf, test):
    multiple_predictors = clf.predict(test)
    return multiple_predictors
