from math import sqrt
from numpy import inf
import pandas as pd
from util import AEE_caculation
import numpy as np
import timeit

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score

def metric_evaluation(flag_evaluation, y_true, y_pred):
    metric_eval = None
    if flag_evaluation == 0:
        result_loss = mean_squared_error(y_true, y_pred)              #MSE
        name_eval = mean_squared_error.__name__
    elif flag_evaluation == 1:
        result_loss = sqrt(mean_squared_error(y_true, y_pred))        #Root MSE  
        name_eval = 'root_mean_squared_error'
    elif flag_evaluation == 2:
        result_loss = mean_absolute_error(y_true, y_pred)             #MAE
        name_eval = mean_absolute_error.__name__
    elif flag_evaluation == 3:
        result_loss = explained_variance_score(y_true, y_pred)      
        name_eval = explained_variance_score.__name__   
    # elif flag_evaluation == 4:
    #     result_loss = max_error(y_true, y_pred)
    #     name_eval = max_error.__name__
    # elif flag_evaluation == 4:
    #     result_loss = mean_squared_log_error(y_true, y_pred)
    #     name_eval = mean_squared_log_error.__name__
    elif flag_evaluation == 4:
        result_loss = r2_score(y_true, y_pred)
        name_eval = r2_score.__name__
    else:
        raise Exception("Not metrics evaluation!!!")
    return result_loss, name_eval

def AEE_metric(y_pred, y_true):
    start = timeit.default_timer()
    loss_abs = abs(y_pred - y_true)
    estimated_value = loss_abs/ y_true
    AEE = {}                                      #13,1
    for i in range(estimated_value.shape[1]):
        AEE_value = AEE_caculation(estimated_value, i)
        AEE.update({'AEE_{}'.format(i):AEE_value})
    stop = timeit.default_timer()
    run_time = stop - start
    AEE.update({"run_time":run_time})
    return AEE

def caculating_AEE_metrics(estimate, grouth_truth):
    start = timeit.default_timer()
    AEE_total = {}
    for i in range(grouth_truth.shape[1]):
        grouth_truth_values = np.array([grouth_truth[:,i][j] for j in range(grouth_truth.shape[0]) if grouth_truth[:,i][j] != 0])
        abs_loss = np.array([abs(estimate[:,i][j] - grouth_truth[:,i][j]) for j in range(grouth_truth.shape[0]) if grouth_truth[:,i][j] != 0])
        N = len(grouth_truth_values)
        AEE_value = 1/N * sum(abs_loss/grouth_truth_values) if N != 0 else 'null'
        AEE_total.update({"AEE_{}".format(i):AEE_value})
    stop = timeit.default_timer()
    run_time = stop - start
    AEE_total.update({"run_time":run_time})
    return AEE_total


