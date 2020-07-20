from math import sqrt
from numpy import inf
import pandas as pd
from util import check_null_value, sum_caculation

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
    grounth_truth_inverst = abs(1/y_true)
    loss_abs = abs(y_pred - y_true)
    estimated_value = grounth_truth_inverst * loss_abs
    AEE = {}                                      #13,1
    df = pd.DataFrame(estimated_value)
    for i in range(estimated_value.shape[1]):
        na_count = check_null_value(df,i)        #Count number null value(na)
        N = estimated_value.shape[0] - na_count
        if N != 0:
            sum_col = sum_caculation(df,i)
            AEE_value = 1/N * sum_col
            AEE.update({'AEE_{}'.format(i):AEE_value})
        else:
            AEE.update({'AEE_{}'.format(i):"null"})
    return AEE



