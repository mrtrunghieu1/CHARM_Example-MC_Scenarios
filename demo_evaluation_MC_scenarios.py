import sys
import os
import util
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from data_helper import excel_file, excel_path, worksheets, num_sample, output, eval_path
from training import learning_algorithm
from util import write_output, write_evaluation
from metrics_evaluation import metric_evaluation, AEE_metric

try:
    from_id = int(sys.argv[1])
    to_id = int(sys.argv[2])
except:
    from_id = 0
    to_id = len(excel_file)

# Choose some multioutput regression
algo_regression_flag = range(0, 23)  # [0, 1, 3, 6, 9, 16, 22]
metric_evaluation_flag = range(0, 5)
combine_flag = ["multi_output", "chain"]


for i_file in range(from_id, to_id):
    print('file ', i_file)
    excel_file_path = os.path.join(excel_path, excel_file[i_file])
    xls = pd.ExcelFile(excel_file_path)

    D_train = None
    D_test = None

    for worksheet in worksheets:
        df = pd.read_excel(xls, worksheet)
        cur_train = df.iloc[:num_sample, :].to_numpy()
        cur_test = df.iloc[num_sample:, :].to_numpy()

        D_train = cur_train if D_train is None else np.concatenate((D_train, cur_train), 0)
        D_test = cur_test if D_test is None else np.concatenate((D_test, cur_test), 0)

    X_train = D_train[:, 0:13]
    Y_train = D_train[:, 13:]

    X_test = D_test[:, 0:13]
    Y_test = D_test[:, 13:]

    excel_result_path = output + '{}\\'.format(excel_file[i_file])
    if not os.path.exists(excel_result_path):
        os.makedirs(excel_result_path)
    
    metric_folder = eval_path + '{}\\'.format(excel_file[i_file])
    if not os.path.exists(metric_folder):
        os.makedirs(metric_folder)
    # Training and predict data
    for flag_algo in algo_regression_flag:
        for flag_combine in combine_flag:
            predict_output, algo_name = learning_algorithm(X_train, Y_train, X_test, flag_algo, flag_combine)
            result_output = np.concatenate((predict_output, Y_test, abs(predict_output-Y_test)), axis=1)  # Save y_predict, truth_label
            # pickle.dump(result_output, open("test.p","wb"))
            '''Evaluation metrics'''
            for flag_i in metric_evaluation_flag:
                result_loss, name_eval = metric_evaluation(flag_i, Y_test, predict_output)
                write_evaluation(name_eval, algo_name, metric_folder, result_loss)
            AEE_value = AEE_metric(predict_output, Y_test)
            write_evaluation('AEE', algo_name, metric_folder, AEE_value)
            write_output(algo_name, excel_result_path, result_output)
