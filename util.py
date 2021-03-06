import pandas as pd
import numpy as np
import os
import json

# from data_helper import output

# def extract_feature(sheet, num_samples):
#     matrix_numpy = []
#     for i in range(num_sample):
#         matrix_numpy.append(sheet.loc[i,:].to_numpy())
#     return matrix_numpy

def write_output(name_algorithm, excel_file_path, data):
    df = pd.DataFrame(data)
    # print("output",output)
    # print("name_algorithm",name_algorithm)
    result_folder = excel_file_path + "{}".format(name_algorithm)
    # print("result_folder",result_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    df.to_excel(os.path.join(result_folder, name_algorithm + '.xlsx'), index=False)

def write_evaluation(name_evaluation, name_algorithm, metric_folder, data):
    result_folder = metric_folder + "/{}".format(name_algorithm)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    txt_file = result_folder + "/{}".format(name_evaluation)
    # if not os.path.exists(txt_file):
    #     os.makedirs(txt_file)
    with open(txt_file, 'w') as outfile:
        json.dump(data, outfile)

# def check_null_value(df,i):
#     count = 0
#     for j in range(4500):
#         if(np.isinf(df[i][j]) or np.isnan(df[i][j])):
#             count+=1
#     return count

# def sum_caculation(df,i):
#     df[i][df[i] == np.inf] = 0
#     sum_col = df[i].sum()
#     return sum_col

def AEE_caculation(c,index_col):
    total = sum([c[:,index_col][i] for i in range(len(c[:,index_col])) if c[:,index_col][i] != np.inf and not np.isnan(c[:,index_col][i])])
    count = sum([1 for i in range(len(c[:, index_col])) if c[:,index_col][i] != np.inf and not np.isnan(c[:,index_col][i])])
    if count != 0:
        AEE = 1/count * total
    else:
        AEE = "null"
    return AEE

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)