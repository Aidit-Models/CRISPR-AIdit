# -*-coding: utf-8 -*-
'''
@author: yan jianfeng
@license: python3.7.3
@contact: yanjianfeng@westlake.edu.cn
@software: pycharm
@file: data.py
@time: 2021/4/1 20:42
@desc: get features
'''
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def is_Exist_file(path):
    import os
    if os.path.exists(path):
        os.remove(path)


def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def walk(path):
    import os
    input_path_list = []
    if not os.path.exists(path):
        return -1
    for root, dirs, names in os.walk(path):
        for filename in names:
            input_path = os.path.join(root, filename)
            input_path_list.append(input_path)
    return input_path_list


# get best_checkpoint_path file
def get_best_checkpoint_path(path):
    file_list = walk(path)
    epoch_dict = {}
    for file in file_list:
        file_p = file.split('-')
        file_p = [s for s in file_p if len(s) != 0]
        epoch = int(file_p[-5])
        epoch_dict[epoch] = file
    epoch_max = max(list(epoch_dict.keys()))
    file_max = epoch_dict[epoch_max]
    return file_max


def best_5_epoches_model(model_path_list, selected_epoch_n=5):
    model_epoch_dict = {}
    for model_path in model_path_list:
        model_file = model_path.split('/')[-1]
        model_count = int(model_file.split('-')[-5])
        model_epoch_dict[model_count] = model_path
    # the highest 5
    model_epoch_list = list(model_epoch_dict.keys())
    model_epoch_list.sort(reverse=True)
    selected_epoches = model_epoch_list[:selected_epoch_n]
    selected_epoches.sort()
    selected_model_path_list = [model_epoch_dict[model_epoch] for model_epoch in selected_epoches]
    return selected_model_path_list


# 深度学习输入1： 序列特征
###########################################
# 生成 Seequence 数据
def find_all(sub, s):
    index = s.find(sub)
    feat_one = np.zeros(len(s))
    while index != -1:
        feat_one[index] = 1
        index = s.find(sub, index + 1)
    return feat_one


# 获取单样本序列数据
def obtain_each_seq_data(seq):
    A_array = find_all('A', seq)
    G_array = find_all('G', seq)
    C_array = find_all('C', seq)
    T_array = find_all('T', seq)
    one_sample = np.array([A_array, G_array, C_array, T_array])
    return one_sample


# 获取序列数据
# 参数说明：
# data：输入的数据，要求含有gRNA列名，该列为原始序列值：target（20） + pam (3) + down (5)
# y: 输入的编辑效率list，对应 data
# layer: 为输出层的label， 用于调整输出数据的shape，可取值['1D', '2D']
# 输出：特征数据 x_data
def obtain_Sequence_data(data, layer_label='1D'):
    x_data = []
    for i, row in data.iterrows():
        seq = row['gRNASeq']
        one_sample = obtain_each_seq_data(seq)
        if layer_label == '1D':  # 用于LSTM or Conv1D, shape=(sample, step, feature)
            one_sample_T = one_sample.T
            x_data.append(one_sample_T)
        else:
            x_data.append(one_sample)
    x_data = np.array(x_data)
    # y = np.array(y)
    if layer_label == '2D':  # 用于 Conv2D shape=(sample, rows, cols, channels)
        x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)
        # y = y.reshape(y.shape[0], 1)
        print('Conv2D: shape=(sample, rows, cols, channels)')
    else:
        print('LSTM or Conv1D: shape=(sample, step, feature)')
    # y = y.astype('float32')
    x_data = x_data.astype('float32')
    print('After transformation, x_data.shape:', x_data.shape)
    return x_data


# get model data
def main_model_data(read_train_data_path, read_val_data_path, ycol, layer_label='1D'):
    train_data = pd.read_csv(read_train_data_path, sep='\t')
    val_data = pd.read_csv(read_val_data_path, sep='\t')
    train_data.rename(columns={'gRNASeq(up20bp+target20bp+pam3bp+down20bp)': 'gRNASeq', ycol: 'regressor_target'}, inplace=True)
    val_data.rename(columns={'gRNASeq(up20bp+target20bp+pam3bp+down20bp)': 'gRNASeq', ycol: 'regressor_target'}, inplace=True)
    # get model data
    # for training data
    x_train = obtain_Sequence_data(train_data, layer_label)
    x_train = x_train.astype('float32')
    y_train = train_data['regressor_target']
    y_train = np.array(y_train)
    y_train = y_train.astype('float32')
    # for validation data
    x_val = obtain_Sequence_data(val_data, layer_label)
    x_val = x_val.astype('float32')
    y_val = val_data['regressor_target']
    y_val = np.array(y_val)
    y_val = y_val.astype('float32')
    return x_train, y_train, x_val, y_val


