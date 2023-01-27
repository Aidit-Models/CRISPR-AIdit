# -*-coding: utf-8 -*-
"""
@author: yan jianfeng
@license: python3.7.3
@contact: yanjianfeng@westlake.edu.cn
@software: pycharm
@file: main_hyperopt.py.py
@time: 2021/4/3 15:52
@desc: 确定参数后训练模型
"""
import time
import math
import tensorflow as tf
from data import *
from models import *
import warnings
warnings.filterwarnings('ignore')


opter = ""


# LR decay
def help_lr_decay(opter):
    if opter == 'Nadam':
        initial_lrate = 0.002
    else:
        initial_lrate = 0.001
    return initial_lrate


def step_decay(epoch):
    import math
    drop = 0.5
    epochs_drop = 50
    initial_lrate = help_lr_decay(opter)
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch)/epochs_drop))
    lrate = max(lrate, 1e-5)
    return lrate


# custom function
##########################################################################
from scipy.stats import spearmanr
def get_spearman_rankcor(y_true, y_pred):
    return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                                       tf.cast(y_true, tf.float32)], Tout=tf.float32))


def spearman(y_true, y_pred):
    import pandas as pd
    y_true = y_true.reshape(y_true.shape[0])
    y_pred = y_pred.reshape(y_true.shape[0])
    sp = pd.Series(y_pred).corr(pd.Series(y_true), method='spearman')
    return sp


# 均方差
def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None
##########################################################################


def model_fit(x_train, y_train, x_val, y_val, model_label, params,
              epochs, early_stopping_patience, save_model_path):
    # intial model
    model = selecting_deeplearning(model_label)(params)
    # compile
    model.compile(loss='mse', optimizer=params['optimizer'], metrics=['mse', get_spearman_rankcor])
    # checkpoint
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
    checkpoint_file = save_model_path
    checkpoint = ModelCheckpoint(filepath=checkpoint_file,
                                 monitor='val_get_spearman_rankcor',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    # LearningRateSchedule
    global opter
    opter = params['optimizer']
    lrate = LearningRateScheduler(step_decay)
    # early stopping
    early_stopping = EarlyStopping(monitor='val_get_spearman_rankcor',
                                   patience=early_stopping_patience,
                                   verbose=0,
                                   mode='max')
    callbacks_list = [checkpoint, lrate, early_stopping]
    # fit
    try:
        model.fit(x_train, y_train,
                  batch_size=params['batch_size'],
                  epochs=epochs,
                  validation_data=(x_val, y_val),
                  callbacks=callbacks_list,
                  shuffle=True,
                  verbose=0)
    except OSError as e:
        print(e)
        pass


# 主函数
def main(read_train_data_path, read_val_data_path, ycol, save_model_dir,
         model_label='BiLSTM', epochs=500, early_stopping_patience=20):
    x_train, y_train, x_val, y_val = main_model_data(read_train_data_path, read_val_data_path, ycol)
    mkdir(save_model_dir)
    save_params_hyperopt_log = save_model_dir + '/summary_val_performance.log'
    is_Exist_file(save_params_hyperopt_log)
    params = selection_model_parameter(model_label)
    # training
    save_model_path = save_model_dir + '/%s-best_weights-improvement-{epoch:03d}-train-{get_spearman_rankcor:.5f}-' \
                                       'test-{val_get_spearman_rankcor:.5f}.hdf5' % model_label
    model_fit(x_train, y_train, x_val, y_val, model_label, params,
              epochs, early_stopping_patience, save_model_path)


if __name__ == "__main__":
    import sys
    main_dir, read_train_data_path, read_val_data_path, ycol, save_model_dir = sys.argv[1:]
    os.chdir(main_dir)
    # read_train_data_path = "./demo datasets/OnTarget_efficiency_train.txt"
    # read_val_data_path = "./demo datasets/OnTarget_efficiency_val.txt"
    # ycol = "Indel_efficiency"
    # save_model_dir = "./AIdit_ON"
    main(read_train_data_path, read_val_data_path, ycol, save_model_dir)
