# -*-coding: utf-8 -*-
"""
@author: yan jianfeng
@license: python3.7.3
@contact: yanjianfeng@westlake.edu.cn
@software: pycharm
@file: Modeling_Analysis_Feature_Sets.py
@time:
@desc: (sequence feature set, MH feature set, engineered feature set)
"""
import os
from DSB_Repair_Engineered_Feature import *
import warnings
warnings.filterwarnings('ignore')


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
        epoch = file_p[2]
        epoch_dict[epoch] = file
    epoch_max = max(list(epoch_dict.keys()))
    file_max = epoch_dict[epoch_max]
    return file_max


# 4.1 Some Evaluation Functions
def spearman(y_true, y_pred):
    import pandas as pd
    y_true = y_true.reshape(y_true.shape[0])
    y_pred = y_pred.reshape(y_true.shape[0])
    sp = pd.Series(y_pred).corr(pd.Series(y_true), method='spearman')
    pcc = pd.Series(y_pred).corr(pd.Series(y_true), method='pearson')
    return (sp, pcc)


# 计算 KL Divergence (Kullback-Leibler)
def asymmetricKL(P, Q):
    """
    Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0.
    """
    epsilon = 0.00001
    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon
    P_sum = sum(P)
    Q_sum = sum(Q)
    P = P / P_sum
    Q = Q / Q_sum
    divergence = np.sum(P * np.log(P / Q))
    return divergence


def symmetricalKL(P, Q):
    """
    P = np.asarray([1.346112,1.337432,1.246655])
    Q = np.asarray([1.033836,1.082015,1.117323])
    print(asymmetricKL(P, Q), asymmetricKL(Q, P))
    print(symmetricalKL(P, Q))
    """
    return (asymmetricKL(P, Q) + asymmetricKL(Q, P)) / 2.00


# Evaluation
# Metrics: pearson & symmetricKL
def evaluation_repair_map(y_train, Y_pred):
    import numpy as np
    from sklearn.metrics import mean_squared_error
    evaluation_dict = {'pearson': [],
                       'symKL': [],
                       'MSE': []}
    sample_n = y_train.shape[0]
    for index in range(sample_n):
        temp_train = y_train[index, :]
        temp_pred = Y_pred[index, :]
        # pearson
        pccs = np.corrcoef(temp_train, temp_pred)[0, 1]
        evaluation_dict['pearson'].append(pccs)
        # symmetricKL
        symKL = symmetricalKL(temp_train, temp_pred)
        evaluation_dict['symKL'].append(symKL)
        # MSE
        mse = mean_squared_error(temp_train, temp_pred)
        evaluation_dict['MSE'].append(mse)
    # DataFrame
    eval_df = pd.DataFrame(evaluation_dict)
    return eval_df


# evaluation: KL-Divergence, MSE, Pearson
def evaluation(model_path, Xdata, ydata):
    # load model
    from keras.models import load_model
    model = load_model(model_path, custom_objects={'my_categorical_crossentropy_2': my_categorical_crossentropy_2})
    # predict
    ypred = model.predict(Xdata)
    result = evaluation_repair_map(ydata, ypred)
    return result


# Algorithm
# 自定义损失函数
def my_categorical_crossentropy_2(labels, logits):
    import tensorflow as tf
    '''
    label = tf.constant([[0,0,1,0,0]], dtype=tf.float32)
    logit = tf.constant([[-1.2, 2.3, 4.1, 3.0, 1.4]], dtype=tf.float32)
    logits = tf.nn.softmax(logit) # 计算softmax
    my_result1 = my_categorical_cross_entropy(labels=label, logits=logits)
    my_result2 = my_categorical_crossentropy_1(label, logits)
    my_result3 = my_categorical_crossentropy_2(label, logits)
    my_result1, my_result2, my_result3
    '''
    return tf.keras.losses.categorical_crossentropy(labels, logits)


# For Predicting the ratio of insertions to deletions
def LRModel_Predicting_Ratio_insertion_to_deletion(X_train):
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    # 构建神经网络模型
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], units=1))
    model.add(Activation('sigmoid'))
    return model


# For Predicting DSB Repair Map
def LRModel_Predicting_DSB_Repair_Map(X_train, y_train):
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    # 构建神经网络模型
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], units=y_train.shape[1]))
    model.add(Activation('softmax'))
    return model


# For one model fitting
def model_fitting(Xtrain, ytrain, Xval, yval, params, save_model_dir, early_stopping_patience):
    # model framework
    model = LRModel_Predicting_DSB_Repair_Map(Xtrain, ytrain)
    # compile
    model.compile(loss=my_categorical_crossentropy_2, optimizer=params['optimizer'])
    # checkpoint
    from keras.callbacks import ModelCheckpoint
    checkpoint_file = save_model_dir + '/model_weights-improvement-{epoch:04d}' \
                                       '-train-{loss:.5f}-test-{val_loss:.5f}.hdf5'
    checkpoint = ModelCheckpoint(filepath=checkpoint_file, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
    # early stoppping
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=0,
                                   mode='min')
    callbacks_list = [checkpoint, early_stopping]
    # # 加载
    # from keras.models import load_mode
    # model = load_model(filepath)
    try:
        # fit
        model.fit(Xtrain, ytrain,
                  batch_size=params['batch_size'],
                  epochs=params['epochs'],
                  validation_data=(Xval, yval),
                  callbacks=callbacks_list,
                  verbose=0)
    except OSError as e:
        print(e)
        pass


def main_fit(Xtrain, ytrain, Xval, yval, save_model_dir, epoch, early_stopping_patience):
    # 拟合模型
    from keras import optimizers
    sgd = optimizers.SGD
    adam = optimizers.Adam
    for optimizer in [sgd]:
        for lr in [0.1]:
            for batch_size in [64]:
                opt = optimizer(lr=lr)
                params = {'optimizer': opt, 'batch_size': batch_size, 'epochs': epoch, 'learning rate': lr}
                model_fitting(Xtrain, ytrain, Xval, yval, params, save_model_dir, early_stopping_patience)


def prepocessing(data, int_data):
    data['gRNASeq_85bp'] = data['TargetSeq(up20bp+target20bp+pam3bp+down20bp)'].apply(lambda x:
                                                                                      x + "GTTTGTATTACCGCCATGCATT")
    del data['TargetSeq(up20bp+target20bp+pam3bp+down20bp)']
    # merge categories
    dmerge = {}
    for nwc in int_data['new category'].unique():
        dmerge[nwc] = []
    for index, row in int_data.iterrows():
        dmerge[row['new category']].append(row['category'])
    ##
    label_data = data[['gRNASeq_85bp']]
    for nwc, values in dmerge.items():
        if len(values) != 1:
            label_data[nwc] = data[values].sum(axis=1)
        else:
            label_data[nwc] = data[values[0]]
    return label_data


# main modeling feature analysis
def main_modeling(cell_line, read_train_data_path, read_val_data_path, read_int_data_path,
                  save_model_dir, epoch=999, early_stopping_patience=50):
    # read data
    train_data = pd.read_csv(read_train_data_path, sep='\t')
    val_data = pd.read_csv(read_val_data_path, sep='\t')
    int_data = pd.read_excel(read_int_data_path)
    train_data = prepocessing(train_data, int_data)
    val_data = prepocessing(val_data, int_data)
    y_cols = list(int_data['new category'].unique())
    # Get feature set
    print("Step 1: Get feature set ...")
    train_seq_data, train_MH_data, train_eng_data = main_xgb_prediction(cell_line, train_data, int_data, save_model_dir)
    val_seq_data, val_MH_data, val_eng_data = main_xgb_prediction(cell_line, val_data, int_data, save_model_dir)
    train_feat_dict = {"seq_feat": train_seq_data, "MH_feat": train_MH_data, "eng_feat": train_eng_data}
    val_feat_dict = {"seq_feat": val_seq_data, "MH_feat": val_MH_data, "eng_feat": val_eng_data}
    ytrain = np.array(train_data[y_cols])
    yval = np.array(val_data[y_cols])
    # ensamble features
    feat_list = ['seq_feat', 'MH_feat', 'eng_feat']
    Tdata = pd.concat([train_feat_dict[feat].iloc[:, 1:] for feat in feat_list], axis=1)
    Vdata = pd.concat([val_feat_dict[feat].iloc[:, 1:] for feat in feat_list], axis=1)
    Xtrain, Xval = np.array(Tdata), np.array(Vdata)
    # fit
    mkdir(save_model_dir)
    main_fit(Xtrain, ytrain, Xval, yval, save_model_dir, epoch, early_stopping_patience)
    print("Finish.")


if __name__ == '__main__':
    import sys
    main_dir, cell_line, read_train_data_path, read_val_data_path, read_int_data_path, save_model_dir = sys.argv[1:]
    os.chdir(main_dir)
    # cell_line = "K562"
    # read_train_data_path = "./demo datasets/DSB_Repair_Map_train.txt"
    # read_val_data_path = "./demo datasets/DSB_Repair_Map_val.txt"
    # read_int_data_path = "./demo datasets/Table S4-K562_integrated_DSB_repair_category.xlsx"
    # save_model_dir = "./AIdit_DSB"
    # train lr
    main_modeling(cell_line, read_train_data_path, read_val_data_path, read_int_data_path, save_model_dir)

