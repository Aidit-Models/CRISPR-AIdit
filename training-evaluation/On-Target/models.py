# -*-coding: utf-8 -*-
'''
@author: yan jianfeng
@license: python3.7.3
@contact: yanjianfeng@westlake.edu.cn
@software: pycharm
@file: models.py
@time: 2021/4/2 19:54
@desc:
'''
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Conv2D
##########################################################################
def Conv2D_model(params, seq_len=63):
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Dense, Dropout
    from keras.layers import Flatten
    visible = Input(shape=(4, seq_len, 1))
    conv2d_1 = Conv2D(params['filters'], params['kernel_size'], strides=1, padding='same', activation='relu')(visible)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2))(conv2d_1)  # 池化层
    # 2nd
    conv2d_2 = Conv2D(params['filters'], (2, 2), strides=1, padding='same', activation='relu')(maxpool_1)
    maxpool_2 = MaxPooling2D(pool_size=(2, 2))(conv2d_2)  # 池化层
    # Flatten + FC
    flat = Flatten()(maxpool_2)
    hidden1 = Dense(params['hidden1'], activation='relu')(flat)
    dropout1 = Dropout(params['dropout1'])(hidden1)
    hidden2 = Dense(params['hidden2'], activation='relu')(dropout1)
    dropout2 = Dropout(params['dropout1'])(hidden2)
    output = Dense(1)(dropout2)
    model = Model(inputs=visible, outputs=output)
    return model


one_BiCNN_params = {
                'filters': 128,
                'kernel_size': 3,
                'hidden1': 256,
                'hidden2': 128,
                'dropout1': 0.2,
                'dropout2': 0.2,
                'batch_size': 128,
                'optimizer': 'Adam'}


# RNN + FC
def BiLSTM_Model(params, seq_len=63):
    from keras.models import Model
    from keras.layers import LSTM, Bidirectional
    from keras.layers import Input
    from keras.layers import Dense, Dropout
    # Model Frame
    visible = Input(shape=(seq_len, 4))
    bi_lstm1 = Bidirectional(LSTM(params['bilstm_hidden1'], dropout=0.2, return_sequences=True))(visible)
    bi_lstm = Bidirectional(LSTM(params['bilstm_hidden'], dropout=0.2))(bi_lstm1)
    # 全连接层 1
    hidden1 = Dense(params['hidden1'], activation='relu')(bi_lstm)
    # 全连接层 2
    dropout = Dropout(params['dropout'])(hidden1)
    output = Dense(1)(dropout)
    #
    model = Model(inputs=visible, outputs=output)
    return model


# one reference param
one_BiLSTM_params = {'bilstm_hidden1': 32,
                      'bilstm_hidden': 64,
                      'hidden1': 64,
                      'dropout': 0.2276,
                      'batch_size': 128,
                      'optimizer': 'Nadam'}


# GRU + FC
def BiGRU_Model(params, seq_len=63):
    from keras.models import Model
    from keras.layers import LSTM, Bidirectional, GRU
    from keras.layers import Input
    from keras.layers import Dense, Dropout
    # Model Frame
    visible = Input(shape=(seq_len, 4))
    gru1 = Bidirectional(GRU(params['bigru_hidden1'], dropout=0.2, return_sequences=True))(visible)
    gru = Bidirectional(GRU(params['bigru_hidden'], dropout=0.2))(gru1)
    # 全连接层 1
    hidden1 = Dense(params['hidden1'], activation='relu')(gru)
    # 全连接层 2
    back1 = Dropout(params['dropout'])(hidden1)
    output = Dense(1)(back1)
    ##
    model = Model(inputs=visible, outputs=output)
    return model


one_BiGRU_params = {'bigru_hidden1': 64,
                      'bigru_hidden': 32,
                      'hidden1': 128,
                      'dropout': 0.4919,
                      'batch_size': 128,
                      'optimizer': 'Nadam'}


## 选择模型参数
def selection_model_parameter(model_label):
    if model_label == 'BiLSTM':
        return one_BiLSTM_params
    elif model_label == 'BiGRU':
        return one_BiGRU_params
    elif model_label == 'BiCNN':
        return one_BiCNN_params
    else:
        print("Error (model: %s; hidden_num: %s), Please check and try again."%(model_label))
        return {}


# 选择模型
def selecting_deeplearning(model_label):
    if model_label == 'BiLSTM':
        return BiLSTM_Model
    elif model_label == 'BiGRU':
        return BiGRU_Model
    elif model_label == 'BiCNN':
        return Conv2D_model
    else:
        print("model_label not in [BiLSTM, BiGRU, Conv2D], over range models. Please check and try again.")
        return np.nan


