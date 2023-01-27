# -*-coding: utf-8 -*-
'''
@author: yan jianfeng
@license: python3.7.3
@contact: yanjianfeng@westlake.edu.cn
@software: pycharm
@file: models.py
@time:
@desc: Lasso, Ridge, Elastic, XGBoost, MLP for off-target modeling
'''
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# For Prediction & Evaluation
def evaluation_for_conventional_meachine_learning(model, x_train, y_train, x_val, y_val):
    from sklearn.metrics import mean_squared_error
    # 0prediction
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)
    # evaluate
    # spearman
    train_spearman = pd.Series(y_train_pred).corr(pd.Series(y_train), method='spearman')
    val_spearman = pd.Series(y_val_pred).corr(pd.Series(y_val), method='spearman')
    # pearson
    train_pccs = pd.Series(y_train_pred).corr(pd.Series(y_train), method='pearson')
    val_pccs = pd.Series(y_val_pred).corr(pd.Series(y_val), method='pearson')
    # MSE
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    return (train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse)


# For Elastic
def Elastic_for_off_target_Modeling(x_train, y_train, x_val, y_val, params):
    from sklearn.linear_model import ElasticNet
    # create model & fit the model
    model = ElasticNet(**params)
    model.fit(x_train, y_train)
    # prediction and evaluation
    train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse = evaluation_for_conventional_meachine_learning(
                                                                                model, x_train, y_train, x_val, y_val)
    return (train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse, model)


# For Ridge
def Ridge_for_off_target_Modeling(x_train, y_train, x_val, y_val, params):
    from sklearn.linear_model import Ridge
    # create model & fit the model
    model = Ridge(**params)
    model.fit(x_train, y_train)
    # prediction and evaluation
    train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse = evaluation_for_conventional_meachine_learning(
                                                                                model, x_train, y_train, x_val, y_val)
    return (train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse, model)


# For Lasso
def Lasso_for_off_target_Modeling(x_train, y_train, x_val, y_val, params):
    from sklearn.linear_model import Lasso
    # create model & fit the model
    model = Lasso(**params)
    model.fit(x_train, y_train)
    # prediction and evaluation
    train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse = evaluation_for_conventional_meachine_learning(
                                                                                model, x_train, y_train, x_val, y_val)
    return (train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse, model)


# For XGBoost
def XGBoost_for_off_target_Modeling(x_train, y_train, x_val, y_val, params):
    from xgboost import XGBRegressor
    # create model & fit the model
    model = XGBRegressor(**params)
    model.fit(x_train, y_train)
    # prediction and evaluation
    train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse = evaluation_for_conventional_meachine_learning(
                                                                                model, x_train, y_train, x_val, y_val)
    return (train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse, model)


# For MLP
def MLP_for_off_target_Modeling(x_train, y_train, x_val, y_val, params):
    from sklearn.neural_network import MLPRegressor
    # create model & fit the model
    hidden_layer_sizes = (
        params['hidden_layer_sizes_1'], params['hidden_layer_sizes_2'], params['hidden_layer_sizes_3'],
        params['hidden_layer_sizes_4'], params['hidden_layer_sizes_5'])
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, alpha=params['alpha'], max_iter=params['max_iter'],
                         random_state=2020, shuffle=True, verbose=False,
                         activation='relu', solver='adam', learning_rate='invscaling')
    model.fit(x_train, y_train)
    # prediction and evaluation
    train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse = evaluation_for_conventional_meachine_learning(
                                                                                model, x_train, y_train, x_val, y_val)
    return (train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse, model)



