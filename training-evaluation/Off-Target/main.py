# -*-coding: utf-8 -*-
'''
@author: yan jianfeng
@license: python3.7.3
@contact: yanjianfeng@westlake.edu.cn
@software: pycharm
@file: main.py
@time:
@desc: Training
'''
from data import *
from models import *
import warnings
warnings.filterwarnings('ignore')


# training
def main_modeling(read_train_path, read_val_path, read_params_path,
                  y_col, save_model_path, model_label='MLP'):
    # get features
    train_data = pd.read_csv(read_train_path, sep='\t')
    val_data = pd.read_csv(read_val_path, sep='\t')
    x_train = off_target_mismatch_feature_engineering(train_data)
    x_val = off_target_mismatch_feature_engineering(val_data)
    x_train, y_train = np.array(x_train), np.array(train_data[y_col])
    x_val, y_val = np.array(x_val), np.array(val_data[y_col])
    # get model parameters
    model_params = pd.read_csv(read_params_path, sep='\t')
    one_params = model_params.loc[model_params['model_label'] == model_label, :]
    index = one_params.index.tolist()[0]
    params = str(one_params.loc[index, 'params'])
    # training
    params = eval(params)
    if model_label == 'Lasso':
        results = Lasso_for_off_target_Modeling(x_train, y_train, x_val, y_val, params)
    elif model_label == 'Ridge':
        results = Ridge_for_off_target_Modeling(x_train, y_train, x_val, y_val, params)
    elif model_label == 'Elastic':
        results = Elastic_for_off_target_Modeling(x_train, y_train, x_val, y_val, params)
    elif model_label == 'XGBoost':
        results = XGBoost_for_off_target_Modeling(x_train, y_train, x_val, y_val, params)
    elif model_label == 'MLP':
        results = MLP_for_off_target_Modeling(x_train, y_train, x_val, y_val, params)
    else:
        results = ''
    train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse, model = results
    import joblib
    save_dir = '/'.join(save_model_path.split("/")[:-1])
    mkdir(save_dir)
    joblib.dump(model, save_model_path)
    return results


# execute -- parameter tunning
if __name__ == "__main__":
    import sys
    main_dir, read_train_path, read_val_path, read_params_path, y_col, save_model_path = sys.argv[1:]
    os.chdir(main_dir)
    # read_train_path = "./demo datasets/OffTarget_efficiency_train.txt"
    # read_val_path = "./demo datasets/OffTarget_efficiency_val.txt"
    # read_params_path = "./demo datasets/off-target_model_parameters.log"
    # y_col = "Efficiency"
    # save_model_path = "./AIdit_OFF/AIdit_OFF.model"
    main_modeling(read_train_path, read_val_path, read_params_path, y_col, save_model_path)







