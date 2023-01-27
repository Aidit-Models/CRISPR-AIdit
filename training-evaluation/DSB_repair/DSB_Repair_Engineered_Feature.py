# -*-coding: utf-8 -*-
"""
@author: yan jianfeng
@license: python3.7.3
@contact: yanjianfeng@westlake.edu.cn
@software: pycharm
@file: DSB_Repair_Engineered_Feature.py
@time:
@desc: Engineered Feature from XGBoost 0prediction
"""
from DSB_Repair_Feature_and_Categories import *
import warnings
warnings.filterwarnings('ignore')


def Obtain_predicting_feature_2nd(data, seq_bp=28, max_len=30):
    # 1. to get sequence feature
    seq_data = obtain_single_sequence_one_hot_feature_2nd(data, seq_bp)
    # 2. to get MH feature
    edit_sites = [34, 35, 36, 37, 38, 39, 40]
    MH_data = main_MH_Feature_2nd(data, edit_sites, max_len)
    return (seq_data, MH_data)


def Obtain_predicting_categories_1(data, y_col):
    # label_list = list(int_data['new category'].unique())
    ydata = np.array(data[y_col])
    return ydata


# 0prediction
def xgb_prediction(Xdata, model_path):
    import joblib
    model = joblib.load(model_path)
    ypred = model.predict(Xdata)
    return ypred


# data.columns: ['sgRNA_name', 'gRNASeq_85bp'] at least
def main_xgb_prediction(Cell_Line, data, int_data, xgb_model_dir, seq_bp=63, max_len=30):
    # Get Xdata
    seq_data, MH_data = Obtain_predicting_feature_2nd(data, seq_bp, max_len)
    Xdata = pd.merge(seq_data, MH_data, how='inner', on=['gRNASeq_85bp'])
    del Xdata['gRNASeq_85bp']
    Xdata = np.array(Xdata)
    print('----------------------')
    print('Xtrain.shape:', Xdata.shape)
    # Get Engineered Feature
    # model_path = "XGB_K562-63bp_%s.model"%("29:40D-12")
    model_path_pattern = xgb_model_dir + "/XGB_%s-%sbp" % (Cell_Line, seq_bp) + '/XGB_%s.model'
    eng_data = seq_data[['gRNASeq_85bp']]
    for model_label in int_data['new category'].unique():
        temp_model_path = model_path_pattern % (model_label.replace(":", "-"))
        ypred = xgb_prediction(Xdata, temp_model_path)
        eng_data[model_label] = ypred
    return (seq_data, MH_data, eng_data)

