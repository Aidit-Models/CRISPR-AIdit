# -*-coding: utf-8 -*-
'''
@author: yan jianfeng
@license: python3.7.3
@contact: yanjianfeng@westlake.edu.cn
@software: pycharm
@file: XGB_For_each_category.py
@time:
@desc:
'''
import os
import time
from DSB_Repair_Feature_and_Categories import *
import warnings
warnings.filterwarnings('ignore')


# 4.1 Some Evaluation Functions
# XGBoost
def XGBoost_fit(Xtrain, ytrain, params):
    from xgboost import XGBRegressor
    # create model & fit the model
    model = XGBRegressor(**params)
    model.fit(Xtrain, ytrain)
    return model


# 0prediction & evaluation
def Evaluation(model, Xdata, ydata):
    # 0prediction
    ypred = model.predict(Xdata)
    # evaluation
    eval_pearson = pd.Series(ypred).corr(pd.Series(ydata), method='pearson')
    eval_spearman = pd.Series(ypred).corr(pd.Series(ydata), method='spearman')
    return eval_pearson, eval_spearman


def main_XGBoost(Xtrain, ytrain, Xval, yval, model_params, save_model_path):
    import joblib
    # training
    model = XGBoost_fit(Xtrain, ytrain, model_params)
    joblib.dump(model, save_model_path)
    # 0prediction & evaluation
    train_pearson, train_spearman = Evaluation(model, Xtrain, ytrain)
    val_pearson, val_spearman = Evaluation(model, Xval, yval)
    print("train_spearman:", train_spearman, "val_spearman", val_spearman)
    return (train_pearson, train_spearman, val_pearson, val_spearman)


def Obtain_model_params(Cell_Line, seq_len=63):
    #  model parameter
    # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    if (Cell_Line == 'K562') & (seq_len == 63):
        model_param = {"n_estimators": 2249,
                       "nthread": 25,
                       "learning_rate": 0.0439,
                       "max_depth": 9,
                       "max_leaf_nodes": 164,
                       "colsample_bytree": 0.819,
                       "subsample": 0.999,
                       "reg_alpha": 1.273,
                       "reg_lambda": 33.017
                       }
    elif (Cell_Line == 'Jurkat') & (seq_len == 63):
        model_param = {"n_estimators": 2600,
                       "nthread": 25,
                       "learning_rate": 0.062,
                       "max_depth": 8,
                       "max_leaf_nodes": 37,
                       "colsample_bytree": 0.797,
                       "subsample": 0.954,
                       "reg_alpha": 1.257,
                       "reg_lambda": 8.984
                       }
    else:
        model_param = {}
        print("Input Error: Cell Line expected in ['K562', 'Jurkat'] and seq_len = 28 0r 63, "
              "but Cell_Line=%s, seq_len=%s" % (Cell_Line, seq_len))
    # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    return model_param


# Get train & test data
def Obtain_predicting_feature(data, seq_bp=28, max_len=30):
    # 1. to get sequence feature
    seq_data = obtain_single_sequence_one_hot_feature_2nd(data, seq_bp)
    # 2. to get MH feature
    edit_sites = [34, 35, 36, 37, 38, 39, 40]
    MH_data = main_MH_Feature_2nd(data, edit_sites, max_len)
    # 3. merge data
    Xdata = pd.merge(seq_data, MH_data, how='inner', on=['gRNASeq_85bp'])
    del Xdata['gRNASeq_85bp']
    return np.array(Xdata)


def Obtain_predicting_categories_1(data, y_col):
    # label_list = list(int_data['new category'].unique())
    ydata = np.array(data[y_col])
    return ydata


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


def main_xgb(cell_line, read_train_data_path, read_val_data_path, read_int_data_path, save_dir,
             seq_bp=63, max_len=30):
    # read data
    train_data = pd.read_csv(read_train_data_path, sep='\t')
    val_data = pd.read_csv(read_val_data_path, sep='\t')
    int_data = pd.read_excel(read_int_data_path)
    train_data = prepocessing(train_data, int_data)
    val_data = prepocessing(val_data, int_data)
    # save path
    save_model_dir = save_dir + '/XGB_%s-%sbp' % (cell_line, seq_bp)
    mkdir(save_model_dir)
    save_summary_path = save_dir + '/Summary_XGBoost_%s-DSB-Modeling.log' % (cell_line)
    for y_col in int_data['new category'].unique():
        save_model_path = save_model_dir + '/XGB_%s.model' % (y_col.replace(":", "-"))
        # Get train & validation data
        Xtrain = Obtain_predicting_feature(train_data, seq_bp, max_len)
        ytrain = Obtain_predicting_categories_1(train_data, y_col)
        Xval = Obtain_predicting_feature(val_data, seq_bp, max_len)
        yval = Obtain_predicting_categories_1(val_data, y_col)
        print('----------------------')
        print('Xtrain.shape:', Xtrain.shape, '; ytrain.shape:', ytrain.shape)
        print('Xval.shape:', Xval.shape, '; yval.shape:', yval.shape)
        Data_Count = Xtrain.shape[0]
        # Training
        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        start = time.time()
        model_param = Obtain_model_params(cell_line, seq_bp)
        train_pearson, train_spearman, val_pearson, val_spearman = main_XGBoost(Xtrain, ytrain, Xval, yval, model_param, save_model_path)
        end = time.time()
        using_time = end - start
        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        # Write
        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        localtime = time.asctime(time.localtime(time.time()))
        if os.path.exists(save_summary_path):
            pass
        else:
            with open(save_summary_path, 'a') as a:
                col_info = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % ('Cell Line', 'Data Count', 'Sequence Length',
                                                                        'y_col',
                                                                     'Train Pearson', 'Train Spearman', 'Val Pearson', 'Val Spearman',
                                                                         'Model Parameter', 'Using Time', 'Writing Time')
                a.write(col_info)
        with open(save_summary_path, 'a') as a:
            # write
            text_info = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (cell_line, Data_Count, seq_bp,
                                                                      y_col,
                                                                      train_pearson, train_spearman, val_pearson, val_spearman,
                                                                      str(model_param), using_time, localtime)
            a.write(text_info)
        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    print("Finish.")


if __name__ == "__main__":
    import sys
    main_dir, cell_line, read_train_data_path, read_val_data_path, read_int_data_path, save_model_dir = sys.argv[1:]
    os.chdir(main_dir)
    # cell_line = "K562"
    # read_train_data_path = "./demo datasets/DSB_Repair_Map_train.txt"
    # read_val_data_path = "./demo datasets/DSB_Repair_Map_val.txt"
    # read_int_data_path = "./demo datasets/Table S4-K562_integrated_DSB_repair_category.xlsx"
    # save_model_dir = "./AIdit_DSB"
    # train xgboost
    main_xgb(cell_line, read_train_data_path, read_val_data_path, read_int_data_path, save_model_dir)


