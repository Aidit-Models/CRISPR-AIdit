import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


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



def correlation(y_true, y_pred):
    import pandas as pd
    y_true = y_true.reshape(y_true.shape[0])
    y_pred = y_pred.reshape(y_true.shape[0])
    sp = pd.Series(y_pred).corr(pd.Series(y_true), method='spearman')
    pr = pd.Series(y_pred).corr(pd.Series(y_true), method='pearson')
    return sp, pr

# Adding Steiger_test for endogenous targets

__author__ = 'psinger'

import numpy as np
from scipy.stats import t, norm
from math import atanh, pow
from numpy import tanh

def rz_ci(r, n, conf_level = 0.95):
    zr_se = pow(1/(n - 3), .5)
    moe = norm.ppf(1 - (1 - conf_level)/float(2)) * zr_se
    zu = atanh(r) + moe
    zl = atanh(r) - moe
    return tanh((zl, zu))

def rho_rxy_rxz(rxy, rxz, ryz):
    num = (ryz-1/2.*rxy*rxz)*(1-pow(rxy,2)-pow(rxz,2)-pow(ryz,2))+pow(ryz,3)
    den = (1 - pow(rxy,2)) * (1 - pow(rxz,2))
    return num/float(den)

def dependent_corr(xy, xz, yz, n, twotailed=True, conf_level=0.95, method='steiger'):
    """
    Calculates the statistic significance between two dependent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between x and z
    @param yz: correlation coefficient between y and z
    @param n: number of elements in x, y and z
    @param twotailed: whether to calculate a one or two tailed test, only works for 'steiger' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'steiger' or 'zou'
    @return: t and p-val
    """
    if method == 'steiger':
        d = xy - xz
        determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
        av = (xy + xz)/2
        cube = (1 - yz) * (1 - yz) * (1 - yz)

        t2 = d * np.sqrt((n - 1) * (1 + yz)/(((2 * (n - 1)/(n - 3)) * determin + av * av * cube)))
        p = 1 - t.cdf(abs(t2), n - 2)
        if twotailed:
            p *= 2

        return t2, p
    elif method == 'zou':
        L1 = rz_ci(xy, n, conf_level=conf_level)[0]
        U1 = rz_ci(xy, n, conf_level=conf_level)[1]
        L2 = rz_ci(xz, n, conf_level=conf_level)[0]
        U2 = rz_ci(xz, n, conf_level=conf_level)[1]
        rho_r12_r13 = rho_rxy_rxz(xy, xz, yz)
        lower = xy - xz - pow((pow((xy - L1), 2) + pow((U2 - xz), 2) - 2 * rho_r12_r13 * (xy - L1) * (U2 - xz)), 0.5)
        upper = xy - xz + pow((pow((U1 - xy), 2) + pow((xz - L2), 2) - 2 * rho_r12_r13 * (U1 - xy) * (xz - L2)), 0.5)
        return lower, upper
    else:
        raise Exception('Wrong method!')

def independent_corr(xy, ab, n, n2 = None, twotailed=True, conf_level=0.95, method='fisher'):
    """
    Calculates the statistic significance between two independent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between a and b
    @param n: number of elements in xy
    @param n2: number of elements in ab (if distinct from n)
    @param twotailed: whether to calculate a one or two tailed test, only works for 'fisher' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'fisher' or 'zou'
    @return: z and p-val
    """

    if method == 'fisher':
        xy_z = 0.5 * np.log((1 + xy)/(1 - xy))
        xz_z = 0.5 * np.log((1 + ab)/(1 - ab))
        if n2 is None:
            n2 = n

        se_diff_r = np.sqrt(1/(n - 3) + 1/(n2 - 3))
        diff = xy_z - xz_z
        import decimal
        from decimal import Decimal
        decimal.getcontext().prec=500
        diff = Decimal(diff)
        se_diff_r = Decimal(se_diff_r)
        z = abs(diff / se_diff_r)
        p = (Decimal(1) - norm.cdf(z))
        if twotailed:
            p *= 2

        return z, p
    elif method == 'zou':
        L1 = rz_ci(xy, n, conf_level=conf_level)[0]
        U1 = rz_ci(xy, n, conf_level=conf_level)[1]
        L2 = rz_ci(ab, n2, conf_level=conf_level)[0]
        U2 = rz_ci(ab, n2, conf_level=conf_level)[1]
        lower = xy - ab - pow((pow((xy - L1), 2) + pow((U2 - ab), 2)), 0.5)
        upper = xy - ab + pow((pow((U1 - xy), 2) + pow((ab - L2), 2)), 0.5)
        return lower, upper
    else:
        raise Exception('Wrong method!')


####################
## plot heatmap
## heatmap for comparation between models
def update_dataset_label(data, update_dataset_dict):
    data['dataset'] = data.index
    data['dataset'] = data['dataset'].apply(lambda x: update_dataset_dict[x] if x in update_dataset_dict else x)
    data.index = data['dataset']
    del data['dataset']
    return data


## 返回 dict: U6 promoter, T7 promoter, small datasets 对应的数据对
## get heatmap data
def obtain_heatmap_data_for_models_comparison(cmp_data, parts_dataset_dict, selective_models_for_comparison,
                                              note_dataset_taining_model, update_dataset_dict,
                                              cell_line=False):
    # 汇总所有 datasets ，方便数据操作
    keep_datasets_for_comparison = []
    for dataset_list in parts_dataset_dict.values():
        keep_datasets_for_comparison += dataset_list

    ## index 赋值
    cmp_data.index = cmp_data.Dataset
    ## 选择比较的数据集 & 选择比较的算法
    cmp_data = cmp_data.loc[keep_datasets_for_comparison, :]
    cmp_data = cmp_data[selective_models_for_comparison]
    ## 把 Xu_Hl60_2015(2076) & XU_Kbm7_2015(2076) 评估结果取正
    ## 对行操作
    try:
        if cell_line == 'K562':
            cmp_data['Xu_Hl60_2015(1774)':'Xu_Hl60_2015(1774)'] = abs(
                cmp_data['Xu_Hl60_2015(1774)':'Xu_Hl60_2015(1774)'])
        elif cell_line == 'Jurkat':
            cmp_data['Xu_Hl60_2015(1776)':'Xu_Hl60_2015(1776)'] = abs(
                cmp_data['Xu_Hl60_2015(1776)':'Xu_Hl60_2015(1776)'])
        elif cell_line == 'H1':
            cmp_data['Xu_Hl60_2015(2026)':'Xu_Hl60_2015(2026)'] = abs(
                cmp_data['Xu_Hl60_2015(2026)':'Xu_Hl60_2015(2026)'])
        else:
            cmp_data['Xu_Hl60_2015(2076)':'Xu_Hl60_2015(2076)'] = abs(
                cmp_data['Xu_Hl60_2015(2076)':'Xu_Hl60_2015(2076)'])
    except KeyError as e:
        pass
    try:
        if cell_line == 'K562':
            cmp_data['XU_Kbm7_2015(1774)':'XU_Kbm7_2015(1774)'] = abs(
                cmp_data['XU_Kbm7_2015(1774)':'XU_Kbm7_2015(1774)'])
        elif cell_line == 'Jurkat':
            cmp_data['XU_Kbm7_2015(1776)':'XU_Kbm7_2015(1776)'] = abs(
                cmp_data['XU_Kbm7_2015(1776)':'XU_Kbm7_2015(1776)'])
        elif cell_line == 'H1':
            cmp_data['XU_Kbm7_2015(2026)':'XU_Kbm7_2015(2026)'] = abs(
                cmp_data['XU_Kbm7_2015(2026)':'XU_Kbm7_2015(2026)'])
        else:
            cmp_data['XU_Kbm7_2015(2076)':'XU_Kbm7_2015(2076)'] = abs(
                cmp_data['XU_Kbm7_2015(2076)':'XU_Kbm7_2015(2076)'])
    except KeyError as e:
        pass

    ## for mask paired with cmp_data
    import copy
    paired_cmp_data = copy.deepcopy(cmp_data)
    paired_cmp_data.loc[:, :] = 0
    # show each model own training dataset
    for model_col, dataset_row_list in note_dataset_taining_model.items():
        for dataset_row in dataset_row_list:
            paired_cmp_data.loc[dataset_row, model_col] = 1.

    ## datasets split three parts: U6, T7 & small datasets
    paired_heatmap_data_dict = {}
    for dataset_label, dataset_list in parts_dataset_dict.items():
        part_cmp_data = cmp_data.loc[dataset_list, :]
        part_paried_cmp_data = paired_cmp_data.loc[dataset_list, :]
        ## update datset
        part_cmp_data = update_dataset_label(part_cmp_data, update_dataset_dict)
        part_paried_cmp_data = update_dataset_label(part_paried_cmp_data, update_dataset_dict)
        paired_heatmap_data_dict[dataset_label] = (part_cmp_data, part_paried_cmp_data)
    ## all
    paired_heatmap_data_dict['All'] = (cmp_data, paired_cmp_data)
    ## update datset
    cmp_data = update_dataset_label(cmp_data, update_dataset_dict)
    paired_cmp_data = update_dataset_label(paired_cmp_data, update_dataset_dict)
    return paired_heatmap_data_dict



## note significant
################################################################
def get_note_for_cripsor(sign_data, dataset):
    try:
        index = sign_data.loc[sign_data['Dataset'] == dataset, :].index.tolist()[0]
    except IndexError as e:
        print("dataset:", dataset)
        raise (e)
    p_value = sign_data.loc[index, 'p_value']
    if p_value < 1e-5:
        note = '***'
    elif p_value < 1e-3:
        note = '**'
    elif p_value < 0.05:
        note = '*'
    else:
        note = 'n.s.'
    return note

def note_significant_for_crispor(ax, data, sign_data):
    x = 0.7
    dataset_list = data.index.tolist()
    for i, dataset in enumerate(dataset_list):
        note = get_note_for_cripsor(sign_data, dataset)
        # ax.hlines(last_y, ix1, ix2, colors="black")
        ax.text(x, i + 0.9, note, fontsize=10)

## plot heatmap
def plot_single_heatmap_for_comparison_adding_note_significant(data, mask_data, model_list,
                                                               title, savefig_path, sign_data,
                                                               vmax=0.67, figsize=(10, 5), cmap='coolwarm'):
    import seaborn as sns;
    sns.set()
    from matplotlib import pyplot as plt

    data = data[model_list]
    mask_data = mask_data[model_list]

    sns.set_style("darkgrid")
    f, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(data, mask=np.array(mask_data), cbar=True, cmap=cmap, vmax=vmax,
                     linewidths=0.01)

    ## text
    cols = data.columns.tolist()
    total_row_num = len(data)
    row_num = -1
    for index, row in data.iterrows():
        row_num += 1
        for index, col in enumerate(cols):
            plt.text(index + 0.25, row_num + 0.5, str(round(row[col], 3)))
    ## note significant
    note_significant_for_crispor(ax, data, sign_data)
    plt.title(title)
    plt.savefig(savefig_path, dpi=300, bbox_inches='tight')
    plt.show()
################################################################
################################################################