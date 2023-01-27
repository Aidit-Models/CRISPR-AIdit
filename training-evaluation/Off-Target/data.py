# -*-coding: utf-8 -*-
"""
@author: yan jianfeng
@license: python3.7.3
@contact: yanjianfeng@westlake.edu.cn
@software: pycharm
@file: data.py
@time:
@desc: get off-target modeling data
"""
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
    else:
        pass


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


# 计算 a pair of sequences 之间的 mismatch 的个数
def compute_mismatch_number(seq1, seq2):
    mismatch_num = 0
    for index, nucle1 in enumerate(seq1):
        nucle2 = seq2[index]
        if nucle1 != nucle2:
            mismatch_num += 1
        else:
            pass
    return mismatch_num


# gRNASeq & offSeq 比对
def alignment_on_off_Sequence(gRNASeq, offSeq):
    gRNASeq = gRNASeq.upper()
    offSeq = offSeq.upper()
    align = []
    for index, nucle0 in enumerate(gRNASeq):
        nucle1 = offSeq[index]
        if nucle1 != '-':
            align.append(nucle0 + nucle1)
        else:
            align.append(nucle0 + nucle0)
    return '-'.join(align)


# alignment: on-off deletion sequence
def alignment_on_off_deletion_sequence(new_offSeq_Target):
    on_off_deltSeq = ''
    for nucle in new_offSeq_Target:
        if nucle != '-':
            on_off_deltSeq = on_off_deltSeq + '.'
        else:
            on_off_deltSeq = on_off_deltSeq + '-'
    return on_off_deltSeq


# 比对确定 off-target insertion sequence
def alignment_on_off_insertion_sequence(offSeq_Target):
    import re
    inser_nucles = re.findall("[acgt]", offSeq_Target)
    inser_nucles = list(set(inser_nucles))
    ##
    inserSeq = ''
    for index, nucle in enumerate(offSeq_Target):
        if nucle not in inser_nucles:
            inserSeq = inserSeq + '.'
        else:
            inserSeq = inserSeq + nucle.upper()
    return inserSeq


# off-target mismatch/insertion/deletion modeling data
def main_off_target_Modeling_data(off_data, mut_type='mismatch'):
    off_data['PAM-NN'] =  off_data['offSeq_63bp'].apply(lambda x: x[41:43])
    off_data['on_off_alignSeq'] = off_data.apply(lambda row: alignment_on_off_Sequence(row['gRNASeq'], row['offSeq_63bp'][20:43]),
                                                 axis=1)
    if mut_type == 'mismatch':
        # compute mismatch number
        off_data['up_mismatch_num'] = 0
        off_data['core_mismatch_num'] = off_data.apply(lambda row: compute_mismatch_number(row['gRNASeq'],
                                                                                           row['offSeq_63bp'][20:43]),
                                                                                           axis=1)
        off_data['down_mismatch_num'] = 0
        cols = ['gRNASeq', 'offSeq_63bp', 'PAM-NN', 'on_off_alignSeq',
                'up_mismatch_num', 'core_mismatch_num', 'down_mismatch_num', 'on_pred', 'off_pred']
    elif mut_type == 'deletion':
        off_data['on_off_deltSeq'] = off_data['offSeq_63bp'].apply(lambda x: alignment_on_off_deletion_sequence(x[20:43]))
        cols = ['offSeq_63bp', 'offSeq_28bp', 'gRNASeq', 'PAM-NN', 'on_off_alignSeq', 'on_off_deltSeq']
    elif mut_type == 'insertion':
        off_data['on_off_inserSeq'] = off_data['offSeq_63bp'].apply(lambda x: alignment_on_off_insertion_sequence(x[20:43]))
        cols = ['offSeq_63bp', 'offSeq_28bp', 'gRNASeq', 'PAM-NN', 'on_off_alignSeq', 'on_off_inserSeq']
    else:
        print("Mutation type not in ['mismatch', 'insertion', 'deletion']. Please check and try again.")
        cols = ['gRNASeq_63bp', 'offSeq_63bp', 'gRNASeq', 'PAM-NN', 'on_off_alignSeq', 'on_pred', 'off_pred']
    data = off_data[cols]
    return data


# **********************************************************************
# ********************* Feature one-hot Encoding ***********************
# 1、序列特征输入： 序列特征
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
    # print(one_sample.shape)
    return one_sample


#  获取序列数据
# 参数说明：
# data：输入的数据，要求含有 gRNA_28bp or gRNASeq_63bp 列名，该列为原始 DNA 序列
# 输出：特征数据 {'data': data}
def obtain_Sequence_data(data, seq_len=63, col='offSeq_63bp'):
    x_data = []
    for i, row in data.iterrows():
        seq = row[col]
        one_sample = obtain_each_seq_data(seq)
        one_sample_reshape = one_sample.T.reshape(seq_len * 4)
        # print(one_sample_reshape.shape)
        x_data.append(one_sample_reshape)
    # reshape
    x_data = np.array(x_data)
    x_data = x_data.astype('float32')
    return x_data


# 2、获得 PAM-NN 特征
def obtain_PAM_Feature(pam_nn, pam_feats=['GG', 'AG', 'GT', 'GC', 'GA', 'TG', 'CG', 'other']):
    """
    pam_feats = ['GG', 'AG', 'GT', 'GC', 'GA', 'TG', 'CG', 'other']
    pam_nn = 'GG'
    pam_list = obtain_PAM_Feature(pam_nn, pam_feats)
    """
    pam_dict = {}
    for pam in pam_feats:
        pam_dict[pam] = 0
    if pam_nn in pam_dict:
        pam_dict[pam_nn] = 1
    else:
        pam_dict['other'] = 1
    pam_list = []
    for pam in pam_feats:
        pam_list.append(pam_dict[pam])
    return pam_list


# 获得 PAM-NN 特征
def main_pam_data(data, pam_feats=['GG', 'AG', 'GT', 'GC', 'GA', 'TG', 'CG', 'other']):
    pam_data = []
    for index, row in data.iterrows():
        pam_nn = row['PAM-NN']
        pam_list = obtain_PAM_Feature(pam_nn, pam_feats)
        pam_data.append(pam_list)
    # pam data
    pam_data = np.array(pam_data)
    pam_data = pam_data.astype('float32')
    return pam_data


# 3、分解到每一个位置的 on-off mismatch feature
# 1、on-off alignment for position-substitution
def helper_each_position_alignSeq(one_pos_alignSeq):
    align_order = ['AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GT', 'TA', 'TC', 'TG']
    align_list = []
    for one_align in align_order:
        if one_align == one_pos_alignSeq:
            align_list.append(1)
        else:
            align_list.append(0)
    return align_list


# one mismatch alignment sequence
def helper_one_alignSeq(alignSeq):
    '''
    alignSeq = 'TT-TG-GC-CA-AG-GC-CA-AC-CC-AA-AA-CC-CC-CC-CC-AA-TT-GG-AA-AA-GG-GG-GG'
    all_align1 = helper_one_alignSeq(alignSeq)
    print(all_align1)
    '''
    alignSeq_list = alignSeq.split('-')
    all_align_list = []
    for alignSeq in alignSeq_list:
        align_list = helper_each_position_alignSeq(alignSeq)
        all_align_list.append(align_list)
    all_align = np.array(all_align_list).T
    return all_align


# 获得 mismatch alignment feature
def main_mismatch_alignment_features(data):
    align_data = []
    for index, row in data.iterrows():
        alignSeq = row['on_off_alignSeq']
        all_align = helper_one_alignSeq(alignSeq)
        all_align = all_align.T.reshape(all_align.shape[0]*all_align.shape[1])
        align_data.append(all_align)
    # align data
    align_data = np.array(align_data)
    align_data = align_data.astype('float32')
    return align_data


# 2、on-off alignment for only position
def helper_one_alignSeq_with_only_position(alignSeq):
    """
    alignSeq = 'TT-TG-GC-CA-AG-GC-CA-AC-CC-AA-AA-CC-CC-CC-CC-AA-TT-GG-AA-AA-GG-GG-GG'
    all_align_list2 = helper_one_alignSeq_with_only_position(alignSeq)
    print(all_align_list2)
    """
    alignSeq_list = alignSeq.split('-')
    all_align_list = []
    for alignSeq in alignSeq_list:
        if alignSeq[0] == alignSeq[1]:
            all_align_list.append(0)
        else:
            all_align_list.append(1)
    return all_align_list


# 获得 mismatch alignment feature with only position
def main_mismatch_alignment_features_with_only_position(data):
    align_data = []
    for index, row in data.iterrows():
        alignSeq = row['on_off_alignSeq']
        all_align_list = helper_one_alignSeq_with_only_position(alignSeq)
        align_data.append(all_align_list)
    # align data
    align_data = np.array(align_data)
    align_data = align_data.astype('float32')
    return align_data


# 4、获得 on-off deletion position distribution
def helper_on_off_deletion_position(on_off_deltSeq):
    deltSeq_list = []
    for m in on_off_deltSeq:
        if m == '.':
            deltSeq_list.append(0)
        else:
            deltSeq_list.append(1)
    return deltSeq_list


# 获得 on-off deletion position feature
def main_on_off_deletion_position(data):
    delt_data = []
    for index, row in data.iterrows():
        on_off_deltSeq = row['on_off_deltSeq']
        deltSeq_list = helper_on_off_deletion_position(on_off_deltSeq)
        delt_data.append(deltSeq_list)
    # delt data
    delt_data = np.array(delt_data)
    delt_data = delt_data.astype('float32')
    return delt_data


# 5、获得 on-off insertion position-nucleotide type
def help_on_off_insertion(on_off_inserSeq):
    ref_dict = {'.': [0, 0, 0, 0],
                'A': [1, 0, 0, 0],
                'C': [0, 1, 0, 0],
                'G': [0, 0, 1, 0],
                'T': [0, 0, 0, 1]}
    inserSeq_list = []
    for m in on_off_inserSeq[1:]:
        inserSeq_list.append(ref_dict[m])
    return inserSeq_list


# 获得 on-off insertion feature
def main_on_off_insertion_feature(data):
    inser_data = []
    for index, row in data.iterrows():
        on_off_inserSeq = row['on_off_inserSeq']
        inserSeq_list = help_on_off_insertion(on_off_inserSeq)
        inserSeq = np.array(inserSeq_list)
        inserSeq = inserSeq.reshape(4*(len(on_off_inserSeq)-1))
        inser_data.append(inserSeq)
    # inser data
    inser_data = np.array(inser_data)
    inser_data = inser_data.astype('float32')
    return inser_data


# 仅考虑 insertion position
def help_on_off_insertion_position(on_off_inserSeq):
    inserSeq_pos = []
    for m in on_off_inserSeq[1:]:
        if m == '.':
            inserSeq_pos.append(0)
        else:
            inserSeq_pos.append(1)
    return inserSeq_pos


# 获得 on-off insertion feature position
def main_on_off_insertion_feature_woth_only_position(data):
    inser_data = []
    for index, row in data.iterrows():
        on_off_inserSeq = row['on_off_inserSeq']
        inserSeq_pos = help_on_off_insertion_position(on_off_inserSeq)
        inser_data.append(inserSeq_pos)
    # inser data
    inser_data = np.array(inser_data)
    inser_data = inser_data.astype('float32')
    return inser_data


# mismatch
# 得到 off-target mismatch Feature Engineering
# nparray_concat_to_one
def array_concat_to_one(collect_feat_data_dict):
    data = pd.DataFrame()
    for feat_label, array in collect_feat_data_dict.items():
        df_array = pd.DataFrame(array)
        cols_n = df_array.shape[1]
        cols = [feat_label + '_%s'%(i + 1) for i in range(cols_n)]
        df_array.columns = cols
        data = pd.concat([data, df_array], axis=1)
    return data


# 得到 off-target deletion Feature Engineering
def off_target_mismatch_feature_engineering(data):
    # geting features: 'gRNASeq', 'PAM-NN', 'on_off_alignSeq'
    data.rename(columns={
                         'TargetSeq(up20bp+target20bp+pam3bp+down20bp)': 'offSeq_63bp'
                         }, inplace=True)
    data = main_off_target_Modeling_data(data, mut_type='mismatch')
    data['offSeq_23bp'] = data['offSeq_63bp'].apply(lambda x: x[20:43])
    pam_feats = ['GG', 'AG', 'GT', 'GC', 'GA', 'TG', 'CG', 'other']
    collect_feat_data_dict = {}
    # fixed feature list
    x_data1 = obtain_Sequence_data(data, seq_len=23, col='offSeq_23bp')
    x_data2 = obtain_Sequence_data(data, seq_len=23, col='gRNASeq')
    x_data3 = np.array(data[['up_mismatch_num', 'core_mismatch_num', 'down_mismatch_num']])
    pred_data = np.array(data[['on_pred', 'off_pred']])
    collect_feat_data_dict['offSeq'] = x_data1
    collect_feat_data_dict['gRNASeq'] = x_data2
    collect_feat_data_dict['mismatch_num_region'] = x_data3
    collect_feat_data_dict['pred_feat'] = pred_data
    # Additional features
    pam_data = main_pam_data(data, pam_feats)
    align_data = main_mismatch_alignment_features(data)
    collect_feat_data_dict['+P'] = pam_data
    collect_feat_data_dict['+M'] = align_data
    # feature concating
    xdata = array_concat_to_one(collect_feat_data_dict)
    return xdata
