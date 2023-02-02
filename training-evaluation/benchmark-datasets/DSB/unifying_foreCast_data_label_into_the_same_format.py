import os
import numpy as np
import pandas as pd
import time
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
        print(path + ' 创建成功')


def walk_files(data_dir):
    import os
    g = os.walk(data_dir)
    data_path_list = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            data_path_list.append(os.path.join(path, file_name))
    return data_path_list


## step 1:
## 整合 Raw train & validation dataset
## merge data
def Integrate_ForeCasT_Processed_Data(read_data_dir, to_save_data_path):
    is_Exist_file(to_save_data_path)
    data_path_list = walk_files(read_data_dir)
    with open(to_save_data_path, 'a') as a:
        column_context = "ID\tCount\tForeCasT_Label\tRead\n"
        a.write(column_context)
        for data_path in data_path_list:
            data_list = []
            with open(data_path, 'r') as f:
                for line in f:
                    line = line.strip('\n')
                    line = line.strip(' ')
                    if line[:3] == "@@@":
                        line_p = line.split("@")
                        ID = line_p[-1]
                    else:
                        line_p = line.split('\t')
                        if len(line_p) == 3:
                            ForeCasT_label = line_p[0]
                            Count = line_p[1]
                            Read = line_p[2]
                            ## Write
                            temp_context = '%s\t%s\t%s\t%s\n' % (ID, Count, ForeCasT_label, Read)
                            a.write(temp_context)
                        else:
                            print("The line is weird, not ID or regular, please check and try. The line = %s" % (line))
                            break
        print("To Merge Data is Finished.")


## Anotation sequence
# 互补序列方法2：python3 translate()方法
def DNA_complement(sequence):
    trantab = str.maketrans('ACGTacgtRYMKrymkVBHDvbhd', 'TGCAtgcaYRKMyrkmBVDHbvdh')
    string = sequence.translate(trantab)
    return string[::-1]


## Update Target Sequence
def update_targetSequence(TargetSequence, Strand):
    if Strand == "FORWARD":
        return TargetSequence
    else:
        return DNA_complement(TargetSequence)


## generate gRANSeq_85bp
def generate_forecast_granseq_85bp(RefSeq, PAM_index):
    up = RefSeq[:(PAM_index - 20)]
    target = RefSeq[(PAM_index - 20):(PAM_index + 3)]
    down = RefSeq[(PAM_index + 3):]
    if len(up) >= 20:
        up = up[-20:]
    else:
        up = '-' * (20 - len(up)) + up
    ## down
    if len(down) >= 42:
        down = down[:42]
    else:
        down = down + '-' * (42 - len(down))
    return up + target + down


def single_forecast_data_get_granseq_85bp(aliData):
    aliData['PAM Index'] = aliData.apply(lambda row: row['Reference Sequence'].index(row['Guide'][1:]) + 19, axis=1)
    aliData['gRNASeq_85bp'] = aliData.apply(lambda row: generate_forecast_granseq_85bp(row['Reference Sequence'],
                                                                                       row['PAM Index']), axis=1)
    aliData['PAM'] = aliData['gRNASeq_85bp'].apply(lambda x: x[40:43])
    print('\nPAM:\n', aliData['PAM'].value_counts())
    return aliData


def main_forecast_get_85bp_reference_sequence(seq_data):
    seq_data.rename(columns={'Guide Sequence': "Guide"}, inplace=True)
    seq_data['Guide'] = seq_data['Guide'].apply(lambda x: x.upper())
    seq_data['TargetSequence'] = seq_data['TargetSequence'].apply(lambda x: x.upper())
    seq_data['num. of Guide'] = seq_data['Guide'].apply(lambda x: len(x))
    seq_data['num. of nucle'] = seq_data['Guide'].apply(
        lambda x: x.count("A") + x.count("G") + x.count("C") + x.count("T"))
    seq_data['filter'] = seq_data.apply(lambda row: 1 if row['num. of Guide'] != row['num. of nucle'] else 0, axis=1)
    seq_data = seq_data.loc[seq_data['filter'] == 0, :]
    seq_data.reset_index(drop=True, inplace=True)
    ## get gRNASeq 85bp column
    seq_data = seq_data[['ID', 'Guide', 'TargetSequence', 'Subset', 'Strand']]
    seq_data['Reference Sequence'] = seq_data.apply(
        lambda row: update_targetSequence(row['TargetSequence'], row['Strand']),
        axis=1)
    #     del seq_data['TargetSequence']
    aData = single_forecast_data_get_granseq_85bp(seq_data)
    aData['num'] = aData['gRNASeq_85bp'].apply(lambda x: x.count('-'))
    aData.sort_values(by='num', ascending=True, inplace=True)
    aData.drop_duplicates('Guide', keep='first', inplace=True)
    del aData['TargetSequence']
    del aData['num']
    return aData


def Anotation_filtering(df, seq_data, oligo_type_list):
    grp_data = df[['ID', 'Count']].groupby('ID').sum()
    grp_data.reset_index(drop=False, inplace=True)
    grp_data = grp_data.loc[grp_data['Count'] >= 100, :]
    IDs = grp_data['ID'].tolist()
    df = df.loc[df['ID'].isin(IDs), :]
    ## merge: Anontation Information
    df = pd.merge(seq_data, df, how='inner', on='ID')
    df.reset_index(drop=True, inplace=True)
    print("The Number of ID:", len(list(df['ID'].unique())))
    df['Read'] = df.apply(lambda row: update_targetSequence(row['Read'], row['Strand']), axis=1)
    return df


## Step 2: Parse ForeCasT Label
## Paning insertion
def Pan_insertion(single_inser, gRNASeq):
    inser_pos = int(single_inser[0][:-1])
    inser_seq = single_inser[1]
    continue_signal = True
    while (inser_pos < 38) & continue_signal:
        inser_nuc = inser_seq[0]
        comp_gRNASeq_nuc = gRNASeq[inser_pos - 1]
        if inser_nuc == comp_gRNASeq_nuc:
            inser_pos += 1
            inser_seq = inser_seq[1:] + inser_seq[0]
            continue_signal = True
        else:
            continue_signal = False
    new_single_inser = ('%sI' % (inser_pos), inser_seq)
    return new_single_inser


## Paning deletion
def Pan_deletion(single_delt, gRNASeq):
    import copy
    raw_single_delt = copy.deepcopy(single_delt)

    delt_inf = int(single_delt[0].split(':')[0])
    delt_sup = int(single_delt[0].split(':')[1][:-1])
    delt_seq = single_delt[1]
    delt_nuc = delt_seq[0]
    try:
        comp_gRNASeq_nuc = gRNASeq[delt_sup]
        while delt_nuc == comp_gRNASeq_nuc:
            delt_inf += 1
            delt_sup += 1
            delt_seq = delt_seq[1:] + delt_seq[0]
            comp_gRNASeq_nuc = gRNASeq[delt_sup]
    except IndexError as e:
        pass
    if delt_inf <= 43:  ## 没移出PAM外，不动
        return raw_single_delt
    else:
        return '0'


## 过滤 virus 中的编辑结果
def filter_edit_interv(vir_edit_inf, vir_edit_sup, define_edit_length=3, cut_site=37):
    ref_inf = cut_site - define_edit_length
    ref_sup = cut_site + define_edit_length
    if (vir_edit_sup < ref_inf) | (vir_edit_inf > ref_sup):
        return 0  ## 不在定义的编辑区间内
    else:
        return 1  ## 在定义的编辑区间内


## Parse ForeCasT Label
## ref_seq & read 都调整到 PAM_index = 40
def adjust_PAM_index(ref_seq, read, PAM_index):
    if PAM_index >= 40:
        ref_seq = ref_seq[(PAM_index - 40):]
        read = read[(PAM_index - 40):]
    else:
        ref_seq = '-' * (40 - PAM_index) + ref_seq
        read = '-' * (40 - PAM_index) + read
    return (ref_seq, read)


## 转换参考坐标系
def Transformer_Reference_System():
    ## Unified Format: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    ## ForeCasT Format: [-17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
    RS_dict = {}
    for i in range(85):
        Uone = i + 1
        Fone = i - 37
        RS_dict[Fone] = Uone
    return RS_dict


## Parse Indel Position
def Parse_Indel_Position(label):
    label_p = label.split('_')
    label_len = int(label_p[0][1:])
    label_pos = label_p[1]
    label_c = label_pos.split('C')  ## 确定是否存在微同源
    if len(label_c) == 2:  ## 存在
        start_pos = int(label_c[0].split('L')[-1])
        end_pos = int(label_pos.split('R')[-1])
        mh_len = int(label_c[1].split('R')[0])
    else:
        start_pos = int(label_pos.split('R')[0][1:])
        end_pos = int(label_pos.split('R')[-1])
        mh_len = 0
    return (label_len, start_pos, end_pos, mh_len)


## Unified Format
def Handle_Unified_Format(single_indel):
    if single_indel == '':
        return single_indel
    elif 'I' in str(single_indel):
        inser_nuc = single_indel[1]
        inser_len = len(inser_nuc)
        if inser_len >= 3:
            return 'I-3+'
        else:
            return 'I-%s_%s' % (inser_len, inser_nuc)
    else:  ## Deletion
        delt_len = len(single_indel[1])
        if delt_len >= 30:
            return 'D-30+'
        else:
            delt_pos = single_indel[0]
            return '%s-%s' % (delt_pos, delt_len)


## 转变 ForeCasT Label to Unified Format
def Transfomer_ForeCasT_Label_To_Unified_Format(label, ref_seq, read, PAM_index):
    ref_seq, read = adjust_PAM_index(ref_seq, read, PAM_index)
    RS_dict = Transformer_Reference_System()
    ## Parse indel position
    label_len, start_pos, end_pos, mh_len = Parse_Indel_Position(label)
    ## Transformer it to Read system
    if (label[0] == 'D') & (label_len >= 30):
        unified_format_indel = 'D-30+'
    else:
        new_start_pos = RS_dict[start_pos]
        new_end_pos = RS_dict[end_pos]
        ref_middle = ref_seq[new_start_pos:(new_end_pos - 1)]  ## only for deletion
        ## 区分 insertion & deletion
        ## Insertion
        if label[0] == 'I':
            read_middle = read[new_start_pos:(new_start_pos + mh_len + label_len)]
            ##
            inser_nucl = read_middle[:label_len]
            inser_pos = new_start_pos + 1
            single_indel = ("%sI" % (inser_pos), inser_nucl)
            ## step 1: 平移
            try:
                new_single_indel = Pan_insertion(single_indel, ref_seq)
            except IndexError as e:
                return 'Wrong'
            new_inser_pos = int(new_single_indel[0][:-1])
            ## step 2: 过滤，确定是否在编辑区间内
            filtered = filter_edit_interv(new_inser_pos, new_inser_pos, 3, 37)
            if filtered == 1:
                pass
            else:
                new_single_indel = ''
        ## Deletion
        elif label[0] == 'D':
            delt_nucl = ref_middle[:label_len]
            delt_inf_pos = new_start_pos + 1
            delt_sup_pos = delt_inf_pos + label_len - 1
            ##
            single_indel = ("%s:%sD" % (delt_inf_pos, delt_sup_pos), delt_nucl)
            ## step 1: 平移
            new_single_indel = Pan_deletion(single_indel, ref_seq)
            if new_single_indel == '0':  ## 移出 PAM 外
                new_single_indel = ''
            else:
                ## step 2: 过滤，确定是否在编辑区间内
                delt_inf = int(new_single_indel[0].split(':')[0])
                delt_sup = int(new_single_indel[0].split(':')[1][:-1])
                filtered = filter_edit_interv(delt_inf, delt_sup, 3, 37)
                if filtered == 1:
                    pass
                else:
                    new_single_indel = ''
        else:
            new_single_indel = ''
        ## 格式化
        unified_format_indel = Handle_Unified_Format(new_single_indel)
    return unified_format_indel


## 计算编辑类型个数: 混合编辑结果由 insertion & deletion
def count_edit_type(label):
    count = 0
    if 'I' in label:
        count += 1
    if 'D' in label:
        count += 1
    return count


## 计算 repair map & reads number
def helper_function(df, id_cols):
    df.reset_index(drop=False, inplace=True)
    df = df.groupby(id_cols + ['Unified_format_label']).sum()
    df.reset_index(drop=False, inplace=True)
    ## repair map
    temp = df[['Unified_format_label', 'Count']]
    temp.sort_values(by='Count', ascending=False, inplace=True)
    temp.index = temp['Unified_format_label']
    repair_map = str(dict(temp['Count']))
    read_num = df['Count'].sum()
    df['repair map'] = repair_map
    df['read_num'] = read_num
    del df['Count']
    del df['Unified_format_label']
    del df['index']
    data = df.loc[:0, :]
    return data


def main_unified_format_for_forecast(data, repair_time=1):
    data['f'] = data['ForeCasT_Label'].apply(lambda x: count_edit_type(x))
    data1 = data.loc[data['f']==repair_time, :]
    data1.reset_index(drop=True, inplace=True)
    data1['Unified_format_label'] = data1.apply(lambda row:
                                                Transfomer_ForeCasT_Label_To_Unified_Format(row['ForeCasT_Label'],
                                                                                            row['Reference Sequence'],
                                                                                            row['Read'],
                                                                                            row['PAM Index']), axis=1)
    return data1


def main(seq_data, read_data_dir, to_save_data_path):
    ## data integration
    Integrate_ForeCasT_Processed_Data(read_data_dir, to_save_data_path)
    ## Anotation
    data = pd.read_csv(to_save_data_path, sep='\t')
    data = Anotation_filtering(data, seq_data, oligo_type_list=[])
    data1 = main_unified_format_for_forecast(data, repair_time=1)
    data1 = data1.loc[data1['Unified_format_label']!='Wrong', :]
    data1.reset_index(drop=True, inplace=True)
    ## print
    print('Ratio of repair time > 1:', (data['Count'].sum() - data1['Count'].sum())*100/data['Count'].sum(), '%')
    kdata = data1.loc[data1['Unified_format_label']=='', :]
    data1 = data1.loc[data1['Unified_format_label']!='', :]
    data1.reset_index(drop=True, inplace=True)
    print("Ratio of not in edit interval:", kdata['Count'].sum()*100/data['Count'].sum(), '%')
    data1['f'] = data1['ForeCasT_Label'].apply(lambda x: int(x.split("_")[0][1:]))
    data1['label'] = data1['ForeCasT_Label'].apply(lambda x: x.split("_")[0][0])
    data1['diff'] = data1.apply(lambda row: abs(len(row['Reference Sequence']) - len(row['Read'])), axis=1)
    a = data1.loc[(data1['f']!=data1['diff']), :][['Count', 'label']].groupby('label').sum().sum()
    print("Ratio of Indel != diff:", a['Count']*100/data['Count'].sum(), '%')
    ## columns clearing
    del data1['f']
    del data1['label']
    del data1['diff']
    ## 汇集 each ID repair map
    id_cols = ['ID', 'Guide', 'Subset']
    data2 = data1[id_cols + ['Count', 'Unified_format_label']].groupby(by=id_cols, as_index=False).apply(lambda df:
                                                                                           helper_function(df, id_cols))
    data2.reset_index(drop=True, inplace=True)
    data3 = data1[['ID', 'Reference Sequence', 'PAM Index', 'gRNASeq_85bp', 'PAM']].drop_duplicates()
    data1 = pd.merge(data2, data3, how='inner', on='ID')
    data1.sort_values(by=['ID', 'read_num'], ascending=True, inplace=True)
    data1.drop_duplicates(subset=['Guide'], keep='last', inplace=True)
    ## save
    data1.to_csv(to_save_data_path, sep='\t', index=False)
    return data1


# execute
if __name__ == '__main__':
    main_dir = "./raw/2019_NBT_Allen"
    os.chdir(main_dir)
    data_dict = {
                 'testA1': "ST_April_2017_K562_800x_6OA_DPI7_Old8",
                 'testA2': "ST_April_2017_K562_800x_6OB_DPI7_Old11",
                 'testB': "ST_Feb_2018_CAS9_12NA_1600X_DPI7",
                 'RPE1': 'ST_Feb_2018_RPE1_500x_7B_DPI7_dec',
                 'CHO-A': 'ST_June_2017_CHO_LV7A_DPI7',
                 'CHO-B': 'ST_June_2017_CHO_LV7B_DPI7',
                 'E14TG2A-A': 'ST_June_2017_E14TG2A_LV7A_DPI7',
                 'E14TG2A-B': 'ST_June_2017_E14TG2A_LV7B_DPI7',
                 'HAP1-A': 'ST_June_2017_HAP1_LV7A_DPI7',
                 'HAP1-B': 'ST_June_2017_HAP1_LV7B_DPI7'
                }
    read_oligo_data_path = './41587_2019_BFnbt4317_MOESM72_ESM.txt'
    oligo_data = pd.read_csv(read_oligo_data_path, sep='\t')
    seq_data = main_forecast_get_85bp_reference_sequence(oligo_data)
    for data_label, read_name in data_dict.items():
        read_data_dir = './%s' % read_name
        to_save_data_path = './%s.log'%(read_name)
        print("\ndata name:", read_name)
        print('========================================')
        data = main(seq_data, read_data_dir, to_save_data_path)