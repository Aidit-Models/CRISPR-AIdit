import os
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')


# get Guide 85bp sequences
def adjust_Reference(Reference):
    a = Reference.split('-')
    b = [part for part in a if part != '']
    raw_reference = ''.join(b)
    return raw_reference


def fix_index(Reference, Raw_reference, Guide_index):
    i = 0
    for j in range(Guide_index):
        nucle = Raw_reference[j]
        while Reference[i] != nucle:
            i += 1
        i += 1
    return i


## generate gRANSeq_85bp
def generate_lindel_granseq_85bp(RefSeq, PAM_index):
    up = RefSeq[:(PAM_index-1)]
    target = RefSeq[(PAM_index-1):(PAM_index+22)]
    down = RefSeq[(PAM_index+22):]
    if len(up) >= 20:
        up = up[-20:]
    else:
        up = '-'*(20-len(up)) + up
    ## down
    if len(down) >= 42:
        down = down[:42]
    else:
        down = down + '-' * (42-len(down))
    return up + target + down


def single_lindel_data_get_granseq_85bp(aliData):
    aliData['PAM Index'] = aliData.apply(lambda row: row['Reference Sequence'].index(row['Guide'][1:]), axis=1)
    aliData['gRNASeq_85bp'] = aliData.apply(lambda row: generate_lindel_granseq_85bp(row['Reference Sequence'],
                                                                                     row['PAM Index']), axis=1)
    aliData['PAM'] = aliData['gRNASeq_85bp'].apply(lambda x: x[40:43])
    print('PAM:\n', aliData['PAM'].value_counts())
    seq_data = aliData[['Guide', 'gRNASeq_85bp']]
    seq_data.drop_duplicates(inplace=True)
    return seq_data


## Get Lindel 85bp guide reference sequence
def get_85bp_reference_sequence(read_data_path):
    import pickle
    with open(read_data_path, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    ## transformer data_dict into dataframe
    data = pd.DataFrame(data_dict)
    ## adding column name
    new_data = data[[6, 9, 10, 11, 12, 0, 1]]
    new_data.rename(columns = {6: 'Guide', 9: 'Indel', 10: 'Start position', 11: 'Length', 12: 'Insertion sequence',
                               0: 'Read', 1: 'Reference'}, inplace=True)
    new_data['Count'] = 1
    ## adjust sequence
    new_data['Raw_reference'] = new_data['Reference'].apply(lambda x: adjust_Reference(x))
    new_data['Guide_index'] = new_data.apply(lambda row: row['Raw_reference'].index(row['Guide']), axis=1)
    new_data['f'] = new_data['Guide_index'].apply(lambda x: 1 if x < 20 else 0)
    new_data = new_data.loc[new_data['f']==0, :]
    new_data.reset_index(drop=True, inplace=True)
    new_data['Reference Sequence'] = new_data.apply(lambda row: row['Raw_reference'][(row['Guide_index']-20):(row['Guide_index'] + 65)],
                                              axis=1)
    ## update Referenece Sequence column
    aliData = new_data[['Guide', 'Reference Sequence']]
    aliData.drop_duplicates(inplace=True)
    aliData.reset_index(drop=False, inplace=True)
    aliData['f'] = aliData["Reference Sequence"].apply(lambda x: len(x))
    a = aliData.loc[aliData['f']==85, ['Guide', 'Reference Sequence']]
    a = a.groupby(['Guide', 'Reference Sequence']).count()
    a.reset_index(drop=False, inplace=True)
    a.index = a['Guide']
    a_dict = dict(a['Reference Sequence'])
    aliData['Reference Sequence'] = aliData.apply(lambda row: a_dict[row['Guide']]if row['Guide'] in a_dict else row['Reference Sequence'], axis=1)
    del aliData['f']
    aliData.drop_duplicates(inplace=True)
    aliData.reset_index(drop=False, inplace=True)
    # get 85bp gRNASeq
    seq_data = single_lindel_data_get_granseq_85bp(aliData)
    return seq_data


# Main Function:get lindel 85bp reference sequence
def main_lindel_get_85bp_reference_sequence():
    ## get lindel 85bp gRNASequence
    seq_data_list = []
    for i in range(1, 4, 1):
        read_data_path = "./NHEJ_rep%s_final_matrix.pkl"%(i)
        seq_data = get_85bp_reference_sequence(read_data_path)
        seq_data_list.append(seq_data)
    # concat
    seq_data = pd.concat(seq_data_list, axis=0)
    seq_data.drop_duplicates(['Guide'], keep='first', inplace=True)
    seq_data.reset_index(drop=True, inplace=True)
    return seq_data


## Get Lindel Training Labels
def get_lindel_labels(feature_index_path):
    import pickle
    with open(feature_index_path, 'rb') as fo:
        dict_data1 = pickle.load(fo, encoding='bytes')
    Lindel_label_dict = dict_data1[1]
    Lindel_labels = [Lindel_label_dict[i] for i in range(557)]
    return Lindel_labels


## Get Lindel Label Data
def get_lindel_label_data(read_data_path, feature_index_path):
    lindel_train = pd.read_csv(read_data_path, sep='\t', header=None)
    Lindel_labels = get_lindel_labels(feature_index_path)
    ##
    label_data = lindel_train.iloc[:, -557:]
    label_data.columns = Lindel_labels
    lindel_label_data = pd.concat([lindel_train[[0]],label_data], axis=1)
    lindel_label_data.rename(columns={0: 'Guide'}, inplace=True)
    return lindel_label_data


## 平移 deletion
def pan_deletion_left(read_left, delt_nucles, read_right):
    while True:
        delt_one_nucle = delt_nucles[-1]
        read_left_one_nucle = read_left[-1]
        if delt_one_nucle == read_left_one_nucle:
            read_left = read_left[:-1]
            delt_nucles = delt_one_nucle + delt_nucles[:-1]
            read_right = delt_one_nucle + read_right
        else:
            break
    return (read_left, delt_nucles, read_right)


def unified_format_For_One_label_dict(one_label_dict, gRNASeq_85bp):
    repair_dict = {}
    for repair_class, ratio in one_label_dict.items():
        repair_class_p = repair_class.split('+')
        if len(repair_class_p) == 2:
            start_pos = int(repair_class_p[0])
            aft = repair_class_p[1]
            try:
                length = int(aft)
                left_seq = gRNASeq_85bp[:(38 + start_pos - 1)]
                delt_seq = gRNASeq_85bp[(38 + start_pos - 1):(38 + start_pos - 1 + length)]
                right_seq = gRNASeq_85bp[(38 + start_pos - 1 + length):]
                ## deletion 平移最左端
                left_seq, delt_seq, right_seq = pan_deletion_left(left_seq, delt_seq, right_seq)
                new_start_pos = len(left_seq) + 1
                new_end_pos = new_start_pos + length -1
                repair_dict['%s:%sD-%s'%(new_start_pos, new_end_pos, length)] = ratio
            except ValueError as e:
                repair_dict['I-%s_%s'%(start_pos, aft)] = ratio
        else:
            repair_dict['I-3+'] = ratio
    return repair_dict


## input：data [Guide, gRNASeq_85bp, Lindel_Modeling_Labels]
## output：sta_data [Guide, gRNASeq_85bp, Lindel_Modeling_Label, DSB_Repair_map]
def unified_lindel_dataset_label(data):
    label_data = data.iloc[:, 2:]
    sta_data_dict = {'Guide': [], 'gRNASeq_85bp': [], 'Lindel_Modeling_Label': [], 'Unified_format': []}
    for index, row in data.iterrows():
        Guide = row['Guide']
        gRNASeq_85bp = row['gRNASeq_85bp']
        temp_s = label_data.loc[index, :]
        temp_s = temp_s[temp_s>0]
        temp_dict = dict(temp_s)
        temp_sorted = sorted(temp_dict.items(), key=lambda kv: kv[1],reverse=True)
        one_label_dict = {}
        for one in temp_sorted:
            one_label_dict[one[0]] = one[1]
        ## input
        sta_data_dict['Guide'].append(Guide)
        sta_data_dict['gRNASeq_85bp'].append(gRNASeq_85bp)
        sta_data_dict['Lindel_Modeling_Label'].append(str(one_label_dict))
        repair_dict = unified_format_For_One_label_dict(one_label_dict, gRNASeq_85bp)
        sta_data_dict['Unified_format'].append(str(repair_dict))
    ## DataFrame
    unidata = pd.DataFrame(sta_data_dict)
    return unidata


def main_unified_lindel_dataset_format(seq_data, read_data_path, feature_index_path, to_save_data_path):
    ## lindel label data
    lindel_label_data = get_lindel_label_data(read_data_path, feature_index_path)
    ## gRNASeq 85bp data
    seq_data['Guide0'] = seq_data['Guide'].apply(lambda x: x[1:])
    lindel_label_data['Guide0'] = lindel_label_data['Guide'].apply(lambda x: x[1:])
    print("Before, data.shape:", lindel_label_data.shape)
    data = pd.merge(seq_data[['Guide0', 'gRNASeq_85bp']], lindel_label_data, how='inner', on='Guide0')
    data.reset_index(drop=True, inplace=True)
    del data['Guide0']
    print("After merging, data.shape:", data.shape)
    ## 统一格式
    unidata = unified_lindel_dataset_label(data)
    ## to save
    unidata.to_csv(to_save_data_path, sep='\t', index=False)
    return unidata


## execute
if __name__ == '__main__':
    main_path = r'./raw/2019_NAR_Chen'
    os.chdir(main_path)

    ## For Lindel_test
    feature_index_path = "./feature_index_all.pkl"
    read_data_path = "./Lindel_test.txt"
    to_save_data_path = './Lindel_Test_DSB_Repair_map.log'
    # get lindel 85bp reference sequence
    lindel_seq_data = main_lindel_get_85bp_reference_sequence()
    data = main_unified_lindel_dataset_format(lindel_seq_data,
                                              read_data_path,
                                              feature_index_path,
                                              to_save_data_path)
