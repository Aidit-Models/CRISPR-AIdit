import pandas as pd
import warnings
warnings.filterwarnings('ignore')

''' 
After Lindel prediction, the function unified_format_for_one_label_dict() can unify each dsb repair outcome predicted by
Lindel into the same format by inputting parameter (one dsb repair outcome predicted by Lindel, 85bp reference sequence).
'''


# deletion
def Pan_deletion_Left(read_left, delt_nucles, read_right):
    while True:
        try:
            delt_one_nucle = delt_nucles[-1]  # right one nucle of deletion
            read_left_one_nucle = read_left[-1]  # right one nucle of read_left
            if delt_one_nucle == read_left_one_nucle:
                read_left = read_left[:-1]
                delt_nucles = delt_one_nucle + delt_nucles[:-1]
                read_right = delt_one_nucle + read_right
            else:
                break
        except IndexError as e:
            break
    return read_left, delt_nucles, read_right


# to unify
def unified_format_for_one_label_dict(one_label_dict, gRNASeq_85bp):
    b_repair_dict = {'label': [], 'ratio': []}
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
                # deletion
                left_seq, delt_seq, right_seq = Pan_deletion_Left(left_seq, delt_seq, right_seq)
                new_start_pos = len(left_seq) + 1
                new_end_pos = new_start_pos + length -1
                b_repair_dict['label'].append('%s:%sD-%s'%(new_start_pos, new_end_pos, length))
                b_repair_dict['ratio'].append(ratio)
            except ValueError as e:
                b_repair_dict['label'].append('I-%s_%s'%(start_pos, aft))
                b_repair_dict['ratio'].append(ratio)
        else:
            b_repair_dict['label'].append('I-3+')
            b_repair_dict['ratio'].append(ratio)
    # DataFrame
    df_repair = pd.DataFrame(b_repair_dict)
    df_repair['ratio'] = df_repair['ratio'].astype('float')
    df_repair = df_repair.groupby('label').sum()
    df_repair.reset_index(drop=False, inplace=True)
    df_repair.sort_values(by='ratio', ascending=False, inplace=True)
    df_repair.index = df_repair['label']
    repair_dict = dict(df_repair['ratio'])
    return repair_dict