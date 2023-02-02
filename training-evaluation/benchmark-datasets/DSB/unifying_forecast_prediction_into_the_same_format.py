import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# Step 1: Parse Predicted Reads Data
################# Batch Mode #################
def Parse_Predicted_Reads_Data(read_pred_reads_path):
    data_dict = {'ID': [], "Reference Sequence": [], "Read": [],  "ForeCasT PredLabel": []}
    with open(read_pred_reads_path, 'r') as f:
        for line in f:
            line = line.strip(' ')
            line = line.strip('\n')
            line_p = line.split('\t')
            if '@'*3 in line: # New ID
                ID = line_p[0].split('@')[-1]
                Ref_Seq = ''
            else:
                index, seq, pred_label = tuple(line_p)
                if int(index) == 0:
                    Ref_Seq = seq
                else:
                    data_dict['ID'].append(ID)
                    data_dict['Reference Sequence'].append(Ref_Seq)
                    data_dict['Read'].append(seq)
                    data_dict['ForeCasT PredLabel'].append(pred_label)
    # DataFrame
    pr_data = pd.DataFrame(data_dict)
    return pr_data


# Parse Predicted Indel Summary
def Parse_Predicted_Indel_Summary_Data(read_pred_indel_summary_path):
    data_dict = {'ID': [], 'ForeCasT PredLabel': [], 'Pred Count': []}
    with open(read_pred_indel_summary_path, 'r') as f:
        for line in f:
            line = line.strip(' ')
            line = line.strip('\n')
            line_p = line.split('\t')
            if '@'*3 in line: # New ID
                ID = line_p[0].split('@')[-1]
            else:
                pred_label, a, count = tuple(line_p)
                if pred_label == '-':
                    pass
                else:
                    data_dict['ID'].append(ID)
                    data_dict['ForeCasT PredLabel'].append(pred_label)
                    data_dict['Pred Count'].append(int(count))
    # DataFrame
    pidata = pd.DataFrame(data_dict)
    return pidata


################# Single Mode #################
# Single Parse Predicted Reads Data
def Single_Parse_Predicted_Reads_Data(read_pred_reads_path, PAM_Index):
    data_dict = {'ID': [], "Reference Sequence": [], "Read": [],  "PAM Index": [], "ForeCasT PredLabel": []}
    with open(read_pred_reads_path, 'r') as f:
        for line in f:
            line = line.strip(' ')
            line = line.strip('\n')
            line_p = line.split('\t')
            index, seq, pred_label = tuple(line_p)
            if int(index) == 0:
                if PAM_Index >= 20:
                    ID = seq[(PAM_Index-20):PAM_Index]
                else:
                    ID = seq[:PAM_Index]
                Ref_Seq = seq
            else:
                data_dict['ID'].append(ID)
                data_dict['Reference Sequence'].append(Ref_Seq)
                data_dict['Read'].append(seq)
                data_dict['PAM Index'].append(PAM_Index)
                data_dict['ForeCasT PredLabel'].append(pred_label)
    # DataFrame
    pr_data = pd.DataFrame(data_dict)
    return pr_data


# Single Parse Predicted Indel Summary
def Sngle_Parse_Predicted_Indel_Summary_Data(read_pred_indel_summary_path):
    data_dict = {'ForeCasT PredLabel': [], 'Pred Count': []}
    with open(read_pred_indel_summary_path, 'r') as f:
        for line in f:
            line = line.strip(' ')
            line = line.strip('\n')
            line_p = line.split('\t')
            pred_label, a, count = tuple(line_p)
            if pred_label == '-':
                pass
            else:
                data_dict['ForeCasT PredLabel'].append(pred_label)
                data_dict['Pred Count'].append(int(count))
    # DataFrame
    pidata = pd.DataFrame(data_dict)
    return pidata


# Step 2: Parse ForeCasT Label
# Paning insertion
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


# Paning deletion
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
    if delt_inf <= 43:
        return raw_single_delt
    else:
        return '0'


def filter_edit_interv(vir_edit_inf, vir_edit_sup, define_edit_length=3, cut_site=37):
    ref_inf = cut_site - define_edit_length
    ref_sup = cut_site + define_edit_length
    if (vir_edit_sup < ref_inf) | (vir_edit_inf > ref_sup):
        return 0
    else:
        return 1


# Parse ForeCasT Label
####################################################################
def adjust_PAM_index(ref_seq, read, PAM_index):
    if PAM_index >= 40:
        ref_seq = ref_seq[(PAM_index - 40):]
        read = read[(PAM_index - 40):]
    else:
        ref_seq = '-' * (40 - PAM_index) + ref_seq
        read = '-' * (40 - PAM_index) + read
    return (ref_seq, read)


def Transformer_Reference_System():
    # Unified Format: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    # ForeCasT Format: [-17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
    RS_dict = {}
    for i in range(85):
        Uone = i + 1
        Fone = i - 37
        RS_dict[Fone] = Uone
    return RS_dict


# Parse Indel Position
def Parse_Indel_Position(label):
    label_p = label.split('_')
    label_len = int(label_p[0][1:])
    label_pos = label_p[1]
    label_c = label_pos.split('C')
    if len(label_c) == 2:
        start_pos = int(label_c[0].split('L')[-1])
        end_pos = int(label_pos.split('R')[-1])
        mh_len = int(label_c[1].split('R')[0])
    else:
        start_pos = int(label_pos.split('R')[0][1:])
        end_pos = int(label_pos.split('R')[-1])
        mh_len = 0
    return (label_len, start_pos, end_pos, mh_len)


# Unified Format
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
    else:  # Deletion
        delt_len = len(single_indel[1])
        if delt_len >= 30:
            return 'D-30+'
        else:
            delt_pos = single_indel[0]
            return '%s-%s' % (delt_pos, delt_len)


# 转变 ForeCasT Label to Unified Format
def Transfomer_ForeCasT_Label_To_Unified_Format(label, ref_seq, read, PAM_index):
    ref_seq, read = adjust_PAM_index(ref_seq, read, PAM_index)
    RS_dict = Transformer_Reference_System()
    # Parse indel position
    label_len, start_pos, end_pos, mh_len = Parse_Indel_Position(label)

    # Transformer it to Read system
    new_start_pos = RS_dict[start_pos]
    new_end_pos = RS_dict[end_pos]
    ref_middle = ref_seq[new_start_pos:(new_end_pos - 1)]  # only for deletion

    # 区分 insertion & deletion
    # Insertion
    if label[0] == 'I':
        read_middle = read[new_start_pos:(new_start_pos + mh_len + label_len)]
        inser_nucl = read_middle[:label_len]
        inser_pos = new_start_pos + 1
        single_indel = ("%sI" % (inser_pos), inser_nucl)
        new_single_indel = Pan_insertion(single_indel, ref_seq)
        new_inser_pos = int(new_single_indel[0][:-1])
        filtered = filter_edit_interv(new_inser_pos, new_inser_pos, 3, 37)
        if filtered == 1:
            pass
        else:
            new_single_indel = ''
    # Deletion
    elif label[0] == 'D':
        delt_nucl = ref_middle[:label_len]
        delt_inf_pos = new_start_pos + 1
        delt_sup_pos = delt_inf_pos + label_len - 1
        single_indel = ("%s:%sD" % (delt_inf_pos, delt_sup_pos), delt_nucl)
        new_single_indel = Pan_deletion(single_indel, ref_seq)
        if new_single_indel == '0':
            new_single_indel = ''
        else:
            delt_inf = int(new_single_indel[0].split(':')[0])
            delt_sup = int(new_single_indel[0].split(':')[1][:-1])
            filtered = filter_edit_interv(delt_inf, delt_sup, 3, 37)
            if filtered == 1:
                pass
            else:
                new_single_indel = ''
    else:
        new_single_indel = ''
    unified_format_indel = Handle_Unified_Format(new_single_indel)
    return unified_format_indel


# Step 3: Integrated DSB Repair Label
# Get gRNASeq 85bp
def Get_ForeCasT_gRNASeq_85bp(TargetSequence, pam_index):
    if pam_index <= 20:
        up = ''
        target = '-'*(20 - pam_index) + TargetSequence[:pam_index]
        pam = TargetSequence[pam_index:(pam_index + 3)]
        down = TargetSequence[(pam_index + 3):]
    else:
        up = TargetSequence[:(pam_index-20)]
        target = TargetSequence[(pam_index-20):pam_index]
        pam = TargetSequence[pam_index:(pam_index + 3)]
        down = TargetSequence[(pam_index + 3):(pam_index + 45)]
    ##
    if len(up) >= 20:
        up = up[-20:]
    else:
        up = "-"*(20 - len(up)) + up
    # Final down stream
    if len(down) >= 42:
        down = down[:42]
    else:
        down = down + "-"*(42 - len(down))
    return up + target + pam + down


# From reads data to ID data
def Get_Integrated_Label_per_ID(kdata, label):
    # 获取 ForeCasT_Label
    Fdata = kdata[['ID', 'Guide', 'gRNASeq_85bp', label, 'Count']]
    Fdata = Fdata.groupby(['ID', 'Guide', 'gRNASeq_85bp', label]).sum()
    Fdata.reset_index(drop=False, inplace=True)
    print("\n获取 %s:"%(label), Fdata.shape, "ID:", len(list(Fdata['ID'].unique())), "Guide:", len(list(Fdata['Guide'].unique())))
    F_dict = {}
    for ID in Fdata.ID.unique():
        temp = Fdata.loc[Fdata['ID'] == ID, :]
        temp.index = temp[label]
        temp_dict = dict(temp['Count'])
        temp_sorted = sorted(temp_dict.items(), key=lambda kv: kv[1],reverse=True)
        one_label_dict = {}
        for one in temp_sorted:
            one_label_dict[one[0]] = one[1]
        # add
        F_dict[ID] = str(one_label_dict)
    # Integrated Data
    Fdata = kdata[['ID', 'Guide', 'gRNASeq_85bp', 'Count']]
    Fdata = Fdata.groupby(['ID', 'Guide', 'gRNASeq_85bp']).sum()
    Fdata.reset_index(drop=False, inplace=True)
    print("获取 Integrated Data", Fdata.shape, "ID:", len(list(Fdata['ID'].unique())), "Guide:", len(list(Fdata['Guide'].unique())))
    Fdata[label] = Fdata['ID'].apply(lambda x: F_dict[x])
    return Fdata


# Final Integrated Data: From reads data to ID data
def main_ForeCasT_Integrated_Data(pdata):
    # Get gRNASeq_85bp & Guide
    pdata['gRNASeq_85bp'] = pdata.apply(lambda row: Get_ForeCasT_gRNASeq_85bp(row['Reference Sequence'], row['PAM Index']), axis=1)
    pdata['Guide'] = pdata['gRNASeq_85bp'].apply(lambda x: x[20:40])
    pdata.rename(columns={'Pred Count': 'Count'}, inplace=True)
    # Integrate: ForeCasT Label
    Fpdata = Get_Integrated_Label_per_ID(pdata, "ForeCasT PredLabel")
    Updata = Get_Integrated_Label_per_ID(pdata, "Unified_Format PredLabel")
    del Fpdata['Count']
    final_data = pd.merge(Fpdata, Updata, how='inner', on=['ID', 'Guide', 'gRNASeq_85bp'])
    final_data.reset_index(drop=True, inplace=True)
    print(final_data.shape)
    return final_data


# Unifing ForeCasT prediciton to the same format
def main_forecast_format_unifing(read_input_data_path, read_pred_indel_summary_path, read_pred_reads_path,
                                 save_PredData_prefix, Mode='Batch', PAM_Index=40):
    # Step 1: Parse Predicted Reads Data
    print("")
    print("============= Mode: %s =============" % (Mode))
    # Parse Predicted Reads & Predicted Indel Summary
    print("\nStep 1: Parse Predicted Reads Data ")
    if Mode == "Batch":
        ipdata = pd.read_csv(read_input_data_path, sep='\t')
        prdata = Parse_Predicted_Reads_Data(read_pred_reads_path)
        pidata = Parse_Predicted_Indel_Summary_Data(read_pred_indel_summary_path)
        print(pidata.shape, prdata.shape)
        pdata = pd.merge(prdata, pidata, how='inner', on=['ID', 'ForeCasT PredLabel'])
        pdata = pd.merge(pdata, ipdata[['ID', 'PAM Index']], how='inner', on=['ID'])
        print(pdata.shape)
    elif Mode == "Single":
        prdata = Single_Parse_Predicted_Reads_Data(read_pred_reads_path, PAM_Index)
        pidata = Sngle_Parse_Predicted_Indel_Summary_Data(read_pred_indel_summary_path)
        print(pidata.shape, prdata.shape)
        pdata = pd.merge(prdata, pidata, how='inner', on='ForeCasT PredLabel')
        print(pdata.shape)
    else:
        pdata = ''
        print("Error (Mode input): Mode just two selections: either Mode or Single. Please check and try again.")
    print('======================================================')

    # Step 2: Parse ForeCasT Label
    print("\nStep 2: Parse ForeCasT Label")
    pdata['Unified_Format PredLabel'] = pdata.apply(
        lambda row: Transfomer_ForeCasT_Label_To_Unified_Format(row['ForeCasT PredLabel'],
                                                                row['Reference Sequence'], row['Read'],
                                                                row['PAM Index']), axis=1)
    pdata = pdata[['ID', 'Reference Sequence', 'Read', 'PAM Index', 'Pred Count', 'ForeCasT PredLabel',
                   'Unified_Format PredLabel']]
    # To save
    save_read_level_ForeCasT_PredData_path = save_PredData_prefix + '_ReadsLevel_ForeCasT_PredData.log'
    pdata.to_csv(save_read_level_ForeCasT_PredData_path, sep='\t', index=False)
    print('======================================================')

    # Step 3: Integrated DSB Repair Label
    print("\nStep 3: Integrated DSB Repair Label")
    final_data = main_ForeCasT_Integrated_Data(pdata)
    final_data = final_data[['ID', 'Guide', 'gRNASeq_85bp', 'Count', 'ForeCasT PredLabel', 'Unified_Format PredLabel']]
    # To save
    save_ForeCasT_PredData_path = save_PredData_prefix + '_Integrated_ForeCasT_PredData.log'
    final_data.to_csv(save_ForeCasT_PredData_path, sep='\t', index=False)
    print('======================================================')
    return (pdata, final_data)


# main function
def main(param):
    '''
    Execute
    # For Single Mode:
    param =  (_predictedindelsummary_file, _predictedindelreads_file, output_prefix, PAM_Index)
    # For Batch Mode:
    param =  (raw_data_file, _predictedindelsummary_file, _predictedindelreads_file, output_prefix)
    :return two files: output_prefix + '_ReadsLevel_ForeCasT_PredData.log'; output_prefix + '_Integrated_ForeCasT_PredData.log'
    pdata, final_data = main(param)
    ##############################################################################################################
    ## Example: Single Mode

    read_pred_indel_summary_path = 'test_output_predictedindelsummary.txt'
    read_pred_reads_path = 'test_output_predictedreads.txt'
    save_PredData_prefix = 'test_unified_format'
    Mode = 'Single'
    PAM_Index = 17
    param = (read_pred_indel_summary_path, read_pred_reads_path, save_PredData_prefix, Mode, PAM_Index)
    pdata, final_data = main(param)
    ####################################################################################
    ## Example: Batch Mode

    read_input_data_path = 'Input_ForeCasT_Validation_RepA1_Data.log'
    read_pred_indel_summary_path = 'Output_ForeCasT_Validation_RepA1_Data_predictedindelsummary.txt'
    read_pred_reads_path = 'Output_ForeCasT_Validation_RepA1_Data_predictedreads.txt'
    save_PredData_prefix = 'ForeCasT_Validation_RepA1'
    param = (read_input_data_path, read_pred_indel_summary_path, read_pred_reads_path, save_PredData_prefix)
    pdata, final_data = main(param)
    '''
    try:  # Single  Mode
        read_pred_indel_summary_path, read_pred_reads_path, save_PredData_prefix, PAM_Index = param
        PAM_Index = int(PAM_Index)
        print('\n==============================================')
        print('================ Mode: Single ===============')
        print('==============================================')
        pdata, final_data = main_forecast_format_unifing('',
                                                         read_pred_indel_summary_path,
                                                         read_pred_reads_path,
                                                         save_PredData_prefix,
                                                         "Single",
                                                         PAM_Index)
    except ValueError as e: # Batch Mode
        print('\n==============================================')
        print('================= Mode: Batch ================')
        print('==============================================')
        read_input_data_path, read_pred_indel_summary_path, read_pred_reads_path, save_PredData_prefix = param
        pdata, final_data = main_forecast_format_unifing(read_input_data_path,
                                                         read_pred_indel_summary_path,
                                                         read_pred_reads_path,
                                                         save_PredData_prefix)
    print("Finish.")
    return pdata, final_data

