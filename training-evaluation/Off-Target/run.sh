#!/usr/bin/env bash

# training AIdit_OFF
main_dir="./"
read_train_path = "./demo datasets/OffTarget_efficiency_train.txt"
read_val_path = "./demo datasets/OffTarget_efficiency_val.txt"
read_params_path = "./demo datasets/off-target_model_parameters.log"
y_col = "Efficiency"
save_model_path = "./AIdit_OFF/AIdit_OFF.model"
python main.py  ${main_dir} ${read_train_path} ${read_val_path} ${read_params_path} ${y_col} ${save_model_path}