#!/usr/bin/env bash

# First: training xgboost
main_dir="./"
cell_line="K562"
read_train_data_path="./demo datasets/DSB_Repair_Map_train.txt"
read_val_data_path="./demo datasets/DSB_Repair_Map_val.txt"
read_int_data_path="./demo datasets/Table S4-K562_integrated_DSB_repair_category.xlsx"
save_model_dir="./AIdit_DSB"

python train_xgboost_for_each_category.py ${main_path} ${cell_line} ${read_train_data_path} ${read_val_data_path} ${read_int_data_path} ${save_model_dir}

# Next: training multiple-category regression
python main.py ${main_dir} ${cell_line} ${read_train_data_path} ${read_val_data_path} ${read_int_data_path} ${save_model_dir}
