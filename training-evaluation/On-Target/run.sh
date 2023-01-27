#!/usr/bin/env bash

# training AIdit_ON
main_dir="./"
read_train_data_path = "./demo datasets/OnTarget_efficiency_train.txt"
read_val_data_path = "./demo datasets/OnTarget_efficiency_val.txt"
ycol = "Indel_efficiency"
save_model_dir = "./AIdit_ON"

python main.py  ${main_dir} ${read_train_data_path} ${read_val_data_path} ${ycol} ${save_model_dir}
