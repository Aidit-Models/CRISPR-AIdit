# Files Description

## For on-target
***
* [On-Target/data.py](On-Target/data.py) contains the code for extracting sequence features.    
* [On-Target/models.py](On-Target/models.py) contains the code for building the final model architecture and obtaining the model hyper-parameters.  
* [On-Target/main.py](On-Target/main.py) contains the code for training the CRISPR/Cas9-gRNA on-target algorithm.  
* [On-Target/evaluation functions.py](On-Target/evaluation%20functions.py) contains the code for evaluation of on-target models.  
* [On-Target/run.sh](On-Target/run.sh) is the script for running the training scripts of the CRISPR/Cas9-gRNA on-target model.
* [On-Target/demo datasets](On-Target/demo%20datasets) contains demo datasets for training CRISPR/Cas9-gRNA on-target model.

## For off-target
***
* [Off-Target/data.py](Off-Target/data.py) contains the code for extracting sequence and biological features.    
* [Off-Target/models.py](Off-Target/models.py) contains the code for building the final off-target model architecture.  
* [Off-Target/main.py](Off-Target/main.py) contains the code for training the CRISPR/Cas9-gRNA off-target algorithm.  
* [Off-Target/demo datasets/off-target_model_parameters.log](Off-Target/demo%20datasets/off-target_model_parameters.log) contains the final off-target model hyper-parameters.  
* [Off-Target/evaluation functions.py](Off-Target/evaluation%20functions.py) contains the code for evaluation of off-target models.  
* [Off-Target/run.sh](Off-Target/run.sh) is the script for running the training scripts of the CRISPR/Cas9-gRNA off-target model.  
* [Off-Target/demo datasets](Off-Target/demo%20datasets) contains some demo datasets for training the CRISPR/Cas9-gRNA off-target model.  

## For DSB repair
***
* [DSB_repair/DSB_Repair_Feature_and_Categories.py](DSB_repair/DSB_Repair_Feature_and_Categories.py) contains the code or extracting sequence and micro-homology features.  
* [DSB_repair/DSB_Repair_Engineered_Feature.py](DSB_repair/DSB_Repair_Engineered_Feature.py) contains the code for extracting xgboosts' prediction features.    
* [DSB_repair/train_xgboost_for_each_category.py](DSB_repair/train_xgboost_for_each_category.py) contains the code for training each xgboost for each DSB-induced repair category.  
* [DSB_repair/main.py](DSB_repair/main.py) contains the code for training the algorithm of the CRISPR/Cas9-gRNA DSB-induced repair outcome.  
* [DSB_repair/evaluation functions.py](DSB_repair/evaluation%20functions.py) contains the code for evaluation of DSB-induced repair models.  
* [DSB_repair/run.sh](DSB_repair/run.sh) is the script for running the training scripts of the CRISPR/Cas9-gRNA DSB-induced repair model.
* [DSB_repair/demo datasets](DSB_repair/demo%20datasets) contains some demo datasets for training the CRISPR/Cas9-gRNA oDSB-induced repair model. 

## Benchmark datasets (used for the corresponding model evaluation and comparison)
***
* [benchmark-datasets/On-Target/2016_GB_Haeussler/data_additional_14.csv](benchmark-datasets/On-Target/2016_GB_Haeussler/data_additional_14.csv) contains collection of guide sequences and their on-target frequencies from all published cleavage efficiency studies, which are collected by Haeussler et al.,.
* [benchmark-datasets/On-Target/self-endogenous on-target dataset.xlsx](benchmark-datasets/On-Target/self-endogenous%20on-target%20dataset.xlsx) contains guide sequences and their frequencies generated by our endogenous experiment.
* [benchmark-datasets/Off-Target/raw](benchmark-datasets/Off-Target/raw) contains raw datasets of off-targets and their frequencies from several off-target studies.
* [benchmark-datasets/Off-Target/processed](benchmark-datasets/Off-Target/processed) contains the corresponding processed off-target datasets and self guide-seq dataset, which are used for off-target model evaluation and comparison.
* [benchmark-datasets/DSB/raw](benchmark-datasets/DSB/raw) contains part of test and validation datasets of DSB repair profiles from Lindel and ForeCasT, respectively. Since the complete datasets are too large to upload, you can download those from the orignal websites found in the [datasets_source.txt](benchmark-datasets/DSB/raw/datasets_source.txt).
* [benchmark-datasets/DSB/raw/datasets_source.txt](benchmark-datasets/DSB/raw/datasets_source.txt) is a file which describes the download source of the test and validation datasets of DSB repair profiles from Lindel and ForeCasT, respectively.
* [benchmark-datasets/DSB/processed](benchmark-datasets/DSB/processed) contains the processed off-target datasets from Lindel and ForeCasT for model comparison and evaluation.
* [benchmark-datasets/DSB/unifying_foreCast_data_label_into_the_same_format.py](benchmark-datasets/DSB/unifying_foreCast_data_label_into_the_same_format.py) contains the code for unifying test or validation datasets from ForeCasT into the same format.
* [benchmark-datasets/DSB/unifying_forecast_prediction_into_the_same_format.py](benchmark-datasets/DSB/unifying_forecast_prediction_into_the_same_format.py) contains the code for unifying ForeCasT prediction into the same format.
* [benchmark-datasets/DSB/unifying_lindel_data_label_into_the_same_format.py](benchmark-datasets/DSB/unifying_lindel_data_label_into_the_same_format.py) contains the code for unifying test or validation datasets from Lindel into the same format.
* [benchmark-datasets/DSB/unifying_lindel_prediction_into_the_same_format.py](benchmark-datasets/DSB/unifying_lindel_prediction_into_the_same_format.py) contains the code for unifying Lindel prediction into the same format.

