# Files Description

## For on-target
***
* [On-Target/data.py](On-Target/data.py) contains the code for extracting sequence features.    
* [On-Target/models.py](On-Target/models.py) contains the code for building the final model architecture and obtaining the model hyper-parameters.  
* [On-Target/main.py](On-Target/main.py) contains the code for training the CRISPR/Cas9-gRNA on-target algorithm.  
* [On-Target/evaluation functions.py](On-Target/evaluation functions.py) contains the code for evaluation of on-target models.  
* [On-Target/run.sh](On-Target/run.sh) is the script for running the training scripts of the CRISPR/Cas9-gRNA on-target model.
* [On-Target/demo datasets](On-Target/demo datasets) contains demo datasets for training CRISPR/Cas9-gRNA on-target model (The completed training dataset can be downloaded from in [the crispt-aidit website](https://crispr-aidit.com)).

## For on-target
***
* [Off-Target/data.py](Off-Target/data.py) contains the code for extracting sequence and biological features.    
* [Off-Target/models.py](Off-Target/models.py) contains the code for building the final off-target model architecture.  
* [Off-Target/main.py](Off-Target/main.py) contains the code for training the CRISPR/Cas9-gRNA off-target algorithm.  
* [Off-Target/demo datasets/off-target_model_parameters.log](Off-Target/demo datasets/off-target_model_parameters.log) contains the final off-target model hyper-parameters.  
* [Off-Target/evaluation functions.py](Off-Target/evaluation functions.py) contains the code for evaluation of off-target models.  
* [Off-Target/run.sh](Off-Target/run.sh) is the script for running the training scripts of the CRISPR/Cas9-gRNA off-target model.  
* [Off-Target/demo datasets](Off-Target/demo datasets) contains some demo datasets for training the CRISPR/Cas9-gRNA off-target model (The completed training dataset can be downloaded from in [the crispt-aidit website](https://crispr-aidit.com)).  

## For on-target
***
* [DSB_repair/DSB_Repair_Feature_and_Categories.py](DSB_repair/DSB_Repair_Feature_and_Categories.py) contains the code or extracting sequence and micro-homology features.  
* [DSB_repair/DSB_Repair_Engineered_Feature.py](DSB_repair/DSB_Repair_Engineered_Feature.py) contains the code for extracting xgboosts' prediction features.    
* [DSB_repair/train_xgboost_for_each_category.py](DSB_repair/train_xgboost_for_each_category.py) contains the code for training each xgboost for each DSB-induced repair category.  
* [DSB_repair/main.py](DSB_repair/main.py) contains the code for training the algorithm of the CRISPR/Cas9-gRNA DSB-induced repair outcome.  
* [DSB_repair/evaluation functions.py](DSB_repair/evaluation functions.py) contains the code for evaluation of DSB-induced repair models.  
* [DSB_repair/run.sh](DSB_repair/run.sh) is the script for running the training scripts of the CRISPR/Cas9-gRNA DSB-induced repair model.
* [DSB_repair/demo datasets](DSB_repair/demo datasets) contains some demo datasets for training the CRISPR/Cas9-gRNA oDSB-induced repair model (The completed training dataset can be downloaded from in [the crispt-aidit website](https://crispr-aidit.com)). 

