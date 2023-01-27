# Files Description

## For on-target
***
* [on-target/data.py](on-target/data.py) contains the code for extracting sequence features.    
* [on-target/on_target_predict.py](on-target/on_target_predict.py) contains the code for predicting CRISPR/Cas9-gRNA on-target activity.  
* [on-target/demo_dataset.txt](on-target/demo_dataset.txt) is a demo dataset for predicting CRISPR/Cas9-gRNA on-target activity.  
* [on-target/run.sh](on-target/run.sh) is the script for an example of running gRNA on-target predictions.

## For off-target
***
* [off-target/off_target_features.py](off-target/off_target_features.py) contains the code for extracting sequence and biological features.    
* [off-target/off_target_predict.py](off-target/off_target_predict.py) contains the code for predicting CRISPR/Cas9-gRNA off-target activity.  
* [off-target/demo_dataset.txt](off-target/demo_dataset.txt) is a demo dataset for predicting CRISPR/Cas9-gRNA off-target activity.  
* [off-target/run.sh](off-target/run.sh) is the script for an example of running gRNA on-target predictions.


## For DSB_repair
***
* [DSB_repair/data.py](DSB_repair/data.py) contains the code for extracting sequence and biological features.    
* [DSB_repair/DSB_repair_predict.py](DSB_repair/DSB_repair_predict.py) contains the code for predicting CRISPR/Cas9-gRNA DSB-induced repair outcomes.  
* [DSB_repair/demo_dataset.txt](DSB_repair/demo_dataset.txt) is a demo dataset for predicting CRISPR/Cas9-gRNA DSB-induced repair outcomes.  
* [DSB_repair/run.sh](DSB_repair/run.sh) is the script for an example of running the prediction of CRISPR/Cas9-gRNA DSB-induced repair outcomes.

## For models
***
* [models/on-target/K562](models/on-target/K562) contains final model files of AIdit_ON (trained by using on-target dataset in K562).  
* [models/on-target/jurkat](models/on-target/jurkat) contains final model files of AIdit_ON_Jurkat (trained by using on-target dataset in Jurkat).  

* [models/off-target/off-target-best-model.model](models/off-target/off-target-best-model.model) is the final model files of AIdit_OFF.    

* [models/DSB_repair/Jurkat](models/DSB_repair/Jurkat) contains final model files of XGBoost predicting each repair category and a multi-category regression regression for Jurkat.  
* [models/DSB_repair/K562](models/DSB_repair/K562) contains final model files of XGBoost predicting each repair category and a multi-category regression for K562.    
* [models/DSB_repair/Integrated_Jurkat_DSB_Repair_Merged_Information.csv](models/DSB_repair/Integrated_Jurkat_DSB_Repair_Merged_Information.csv) is used to integrate DSB repair categories of Jurkat as described in the article.  
* [models/DSB_repair/Integrated_K562_DSB_Repair_Merged_Information.csv](models/DSB_repair/Integrated_K562_DSB_Repair_Merged_Information.csv) is used to integrate DSB repair categories of K562 as described in the article.



