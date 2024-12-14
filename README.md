# GML-Group-Project

## Datasets

The dataset we utilized is from [this website](https://datadryad.org/stash/landing/show?id=doi%3A10.5061%2Fdryad.g4f4qrfrc).

## Environment Setup
1. Please install and activate the environment as specified in _environment.yml_ as follows.
   ```
   conda env create -f environment.yml
   conda activate pyg_env
   ```
## Run Experiments
2. Now, if you are running in a _slurm_ based computing cluster, then please run the sbatch job script. Adjust the job description according to your resources, then you may skip to step 4; otherwise, if you are running on a computer, please go to step 3.
   ```
   sbatch HANmodel_training.job
   sbatch RGCNmodel_training.job
   sbatch GTNmodel_training.job
   ```
(optional) We utilized scanorama method for batch correction between different tissue regions. To obtain corrected expression matrix for training and evaluation, please run  
   ```
   python batch_correction.py B004_training_dryad.csv
   ```
If you want to run the cross-tissue annotation experiments, please run
   ```
   python HAN_CL_SB_training.py
   python RGCN_CL_SB_training.py
   ```
3. Please navigate to the working directory and run scripts GTNmodel.py, HANmodel.py, and RGCNmodel.py models to replicate the experimental results. These scripts will run a experiment with $threshold distance =$ {10,20,30,40,50}, and $topk$ = 10 by default. 
   ```
   python GTNmodel.py
   python HANmodel.py
   python RGCNmodel.py
   ```
   Please be reminded that the script was designed to run in a computing cluster, and the $topk$ value (default = 10) is bound to the environmental variable **SLURM_ARRAY_TASK_ID**, which is default to 0 if not found. If you want to change the $topk$ value, you can ignore the **SLURM_ARRAY_TASK_ID** and set it manually.

## View Results
4. Please check /result/result.csv for the experimental result in csv format. The scripts will also store the model parameters in a directory named **model** under the working directory.

