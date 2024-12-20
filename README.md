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
   ```
   (optional) We utilized scanorama method for batch correction between different tissue regions. To obtain corrected expression matrix for training and evaluation, please run  
      ```
      python batch_correction.py B004_training_dryad.csv
      ```
3. Please navigate to the working directory, if you want to run the **cross-tissue** annotation experiments, please run
      ```
      python HAN_CL_SB_training.py
      python RGCN_CL_SB_training.py
      ```
   To reproduce experimental result for **intra-region** annotation, please run
      ```
      python han_training.py
      python rgcn_training.py
      ```
## View Results
4. Please check /result/result.csv for the experimental result in csv format. The scripts will also store the model parameters in a directory named **model** under the working directory.

