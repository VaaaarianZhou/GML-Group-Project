# GML-Group-Project
1. Please install and activate the environment as specified in _environment.yml_ as follows.
   ```
   conda env create -f environment.yml
   conda activate pyg_env
   ```
2. Please navigate to the working directory and run scripts GTNmodel.py, HANmodel.py, and RGCNmodel.py models to replicate the experimental results. These scripts will run a experiment with $threshold distance =$ {10,20,30,40,50}, and $topk$ = 10 by default.
```
   python GTNmodel.py
   python HANmodel.py
   python RGCNmodel.py
```
3. Please check /result/result.csv for the experimental result in csv format.
Please be reminded that the script is designed to run in a computing cluster, and the topk value is bound to the environmental variable **SLURM_ARRAY_TASK_ID**, which is default to 0 if not found. If you want to change the $topk$ value, you can ignore the **SLURM_ARRAY_TASK_ID** and set the first value manually.
