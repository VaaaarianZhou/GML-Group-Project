# GML-Group-Project
Please run scripts GTNmodel.py, HANmodel.py, and RGCNmodel.py models to replicate the experimental results. These scripts will run a experiment with $threshold distance =$ {10,20,30,40,50}, and $topk$ = 10 by default. 

Please be reminded that the script is designed to run in a computing cluster, and the topk value is bound to the environmental variable **SLURM_ARRAY_TASK_ID**, which is default to 0 if not found. If you want to change the $topk$ value, you can ignore the **SLURM_ARRAY_TASK_ID** and set the first value manually.
