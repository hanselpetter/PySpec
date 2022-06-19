#!/software/anaconda3/2020.11/bin/python
#SBATCH -p action
#SBATCH -o 3e2_5_20_mu_1e4_action.log
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

import os, sys
import subprocess
import time
import numpy as np
from pathlib import Path


NARRAY = str(100) # number of jobs
filename = "ind_traj_pldm"

# JOBID = str(9591013)
JOBID = str(os.environ["SLURM_JOB_ID"]) # get ID of this job
Path("tmpdir_" + JOBID).mkdir(parents=True, exist_ok=True) # make temporary directory for individual job files
os.chdir("tmpdir_" + JOBID) # change to temporary directory
command = str("sbatch -W --array [1-" + NARRAY + "] ../sbatcharray.py") # command to submit job array

open(filename,'a').close()

t0 = time.time()

# ARRAYJOBID = str(9591014)
ARRAYJOBID = str(subprocess.check_output(command, shell=True)).replace("b'Submitted batch job ","").replace("\\n'","") # runs job array and saves job ID of the array

t1 = time.time()
print("Array time: ",t1-t0)

os.chdir("..") # go back to original directory

# os.system("rm -rf tmpdir_" + JOBID) # delete temporary folder (risky since this removes original job file data)