#!/software/anaconda3/2020.11/bin/python
#SBATCH -p action
#SBATCH --job-name=specproj
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -t 5-00:00:00
#SBATCH --output=test_%A_%a.out
#SBATCH --error=test_%A_%a.err

import sys, os
from tkinter import N
sys.path.insert(0, "/scratch/mmondal/specTest/PLDM")
sys.path.append(os.popen("pwd").read().split("/tmpdir")[0]) # include parent directory which has method and model files
#-------------------------

import coupled_dimer as model
import linearSpectra as lS
import time
import numpy as np

t0 = time.time()

JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"]) # get ID of this job
TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"]) # get ID of this task within the array

TrajFolder = "/scratch/mmondal/specTest/linearSpec/Trajectories/"
NArray = 100
NTraj = model.NTraj
NStates = model.NStates
NTasks = NTraj//NArray

for itraj in range((int(TASKID)-1)*NTasks,int(TASKID)*NTasks):
    ρTraj = lS.simulate(itraj,TrajFolder)
    for i in ρTraj.keys():
        ρij = ρTraj[i]
        PijFile  = TrajFolder +  f"{itraj+1}/{i[0]}{i[1]}.txt"
        np.savetxt(PijFile,ρij)

t1 = time.time()
print("Total time: ", t1-t0)