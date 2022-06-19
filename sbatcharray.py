#!/software/anaconda3/2020.11/bin/python
#SBATCH -p action
#SBATCH --job-name=ind_traj_pldm
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 48:00:00
#SBATCH --output=test_%A_%a.out
#SBATCH --error=test_%A_%a.err

import sys, os
sys.path.append(os.popen("pwd").read().split("/tmpdir")[0]) # include parent directory which has method and model files
#-------------------------
import coupled_dimer as model
import time
import numpy as np
import run_pldm as rp

t0 = time.time()

JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"]) # get ID of this job
TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"]) # get ID of this task within the array

TrajFolder = "/scratch/mmondal/2DES_focused/Trajectories/"
NArray = 100
NTraj = model.NTraj
NStates = model.NStates
NTasks = NTraj//NArray
pulseNumber = 1

for itraj in range((int(TASKID)-1)*NTasks,int(TASKID)*NTasks):
    ρ_traj  = rp.simulate(pulseNumber,itraj,TrajFolder)
    for matEl in ρ_traj.keys():
        PijFile = TrajFolder + f"{itraj+1}/Pij_{matEl[0]}{matEl[1]}_{itraj+1}.txt"
        np.savetxt(PijFile,ρ_traj[matEl])
    
t1 = time.time()
print("Total time: ", t1-t0)