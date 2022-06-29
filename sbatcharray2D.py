#!/software/anaconda3/2020.11/bin/python
#SBATCH -p exciton -A exciton
#SBATCH --job-name=ind_traj_pldm
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 48:00:00
#SBATCH --output=test_%A_%a.out
#SBATCH --error=test_%A_%a.err

import sys, os
from tkinter import N
sys.path.append(os.popen("pwd").read().split("/tmpdir")[0]) # include parent directory which has method and model files
#-------------------------
from PLDM import coupled_dimer as model
import nonlinearSpectra as nlS
# from PLDM import run_pldm as rp
import time
import numpy as np


t0 = time.time()

JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"]) # get ID of this job
TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"]) # get ID of this task within the array

TrajFolder = "/scratch/mmondal/2DES_focused/Trajectories/"
NArray = 1
NTraj = model.NTraj
NStates = model.NStates
NTasks = NTraj//NArray

for itraj in range((int(TASKID)-1)*NTasks,int(TASKID)*NTasks):
    R3_traj  = nlS.simulate(itraj,TrajFolder)
    R3FileRe = TrajFolder + f"{itraj+1}/R3ij_{itraj+1}_Re.txt"
    R3FileIm = TrajFolder + f"{itraj+1}/R3ij_{itraj+1}_Im.txt"
    # for matEl in R3_traj.keys():
    #     R3FileRe = TrajFolder + f"{itraj+1}/R3ij_{matEl[0]}{matEl[1]}_{itraj+1}_Re.txt"
    #     R3FileIm = TrajFolder + f"{itraj+1}/R3ij_{matEl[0]}{matEl[1]}_{itraj+1}_Im.txt"
    #     # np.savetxt(R3FileRe,np.array(R3_traj[matEl],dtype=int))
    np.savetxt(R3FileRe,np.real(R3_traj.reshape(model.NSteps1*model.NSteps2*model.NSteps3)))
    np.savetxt(R3FileIm,np.imag(R3_traj.reshape(model.NSteps1*model.NSteps2*model.NSteps3)))
t1 = time.time()
print("Total time: ", t1-t0)