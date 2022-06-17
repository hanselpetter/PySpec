#!/software/anaconda3/2020.11/bin/python
#SBATCH -p standard
#SBATCH --job-name=ind_traj_pldm
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 48:00:00
#SBATCH --output=test_%A_%a.out
#SBATCH --error=test_%A_%a.err

import sys, os
sys.path.append(os.popen("pwd").read().split("/tmpdir")[0]) # include parent directory which has method and model files
#-------------------------
import pldm as method
import coupled_dimer as model
import time
import numpy as np

t0 = time.time()

JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"]) # get ID of this job
TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"]) # get ID of this task within the array

TrajFolder = "/scratch/mmondal/2DES_focused/Trajectories/"
NTraj = model.NTraj
NStates = model.NStates
NArray = 25
NTasks = NTraj//NArray

for itraj in range((int(TASKID)-1)*NTasks,int(TASKID)*NTasks):
    init_bath = np.loadtxt(TrajFolder + f"initial_bath_{itraj+1}.txt")
    initR = init_bath[:,0]
    initP = init_bath[:,1]

    os.chdir(TrajFolder)
    ρ_traj  = method.runTraj(initR,initP,itraj+1)
    PijFile = TrajFolder + f"Pij_{itraj+1}.txt"
    np.savetxt(PijFile,ρ_traj)
    os.chdir("../")

t1 = time.time()
print("Total time: ", t1-t0)