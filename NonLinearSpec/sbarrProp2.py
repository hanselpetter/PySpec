#!/software/anaconda3/2020.11/bin/python
#SBATCH -p exciton -A exciton
#SBATCH --job-name=2DmanPar
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 5-00:00:00
#SBATCH --output=test_%A_%a.out
#SBATCH --error=test_%A_%a.err

import sys, os
sys.path.insert(0, "/scratch/mmondal/specTest/PLDM")
sys.path.append(os.popen("pwd").read().split("/tmpdir")[0]) # include parent directory which has method and model files
#-------------------------
import coupled_dimer as model
import specFunctions as sF
import secondLaser as sL

import time
import numpy as np

t0 = time.time()

JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"]) # get ID of this job
TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"]) # get ID of this task within the array

TrajFolder = "/scratch/mmondal/specTest/spectra/Trajectories/"
NArray = 100
NJobs = model.NSteps1
NTasks = NJobs//NArray
NRem = NJobs - (NTasks*NArray)
TaskArray = [i for i in range((int(TASKID)-1) * NTasks , int(TASKID) * NTasks)]
if (int(TASKID)-1) in range(NRem):
    extJobIndex = NTasks*NArray + int(TASKID)-1
    TaskArray.append(extJobIndex)

pl1 = sF.savStpRem(model.NSteps1, model.nskip1)
pl2 = sF.savStpRem(model.NSteps2, model.nskip2)
pl3 = sF.savStpRem(model.NSteps3, model.nskip3)
R3dim1 = (model.NSteps1//model.nskip1) + pl1
R3dim2 = (model.NSteps2//model.nskip2) + pl2
R3dim3 = (model.NSteps3//model.nskip3) + pl3
R3 = np.zeros((R3dim1, R3dim2, R3dim3), dtype=complex)
for ij in range(sL.n0.shape[0]):
    R3 += sL.simulate(ij, 0, TrajFolder, TaskArray)

np.savetxt(TrajFolder + f"1/R3_{TASKID}_Re.txt",np.real(R3).flatten()) 
np.savetxt(TrajFolder + f"1/R3_{TASKID}_Im.txt",np.imag(R3).flatten())

t1 = time.time()
print("Total time: ", t1-t0)