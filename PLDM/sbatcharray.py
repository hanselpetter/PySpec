#!/software/anaconda3/2020.11/bin/python
#SBATCH -p action
#SBATCH --job-name=sbatcharray
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=5GB
#SBATCH -t 1:00:00
#SBATCH --output=test_%A_%a.out
#SBATCH --error=test_%A_%a.err

import sys, os
sys.path.append(os.popen("pwd").read().split("/tmpdir")[0]) # include parent directory which has method and model files
#-------------------------
import LMFE as method
import model_PMET as model
import time
import numpy as np

t0 = time.time()

JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"]) # get ID of this job
TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"]) # get ID of this task within the array

result = method.runTraj(model.parameters()) # run the method

# Write individual job data to files

NStates = model.NStates
nskip = model.nskip
NTraj = model.NTraj
NR = model.NR
NSteps = model.NSteps

rho_sum = np.zeros(result[0].shape, dtype = result[0].dtype)
psiD = np.zeros((NStates,(NSteps//nskip)), dtype = result[1].dtype)
R_t = np.zeros((NR+1,(NSteps//nskip)), dtype = result[2].dtype)
rhist = np.zeros(result[3].shape, dtype = result[3].dtype)
Ravg = np.zeros(result[4].shape, dtype = result[4].dtype)
#rho_1traj = np.zeros(result[5].shape, dtype = result[5].dtype)

for ii in range(NTraj):
    psiD[:,ii*(NSteps//nskip) : (ii+1)*(NSteps//nskip)] = result[1][ii,:,:]
    R_t[:,ii*(NSteps//nskip) : (ii+1)*(NSteps//nskip)] = result[2][ii,:,:]
for t in range(result[0].shape[-1]):
    for ii in range(NStates):
        rho_sum[ii,ii,t] += result[0][ii,ii,t]
        for jj in range(NStates):
            if(ii!=jj):
                rho_sum[ii,jj,t] += result[0][ii,jj,t]
    #rhist[:,t] += result[3][:,t]
    #Ravg[t]    += result[4][t]
    #rho_1traj[t,:,:]    += result[i][5][t,:,:]

PiiDFile = open(f"./PiiD_{JOBID}_{TASKID}.txt","w")
#PijDFile = open(f"{fold}/PijD_{JOBID}_{TASKID}.txt","w")
psiDFile = np.savetxt(f"./psiD_{JOBID}_{TASKID}.txt",psiD) 
#rho_1trajFile = open(f"{fold}/rho_1traj_sampled_{JOBID}_{TASKID}.txt","w")
#rhistFile = np.savetxt(f"{fold}/rhist_{JOBID}_{TASKID}.txt",rhist.T) 
#Ravgfile = np.savetxt(f"{fold}/Ravgt_{JOBID}_{TASKID}.txt",Ravg/(NTraj))
R_tfile = np.savetxt(f"./R_t_{JOBID}_{TASKID}.txt",R_t)
for t in range(result[0].shape[-1]):
    PiiDFile.write(f"{t * model.nskip * model.dtN} \t")
    #PijDFile.write(f"{t * model.parameters.nskip * model.parameters.dtN} \t")
    #rho_1trajFile.write(f"{t * model.parameters.nskip * model.parameters.dtN} \t")
    for i in range(model.NStates):
        PiiDFile.write(str(rho_sum[i,i,t].real / NTraj) + "\t")
        #for j in range(NStates):
        #    if(i!=j):
        #        PijDFile.write(str(rho_sum[i,j,t].imag / NTraj) + "\t")
    #for k in range(NCluster*NStates):
        #rho_1trajFile.write(str(rho_1traj[t,k,0].real) + "\t")
        #rho_1trajFile.write(str(rho_1traj[t,k,1].real) + "\t")
    PiiDFile.write("\n")
    #PijDFile.write("\n")
    #rho_1trajFile.write("\n")
PiiDFile.close()
#PijDFile.close()
#rho_1trajFile.close()

t1 = time.time()
print("Total time: ", t1-t0)