#!/software/anaconda3/2020.11/bin/python
#SBATCH -p action
#SBATCH -o output_multipar.log
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

NARRAY = str(100) # number of jobs
filename = "cpldimr"

import os, sys
import subprocess
import time
import numpy as np
from pathlib import Path

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

# Gather data

psiD = np.loadtxt("tmpdir_" + JOBID + "/psiD_" + ARRAYJOBID + "_1.txt", dtype = np.complex) # load first job to get parameters
R_t = np.loadtxt("tmpdir_" + JOBID + "/R_t_" + ARRAYJOBID + "_1.txt") # load first job to get parameters
steps = len(psiD[0,:]) # number of printed steps (NSteps//nskip)
psiD = np.zeros((len(psiD[:,0]),steps*int(NARRAY)), dtype = psiD.dtype) # initialize zeros matrix using parameters
R_t = np.zeros((len(R_t[:,0]),steps*int(NARRAY)), dtype = R_t.dtype) # initialize zeros matrix using parameters

for i in range(int(NARRAY)):
    if(i==0):
        PiiD = np.loadtxt("tmpdir_" + JOBID + "/PiiD_" + ARRAYJOBID + "_1.txt") # load first job
        #PijD = np.loadtxt("tmpdir_" + JOBID + "/PijD_" + ARRAYJOBID + "_1.txt")
    else:
        PiiD += np.loadtxt("tmpdir_" + JOBID + "/PiiD_" + ARRAYJOBID + "_" + str(i+1) + ".txt") # add each job together
        #PijD += np.loadtxt("tmpdir_" + JOBID + "/PijD_" + ARRAYJOBID + "_" + str(i+1) + ".txt")
    psiD[:,i*steps : (i+1)*steps] = np.loadtxt("tmpdir_" + JOBID + "/psiD_" + ARRAYJOBID + "_" + str(i+1) + ".txt", dtype = np.complex) # append each line with next trajectory(s)
    R_t[:,i*steps : (i+1)*steps] = np.loadtxt("tmpdir_" + JOBID + "/R_t_" + ARRAYJOBID + "_" + str(i+1) + ".txt") # append each line with next trajectory(s)

psiDFile = np.savetxt(f"./psiD_{filename}.txt",psiD)
R_tFile = np.savetxt(f"./R_t_{filename}.txt",R_t)
    
PiiD = PiiD / int(NARRAY) # divide to go from sum to average
#PijD = PijD / int(NARRAY)
PiiDFile = np.savetxt(f"./PiiD_{filename}.txt",PiiD)
#PijDFile = np.savetxt(f"./PijD_{filename}.txt",PijD)

# os.system("rm -rf tmpdir_" + JOBID) # delete temporary folder (risky since this removes original job file data)