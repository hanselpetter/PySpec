#!/software/anaconda3/2020.11/bin/python
#SBATCH -p exciton -A exciton
#SBATCH --job-name=2Danalyse
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=16GB
#SBATCH -t 5-00:00:00
#SBATCH --output=test_%A_%a.out
#SBATCH --error=test_%A_%a.err

import numpy as np
import sys
sys.path.insert(0, "/scratch/mmondal/specTest/PLDM")

import coupled_dimer as model

R3_Re = np.zeros((model.NSteps1, model.NSteps2, model.NSteps3)).flatten()
R3_Im = np.zeros((model.NSteps1, model.NSteps2, model.NSteps3)).flatten()

TrajFolder = "/scratch/mmondal/specTest/spectra/Trajectories/"
NArray = 20

for itask in range(NArray):
    print(itask+1)
    R3_Re += np.loadtxt(TrajFolder + f"1/R3_{itask+1}_Re.txt")
    R3_Im += np.loadtxt(TrajFolder + f"1/R3_{itask+1}_Im.txt")

np.savetxt(TrajFolder + f"1/R3Re.txt",R3_Re)
np.savetxt(TrajFolder + f"1/R3Im.txt",R3_Im)

