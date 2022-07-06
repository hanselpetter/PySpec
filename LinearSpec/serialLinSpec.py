import numpy as np
import sys
sys.path.insert(0, "/scratch/mmondal/specTest/PLDM")

import coupled_dimer as model
import linearSpectra as lS

TrajFolder = "/scratch/mmondal/specTest/Trajectories/"
for itraj in range(model.NTraj):
    ρTraj = lS.simulate(itraj,TrajFolder)
    for i in ρTraj.keys():
        ρij = ρTraj[i]
        PijFile  = TrajFolder +  f"{itraj+1}/{i[0]}{i[1]}.txt"
        np.savetxt(PijFile,ρij)