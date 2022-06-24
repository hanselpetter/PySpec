import numpy as np
import coupled_dimer as model
import runPLDM as rP

TrajFolder = "/scratch/mmondal/2DES_focused/Trajectories/"
for itraj in range(4):
    ρ_traj  = rP.simulate(itraj,TrajFolder)
    for matEl in ρ_traj.keys():
        PijFile = TrajFolder + f"{itraj+1}/Pij_{matEl[0]}{matEl[1]}_{itraj+1}.txt"
        np.savetxt(PijFile,ρ_traj[matEl])