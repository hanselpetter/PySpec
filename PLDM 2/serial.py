import numpy as np
import coupled_dimer as model
import runPLDM as rp

TrajFolder = "/scratch/mmondal/2DES_focused/PLDM/Trajectories/"
for itraj in range(model.NTraj):
    ρ_traj  = rp.simulate(itraj,TrajFolder)
    PijFileRe = TrajFolder + f"{itraj+1}/Pij_{itraj+1}_Re.txt"
    PijFileIm = TrajFolder + f"{itraj+1}/Pij_{itraj+1}_Im.txt"
    PijRe, PijIm = np.real(ρ_traj[:,1:]), np.imag(ρ_traj[:,1:])
    np.savetxt(PijFileRe,PijRe)
    np.savetxt(PijFileIm,PijIm)