import numpy as np
import pldm as method

TrajFolder = "/scratch/mmondal/2DES_focused/Trajectories/"
for itraj in range(10):
    init_bath = np.loadtxt(TrajFolder + f"/traj{itraj+1}/initial_bath_{itraj+1}.txt")
    initR = init_bath[:,0]
    initP = init_bath[:,1]

    ρ_traj  = method.runTraj(initR,initP,itraj+1)
    PijFile = TrajFolder + f"/traj{itraj+1}/Pij_{itraj+1}.txt"
    np.savetxt(PijFile,ρ_traj)
