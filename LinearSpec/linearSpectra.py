import numpy as np
import time
import sys
sys.path.insert(0, "/scratch/mmondal/specTest/PLDM")

import pldm as method
import coupled_dimer as model
import linSpecFunctions as sf

def simulate(itraj,TrajFolder):
    ρ = model.ρ0()
    iE = sf.impElement(model,ρ)
    ρi = {}
    st = time.time()
    for i in range(iE.shape[0]):
        iF, iB = iE[i][0], iE[i][1]
        iBath = np.loadtxt(TrajFolder + f"{itraj+1}/initial_bath_{itraj+1}.txt")
        iR, iP = iBath[:,0], iBath[:,1]
        ρi[tuple(iE[i])], Rarr, Parr = method.runTraj(iR,iP,iF,iB,itraj,model.NSteps1)
    et = time.time()
    print(f"{et-st} seconds for traj{itraj+1}")
    return ρi