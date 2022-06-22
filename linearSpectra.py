import numpy as np
from PLDM import pldm as method
from PLDM import coupled_dimer as model
import spec_functions as sf

def simulate(pulseNumber,itraj,TrajFolder):
    ρ = model.ρ0()
    iE = sf.impElement(pulseNumber,ρ)
    ρi = {}
    for i in range(iE.shape[1]):
        iF, iB = iE[i][0], iE[i][1]
        iBath = np.loadtxt(TrajFolder + f"{itraj+1}/initial_bath_{itraj+1}.txt")
        iR, iP = iBath[:,0], iBath[:,1]
        ρi[tuple(iE[i])] = method.runTraj(iR,iP,iF,iB,itraj)
    return ρi
