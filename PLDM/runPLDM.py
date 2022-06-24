import numpy as np
import pldm as method
import coupled_dimer as model

def simulate(itraj,TrajFolder):
    iF, iB = model.initStateF, model.initStateB
    iBath = np.loadtxt(TrajFolder + f"{itraj+1}/initial_bath_{itraj+1}.txt")
    iR, iP = iBath[:,0], iBath[:,1]
    ρ = method.runTraj(iR,iP,iF,iB,itraj,model.NSteps)
    return ρ
