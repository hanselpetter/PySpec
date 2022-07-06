import numpy as np
import time
import pldm as method
import coupled_dimer as model

def simulate(itraj,TrajFolder):
    iF, iB = model.initStateF, model.initStateB
    iBath = np.loadtxt(TrajFolder + f"{itraj+1}/initial_bath_{itraj+1}.txt")
    iR, iP = iBath[:,0], iBath[:,1]
    st = time.time()
    ρ, RArr, PArr = method.runTraj(iR,iP,iF,iB,itraj,model.NSteps1)
    et = time.time()
    print(f"it took {et-st} seconds to run trajectory {itraj+1}")
    return ρ
