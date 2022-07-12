from decimal import ROUND_HALF_DOWN
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
    st = time.time()
    NStates = model.NStates

    pl = sf.savStpRem(model.NSteps1, model.nskip)
    timeSteps = np.arange(0,(model.NSteps1//model.nskip + pl)*model.dtN,model.dtN)
    R1 = np.zeros(len(timeSteps), dtype=np.complex128)

    for i in range(iE.shape[0]):
        """PLDM simulation for each surrived coherence term"""
        iF, iB = iE[i][0], iE[i][1]
        iBath = np.loadtxt(TrajFolder + f"{itraj+1}/initial_bath_{itraj+1}.txt")
        iR, iP = iBath[:,0], iBath[:,1]
        ρi, Rarr, Parr = method.runTraj(iR,iP,iF,iB,model.NSteps1)

        """Response from each coherence"""
        # for tStep in range(ρi.shape[0]):
        #     ρtRe = ρi[tStep,1:(NStates*NStates)+1].reshape(NStates,NStates)
        #     ρtIm = ρi[tStep,(NStates*NStates)+1:].reshape(NStates,NStates)
        #     ρt = ρtRe + 1j*ρtIm
        #     ρt = ρt + np.conjugate(ρt).T

        #     μxt = ρt * sf.commutator(model.μ(),ρ)[iE[i][0],iE[i][1]]
        #     R1[tStep] += 1j * np.trace(model.μ() @ μxt)
        R1 = sf.linResponse(ρi,NStates,R1,iE[i])
    
    et = time.time()
    print(f"{et-st} seconds for traj{itraj+1}")

    return R1