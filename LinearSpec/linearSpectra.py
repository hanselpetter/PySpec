import numpy as np
from numba import jit

import pldm as method
import coupled_dimer as model
import initBath as iBth

@jit(nopython=True)
def linResponse(ρi,NStates,el):
    
    R1 = np.zeros(ρi.shape[0],dtype=np.complex128)
    for tStep in range(ρi.shape[0]):
        ρtRe = ρi[tStep,1:(NStates*NStates)+1].reshape(NStates,NStates)
        ρtIm = ρi[tStep,(NStates*NStates)+1:].reshape(NStates,NStates)
        ρt = ρtRe + 1j*ρtIm
        
        μx = np.dot(model.μ(),model.ρ0())-np.dot(model.ρ0(),model.μ())
        μt = ρt * (model.μ()[el[0],el[1]])
        R1[tStep] += 1j * np.trace(np.dot(μt,μx))
    
    return R1

@jit(nopython=True)
def simulate(El):
    NStates = model.NStates

    """PLDM simulation for ρ[El[0], El[1]]"""
    iF, iB = El[0], El[1]
    iR, iP = iBth.initR()
    ρi, Rarr, Parr = method.runTraj(iR,iP,iF,iB,model.NSteps1)

    return linResponse(ρi,NStates,El)