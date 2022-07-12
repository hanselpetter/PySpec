import numpy as np
from numba import jit
import sys
sys.path.insert(0, "/scratch/mmondal/specTest/PLDM")

import coupled_dimer as model

μ_t0 = model.μ()
ρ_t0 = model.ρ0()
ρlen = ρ_t0.shape[0]*ρ_t0.shape[1]

def commutator(A,B):
    "commutator of two matrix elements"
    return np.dot(A,B) - np.dot(B,A)

def μx(model,ρ):
    "operating μ on density matrix ρ"
    return commutator(model.μ(),ρ)

def non0(mat):
    "Finding the indices of nonzero matrix elements"
    n0 = np.array(np.where(mat!=0))
    return n0[:,:(n0.shape[1])].T

def genRho(ρ):
    "Generating the whole rho(n) from single matrix element propagation"
    return ρ + np.conjugate(ρ.T)

# @jit(nopython=True)
def impElement(model,ρ):
    # μx = commutator(model.μ(),ρ)
    μx = model.μ()
    return np.array(np.where(μx!=0)).T

def savStpRem(NSteps, nskip):
    if NSteps%nskip == 0:
            pl = 0
    else :
        pl = 1
    return pl

@jit(nopython=True)
def linResponse(ρi,NStates,R1,el):
    
    for tStep in range(ρi.shape[0]):
        ρtRe = ρi[tStep,1:(NStates*NStates)+1].reshape(NStates,NStates)
        ρtIm = ρi[tStep,(NStates*NStates)+1:].reshape(NStates,NStates)
        ρt = ρtRe + 1j*ρtIm
        ρt = ρt + np.conjugate(ρt).T
        μx = np.dot(model.μ(),model.ρ0())-np.dot(model.ρ0(),model.μ())
        μt = ρt * model.μ()[el[0],el[1]]
        R1[tStep] += 1j * np.trace(np.dot(μt,μx))
    
    return R1
