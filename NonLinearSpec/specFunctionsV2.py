import numpy as np
from numba import jit
import focus2D as foc

import pldm as method
import coupled_dimer as model
import feynDiag as fD

μ_t0 = model.μ()
ρ_t0 = model.ρ0()
ρlen = ρ_t0.shape[0]*ρ_t0.shape[1]

@jit(nopython=True)
def commutator(A,B):
    "commutator of two matrix elements"
    return np.dot(A,B) - np.dot(B,A)

@jit(nopython=True)
def μx(model,ρ):
    "operating μ on density matrix ρ"
    return commutator(model.μ(),ρ)

# @jit(nopython=True)
def non0(mat):
    "Finding the indices of nonzero matrix elements"
    n0 = np.array(np.where(mat!=0))
    return n0[:,:(n0.shape[1])].T

@jit(nopython=True)
def savStpRem(NSteps, nskip):
    if NSteps%nskip == 0:
            pl = 0
    else :
        pl = 1
    return pl

@jit(nopython=True)
def thirdPropagation(ij, ρij_t3, wt3, KSigns, R3, t1Index, t2Index):
    """calculating and saving response function for every nskip time steps"""
    
    for t3Index in range(model.NSteps3):
        if (t1Index%model.nskip1 == 0) and (t2Index%model.nskip2 == 0) and (t3Index%model.nskip3 == 0):
            """extracting ρ(t2Index) from ρij_t2"""
            ρ_t3Re = ρij_t3[t3Index,1:(ρlen+1)].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
            ρ_t3Im = ρij_t3[t3Index,(ρlen+1):].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
            ρ_t3 = ρ_t3Re + 1j*ρ_t3Im

            """3rd order response calculation with all weights"""
            rt3 = (1j**3) * np.trace(np.dot(μ_t0,(ρ_t3*wt3)))
            KVec= np.array([KSigns[ij,0,0],KSigns[ij,t1Index,0],KSigns[ij,t1Index,t2Index]],dtype=np.float64)
            diagram = fD.getDiag(KVec)
            R3[diagram, t1Index//model.nskip1, t2Index//model.nskip2, t3Index//model.nskip3] += rt3

    return R3

@jit(nopython=True)
def secondPropagation(ij, ρij_t2, R_t2, P_t2, wt, KSigns, R3, t1Index):

    """3rd laser pulse"""
    for t2Index in range(model.NSteps2):
        if (t2Index%model.nskip2 == 0):
            """extracting ρ(t2Index) from ρij_t2"""
            ρ_t2Re = ρij_t2[t2Index,1:(ρlen+1)].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
            ρ_t2Im = ρij_t2[t2Index,(ρlen+1):].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
            ρ_t2 = ρ_t2Re + 1j*ρ_t2Im

            """Focusing to get the iF and iB for next PLDM and traj weight of current trajectory run"""
            μx_t2 = np.dot(μ_t0,ρ_t2)-np.dot(ρ_t2,μ_t0)
            focusedEl_t2, trajWeight_t2 = foc.focusing(μx_t2)
            iF_t2, iB_t2 = focusedEl_t2
            wt[ij,t1Index,t2Index] *= trajWeight_t2

            """Determining the sign of k for the pulse"""
            iFmax, iFmin, iBmax, iBmin = fD.extremeIndices(ρ_t2)
            KSigns[ij,t1Index,t2Index] = fD.kSign(iF_t2, iB_t2, iFmax, iFmin, iBmax, iBmin)

            """position and momentum of bath at t∈[t1,t2]"""
            iR_t2, iP_t2 = R_t2[t2Index,:], P_t2[t2Index,:]

            """"For each t∈[t1,t2], running PLDM t->t3 by focusing ρ at t"""
            ρij_t3, R_t3, P_t3 = method.runTraj(iR_t2, iP_t2, iF_t2, iB_t2, model.NSteps3)
            wt3 = wt[ij,t1Index,t2Index]

            R3third = thirdPropagation(ij, ρij_t3, wt3, KSigns, R3, t1Index, t2Index)
            R3 = R3third

    return R3