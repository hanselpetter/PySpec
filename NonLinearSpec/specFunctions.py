import numpy as np
from numba import jit
import sys
sys.path.insert(0, "/scratch/mmondal/specTest/PLDM")

import coupled_dimer as model
import pldm as method
import focus2D as foc

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

def impElement(model,ρ):
    μx = commutator(model.μ(),ρ)
    return np.array(np.where(μx!=0)).T

def savStpRem(NSteps, nskip):
    if NSteps%nskip == 0:
            pl = 0
    else :
        pl = 1
    return pl

@jit(nopython=True)
def thirdPropagation(ρij_t3,wt3,R3,t1Index,t2Index):
    """calculating and saving response function for every nskip time steps"""
    
    for t3Index in range(model.NSteps3):
        if (t1Index%model.nskip1 == 0) and (t2Index%model.nskip2 == 0) and (t3Index%model.nskip3 == 0):
            """extracting ρ(t2Index) from ρij_t2"""
            ρ_t3Re = ρij_t3[t3Index,1:(ρlen+1)].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
            ρ_t3Im = ρij_t3[t3Index,(ρlen+1):].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
            ρ_t3 = ρ_t3Re + 1j*ρ_t3Im
            ρ_t3 = ρ_t3 + np.conjugate(ρ_t3).T

            """3rd order response calculation with all weights"""
            rt3 = (1j**3) * np.trace(np.dot(μ_t0,(ρ_t3*wt3))) #* ((-1.0)**np.sum(rsideVec))
            R3[t1Index//model.nskip1,t2Index//model.nskip2,t3Index//model.nskip3] += rt3

    return R3

@jit(nopython=True)
def secondPropagation(ij, ρij_t2, r_t2, p_t2, wt, R3, t1Index):

    """3rd laser pulse"""
    for t2Index in range(model.NSteps2):
        """extracting ρ(t2Index) from ρij_t2"""
        ρ_t2Re = ρij_t2[t2Index,1:(ρlen+1)].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
        ρ_t2Im = ρij_t2[t2Index,(ρlen+1):].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
        ρ_t2 = ρ_t2Re + 1j*ρ_t2Im
        ρ_t2 = ρ_t2 + np.conjugate(ρ_t2).T

        """Focusing to get the iF and iB for next PLDM and traj weight of current trajectory run"""
        μx_t2 = np.dot(μ_t0,ρ_t2)-np.dot(ρ_t2,μ_t0)
        focusedEl_t2 = foc.focusing(μx_t2)
        trajWeight_t2 = foc.trajWt(μx_t2)
        iF_t2, iB_t2 = focusedEl_t2
        wt[ij,t1Index,t2Index] *= trajWeight_t2

        """position and momentum of bath at t∈[t1,t2]"""
        iR_t2, iP_t2 = r_t2[t2Index,:], p_t2[t2Index,:]

        """"For each t∈[t1,t2], running PLDM t->t3 by focusing ρ at t"""
        ρij_t3, r_t3, p_t3 = method.runTraj(iR_t2, iP_t2, iF_t2, iB_t2, model.NSteps3)
        wt3 = wt[ij,t1Index,t2Index]

        R3third = thirdPropagation(ρij_t3,wt3,R3,t1Index,t2Index)
        R3 = R3third

    return R3

@jit(nopython=True)
def firstPropagation(ij, ρij_t1, r_t1, p_t1, wt, R3, itraj):

    for t1Index in range(model.NSteps1):                                                                                    # Can be parallelised at this level
        """Second laser pulse"""

        """extracting ρ(t1Index) from ρij_t1"""
        ρ_t1Re = ρij_t1[t1Index,1:(ρlen+1)].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
        ρ_t1Im = ρij_t1[t1Index,(ρlen+1):].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
        ρ_t1 = ρ_t1Re + 1j*ρ_t1Im
        ρ_t1 = ρ_t1 + np.conjugate(ρ_t1).T

        """Focusing to get the iF and iB for next PLDM and traj weight of current trajectory run"""
        μx_t1 = np.dot(μ_t0,ρ_t1) - np.dot(ρ_t1,μ_t0)
        focusedEl_t1  = foc.focusing(μx_t1)
        trajWeight_t1 = foc.trajWt(μx_t1)
        iF_t1, iB_t1 = focusedEl_t1
        wt[ij,t1Index,:] *= np.ones(wt.shape[2],dtype=np.complex128) * trajWeight_t1 

        """position and momentum of bath at t∈[0,t1]"""
        iR_t1, iP_t1 = r_t1[t1Index,:], p_t1[t1Index,:]

        """"For each t∈[0,t1], running PLDM t->t2 by focusing ρ at t"""
        ρij_t2, r_t2, p_t2 = method.runTraj(iR_t1, iP_t1, iF_t1, iB_t1, model.NSteps2)

        R3sec = secondPropagation(ij, ρij_t2, r_t2, p_t2, wt, R3, t1Index, itraj)
        R3 = R3sec

    
    return R3