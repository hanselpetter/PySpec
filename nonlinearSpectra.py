import numpy as np
from PLDM import pldm as method
from PLDM import coupled_dimer as model
import specFunctions as sF
import focus2D as foc
import feynDiag as fD

μ_t0 = model.μ()
ρ_t0 = model.ρ0()
μx_t0 = sF.commutator(μ_t0,ρ_t0)
n0 = sF.non0(μx_t0)                         # Non-zero indices of [μ,ρ]
ρlen = ρ_t0.shape[0]*ρ_t0.shape[1]

def simulate(itraj,TrajFolder):

    """Keeping track of feynman diagrams,
       For all times in 0->t1, ksign and kside will be the same"""
    kside = np.zeros(n0.shape[1], model.NSteps1, model.NSteps2, dtype=int)
    ksign = np.zeros(n0.shape[1], model.NSteps1, model.NSteps2, dtype=int)

    """Initialising weight for each response"""
    wt = np.zeros(n0.shape[1], model.NSteps1, model.NSteps2, dtype=complex)  

    """first loop over the non zero elements in [μ,ρ0]"""
    for ij in range(n0.shape[1]):

        """Initialising response and weight matrix"""
        R3 = np.zeros(model.NSteps1, model.NSteps2, model.NSteps3, dtype=complex)          # Response array
        
        """Initialising the Feynman diagrams"""
        kside[ij,:,:] = fD.KSide(np.array([0,0]),n0[ij])                                   # side of perturbation
        ksign[ij,:,:] = fD.KSign(np.array([0,0]),n0[ij])                                   # sign of K-vector

        """first PLDM run for first laser pulse"""
        iF0, iB0 = n0[ij][0], n0[ij][1]
        wt[ij,:,:] = np.ones(wt.shape[0])*μx_t0[iF0,iB0]                                   # initial weights according to μx_t0 elements
        iBath = np.loadtxt(TrajFolder + f"{itraj+1}/initial_bath_{itraj+1}.txt")
        iR, iP = iBath[:,0], iBath[:,1]
        ρijT1 = method.runTraj(iR, iP, iF0, iB0, itraj, model.NSteps1)

        for t1Index in range(model.NSteps1):                                               # Can be parallelised at this level
            """"For each t in 0->t1, do PLDM t->t2 by focusing ρ at t"""

            """extracting ρ(t1Index) from ρijT1"""
            ρT1Re = ρijT1[t1Index,1:(ρlen+1)].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
            ρT1Im = ρijT1[t1Index,(ρlen+1):].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
            ρT1 = ρT1Re + 1j*ρT1Im

            """Focusing to get the iF and iB for next PLDM and traj weight of current trajectory run"""
            focusedElT1, trajWeight = foc.focus(ρT1)
            iFT1, iBT1 = focusedElT1
            wt[ij,t1Index,:] = wt[ij,:,:]*trajWeight  

            """2nd light-matter perturbation interation in Feynman diagram"""
            kside[ij,t1Index,:] = fD.KSide(n0[ij],focusedElT1)
            ksign[ij,t1Index,:] = fD.KSign(n0[ij],focusedElT1)