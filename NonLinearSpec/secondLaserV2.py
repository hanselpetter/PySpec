import numpy as np

import pldm as method
import coupled_dimer as model
import specFunctionsV2 as sF
import focus2D as foc
import feynDiag as fD

μ_t0 = model.μ()
ρ_t0 = model.ρ0()
μx_t0 = sF.commutator(μ_t0,ρ_t0)
n0 = sF.non0(μx_t0)                                                                                                             # Non-zero indices of [μ,ρ]
ρlen = ρ_t0.shape[0]*ρ_t0.shape[1]

pl1 = sF.savStpRem(model.NSteps1, model.nskip1)
pl2 = sF.savStpRem(model.NSteps2, model.nskip2)
pl3 = sF.savStpRem(model.NSteps3, model.nskip3)
R3dim1 = (model.NSteps1//model.nskip1) + pl1
R3dim2 = (model.NSteps2//model.nskip2) + pl2
R3dim3 = (model.NSteps3//model.nskip3) + pl3

def simulate(ij, itraj, TrajFolder, t1IndexList):

    ρij_t1 = np.loadtxt(TrajFolder+f"{itraj+1}/Pijt1_{n0[ij][0]}{n0[ij][1]}_{itraj+1}.txt")
    R_t1 = np.loadtxt(TrajFolder+f"{itraj+1}/Rt1_{n0[ij][0]}{n0[ij][1]}_{itraj+1}.txt")
    P_t1 = np.loadtxt(TrajFolder+f"{itraj+1}/Pt1_{n0[ij][0]}{n0[ij][1]}_{itraj+1}.txt")
    wijt1 = np.loadtxt(TrajFolder+f"{itraj+1}/wt1_{n0[ij][0]}{n0[ij][1]}_{itraj+1}.txt")

    wt = np.zeros((n0.shape[0], model.NSteps1, model.NSteps2), dtype=np.complex128) 
    wt[ij,:,:] = np.ones((wt.shape[1],wt.shape[2]),dtype=np.complex128) * (wijt1[0] + 1j*wijt1[1])
    R3 = np.zeros((8, R3dim1, R3dim2, R3dim3), dtype=np.complex128)

    KSigns = np.zeros((n0.shape[0], model.NSteps1, model.NSteps2))
    fF, fB = n0[ij][0],n0[ij][1]
    iFmax, iFmin, iBmax, iBmin = 0, 0, 0, 0
    KSigns[ij,0,0] = fD.kSign(fF, fB, iFmax, iFmin, iBmax, iBmin)

    for t1Index in t1IndexList:                                                                                    # Can be parallelised at this level
        """Second laser pulse"""

        """extracting ρ(t1Index) from ρij_t1"""
        ρ_t1Re = ρij_t1[t1Index,1:(ρlen+1)].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
        ρ_t1Im = ρij_t1[t1Index,(ρlen+1):].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
        ρ_t1 = ρ_t1Re + 1j*ρ_t1Im

        """Focusing to get the iF and iB for next PLDM and traj weight of current trajectory run"""
        μx_t1 = sF.commutator(μ_t0,ρ_t1)
        focusedEl_t1, trajWeight_t1  = foc.focusing(μx_t1)
        iF_t1, iB_t1 = focusedEl_t1
        wt[ij,t1Index,:] *= np.ones(wt.shape[2],dtype=np.complex128) * trajWeight_t1 

        """Determining the sign of k for the pulse"""
        iFmax, iFmin, iBmax, iBmin = fD.extremeIndices(ρ_t1)
        KSigns[ij,t1Index,0] = fD.kSign(iF_t1, iB_t1, iFmax, iFmin, iBmax, iBmin)

        """position and momentum of bath at t∈[0,t1]"""
        iR_t1, iP_t1 = R_t1[t1Index,:], P_t1[t1Index,:]

        """For each t∈[0,t1], running PLDM t->t2 by focusing ρ at t"""
        ρij_t2, R_t2, P_t2 = method.runTraj(iR_t1, iP_t1, iF_t1, iB_t1, model.NSteps2)
        
        R3sec = sF.secondPropagation(ij, ρij_t2, R_t2, P_t2, wt, KSigns, R3, t1Index)
        R3 = R3sec
    
    return R3