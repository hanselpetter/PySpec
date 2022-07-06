import numpy as np
import time
import sys
sys.path.insert(0, "/scratch/mmondal/specTest/PLDM")

import pldm as method
import coupled_dimer as model
import specFunctions as sF
import focus2D as foc

μ_t0 = model.μ()
ρ_t0 = model.ρ0()
μx_t0 = sF.commutator(μ_t0,ρ_t0)
comm_t1 = sF.commutator(μ_t0,μx_t0)
comm_t2 = sF.commutator(μ_t0,comm_t1)
n0 = sF.non0(μx_t0)                                                                                                             # Non-zero indices of [μ,ρ]
ρlen = ρ_t0.shape[0]*ρ_t0.shape[1]

def simulate(itraj,TrajFolder):

    """Initialising weight for each response"""
    wt = np.zeros((n0.shape[0], model.NSteps1, model.NSteps2), dtype=complex)  
    R3 = np.zeros((model.NSteps1, model.NSteps2, model.NSteps3), dtype=complex)
    
    """first loop over the non zero elements in [μ,ρ0]"""
    for ij in range(n0.shape[0]):

        """first PLDM run for first laser pulse"""
        iF0, iB0 = n0[ij][0], n0[ij][1]
        wt[ij,:,:] = np.ones((wt.shape[1],wt.shape[2]),dtype=complex) * μx_t0[iF0,iB0]                                          # initial weights according to μx_t0 elements
        iBath = np.loadtxt(TrajFolder + f"{itraj+1}/initial_bath_{itraj+1}.txt")
        iR, iP = iBath[:,0], iBath[:,1]
        ρij_t1, r_t1, p_t1 = method.runTraj(iR, iP, iF0, iB0, itraj, model.NSteps1)


        for t1Index in range(model.NSteps1):                                                                                    # Can be parallelised at this level
            start_time = time.time()
            """Second laser pulse"""

            """extracting ρ(t1Index) from ρij_t1"""
            ρ_t1Re = ρij_t1[t1Index,1:(ρlen+1)].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
            ρ_t1Im = ρij_t1[t1Index,(ρlen+1):].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
            ρ_t1 = ρ_t1Re + 1j*ρ_t1Im
            ρ_t1 = ρ_t1 + np.conjugate(ρ_t1).T

            """Focusing to get the iF and iB for next PLDM and traj weight of current trajectory run"""
            μx_t1 = sF.commutator(μ_t0,ρ_t1)
            focusedEl_t1  = foc.focusing(μx_t1)
            trajWeight_t1 = foc.trajWt(μx_t1)
            iF_t1, iB_t1 = focusedEl_t1
            wt[ij,t1Index,:] *= np.ones(wt.shape[2],dtype=complex) * trajWeight_t1 

            """position and momentum of bath at t∈[0,t1]"""
            iR_t1, iP_t1 = r_t1[t1Index,:], p_t1[t1Index,:]

            """"For each t∈[0,t1], running PLDM t->t2 by focusing ρ at t"""
            ρij_t2, r_t2, p_t2 = method.runTraj(iR_t1, iP_t1, iF_t1, iB_t1, itraj, model.NSteps2)
            
            R3sec = sF.secondPropagation(ij, ρij_t2, r_t2, p_t2, wt, R3, t1Index, itraj)
            R3 = R3sec
            
            end_time = time.time()
            print(f"It took {end_time-start_time} seconds to do calculation for t1 = {t1Index} step")
    
    return R3