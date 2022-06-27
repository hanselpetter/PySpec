import numpy as np
from PLDM import pldm as method
from PLDM import coupled_dimer as model
import specFunctions as sF
import focus2D as foc
import feynDiag as fD

μ_t0 = model.μ()
ρ_t0 = model.ρ0()
μx_t0 = sF.commutator(μ_t0,ρ_t0)
n0 = sF.non0(μx_t0)                                                                                                             # Non-zero indices of [μ,ρ]
ρlen = ρ_t0.shape[0]*ρ_t0.shape[1]

def simulate(itraj,TrajFolder):

    """Keeping track of feynman diagrams,
       For all times in 0->t1, ksign and kside will be the same"""
    kside = np.zeros((n0.shape[1], model.NSteps1, model.NSteps2), dtype=int)
    ksign = np.zeros((n0.shape[1], model.NSteps1, model.NSteps2), dtype=int)

    """Initialising weight for each response"""
    wt = np.zeros((n0.shape[1], model.NSteps1, model.NSteps2), dtype=complex)  
    R3_all = {}
    
    """first loop over the non zero elements in [μ,ρ0]"""
    for ij in range(n0.shape[1]):

        """Initialising response and weight matrix"""
        R3 = np.zeros((model.NSteps1, model.NSteps2, model.NSteps3), dtype=complex)                                             # Response array
        
        """Initialising the Feynman diagrams"""
        kside[ij,:,:] = np.ones((kside.shape[1],kside.shape[2]),dtype=int) * fD.KSide(np.array([0,0]),n0[ij])                   # side of perturbation
        ksign[ij,:,:] = np.ones((ksign.shape[1],ksign.shape[2]),dtype=int) * fD.KSign(np.array([0,0]),n0[ij])                   # sign of K-vector

        """first PLDM run for first laser pulse"""
        iF0, iB0 = n0[ij][0], n0[ij][1]
        wt[ij,:,:] = np.ones((wt.shape[1],wt.shape[2]),dtype=complex) * μx_t0[iF0,iB0]                                            # initial weights according to μx_t0 elements
        iBath = np.loadtxt(TrajFolder + f"{itraj+1}/initial_bath_{itraj+1}.txt")
        iR, iP = iBath[:,0], iBath[:,1]
        ρij_t1, r_t1, p_t1 = method.runTraj(iR, iP, iF0, iB0, itraj, model.NSteps1)


        for t1Index in range(model.NSteps1):                                                                                    # Can be parallelised at this level
            """Second laser pulse"""

            """extracting ρ(t1Index) from ρij_t1"""
            ρ_t1Re = ρij_t1[t1Index,1:(ρlen+1)].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
            ρ_t1Im = ρij_t1[t1Index,(ρlen+1):].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
            ρ_t1 = ρ_t1Re + 1j*ρ_t1Im

            """Focusing to get the iF and iB for next PLDM and traj weight of current trajectory run"""
            μx_t1 = sF.commutator(μ_t0,ρ_t1)
            focusedEl_t1, trajWeight_t1 = foc.focus(μx_t1)
            iF_t1, iB_t1 = focusedEl_t1
            wt[ij,t1Index,:] *= np.ones(wt.shape[2],dtype=complex) * trajWeight_t1 * μx_t1[iF_t1,iB_t1]  

            """2nd light-matter perturbation interation in Feynman diagram"""
            kside[ij,t1Index,:] = np.ones(kside.shape[2]) * fD.KSide(n0[ij],focusedEl_t1)
            ksign[ij,t1Index,:] = np.ones(ksign.shape[2]) * fD.KSign(n0[ij],focusedEl_t1)

            """position and momentum of bath at t∈[0,t1]"""
            iR_t1, iP_t1 = r_t1[t1Index,:], p_t1[t1Index,:]

            """"For each t∈[0,t1], running PLDM t->t2 by focusing ρ at t"""
            ρij_t2, r_t2, p_t2 = method.runTraj(iR_t1, iP_t1, iF_t1, iB_t1, itraj, model.NSteps2)


            for t2Index in range(model.NSteps2):
                """3rd laser pulse"""

                """extracting ρ(t2Index) from ρij_t2"""
                ρ_t2Re = ρij_t2[t2Index,1:(ρlen+1)].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
                ρ_t2Im = ρij_t2[t2Index,(ρlen+1):].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
                ρ_t2 = ρ_t2Re + 1j*ρ_t2Im

                """Focusing to get the iF and iB for next PLDM and traj weight of current trajectory run"""
                μx_t2 = sF.commutator(μ_t0,ρ_t2)
                focusedEl_t2, trajWeight_t2 = foc.focus(μx_t2)
                iF_t2, iB_t2 = focusedEl_t2
                wt[ij,t1Index,t2Index] *= trajWeight_t2 * μx_t1[iF_t2,iB_t2]

                """3rd light-matter perturbation interation in Feynman diagram"""
                kside[ij,t1Index,t2Index] = fD.KSide(n0[ij],focusedEl_t2)
                ksign[ij,t1Index,t2Index] = fD.KSign(n0[ij],focusedEl_t2)

                """position and momentum of bath at t∈[t1,t2]"""
                iR_t2, iP_t2 = r_t2[t2Index,:], p_t2[t2Index,:]

                """"For each t∈[t1,t2], running PLDM t->t3 by focusing ρ at t"""
                ρij_t3, r_t3, p_t3 = method.runTraj(iR_t2, iP_t2, iF_t2, iB_t2, itraj, model.NSteps3)
                wt3 = wt[ij,t1Index,t2Index]
                ksideVec = np.array([kside[ij,t1Index,t2Index],kside[ij,t1Index,0],kside[ij,0,0]])

                for t3Index in range(model.NSteps3):
                    print(t1Index, t2Index, t3Index)
                    """extracting ρ(t2Index) from ρij_t2"""
                    ρ_t3Re = ρij_t3[t3Index,1:(ρlen+1)].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
                    ρ_t3Im = ρij_t3[t3Index,(ρlen+1):].reshape(ρ_t0.shape[0],ρ_t0.shape[1])
                    ρ_t3 = ρ_t3Re + 1j*ρ_t3Im

                    """3rd order response calculation with all weights"""
                    rt3 = (1j**3) * np.trace(μ_t0 @ (ρ_t3*wt3)) * ((-1.0)**np.sum(ksideVec))
                    R3[t1Index,t2Index,t3Index] = rt3
        
        R3_all[tuple(n0[ij])] = [R3,ksign]
    
    return R3_all