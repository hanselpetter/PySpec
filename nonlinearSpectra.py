import numpy as np
from PLDM import pldm as method
from PLDM import coupled_dimer as model
import specFunctions as sF

μ_t0 = model.μ()
ρ_t0 = model.ρ0()
μx_t0 = sF.commutator(μ_t0,ρ_t0)
n0 = sF.non0(μx_t0)

def simulate(itraj,TrajFolder):

    """Initialising response matrix"""
    R3 = np.zeros(model.NSteps1, model.NSteps2, model.NSteps3) 

    """first loop over the non zero elements in [μ,ρ0]"""
    for ij in range(n0.shape[1]):

        """Forward and backward focused states for first PLDM run"""
        iF0, iB0 = n0[ij][0], n0[ij][1]
        iBath = np.loadtxt(TrajFolder + f"{itraj+1}/initial_bath_{itraj+1}.txt")
        iR, iP = iBath[:,0], iBath[:,1]
        R3[:,0,0] = method.runTraj(iR, iP, iF0, iB0, itraj, model.NSteps1)



