import numpy as np
from mpi4py import MPI
import sys

import pldm as method
import coupled_dimer as model
import specFunctions as sF

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TrajFolder = sys.argv[1]
NTraj = model.NTraj
NTasks = NTraj//size
NRem = NTraj - (NTasks*size)
TaskArray = [i for i in range(rank * NTasks , (rank+1) * NTasks)]
for i in range(NRem):
    if i == rank: 
        TaskArray.append((NTasks*size)+i)

μ_t0 = model.μ()
ρ_t0 = model.ρ0()
μx_t0 = sF.commutator(μ_t0,ρ_t0)
n0 = sF.non0(μx_t0)                                                                                                             # Non-zero indices of [μ,ρ]
ρlen = ρ_t0.shape[0]*ρ_t0.shape[1]

def firstPropagation(itraj,TrajFolder):
    
    """Initialising weight for each response after interactionwith the first laser"""
    wt = np.zeros(n0.shape[0], dtype=complex)
    ijData = {}

    """first loop over the non zero elements in [μ,ρ0]"""
    for ij in range(n0.shape[0]):

        iF0, iB0 = n0[ij][0], n0[ij][1]           # setting the first forward and backward indices
        wt[ij] = μx_t0[iF0,iB0]                   # initial weights for ij = μx_t0[i,j] 

        """Initialising the bath"""
        iBath = np.loadtxt(TrajFolder + f"{itraj+1}/initial_bath_{itraj+1}.txt")
        iR, iP = iBath[:,0], iBath[:,1]

        """first PLDM run for the coherence terms in [μ,ρ0]"""
        ρij_t1, R_t1, P_t1 = method.runTraj(iR, iP, iF0, iB0, model.NSteps1)
        ijData[tuple(n0[ij])] = ρij_t1, R_t1, P_t1, wt[ij]

    return ijData

for itraj in TaskArray:
    print(itraj+1)
    firstLaserProp = firstPropagation(itraj,TrajFolder)
    for trajKey in firstLaserProp.keys():

        ρt1 = firstLaserProp[trajKey][0]
        Rt1 = firstLaserProp[trajKey][1]
        Pt1 = firstLaserProp[trajKey][2]
        wt1 = firstLaserProp[trajKey][3]
        
        np.savetxt(TrajFolder+f"{itraj+1}/Pijt1_{trajKey[0]}{trajKey[1]}_{itraj+1}.txt",ρt1)
        np.savetxt(TrajFolder+f"{itraj+1}/Rt1_{trajKey[0]}{trajKey[1]}_{itraj+1}.txt",Rt1)
        np.savetxt(TrajFolder+f"{itraj+1}/Pt1_{trajKey[0]}{trajKey[1]}_{itraj+1}.txt",Pt1)
        np.savetxt(TrajFolder+f"{itraj+1}/wt1_{trajKey[0]}{trajKey[1]}_{itraj+1}.txt",np.array([wt1.real,wt1.imag]))