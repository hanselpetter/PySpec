import numpy as np
from mpi4py import MPI
import sys

import pldm as method
import coupled_dimer as model
import initBath as iBth

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

print(f"Rank {rank} has {len(TaskArray)} number of trajectories")

if model.NSteps1%model.nskip == 0:
    pl = 0
else :
    pl = 1

ρFlatLen = (model.NSteps1//model.nskip + pl)*(2*model.NStates*model.NStates+1)
ρchain = np.zeros(ρFlatLen*model.NStates)
recvBuff = np.zeros_like(ρchain)

st = MPI.Wtime()
for iState in range(model.NStates):
    ρState = np.zeros((model.NSteps1//model.nskip + pl, 2*model.NStates*model.NStates+1))
    ρState[:,0] = np.arange(0,(model.NSteps1//model.nskip + pl)*model.dtN,model.dtN)
    for itraj in TaskArray:
        iF, iB = iState, iState
        iR, iP = iBth.initR()
        ρi, Rarr, Parr = method.runTraj(iR,iP,iF,iB,model.NSteps1)
        ρState[:,1:] += ρi[:,1:]
    ρStateTaskArr = ρState.flatten()
    ρchain[iState*ρFlatLen:(iState+1)*ρFlatLen] = ρStateTaskArr

comm.Reduce(ρchain, recvBuff, op=MPI.SUM, root=0)
ed = MPI.Wtime()
print(f"jobs for rank {rank} finished in {ed-st} seconds")

if comm.rank == 0:
    for iState in range(model.NStates):
        ρiState = recvBuff[iState*ρFlatLen:(iState+1)*ρFlatLen]/NTraj
        ρiState = ρiState.reshape((model.NSteps1//model.nskip + pl,2*model.NStates*model.NStates+1))
        ρiState[:,0] *= NTraj
        np.savetxt(TrajFolder + f"ρ{iState}{iState}.txt",ρiState)

