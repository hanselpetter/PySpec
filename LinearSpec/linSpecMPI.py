import numpy as np
from mpi4py import MPI
import sys

import coupled_dimer as model
import linearSpectra as lS

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

def savStpRem(NSteps, nskip):
    if NSteps%nskip == 0:
            pl = 0
    else :
        pl = 1
    return pl

def impElement():
    μxρ0 = np.dot(model.μ(),model.ρ0())-np.dot(model.ρ0(),model.μ())
    return np.array(np.where(μxρ0!=0)).T

pl = savStpRem(model.NSteps1, model.nskip)
timeSteps = np.arange(0,(model.NSteps1//model.nskip + pl)*model.dtN,model.dtN)
iE = impElement()

R1chain = np.zeros(len(timeSteps)*iE.shape[0], dtype=np.complex128)
recvBuff = np.zeros(len(R1chain), dtype=np.complex128)

st = MPI.Wtime()
for el in range(iE.shape[0]):
    El = iE[el]   
    for itraj in TaskArray: 
        R1chain[el*len(timeSteps):(el+1)*len(timeSteps)] += lS.simulate(El)

comm.Reduce(R1chain, recvBuff, op=MPI.SUM, root=0)
ed = MPI.Wtime()
print(f"jobs for rank {rank} finished in {ed-st} seconds")

if comm.rank == 0:
    for el in range(iE.shape[0]):
        rankEl = iE[el]
        rankR1El = recvBuff[el*len(timeSteps):(el+1)*len(timeSteps)]/NTraj
        rankR1ElRe, rankR1ElIm = np.real(rankR1El), np.imag(rankR1El)
        np.savetxt(TrajFolder + f"R1_{rankEl[0]}{rankEl[1]}.txt", np.array([rankR1ElRe,rankR1ElIm]).T)