import time
import numpy as np
from mpi4py import MPI
import sys

import coupled_dimer as model
import specFunctions as sF
import secondLaser as sL

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

pl1 = sF.savStpRem(model.NSteps1, model.nskip1)
pl2 = sF.savStpRem(model.NSteps2, model.nskip2)
pl3 = sF.savStpRem(model.NSteps3, model.nskip3)
R3dim1 = (model.NSteps1//model.nskip1) + pl1
R3dim2 = (model.NSteps2//model.nskip2) + pl2 
R3dim3 = (model.NSteps3//model.nskip3) + pl3
R3 = np.zeros((8, R3dim1, R3dim2, R3dim3), dtype=complex)
recvBuff = np.zeros_like(R3, dtype=complex)

st = time.time()
for itraj in TaskArray:
    print(f"{itraj}")
    R3traj = np.zeros((8, R3dim1, R3dim2, R3dim3), dtype=complex)
    for ij in range(sL.n0.shape[0]):
        R3traj += sL.simulate(ij, itraj, TrajFolder, [istep for istep in range(model.NSteps1)])
        R3 += R3traj

comm.Reduce(R3,recvBuff,op=MPI.SUM,root=0)
    
et = time.time()

if comm.rank == 0:
    np.savetxt(TrajFolder + f"R3_dimer.txt",np.array([np.real(recvBuff).flatten(),np.imag(recvBuff).flatten()]).T) 
    print(f"Time taken = {et-st} seconds")