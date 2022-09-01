import numpy as np
import time
import scipy.constants as sc
from numba import jit
import coupled_dimer as model
import specFunctions as sF
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters for Fourier transform
cminv2au = 4.55633*1e-6
fs2au = 41.341374575751
ω1_max, ω1_min = 10600*2*np.pi*sc.c/(10**13), 9400*2*np.pi*sc.c/(10**13)
ω3_max, ω3_min = 10600*2*np.pi*sc.c/(10**13), 9400*2*np.pi*sc.c/(10**13)
lenω1, lenω3 = 1000, 1000
ω1 = np.linspace(ω1_min,ω1_max,lenω1)
ω3 = np.linspace(-ω3_min,-ω3_max,lenω3)
dω1 = ω1[1] - ω1[0]
dω3 = ω3[1] - ω3[0]

# Division of tasks on multiple processes
NTasks = len(ω1)//size
NRem = len(ω1) - (NTasks*size)
TaskArray = np.array([i for i in range(rank * NTasks , (rank+1) * NTasks)])
for i in range(NRem):
    if i == rank: 
        TaskArray.append((NTasks*size)+i)
print("TaskArray for rank ",rank," is ",TaskArray)

# time axis
pl1 = sF.savStpRem(model.NSteps1, model.nskip1)
pl2 = sF.savStpRem(model.NSteps2, model.nskip2)
pl3 = sF.savStpRem(model.NSteps3, model.nskip3)
R3dim1 = (model.NSteps1//model.nskip1) + pl1
R3dim2 = (model.NSteps2//model.nskip2) + pl2
R3dim3 = (model.NSteps3//model.nskip3) + pl3

t1 = np.arange(0,model.dtN*R3dim1,model.dtN)
print("t1 dimesnion = ",t1.shape[0])
t2 = np.arange(0,model.dtN*R3dim2,model.dtN)
print("t2 dimesnion = ",t2.shape[0])
t3 = np.arange(0,model.dtN*R3dim3,model.dtN)
print("t3 dimesnion = ",t3.shape[0])

# Leading time dependent response
stR = time.time()
print("loading response data")
R3 = np.loadtxt("R3_dimer.txt")
print(f"response loaded, took {time.time()-stR}")

R3 = R3[:,0] + 1j*R3[:,1]
R3 = R3.reshape(8,R3dim1,R3dim2,R3dim3)

@jit(nopython=True)
def fourier2DRephasing(taskArray,ω1dummy,ω3dummy,t1,t3,R3):
    R3ω = np.zeros((len(ω1dummy),len(ω3dummy)),dtype=np.complex128)    
    for ω1Index in taskArray:
        for ω3Index in range(len(ω3dummy)):
            for t1Index in range(len(t1)):
                exp1 = np.exp(1j*ω1dummy[ω1Index]*t1[t1Index]/fs2au)
                for t3Index in range(len(t3)):
                    exp3 = np.exp(1j*ω3dummy[ω3Index]*t3[t3Index]/fs2au)
                    R3ω[ω1Index,ω3Index] += R3[t1Index,t3Index]*exp1*exp3*model.dtN*model.dtN
    return R3ω

stD = time.time()
dr1 = np.array([0,1])
dr2 = np.linspace(1,2,2,dtype=np.complex128)
R3dummy1 = fourier2DRephasing(dr1,dr2,dr2,dr2,dr2,np.linspace(1,2,4,dtype=np.complex128).reshape(2,2))
print(f"R3dummy done, took {time.time()-stD} seconds")

stR = time.time()
print("running job for ",rank)

R3Rephasing = R3[0,:,:,:] + R3[1,:,:,:] + R3[2,:,:,:] + R3[3,:,:,:]       # Diagrams 0 and 1 correspond to rephasing signals
R3ω1t2ω3Rephasing = fourier2DRephasing(TaskArray,ω1,ω3,t1,t3,R3Rephasing[:,0,:])

R3ω1t2ω3 = R3ω1t2ω3Rephasing
print(f"R3 FT done, took {time.time()-stR} seconds")

recvBuff = np.zeros_like(R3ω1t2ω3, dtype=np.complex128)
comm.Reduce(R3ω1t2ω3,recvBuff,op=MPI.SUM,root=0)

if comm.rank == 0:
    etR = time.time()
    np.savetxt("R3omega_dimer.txt",np.array([np.real(recvBuff.flatten()),np.imag(recvBuff.flatten())]).T)
    print(f"Time taken = {etR-stR} seconds")