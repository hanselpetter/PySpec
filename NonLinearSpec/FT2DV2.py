import numpy as np
import time
import scipy.constants as sc
from numba import jit
import coupled_dimer as model
import specFunctionsV2 as sF
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cminv2au = 4.55633*1e-6
fs2au = 41.341374575751
ω_max = 10600*2*np.pi*sc.c/(10**13)
ω_min = 9400*2*np.pi*sc.c/(10**13)
ω = np.linspace(ω_min,ω_max,1000)
dω = ω[1] - ω[0]

NTasks = len(ω)//size
NRem = len(ω) - (NTasks*size)
TaskArray = np.array([i for i in range(rank * NTasks , (rank+1) * NTasks)])
for i in range(NRem):
    if i == rank: 
        TaskArray.append((NTasks*size)+i)
print("TaskArray for rank ",rank," is ",TaskArray)

stR = time.time()
print("loading response data")
R3 = np.loadtxt("R3_dimer_40_5e3_80_1e4_3e2.txt")
print(f"response loaded, took {time.time()-stR}")

pl1 = sF.savStpRem(model.NSteps1, model.nskip1)
pl2 = sF.savStpRem(model.NSteps2, model.nskip2)
pl3 = sF.savStpRem(model.NSteps3, model.nskip3)
R3dim1 = (model.NSteps1//model.nskip1) + pl1
R3dim2 = (model.NSteps2//model.nskip2) + pl2
R3dim3 = (model.NSteps3//model.nskip3) + pl3

R3 = R3[:,0] + 1j*R3[:,1]
R3 = R3.reshape(8,R3dim1,R3dim2,R3dim3)

t1 = np.arange(0,model.dtN*R3dim1,model.dtN)
print("t1 dimesnion = ",t1.shape[0])
t2 = np.arange(0,model.dtN*R3dim2,model.dtN)
print("t2 dimesnion = ",t2.shape[0])
t3 = np.arange(0,model.dtN*R3dim3,model.dtN)
print("t3 dimesnion = ",t3.shape[0])

@jit(nopython=True)
def fourier2DRephasing(taskArray,ωdummy,t1,t2,t3,R3):
    R3ω = np.zeros((len(ωdummy),len(t2),len(ωdummy)),dtype=np.complex128)    
    for ω1Index in taskArray:
        for ω3Index in range(len(ωdummy)):
            for t1Index in range(len(t1)):
                exp1 = np.exp(1j*ωdummy[ω1Index]*t1[t1Index]/fs2au)
                for t3Index in range(len(t3)):
                    exp3 = np.exp(1j*ωdummy[ω3Index]*t3[t3Index]/fs2au)
                    R3ω[ω1Index,:,ω3Index] += R3[t1Index,:,t3Index]*exp1*exp3*model.dtN*model.dtN
    return R3ω

@jit(nopython=True)
def fourier2DNonRephasing(taskArray,ωdummy,t1,t2,t3,R3):
    R3ω = np.zeros((len(ωdummy),len(t2),len(ωdummy)),dtype=np.complex128)    
    for ω1Index in taskArray:
        for ω3Index in range(len(ωdummy)):
            for t1Index in range(len(t1)):
                exp1 = np.exp(-1j*ωdummy[ω1Index]*t1[t1Index]/fs2au)
                for t3Index in range(len(t3)):
                    exp3 = np.exp(1j*ωdummy[ω3Index]*t3[t3Index]/fs2au)
                    R3ω[ω1Index,:,ω3Index] += R3[t1Index,:,t3Index]*exp1*exp3*model.dtN*model.dtN
    return R3ω

stD = time.time()
dr1 = np.array([0,1])
dr2 = np.linspace(1,2,2)
R3dummy1 = fourier2DRephasing(dr1,dr1,dr2,dr2,dr2,np.linspace(1,2,8).reshape(2,2,2))
R3dummy2 = fourier2DNonRephasing(dr1,dr1,dr2,dr2,dr2,np.linspace(1,2,8).reshape(2,2,2))
print(f"R3dummy done, took {time.time()-stD} seconds")

stR = time.time()
print("running job for ",rank)

R3Rephasing = R3[0,:,:,:] + R3[1,:,:,:]        # Diagrams 0 and 1 correspond to rephasing signals
R3ω1t2ω3Rephasing = fourier2DRephasing(TaskArray,ω,t1,t2,t3,R3Rephasing)

R3NonRephasing = R3[2,:,:,:] + R3[3,:,:,:]     # Diagrams 2 and 3 correspond to non-rephasing signals
R3ω1t2ω3NonRephasing = fourier2DNonRephasing(TaskArray,ω,t1,t2,t3,R3NonRephasing)

R3ω1t2ω3 = R3ω1t2ω3Rephasing + R3ω1t2ω3NonRephasing
print(f"R3 FT done, took {time.time()-stR} seconds")

recvBuff = np.zeros_like(R3ω1t2ω3, dtype=np.complex128)
comm.Reduce(R3ω1t2ω3,recvBuff,op=MPI.SUM,root=0)

if comm.rank == 0:
    etR = time.time()
    np.savetxt("R3omega_dimer_40_5e3_80_1e4_3e2.txt",np.array([np.real(recvBuff.flatten()),np.imag(recvBuff.flatten())]).T)
    print(f"Time taken = {etR-stR} seconds")