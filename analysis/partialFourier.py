import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/scratch/mmondal/specTest/PLDM")
sys.path.insert(0, "/scratch/mmondal/specTest/linearSpec")

import coupled_dimer as model
import linSpecFunctions as sf

cminv2au = 4.55633*1e-6
fs2au = 41.341374575751
ω_max = 11000*2*np.pi*sc.c/(10**13)
ω_min = 9000*2*np.pi*sc.c/(10**13)
ω = np.linspace(ω_min,ω_max,2001)
dω = ω[1] - ω[0]

NSteps = model.NSteps1
NStates = model.NStates
nskip = model.nskip
if NSteps%nskip == 0:
    pl = 0
else :
    pl = 1
timeSteps = np.arange(0,(NSteps//nskip + pl)*model.dtN,model.dtN)
ρ = model.ρ0()
iE = sf.impElement(model,ρ)
TrajFolder = "/scratch/mmondal/specTest/linearSpec/Trajectories/"

"""Averaging the elements over all trajectories"""
# for El in range(iE.shape[0]):
#     ρij = np.zeros((NSteps//nskip + pl, 2*NStates*NStates)) 
#     for itraj in range(model.NTraj):
#         trajIndex = itraj + 1
#         print("traj = ",trajIndex)
#         itrajElFile = TrajFolder + f"{trajIndex}/{iE[El][0]}{iE[El][1]}.txt"
#         ρij += np.loadtxt(itrajElFile)[:,1:]
#     ρij /= trajIndex
#     np.savetxt(f"{iE[El][0]}{iE[El][1]}_avg.txt",ρij)

"""Calculating Response from each coherence element"""
R1t = np.zeros(NSteps)
for El in range(iE.shape[0]):
    ρijData = np.loadtxt(f"{iE[El][0]}{iE[El][1]}_avg.txt")
    ρij = ρijData[:,:NStates*NStates] + (1j * ρijData[:,NStates*NStates:])
    for tStep in range(len(R1t)):
        print("tStep = ",tStep)
        ρijtStep = ρij[tStep,:].reshape((NStates,NStates))
        ρijtStep += np.conjugate(ρijtStep).T
        μt = model.μ()[iE[El][0],iE[El][1]] * ρijtStep
        R1t[tStep] += 1j*np.trace(np.dot(μt,sf.commutator(model.μ(),model.ρ0())))

np.savetxt("R1Re.txt",np.array([timeSteps,np.real(R1t)]).T)
np.savetxt("R1Im.txt",np.array([timeSteps,np.imag(R1t)]).T)

R1ReData = np.loadtxt("R1Re.txt")
R1ImData = np.loadtxt("R1Im.txt")
time = R1ReData[:,0]/fs2au
R1Re = R1ReData[:,1]
R1Im = R1ImData[:,1]
R1ωRe = np.zeros(len(ω))
R1ωIm = np.zeros(len(ω))

"""Fourier transform over the t axis"""
for omega in range(len(ω)):
    print(ω[omega], " cm-1")
    exp_fact = np.exp(1j*ω[omega]*timeSteps/fs2au)
    # cos_fact = np.cos(np.pi*t/(2*np.max(t)))
    int_func = exp_fact*(R1Re+1j*R1Im)
    R1ωRe[omega] = np.real(-2*np.sum(int_func)*dω)
    R1ωIm[omega] = np.imag(-2*np.sum(int_func)*dω)

np.savetxt("Response_Re.txt",np.array([ω,R1ωRe]).T)
np.savetxt("Response_Im.txt",np.array([ω,R1ωIm]).T)

# plt.plot(ω,np.abs(R1ωRe))
# plt.savefig("Romega_Re.png")
plt.plot(ω,np.abs(R1ωIm))
plt.savefig("Romega_Im.png")

