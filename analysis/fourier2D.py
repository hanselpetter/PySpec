import numpy as np
import scipy.constants as sc
from PLDM import coupled_dimer as model

cminv2au = 4.55633*1e-6
fs2au = 41.341374575751
ω_max = 800*2*np.pi*sc.c/(10**13)
ω_min = -ω_max
ω = np.linspace(0,4,1001)
dω = ω[1] - ω[0]

R3 = np.zeros((model.NSteps1, model.NSteps2, model.NSteps3),dtype=complex)
t1 = np.arange(0,model.Sim1Time,model.dtN/fs2au)
t2 = np.arange(0,model.Sim2Time,model.dtN/fs2au)
t3 = np.arange(0,model.Sim3Time,model.dtN/fs2au)
R3t1t2ω3 = np.zeros((len(t1),len(t2),len(ω)),dtype=complex)
# R3ω1t2t3 = np.zeros((len(ω),len(t2),len(t3)),dtype=complex)
R3ω1t2ω3 = np.zeros((len(ω),len(t2),len(ω)),dtype=complex)


"""First fourier transform over the t3 axis"""
for i in range(len(t1)):
    for j in range(len(t2)):
        for omega in range(len(ω)):
            # print(ω[omega], " cm-1")
            exp_fact = np.exp(1j*ω[omega]*t3)
            # cos_fact = np.cos(np.pi*t/(2*np.max(t)))
            int_func = exp_fact*R3[i,j,:]
            R3t1t2ω3[i,j,omega] = -2*np.sum(int_func)*dω

for i in range(len(t2)):
    for j in range(len(ω)):
        for omega in range(len(ω)):
            # print(ω[omega], " cm-1")
            exp_fact = np.exp(1j*ω[omega]*t1)
            # cos_fact = np.cos(np.pi*t/(2*np.max(t)))
            int_func = exp_fact*R3t1t2ω3[:,i,j]
            R3ω1t2ω3[omega,i,j] = -2*np.sum(int_func)*dω

np.savetxt("Response.txt",R3ω1t2ω3.flatten())
