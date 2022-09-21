import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import coupled_dimer as model

fs2au = 41.341374575751
cminv2au = 4.55633*1e-6
eV2au = 0.036749405469679
K2au = 0.00000316678

def savStpRem(NSteps, nskip):
    if NSteps%nskip == 0:
            pl = 0
    else :
        pl = 1
    return pl

NSteps = int(model.SimTime1/(model.dtN/fs2au)) + 1
NStates = model.NStates
nskip = model.nskip1
pl = savStpRem(NSteps, nskip)
Rdim = (NSteps//nskip) + pl
t = np.arange(0, model.dtN*Rdim, model.dtN)/fs2au

R1_01 = np.loadtxt("dimer/R1_01.txt")
R1_10 = np.loadtxt("dimer/R1_10.txt")
R1_02 = np.loadtxt("dimer/R1_02.txt")
R1_20 = np.loadtxt("dimer/R1_20.txt")
R = (R1_01+R1_10+R1_02+R1_20)[:,0] + 1j*(R1_01+R1_10+R1_02+R1_20)[:,1]

# Checking the time-dependent response
# plt.plot(t, np.real(R), label="Real")
# plt.plot(t, np.imag(R), label="Imag")
# plt.legend()
# plt.show()

"""Parameters for Fourier transform"""
ωMax = 11000*2*np.pi*sc.c/(10**13)*2*np.pi*sc.c/(10**13)
ωMin = 9000*2*np.pi*sc.c/(10**13)*2*np.pi*sc.c/(10**13)
ω = np.linspace(ωMin, ωMax, 4001)
dω = ω[1]-ω[0]

smoothing = np.cos(np.pi*t/(2*np.max(t)))           # smoothing function for linear response
ot = np.outer(ω,t)                                  # outer product of ω and t
expot = np.exp(1j*ot)
expotsmooth = np.einsum("ij,j->ij",expot,smoothing)

"""Doing the Fourier transform"""
Rω = expotsmooth @ (R.reshape(len(R),1))

# checking the response
# ωcminvJ = (ω/(2*np.pi*sc.c/(10**13)))#-10050
# fig, ax = plt.subplots()
# fig.set_size_inches(10,8,True)
# ax.plot(ωcminvJ, -Rω.imag/np.max(np.abs(Rω)), label="No-cavity")
# plt.legend()
# plt.show()