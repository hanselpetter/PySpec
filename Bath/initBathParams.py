import numpy as np
import matplotlib.pyplot as plt

fs2au = 41.341374575751
cminv2au = 4.55633*1e-6
ωc = 18*cminv2au                # cutoff frequency
λ = 50*cminv2au                 # Reorganisation energy
N = 100                         # Number of modes needed
ω = np.linspace(0.00000001,250*ωc,30000)
dω = ω[1]-ω[0]

def J(ω,ωc,λ):  
    f1 = 2*λ*ωc*ω
    f2 = (ωc**2) + (ω**2)
    return f1/f2

Fω = np.zeros(len(ω))
for i in range(len(ω)):
    Fω[i] = (4/np.pi) * np.sum(J(ω[:i],ωc,λ)/ω[:i]) * dω

λs = Fω[-1]
ωj = np.zeros(N)
for i in range(N):
    costfunc = np.abs(Fω-(((i+0.5)/N)*λs))
    ωj[i] = ω[np.where(costfunc == np.min(costfunc))[0]]
cj = ωj * ((λs/(2*N))**0.5)

np.savetxt("ωj_50.txt",ωj)
np.savetxt("cj_50.txt",cj)