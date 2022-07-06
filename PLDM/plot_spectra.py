import numpy as np
import matplotlib.pyplot as plt 

berkelbach = np.loadtxt("exact_spectra.txt")
provozza = np.loadtxt("provozza_spectra.txt")
mannouch = np.loadtxt("spin-PLDM_HEOM_exact_spectra_1050.txt")
ωb = berkelbach[:,0]
Rb = berkelbach[:,1]
ωp = provozza[:,0]
Rp = provozza[:,1]
ωm = mannouch[:,0]
Rm = mannouch[:,1]


plt.plot(ωb,Rb/max(Rb),label="berkelbach")
plt.plot(ωp,Rp/max(Rp),label="provozza")
plt.plot(ωm-1050,Rm/max(Rm),label="mannouch")
plt.legend()
plt.show()