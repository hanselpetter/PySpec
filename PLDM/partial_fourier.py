import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

cminv2au = 4.55633*1e-6
fs2au = 41.341374575751
ω_max = 800*2*np.pi*sc.c/(10**13)
ω_min = -ω_max
ω = np.linspace(-4,4,20001)
dω = ω[1] - ω[0]
μ10 = -5+(1j*0)
μ20 = 1+(1j*0)

ρ0 = np.zeros((4,4),dtype=complex)
ρ0[0,0] = 1

μ0 = np.zeros((4,4),dtype=complex)
μ0[1,0], μ0[0,1] = μ10, np.conjugate(μ10)
μ0[2,0], μ0[0,2] = μ20, np.conjugate(μ20)  

def commutator(A, B):
    return A@B - B@A

μx = commutator(μ0,ρ0)

ρ10_data = np.loadtxt("eavg_20000/PijD_cpldimr_300_5_100_10_eavg20000.txt",dtype=complex)
ρ20_data = np.loadtxt("eavg_20000/PijD_cpldimr_300_5_100_20_eavg20000.txt",dtype=complex)
# ρ10_adata = np.loadtxt("PijD_CplDimr_10_aamod.txt",dtype=complex)
# ρ20_adata = np.loadtxt("PijD_CplDimr_20_aamod.txt",dtype=complex)
time = ρ10_data[:,0]/fs2au

ρ1l = μ10*(ρ10_data[:,1::2] + 1j*ρ10_data[:,2::2]).reshape(time.shape[0],4,4)
ρ1u = np.conjugate(μ10)*np.array([np.conjugate(ρ1l[i].T) for i in range(time.shape[0])])
ρ2l = μ20*(ρ20_data[:,1::2] + 1j*ρ20_data[:,2::2]).reshape(time.shape[0],4,4)
ρ2u = np.conjugate(μ20)*np.array([np.conjugate(ρ2l[i].T) for i in range(time.shape[0])])
μt = ρ1l+ρ1u+ρ2l+ρ2u

berkelbach = np.loadtxt("exact_spectra.txt")
provozza = np.loadtxt("provozza_spectra.txt")
mannouch = np.loadtxt("spin-PLDM_HEOM_exact_spectra_1050.txt")
ωb = berkelbach[:,0]
Rb = berkelbach[:,1]
ωp = provozza[:,0]
Rp = provozza[:,1]
ωm = mannouch[:,0]
Rm = mannouch[:,1]

# ρ1l_a = μ10*(ρ10_adata[:,1::2] + 1j*ρ10_adata[:,2::2]).reshape(time.shape[0],4,4)
# ρ1u_a = np.conjugate(μ10)*np.array([np.conjugate(ρ1l_a[i].T) for i in range(time.shape[0])])
# ρ2l_a = μ20*(ρ20_adata[:,1::2] + 1j*ρ20_adata[:,2::2]).reshape(time.shape[0],4,4)
# ρ2u_a = np.conjugate(μ20)*np.array([np.conjugate(ρ2l_a[i].T) for i in range(time.shape[0])])
# μt_a = ρ1l_a+ρ1u_a+ρ2l_a+ρ2u_a

R1 = np.array([1j*np.trace(μt[i] @ μx) for i in range(time.shape[0])])
# R1_a = np.array([1j*np.trace(μt_a[i] @ μx) for i in range(time.shape[0])])

# plt.plot(time,R1)
# plt.show()
# exit()

α = np.zeros(len(ω),dtype=complex)
# α_a = np.zeros(len(ω),dtype=complex)
for omega in range(len(ω)):
    print(ω[omega], " cm-1")
    exp_fact = np.exp(1j*ω[omega]*(time))
    cos_fact = np.cos(np.pi*time/(2*np.max(time)))
    int_func = exp_fact*R1*cos_fact
    # int_func_a = exp_fact*R1_a*cos_fact
    α[omega] = -2*np.sum(int_func)*dω
    # α_a[omega] = -2*np.sum(int_func_a)*dω

# plt.plot(ω/(2*np.pi*sc.c/(10**13))-10000,np.imag(α_a),label="cos")
plt.plot(ωb,Rb/max(Rb),label="berkelbach")
plt.plot(ωp,Rp/max(Rp),label="provozza")
plt.plot(ωm-1050,Rm/max(Rm),label="mannouch")
plt.plot(ω/(2*np.pi*sc.c/(10**13))-20000,np.imag(α)/max(np.imag(α)[len(α)//2:]),label="my_sim")
plt.xlim(-800,800)
plt.ylim(0,1)
# plt.plot(ω,np.real(α),label="Re")
# plt.plot(ω,np.abs(α)**2,label="Abs")
plt.xlabel(r"$ω-\bar{ε}$ $(cm^{-1})$ ----->")
plt.ylabel("Absorption ----->")
plt.legend()
plt.show()

# np.savetxt("aamod_model_Abs_real.txt",np.real(α))
np.savetxt("my_model_Abs_imag.txt",np.imag(α))