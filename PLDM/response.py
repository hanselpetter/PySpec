import numpy as np
import matplotlib.pyplot as plt

fs2au = 41.341374575751
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
time = ρ10_data[:,0]/fs2au

ρ10 = μ10*(ρ10_data[:,1::2] + 1j*ρ10_data[:,2::2]).reshape(time.shape[0],4,4)
ρ01 = np.conjugate(μ10)*np.array([np.conjugate(ρ10[i].T) for i in range(time.shape[0])])
ρ20 = μ20*(ρ20_data[:,1::2] + 1j*ρ20_data[:,2::2]).reshape(time.shape[0],4,4)
ρ02 = np.conjugate(μ20)*np.array([np.conjugate(ρ20[i].T) for i in range(time.shape[0])])
μt = ρ10+ρ01+ρ20+ρ02

R1 = np.array([1j*np.trace(μt[i] @ μx) for i in range(time.shape[0])])

plt.plot(time,R1)
# plt.xlim(0,300)
plt.xlabel("time (fs) ----->")
plt.ylabel(r"$R^{(1)}$(t) ----->")
plt.show()
# r10 = rho1[:,4]
# r20 = rho2[:,8]
# plt.plot(time,np.real(r10))
# plt.plot(time,np.abs(r20))
# plt.show()
exit()

print(r10[10]+np.conjugate(r10[10]))
exit()
t1 = (mu10**2)*np.imag((r10+np.conjugate(r10))*r10[0])
t2 = (mu20**2)*np.imag((r20+np.conjugate(r20))*r20[0])
t3 = (mu10*mu20)*np.imag((r10+np.conjugate(r10))*r20[0])
t4 = (mu10*mu20)*np.imag((r20+np.conjugate(r20))*r10[0])

Cmm = t1+t2+t3+t4
plt.plot(time,t1)
plt.xlabel("t(fs)")
plt.ylabel(r"$R_1(t)$")
plt.show()
exit()
au2cminv = 1/(4.55633*1e-6)

Rshift = fftshift(Cmm)
fourier = fftshift(fft(Rshift,norm='ortho'))
omega = 2*np.pi*fftshift(fftfreq(len(Rshift), (max(Rshift)-min(Rshift))/len(Rshift)))
# plt.plot(omega,np.imag(fourier),label="imag")
plt.plot(omega,np.real(fourier),label="real")
plt.legend()
plt.xlim(-2000,2000)
# plt.xlim(1300,1800)
plt.show()

