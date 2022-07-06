import numpy as np
import matplotlib.pyplot as plt


traj1_1e5 = np.loadtxt("/Users/quantised_elious/Desktop/work/codes/PLDM/eavg_20000/1e3_1_1e2_11_1e5_sampled/PijD_11129417_1.txt",dtype=complex)
traj1_5e4 = np.loadtxt("/Users/quantised_elious/Desktop/work/codes/PLDM/eavg_20000/1e3_1_1e2_11_5e4_sampled/PijD_11141227_1.txt",dtype=complex)
ρ11_1e5 = traj1_1e5[:,11] + 1j*traj1_1e5[:,12]
ρ11_5e4 = traj1_5e4[:,11] + 1j*traj1_5e4[:,12]
fs2au = 41.341374575751
t = np.real(traj1_1e5[:,0])/fs2au

for traj in range(2,251):
    print(f"analysing trajectory {traj}")
    traj_1e5 = np.loadtxt(f"/Users/quantised_elious/Desktop/work/codes/PLDM/eavg_20000/1e3_1_1e2_11_1e5_sampled/PijD_11129417_{traj}.txt",dtype=complex)
    traj_5e4 = np.loadtxt(f"/Users/quantised_elious/Desktop/work/codes/PLDM/eavg_20000/1e3_1_1e2_11_5e4_sampled/PijD_11141227_{traj}.txt",dtype=complex)
    ρ11_1e5 += traj_1e5[:,11] + 1j*traj_1e5[:,12]
    ρ11_5e4 += traj_5e4[:,11] + 1j*traj_5e4[:,12]

np.savetxt("1e3_1_1e2_11_1e5_sampled.txt",np.real(ρ11_1e5/250))
np.savetxt("1e3_1_1e2_11_5e4_sampled.txt",np.real(ρ11_5e4/250))

plt.plot(t,np.real(ρ11_1e5/250),label="1e5")
plt.plot(t,np.real(ρ11_5e4/250),label="5e4")
plt.show()
