import numpy as np 
import matplotlib.pyplot as plt 

fs2au = 41.341374575751
cminv2au = 4.55633*1e-6

exact = np.loadtxt("exact_population.txt")
t = np.loadtxt("time_1e3_1_1e2_11.txt")
ρs_1e5 = np.loadtxt("1e3_1_1e2_11_1e5_sampled.txt")
ρs_5e4 = np.loadtxt("1e3_1_1e2_11_5e4_sampled.txt")

plt.plot(t,ρs_1e5,label="sampled_1e5_dN=1")
plt.plot(t,ρs_5e4,label="sampled_5e4_dN=1")
plt.plot(exact[:,0],exact[:,1],label="provozza")

plt.xlabel("time (fs) ----->")
plt.ylabel(r"$\rho$ ----->")
plt.legend()
plt.show()