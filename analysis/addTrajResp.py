import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/scratch/mmondal/specTest/PLDM")
import coupled_dimer as model
import linSpecFunctions as sf

fs2au = 41.341374575751
TrajFolder = "/scratch/mmondal/specTest/linearSpec/Trajectories/"
pl = sf.savStpRem(model.NSteps1, model.nskip)
timeSteps = np.arange(0,(model.NSteps1//model.nskip + pl)*model.dtN,model.dtN)
R1Re = np.zeros(len(timeSteps))
R1Im = np.zeros(len(timeSteps))

for traj_index in range(model.NTraj):
    itraj = traj_index+1
    print(itraj)
    R1Re += np.loadtxt(TrajFolder + f"{itraj}/R1Re.txt")
    R1Im += np.loadtxt(TrajFolder + f"{itraj}/R1Im.txt")

np.savetxt("R1ReDimr.txt",np.array([timeSteps/fs2au,R1Re/itraj]).T)
np.savetxt("R1ImDimr.txt",np.array([timeSteps/fs2au,R1Im/itraj]).T)

plt.plot(timeSteps,R1Re)
plt.savefig("R1re.png")

