import numpy as np
from numpy import random as rd
from numba import jit
import coupled_dimer as model
import os

@jit(nopython=True)
def initR():
    '''Sampling the initial position and velocities of bath parameters from 
       wigner distribution'''

    Kb = 8.617333262*1e-5*model.eV2au/model.K2au    # Kb = 8.617333262 x 10^-5 eV/K
    T = 300*model.K2au                              # Temperature in au
    β = 1/(Kb*T)

    σR_wigner = 1/np.sqrt(2*model.ωj*np.tanh(β*model.ωj*0.5))
    σP_wigner = np.sqrt(model.ωj/(2*np.tanh(β*model.ωj*0.5)))
    μR_wigner = 0
    μP_wigner = 0

    R = np.zeros(2*model.NModes)
    P = np.zeros(2*model.NModes)
    for dof in range(model.NModes):
        R[dof],R[model.NModes+dof] = rd.normal(μR_wigner, σR_wigner[dof]), rd.normal(μR_wigner, σR_wigner[dof])
        P[dof],P[model.NModes+dof] = rd.normal(μP_wigner, σP_wigner[dof]), rd.normal(μP_wigner, σP_wigner[dof])

    return R, P

os.makedirs("Trajectories", exist_ok=True)
os.chdir("Trajectories")
for itraj in range(model.NTraj):
    print(itraj)
    os.makedirs(f"{itraj+1}", exist_ok=True)
    os.chdir(f"{itraj+1}")
    itrajR, itrajP = initR()
    np.savetxt(f"initial_bath_{itraj+1}.txt",np.array([itrajR,itrajP]).T)
    os.chdir("../")