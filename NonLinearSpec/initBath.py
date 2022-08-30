import numpy as np
from numpy import random as rd
from numba import jit

import coupled_dimer as model

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

    R = np.zeros(model.NR)
    P = np.zeros(model.NR)
    for dof in range(model.NModes):
        for site in range(model.NR//model.NModes):
            R[site*model.NModes + dof] = rd.normal(μR_wigner, σR_wigner[dof])
            P[site*model.NModes + dof] = rd.normal(μP_wigner, σP_wigner[dof])
    return R, P