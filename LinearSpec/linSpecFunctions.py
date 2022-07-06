import numpy as np
from numba import jit
import sys
sys.path.insert(0, "/scratch/mmondal/specTest/PLDM")

import coupled_dimer as model
import pldm as method

μ_t0 = model.μ()
ρ_t0 = model.ρ0()
ρlen = ρ_t0.shape[0]*ρ_t0.shape[1]

def commutator(A,B):
    "commutator of two matrix elements"
    return np.dot(A,B) - np.dot(B,A)

def μx(model,ρ):
    "operating μ on density matrix ρ"
    return commutator(model.μ(),ρ)

def non0(mat):
    "Finding the indices of nonzero matrix elements"
    n0 = np.array(np.where(mat!=0))
    return n0[:,:(n0.shape[1])].T

def genRho(ρ):
    "Generating the whole rho(n) from single matrix element propagation"
    return ρ + np.conjugate(ρ.T)

# @jit(nopython=True)
def impElement(model,ρ):
    μx = commutator(model.μ(),ρ)
    return np.array(np.where(μx!=0)).T

