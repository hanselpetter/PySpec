import numpy as np
from PLDM import coupled_dimer as model

def commutator(A,B):
    "commutator of two matrix elements"
    return A@B - B@A

def μx(ρ):
    "operating μ on density matrix ρ"
    return commutator(model.μ(),ρ)

def non0(mat):
    "Finding the indices of nonzero matrix elements"
    n0 = np.array(np.where(mat!=0))
    return n0[:,:(n0.shape[1])].T

def genRho(ρ):
    "Generating the whole rho(n) from single matrix element propagation"
    return ρ + np.conjugate(ρ.T)
