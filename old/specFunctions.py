import numpy as np

def commutator(A,B):
    "commutator of two matrix elements"
    return A@B - B@A

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

def impElement(model,ρ):
    μx = commutator(model.μ(),ρ)
    return np.array(np.where(μx!=0)).T
