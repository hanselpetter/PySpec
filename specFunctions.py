import numpy as np
from PLDM import coupled_dimer as model

def commutator(A,B):
    "commutator of two matrix elements"
    return A@B - B@A

def μx(ρ):
    "operating μ on density matrix ρ"
    return commutator(model.μ(),ρ)

def impElement(pulseNumber,ρ):
    "Finding the indices of the most important matrix elements"
    mat = μx(ρ)
    if pulseNumber == 1:
        "Basically first pump or just needed for linear absorption"
        non0 = np.array(np.where(mat!=0))
        return non0[:,:(non0.shape[1]//2)].T
    else:
        "To be modified for selecting important element for 2nd pulse onwards"
        return 0