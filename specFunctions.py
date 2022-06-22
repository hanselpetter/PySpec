import numpy as np
import cmath
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

def genRho(rho):
    "Generating the whole rho(n) from single matrix element propagation"
    return rho + np.conjugate(rho.T)

def cdf(mat):
    "Generating a Cumulative Distribution of a matrix"
    cum = np.cumsum(mat)
    maxVal = np.max(cum)
    cum /= maxVal
    return cum, maxVal

def pol(mat):
    """complex matrix is polar form
     input -> matrix,
     output -> magnitude matrix, phase matrix of input matrix"""
    r = np.zeros((mat.shape[0],mat.shape[1]))
    theta = np.zeros((mat.shape[0],mat.shape[1]))
    for i in mat.shape[0]:
        for j in mat.shape[1]:
            r[i,j], theta[i,j] = cmath.polar(mat[i,j])
    return r,theta

def focusedMCSample(cdfMat):
    """Finding the index of most important density matrix element based on
        Monte Carlo sampling"""
    randNum = np.random.uniform()
    cdfMatShift = cdfMat-randNum
    posIndices = np.where(cdfMatShift >= 0)[0]
    return posIndices[0]

def one2two(mat,num):
    """converting 1D array index to 2D array index"""
    fac = num//mat.shape[0]
    rem = num%mat.shape[1]
    return np.array([fac-1,rem-1])

def focus(mat):
    """getting the initStateF and initStateB for the given matrix"""
    rMat, thetaMat = pol(mat)
    rcdf, rcdfMax = cdf(rMat)
    impEl = focusedMCSample(rcdf)
    impElPhase = thetaMat.flatten()[impEl]
    trajWeight = rcdfMax*np.exp(1j*impElPhase)
    focusedEl = one2two(mat,impEl)
    return focusedEl, trajWeight

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