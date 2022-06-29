import numpy as np
import specFunctions as sF
import cmath

def cdf(mat):
    "Generating a Cumulative Distribution of a matrix with its elements"
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
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
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
    rowIndex = num//(mat.shape[0])
    columIndex = num%(mat.shape[0])
    return np.array([rowIndex,columIndex])

def focusing(mat):
    """getting the initStateF and initStateB for the given matrix"""
    rMat, thetaMat = pol(mat)
    rcdf, rcdfMax = cdf(rMat)
    impEl = focusedMCSample(rcdf)
    impElPhase = thetaMat.flatten()[impEl]
    trajWeight = rcdfMax*np.exp(1j*impElPhase)
    focusedEl = one2two(mat,impEl)
    return focusedEl, trajWeight