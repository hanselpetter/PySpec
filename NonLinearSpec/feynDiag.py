import numpy as np
from numba import jit

@jit(nopython=True)
def non0Indices(matrix):
    """Finding the non-zero indices of a matrix as np.where not working with numba"""
    rows, columns = matrix.shape
    non0indices = np.array([0]) 
    for i in range(rows):
        for j in range(columns):
            if matrix[i,j] != 0:
                non0indices = np.hstack((non0indices,np.array([i,j])))
    return non0indices[1:].reshape(len(non0indices[1:])//2,2).T

@jit(nopython=True)
def extremeIndices(ρ):
    """input -> density matrix ρ
       output -> Fmax, Fmin = maximum and minimum forward initial indices of nonzero terms in ρ
                 Bmax, Bmin = maximum and minimum backward initial indices of nonzero terms in ρ"""
    ρabs = np.abs(ρ)
    non0 = non0Indices(ρabs)  # check after first laser about other matrix elements
    Fmax, Fmin = np.max(non0[0,:]), np.min(non0[0,:])
    Bmax, Bmin = np.max(non0[1,:]), np.min(non0[1,:])
    return Fmax, Fmin, Bmax, Bmin

@jit(nopython=True)
def kSign(fF, fB, iFmax, iFmin, iBmax, iBmin):
    """NB -> check this for higher excitation manifolds and higher dimensionalities"""
    """input -> fF, fB = Final forward and backward indices
                iFmax, iFmin = maximum and minimum forward initial indices of nonzero terms in ρ
                iBmax, iBmin = maximum and minimum backward initial indices of nonzero terms in ρ
       output -> sign of k based on the manifold of transition"""    
    if (fF > iFmax) or (fB < iBmin):
        return 1
    if (fF < iFmin) or (fB > iBmax):
        return -1

@jit(nopython=True)
def testK(k1,k2):
    """Testing whether the k1 and k2 are equal
       input -> k1 and k2 vectors of length 3 and containing either 1s or -1s,
                the data type should be float for np.dot to work
       output -> True if k1 == k2 and False if k1 != k2"""
    return np.dot(k1,k2.reshape(3,1)) == 3

@jit(nopython=True)
def getDiag(k):
    """Returns the kind of feynman diagram for a given (k1,k2,k3) combinamtion"""

    # Rephasing signals
    if testK(k,np.array([-1.,1.,1.])):
        diagram =  0
    elif testK(k,np.array([1.,-1.,-1.])):
        diagram =  1

    # Non-Rephasing signals
    elif testK(k,np.array([1.,-1.,1.])):
        diagram =  2
    elif testK(k,np.array([-1.,1.,-1.])):
        diagram =  3

    # Double quantum signals
    elif testK(k,np.array([1.,1.,-1.])):
        diagram =  4
    elif testK(k,np.array([-1.,-1.,1.])):
        diagram =  5

    # other signals
    elif testK(k,np.array([1.,1.,1.])):
        diagram =  6
    elif testK(k,np.array([-1.,-1.,-1.])):
        diagram =  7
    
    return diagram

if __name__ == "__main__":

    ρ = np.zeros((4,4),dtype=np.complex128)
    ρ[1,0], ρ[2,0] = 0.76+1j*0.09, 0.12-1j*0.25
    iFmax, iFmin, iBmax, iBmin = extremeIndices(ρ)
    ks1 = kSign(2, 1, iFmax, iFmin, iBmax, iBmin)
    ks2 = kSign(1, 2, iFmax, iFmin, iBmax, iBmin)
    ks3 = kSign(2, 2, iFmax, iFmin, iBmax, iBmin)
    print(f"({ks1}, {ks2}, {ks3}) corresponds to diagram {getDiag(np.array([ks1,ks2,ks3],dtype=np.float64))}")

    ρdagger = np.conjugate(ρ).T
    iFmax, iFmin, iBmax, iBmin = extremeIndices(ρdagger)
    ks1 = kSign(2, 1, iFmax, iFmin, iBmax, iBmin)
    ks2 = kSign(1, 2, iFmax, iFmin, iBmax, iBmin)
    ks3 = kSign(2, 2, iFmax, iFmin, iBmax, iBmin)
    print(f"({ks1}, {ks2}, {ks3}) corresponds to diagram {getDiag(np.array([ks1,ks2,ks3],dtype=np.float64))}")