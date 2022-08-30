import numpy as np

def KSide(prevEl,currEl):
    """returns +1 if bra side and -1 if ket side
       Need to be rigoruosly studied for larger hilbert spaces"""
    ketDiff = currEl[0]-prevEl[0]
    braDiff = currEl[1]-prevEl[1]
    relDiff = np.abs(braDiff)-np.abs(ketDiff)
    return int(np.sign(relDiff/np.abs(relDiff)))

def KSign(prevEl,currEl):
    """returns +1 if +k and -1 if -k
       Need to be rigoruosly studied for larger hilbert spaces"""
    kside = KSide(prevEl,currEl)
    ketDiff = prevEl[0]-currEl[0]
    braDiff = prevEl[1]-currEl[1]
    netDiff = ketDiff+braDiff
    return int(np.sign((kside*netDiff)/np.abs(kside*netDiff)))

def sigCombo():
    """possible combination of signals"""
    signs = np.array([1,-1])
    sigComb = []
    for i in signs:
        for j in signs:
            for k in signs:
                sigComb.append(np.array([i,j,k]))
    return np.array(sigComb)

def rightInt(prevEl, currEl):
    ketDiff = currEl[1] - prevEl[1]
    return np.sign(ketDiff)**2