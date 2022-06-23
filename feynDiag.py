import numpy as np

def numDiags():
    """generates the possible diagrams for 2D process"""
    signs = np.array([1,-1])
    diagrams = []
    for i in signs:
        for j in signs:
            for k in signs:
                diagrams.append(np.array([i,j,k]))
    return np.array(diagrams)

def KSide(prevEl,currEl):
    """returns +1 if bra side and -1 if ket side"""
    ketDiff = currEl[0]-prevEl[0]
    braDiff = currEl[1]-prevEl[1]
    relDiff = np.abs(braDiff)-np.abs(ketDiff)
    return relDiff//np.abs(relDiff)

def KSign(prevEl,currEl):
    """returns +1 if +k and -1 if -k"""
    kside = KSide(prevEl,currEl)
    ketDiff = prevEl[0]-currEl[0]
    braDiff = prevEl[1]-currEl[1]
    netDiff = ketDiff+braDiff
    return (kside*netDiff)//np.abs(kside*netDiff)
