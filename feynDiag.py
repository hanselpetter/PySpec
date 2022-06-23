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
    """returns positive value if right side and negative value if left side"""
    ketDiff = currEl[0]-prevEl[0]
    braDiff = currEl[1]-prevEl[1]
    relDiff = np.abs(braDiff)-np.abs(ketDiff)
    return np.sign(relDiff)