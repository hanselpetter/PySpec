import numpy as np

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# Initialization of the mapping Variables
def initMapping(Nstates, initStateF = 0, initStateB = 0, stype = "focused"):
    #global qF, qB, pF, pB, qF0, qB0, pF0, pB0
    qF = np.zeros((Nstates))
    qB = np.zeros((Nstates))
    pF = np.zeros((Nstates))
    pB = np.zeros((Nstates))
    if (stype == "focused"):
        qF[initStateF] = 1.0
        qB[initStateB] = 1.0
        pF[initStateF] = 1.0
        pB[initStateB] = -1.0 # This minus sign allows for backward motion of fictitious oscillator
    elif (stype == "sampled"):
       qF = np.array([ np.random.normal() for i in range(Nstates)]) 
       qB = np.array([ np.random.normal() for i in range(Nstates)]) 
       pF = np.array([ np.random.normal() for i in range(Nstates)]) 
       pB = np.array([ np.random.normal() for i in range(Nstates)]) 
    return qF, qB, pF, pB 

def Umap(qF, qB, pF, pB, dt, VMat):
    qFin, qBin, pFin, pBin = qF * 1.0, qB * 1.0, pF * 1.0, pB * 1.0 # Store input position and momentum for verlet propogation
    # Store initial array containing sums to use at second derivative step

    VMatxqB =  VMat @ qBin #np.array([np.sum(VMat[k,:] * qBin[:]) for k in range(NStates)])
    VMatxqF =  VMat @ qFin #np.array([np.sum(VMat[k,:] * qFin[:]) for k in range(NStates)])

    # Update momenta using input positions (first-order in dt)
    pB -= 0.5 * dt * VMatxqB  # VMat @ qBin  
    pF -= 0.5 * dt * VMatxqF  # VMat @ qFin  
    # Now update positions with input momenta (first-order in dt)
    qB += dt * VMat @ pBin  
    qF += dt * VMat @ pFin  
    # Update positions to second order in dt
    qB -=  (dt**2/2.0) * VMat @ VMatxqB 
    qF -=  (dt**2/2.0) * VMat @ VMatxqF
       #-----------------------------------------------------------------------------
    # Update momenta using output positions (first-order in dt)
    pB -= 0.5 * dt * VMat @ qB  
    pF -= 0.5 * dt * VMat @ qF  

    return qF, qB, pF, pB

def Force(dat):

    dH = dat.dHij #dHel(R) # Nxnxn Matrix, N = Nuclear DOF, n = NStates 
    dH0 = dat.dH0
    qF, pF, qB, pB =  dat.qF, dat.pF, dat.qB, dat.pB
    # F = np.zeros((len(dat.R)))
    F = -dH0
    for i in range(len(qF)):
        F -= 0.25 * dH[i,i,:] * ( qF[i] ** 2 + pF[i] ** 2 + qB[i] ** 2 + pB[i] ** 2)
        for j in range(i+1, len(qF)):
            F -= 0.5 * dH[i,j,:] * ( qF[i] * qF[j] + pF[i] * pF[j] + qB[i] * qB[j] + pB[i] * pB[j])
    return F

def Force_vctorised(dat):
    '''Force on bath due to electronic states mapping variables'''

    qF, pF, qB, pB =  dat.qF, dat.pF, dat.qB, dat.pB
    qFmat, pFmat, qBmat, pBmat = qF.T@qF, pF.T@pF, qB.T@qB, pB.T@pB
    F = - dat.dH0 - 0.25*np.einsum("ijk,ij->k",dat.dHij,qFmat+pFmat+qBmat+pBmat)

    return F


def VelVer(dat) : # R, P, qF, qB, pF, pB, dtI, dtE, F1, Hij,M=1): # Ionic position, ionic velocity, etc.
 
    # data 
    qF, qB, pF, pB = dat.qF * 1.0, dat.qB *  1.0, dat.pF * 1.0, dat.pB * 1.0
    par =  dat.param
    v = dat.P/par.M
    EStep = int(par.dtN/par.dtE)
    dtE = par.dtN/EStep
    
    # half-step mapping
    for t in range(int(np.floor(EStep/2))):
        qF, qB, pF, pB = Umap(qF, qB, pF, pB, dtE, dat.Hij)
    dat.qF, dat.qB, dat.pF, dat.pB = qF * 1, qB * 1, pF * 1, pB * 1 

    # ======= Nuclear Block ==================================
    F1    =  Force(dat) # force with {qF(t+dt/2)} * dH(R(t))
    dat.R += v * par.dtN + 0.5 * F1 * par.dtN ** 2 / par.M
    
    #------ Do QM ----------------
    dat.Hij  = par.Hel(dat.R)
    dat.dHij = par.dHel(dat.R)
    dat.dH0  = par.dHel0(dat.R)
    #-----------------------------
    F2 = Force(dat) # force with {qF(t+dt/2)} * dH(R(t+ dt))
    v += 0.5 * (F1 + F2) * par.dtN / par.M

    dat.P = v * par.M
    # =======================================================
    
    # half-step mapping
    dat.Hij = par.Hel(dat.R) # do QM
    for t in range(int(np.ceil(EStep/2))):
        qF, qB, pF, pB = Umap(qF, qB, pF, pB, dtE, dat.Hij)
    dat.qF, dat.qB, dat.pF, dat.pB = qF, qB, pF, pB 
    
    return dat

def pop(dat):
    return np.outer(dat.qF + 1j * dat.pF, dat.qB-1j*dat.pB) * dat.rho0

def runTraj(model):
    #------- Seed --------------------
    try:
        np.random.seed(model.SEED)
    except:
        pass
    #------------------------------------
    ## Parameters -------------
    NSteps = model.NSteps
    NTraj = model.NTraj
    NStates = model.NStates
    initStateF = model.initStateF  # Forward intial state
    initStateB = model.initStateB  # Backward initial state
    stype = model.stype
    nskip = model.nskip
    #---------------------------
    if NSteps%nskip == 0:
        pl = 0
    else :
        pl = 1
    rho_ensemble = np.zeros((NStates,NStates,NSteps//nskip + pl), dtype=complex)
    # Ensemble
    for itraj in range(NTraj): 
        # Trajectory data
        print("Simulating trajectory ",itraj)
        dat = Bunch(param =  model)
        dat.R, dat.P = model.initR()

        # set propagator
        vv  = VelVer

        # Call function to initialize mapping variables
        dat.qF, dat.qB, dat.pF, dat.pB = initMapping(NStates, initStateF, initStateB, stype) 

        # Set initial values of fictitious oscillator variables for future use
        qF0, qB0, pF0, pB0 = dat.qF[initStateF], dat.qB[initStateB], dat.pF[initStateF], dat.pB[initStateB] 
        dat.rho0 = 0.25 * (qF0 - 1j*pF0) * (qB0 + 1j*pB0)

        #----- Initial QM --------
        dat.Hij  = parameters.Hel(dat.R)
        dat.dHij = parameters.dHel(dat.R)
        dat.dH0  = parameters.dHel0(dat.R)
        #----------------------------
        iskip = 0 # please modify
        for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                rho_ensemble[:,:,iskip] += pop(dat)
                iskip += 1
            #-------------------------------------------------------
            dat = vv(dat)

    return rho_ensemble