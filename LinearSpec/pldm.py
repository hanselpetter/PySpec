import numpy as np
from numba import jit
import coupled_dimer as model

# Initialization of the mapping Variables
@jit(nopython=True)
def initMapping(Nstates, initStateF = 0, initStateB = 0, stype = 0):
    #global qF, qB, pF, pB, qF0, qB0, pF0, pB0
    qF = np.zeros((Nstates))
    qB = np.zeros((Nstates))
    pF = np.zeros((Nstates))
    pB = np.zeros((Nstates))
    if (stype == 0):
        qF[initStateF] = 1.0
        qB[initStateB] = 1.0
        pF[initStateF] = 1.0
        pB[initStateB] = -1.0 # This minus sign allows for backward motion of fictitious oscillator
    elif (stype == 1):
       qF = np.array([ np.random.normal() for i in range(Nstates)]) 
       qB = np.array([ np.random.normal() for i in range(Nstates)]) 
       pF = np.array([ np.random.normal() for i in range(Nstates)]) 
       pB = np.array([ np.random.normal() for i in range(Nstates)]) 
    return qF, qB, pF, pB 

@jit(nopython=True)
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

@jit(nopython=True)
def Force(R, qF, pF, qB, pB):
    dH = model.dHel(R) #dHel(R) # Nxnxn Matrix, N = Nuclear DOF, n = NStates 
    dH0 = model.dHel0(R)

    F = -dH0
    for i in range(len(qF)):
        F -= 0.25 * dH[i,i,:] * ( qF[i] ** 2 + pF[i] ** 2 + qB[i] ** 2 + pB[i] ** 2)
        for j in range(i+1, len(qF)):
            F -= 0.5 * dH[i,j,:] * ( qF[i] * qF[j] + pF[i] * pF[j] + qB[i] * qB[j] + pB[i] * pB[j])

    return F

@jit(nopython=True)
def VelVer(R, P, qF, qB, pF, pB) : # R, P, qF, qB, pF, pB, dtI, dtE, F1, Hij,M=1): # Ionic position, ionic velocity, etc. 
    # data 
    qF, qB, pF, pB = qF * 1.0, qB *  1.0, pF * 1.0, pB * 1.0
    v = P/model.M
    EStep = int(model.dtN/model.dtE)
    dtE = model.dtN/EStep
    
    # half-step mapping
    Hij  = model.Hel(R)
    for t in range(int(np.floor(EStep/2))):
        qF, qB, pF, pB = Umap(qF, qB, pF, pB, dtE, Hij)
    qF, qB, pF, pB = qF * 1, qB * 1, pF * 1, pB * 1 

    # ======= Nuclear Block ==================================
    F1    =  Force(R, qF, pF, qB, pB) # force with {qF(t+dt/2)} * dH(R(t))
    R += v * model.dtN + 0.5 * F1 * model.dtN ** 2 / model.M
    
    F2 = Force(R, qF, pF, qB, pB) # force with {qF(t+dt/2)} * dH(R(t+ dt))
    v += 0.5 * (F1 + F2) * model.dtN / model.M

    P = v * model.M
    # =======================================================
    
    # half-step mapping
    Hij = model.Hel(R) # do QM
    for t in range(int(np.ceil(EStep/2))):
        qF, qB, pF, pB = Umap(qF, qB, pF, pB, dtE, Hij)
    qF, qB, pF, pB = qF, qB, pF, pB 
    
    return R, P, qF, qB, pF, pB 

@jit(nopython=True)
def pop(qF, pF, qB, pB, ρ0):
    return np.outer(qF + 1j * pF, qB-1j*pB) * ρ0

@jit(nopython=True)
def runTraj(iR,iP,iF,iB,NSteps):
    ## Parameters -------------
    NStates = model.NStates
    stype = model.stype
    nskip = model.nskip
    #---------------------------
    if NSteps%nskip == 0:
        pl = 0
    else :
        pl = 1

    RArr = np.zeros((NSteps//nskip + pl, model.NR))
    PArr = np.zeros((NSteps//nskip + pl, model.NR))
    ρ = np.zeros((NSteps//nskip + pl, 2*NStates*NStates+1))        # 2 is to store real and imaginary component seperately
    ρ[:,0] = np.arange(0,(NSteps//nskip + pl)*model.dtN,model.dtN)
    
    # Trajectory data
    R, P = iR, iP

    # set propagator
    vv  = VelVer

    # Call function to initialize mapping variables
    qF, qB, pF, pB = initMapping(NStates, iF, iB, stype) 

    # Set initial values of fictitious oscillator variables for future use
    qF0, qB0, pF0, pB0 = qF[iF], qB[iB], pF[iF], pB[iB] 
    ρ0 = 0.25 * (qF0 - 1j*pF0) * (qB0 + 1j*pB0)

    iskip = 0 # please modify
    for i in range(NSteps): # One trajectory
        #------- ESTIMATORS-------------------------------------
        if (i % nskip == 0):
            Pijt = pop(qF,pF,qB,pB,ρ0)
            PijtReal, PijtImag = Pijt.real, Pijt.imag
            ρ[iskip,1:] += np.hstack((PijtReal.flatten(),PijtImag.flatten()))
            iskip += 1
        #-------------------------------------------------------
        RArr[i,:], PArr[i,:] = R, P
        R, P, qF, qB, pF, pB = vv(R, P, qF, qB, pF, pB)
    
    return ρ, RArr, PArr