import numpy as np
from numba import jit

fs2au = 41.341374575751
cminv2au = 4.55633*1e-6
eV2au = 0.036749405469679
K2au = 0.00000316678

'''dtN -> nuclear time step (au)
    Nsteps -> number of nuclear time steps
    Total simulation time = Nsteps x dtN (au)'''
Sim1Time = 300             # in fs
Sim2Time = 0.5             # in fs
Sim3Time = 300             # in fs
dtN = 40.0                
NSteps1 = int(Sim1Time/(dtN/fs2au)) + 1  
NSteps2 = int(Sim2Time/(dtN/fs2au)) + 1
NSteps3 = int(Sim3Time/(dtN/fs2au)) + 1

'''Esteps -> number of electronic time steps per nuclear time step
    dtE -> electronic time step (au)'''        
ESteps = 120              
dtE = dtN/ESteps 
    
NStates = 4                 # number of electronic states
M = 1                       # mass of nuclear particles (au)
NTraj = 10000                 # number of trajectories
nskip = 1                   # save data every nskip steps of PLDM simulation

# skip steps for saving data of the response function
nskip1 = 1                 
nskip2 = 5000
nskip3 = 1

stype = 0                   # Focused = 0, sampled = 1
initStateF = 1
initStateB = 0

'''Bath parameters'''
cj = np.loadtxt("/scratch/mmondal/specTest/Bath/cj_50.txt")
ωj = np.loadtxt("/scratch/mmondal/specTest/Bath/ωj_50.txt")
NModes = len(cj)           # Number of bath modes per site
NR = 2*NModes              # Total number of bath DOF 

@jit(nopython=True)
def Hel(R):
    '''Electronic diabatic Hamiltonian'''
    Vij = np.zeros((NStates,NStates))

    ε = (np.array([-50,50])+10000)*cminv2au
    J12 = 100*cminv2au                      # electronic coupling between 10 and 01
    SB1 = np.sum(cj[:]*R[:NModes])             # system bath coupling for site 1
    SB2 = np.sum(cj[:]*R[NModes:])     # system bath coupling for site 2

    Vij[1,1] = ε[1] + SB2
    Vij[2,2] = ε[0] + SB1
    Vij[3,3] = np.sum(ε) + SB1 + SB2
    Vij[1,2], Vij[2,1] = J12, J12 

    return Vij

@jit(nopython=True)
def dHel0(R):
    '''bath derivative of the state independent part of the Hamiltonian'''
    return (np.hstack((ωj,ωj))**2)*R

@jit(nopython=True)
def dHel(R):
    '''bath derivative of the state dependent part of Hamiltonian'''
    dVij = np.zeros((NStates,NStates,2*NModes))
    dVij[1,1,NModes:] = cj[:]
    dVij[2,2,:NModes] = cj[:]
    dVij[3,3,:] = np.hstack((cj,cj))[:]

    return dVij

@jit(nopython=True)
def μ():
    "Dipole operator matrix"
    μmat = np.zeros((NStates,NStates),dtype=np.complex128)
    μmat[0,1], μmat[0,2], μmat[3,1], μmat[3,2] = -1, 0.2, 0.2, -1
    # μmat[0,1], μmat[0,2] = 5, -1

    return μmat + μmat.T 

@jit(nopython=True)
def ρ0():
    "Initial density matrix"
    ρ0mat = np.zeros((NStates,NStates),dtype=np.complex128)
    ρ0mat[0,0] = 1

    return ρ0mat 