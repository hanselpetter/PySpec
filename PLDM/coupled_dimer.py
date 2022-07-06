import numpy as np
from numpy import random as rd

fs2au = 41.341374575751
cminv2au = 4.55633*1e-6
eV2au = 0.036749405469679
K2au = 0.00000316678

'''dtN -> nuclear time step (au)
    Nsteps -> number of nuclear time steps
    Total simulation time = Nsteps x dtN (au)'''
totalSim = 1000             # in fs
dtN = 5.0                
NSteps = int(totalSim/(dtN/fs2au)) + 1  

'''Esteps -> number of electronic time steps per nuclear time step
    dtE -> electronic time step (au)'''        
ESteps = 20              
dtE = dtN/ESteps 
    
NStates = 4                 # number of electronic states
M = 1                       # mass of nuclear particles (au)
NTraj = 1000                 # number of trajectories
nskip = 1                   # save data every nskip steps

stype = "focused"
initStateF = 1
initStateB = 1

'''Bath parameters'''
cj = np.loadtxt("/Users/quantised_elious/Desktop/Project1/codes/Semiclassical_methods/SemiClassical-NAMD/bath_parameters/cj_50.txt")
ωj = np.loadtxt("/Users/quantised_elious/Desktop/Project1/codes/Semiclassical_methods/SemiClassical-NAMD/bath_parameters/ωj_50.txt")
NModes = len(cj)           # Number of bath modes per site
NR = 2*NModes              # Total number of bath DOF 


def Hel(R):
    '''Electronic diabatic Hamiltonian'''

    Nstates = parameters.NStates
    cj = parameters.cj
    NModes = len(cj)
    Vij = np.zeros((Nstates,Nstates))

    ε = (np.array([-50,50])+10000)*cminv2au
    J12 = 100*cminv2au                      # electronic coupling between 10 and 01
    SB1 = np.sum(cj[:]*R[:NModes])             # system bath coupling for site 1
    SB2 = np.sum(cj[:]*R[NModes:])     # system bath coupling for site 2

    Vij[1,1] = ε[1] + SB2
    Vij[2,2] = ε[0] + SB1
    Vij[3,3] = np.sum(ε) + SB1 + SB2
    Vij[1,2], Vij[2,1] = J12, J12 

    return Vij

def dHel0(R):
    '''bath derivative of the state independent part of the Hamiltonian'''

    ωj = parameters.ωj
    return (np.hstack((ωj,ωj))**2)*R

def dHel(R):
    '''bath derivative of the state dependent part of Hamiltonian'''

    NStates = parameters.NStates
    cj = parameters.cj
    NModes = len(cj)
    dVij = np.zeros((NStates,NStates,2*NModes))
    dVij[1,1,NModes:] = cj[:]
    dVij[2,2,:NModes] = cj[:]
    dVij[3,3,:] = np.hstack((cj,cj))[:]

    return dVij

def initR():
    '''Sampling the initial position and velocities of bath parameters from 
       wigner distribution'''

    Kb = 8.617333262*1e-5*eV2au/K2au        # Kb = 8.617333262 x 10^-5 eV/K
    T = 300*K2au                            # Temperature in au
    β = 1/(Kb*T)
    ωj = parameters.ωj
    NModes = len(ωj)

    σR_wigner = 1/np.sqrt(2*ωj*np.tanh(β*ωj*0.5))
    σP_wigner = np.sqrt(ωj/(2*np.tanh(β*ωj*0.5)))
    μR_wigner = 0
    μP_wigner = 0

    R = np.zeros(2*NModes)
    P = np.zeros(2*NModes)
    for dof in range(NModes):
        R[dof],R[NModes+dof] = rd.normal(μR_wigner, σR_wigner[dof]), rd.normal(μR_wigner, σR_wigner[dof])
        P[dof],P[NModes+dof] = rd.normal(μP_wigner, σP_wigner[dof]), rd.normal(μP_wigner, σP_wigner[dof])

    return R, P