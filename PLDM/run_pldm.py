import numpy as np
import coupled_dimer as model
import pldm as method

NSteps = model.parameters.NSteps
NTraj = model.parameters.NTraj
NStates = model.parameters.NStates

par = model.parameters() 
par.dHel = model.dHel
par.dHel0 = model.dHel0
par.initR = model.initR
par.Hel   = model.Hel
par.stype = "focused"
fState = model.parameters.initStateF
bState = model.parameters.initStateB

rho_sum  = method.runTraj(par)
PijFile = open("Pij_11_1000fs_1000.txt","w")

for t in range(rho_sum.shape[-1]):
    PijFile.write(f"{t * model.parameters.nskip * model.parameters.dtN} \t")
    # for i in range(NStates):
    PijFile.write(str(rho_sum[fState,bState,t].real / ( NTraj ) ) + "\t")
    PijFile.write(str(rho_sum[fState,bState,t].imag / ( NTraj ) ) + "\t")
    PijFile.write(str(rho_sum[fState+1,bState+1,t].real / ( NTraj ) ) + "\t")
    PijFile.write(str(rho_sum[fState+1,bState+1,t].imag / ( NTraj ) ) + "\t")
    PijFile.write("\n")
PijFile.close()