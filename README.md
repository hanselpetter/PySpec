# PySpec

PySpec is a python based code for simulating linear and multi-dimensional spectroscopy for a given Diabatic 
Hamiltonian. The dynamics for the quantum subsystem under a given bath is done currently being done by 
focused-PLDM. 

### Features
- Simple PLDM population and operator dynamics (serial and parallel)
- Linear Absorption (serial and parallel)
- 2D spectra (serial and parallel)

### simple PLDM calculation steps
- Create a model (Hamiltonian) file in PLDM folder with all parameters
- Specify a bath in Bath/initBathParams.py and generate bath parameters
- run Bath/initBath.py to generate Trajectory folder containing seperate folder for each trajectory
- run PLDM/serial.py

### Linear absorption calculation steps
- Create a model (Hamiltonian) file in PLDM folder with all parameters
- Specify a bath in Bath/initBathParams.py and generate bath parameters
- run Bath/initBath.py to generate Trajectory folder containing seperate folder for each trajectory
- For serial calculation, run LinearSpec/serialLinSpec.py
- For parallel calculation, run LinearSpec/multiparLinear.py



