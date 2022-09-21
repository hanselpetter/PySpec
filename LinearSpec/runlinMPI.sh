#!/bin/bash
#SBATCH -p action
#SBATCH --job-name=1Dμ   # create a name for your job
#SBATCH --ntasks=100               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=1G         # memory per cpu-core
#SBATCH -t 5-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=LS.out
#SBATCH --error=LS.err

srun python linSpecMPI.py /scratch/mmondal/specTest/DimerSimulations/Dimer_In_Cavity/asymmetricDimer/5_40_1e4_5e2/dimer/