#!/bin/bash
#SBATCH -p action
#SBATCH --job-name=μxρ0   # create a name for your job
#SBATCH --ntasks=250               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=1G         # memory per cpu-core
#SBATCH -t 1-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=FirstLaser_dimer_40_5e3_80_1e4_3e2.out
#SBATCH --error=FirstLaser_dimer_40_5e3_80_1e4_3e2.err

srun python firstLaserMPI.py /scratch/mmondal/specTest/DimerSimulations/40_5e3_80_1e4_3e2/dimer_40_5e3_80_1e4_3e2/