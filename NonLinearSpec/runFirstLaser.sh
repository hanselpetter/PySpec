#!/bin/bash
#SBATCH -p action
#SBATCH --job-name=FLMPIdim   # create a name for your job
#SBATCH --ntasks=250               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=1G         # memory per cpu-core
#SBATCH -t 1-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=FirstLaser.out
#SBATCH --error=FirstLaser.err

srun python firstLaserMPI.py /scratch/mmondal/specTest/DimerSimulations/v2sims/40_120_4e4_3e2/dimer_40_120_4e4_3e2/