#!/bin/bash
#SBATCH -p action
#SBATCH --job-name=1Dμ   # create a name for your job
#SBATCH --ntasks=200               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=1G         # memory per cpu-core
#SBATCH -t 5-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=spec_20_20_5e4_5e2.out
#SBATCH --error=spec_20_20_5e4_5e2.err

srun python linSpecMPI.py /scratch/mmondal/specTest/DimerSimulations/timeScaleAnalysis/20au/20_20_5e4_5e2/dimer_20_20_5e4_5e2/