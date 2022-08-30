#!/bin/bash
#SBATCH -p action
#SBATCH --job-name=ft2d
#SBATCH --ntasks=100               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=2G 
#SBATCH -t 1-00:00:00
#SBATCH --output=ft2d_dimer_40_5e3_80_1e4_3e2.out
#SBATCH --error=ft2d_dimer_40_5e3_80_1e4_3e2.err

srun python FT2DV2.py