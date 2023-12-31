#!/bin/bash
#SBATCH -p action
#SBATCH --job-name=ft2d
#SBATCH --ntasks=100               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=1G 
#SBATCH -t 1-00:00:00
#SBATCH --output=ft2d.out
#SBATCH --error=ft2d.err

srun python FT2Dt2.py