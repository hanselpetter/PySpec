#!/bin/bash
#SBATCH -p exciton -A exciton
#SBATCH --job-name=1DpartialFourier
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 1:00:00
#SBATCH --output=test_%A_%a.out
#SBATCH --error=test_%A_%a.err

python partialFourier.py