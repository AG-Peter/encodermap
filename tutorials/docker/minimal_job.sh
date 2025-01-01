#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=res.txt
#SBATCH --partition=debug
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100

srun hostname
srun pwd
srun sleep 2
