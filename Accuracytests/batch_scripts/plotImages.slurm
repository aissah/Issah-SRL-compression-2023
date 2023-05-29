#!/bin/bash 
#SBATCH --job-name="plot images"
#SBATCH --mem=24GB
#SBATCH --time=2:00:00
#SBATCH --export=ALL

module load apps/python3/2020.02

source activate ny_env

# python /u/st/by/aissah/scratch/summer2022exp/Accuracytests/plotOutputInFiles.py
python /u/st/by/aissah/scratch/summer2022exp/Accuracytests/plot_error_norm.py
