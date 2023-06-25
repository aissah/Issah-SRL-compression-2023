#!/bin/bash 
#SBATCH --job-name="wl Accuracy Tests"
#SBATCH --mem=24GB
#SBATCH --time=45:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL

module load apps/python3/2020.02

source activate ny_env

for i in 2 4
do
# python /u/st/by/aissah/scratch/summer2022exp/Accuracytests/accuracyTests_foresee.py selection_size number_of_groups "$i"(batch number) compresssion_type(wavelet, zfp, svd) checkpoint_flag(1=from beginning, 2=start from checkpoint)
srun --exclusive --ntasks=1 python /u/st/by/aissah/scratch/summer2022exp/Accuracytests/accuracyTests_foresee.py 400 4 "$i" wavelet 1 &
done
wait
echo "end of run"