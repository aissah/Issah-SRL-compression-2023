#!/bin/bash 
#SBATCH --job-name="Accuracy Tests version 2"
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL

module load apps/python3/2020.02

source activate ny_env

for i in {1..4}
do
# python /u/st/by/aissah/scratch/summer2022exp/Accuracytests/accuracyTests_foresee.py selection_size number_of_groups "$i"(batch number) compresssion_type(wavelet, zfp, svd) checkpoint_flag(1=from beginning, 2=start from checkpoint)
srun --exclusive --ntasks=1 python %~dp0..\accuracyTests_foresee.py 400 4 "$i" svd 1 &
done
wait
echo "end of run"