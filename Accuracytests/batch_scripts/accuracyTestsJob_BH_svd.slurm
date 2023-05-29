#!/bin/bash 
#SBATCH --job-name="svd BH Accuracy Tests"
#SBATCH --mem=12GB
#SBATCH --time=45:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL

module load apps/python3/2020.02

source activate ny_env

for i in {1..4}
do
# python /u/st/by/aissah/scratch/summer2022exp/Accuracytests/accuracyTests_forsee.py selection_size number_of_groups "$i"(batch number) compresssion_type(wavelet, zfp, svd) checkpoint_flag(1=from beginning, 2=start from checkpoint)
srun --exclusive --ntasks=1 python /u/st/by/aissah/scratch/summer2022exp/Accuracytests/accuracyTests_BH.py 2400 4 "$i" svd 1 &
done
wait
echo "end of run"