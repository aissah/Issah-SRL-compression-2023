#!/bin/bash
#SBATCH --job-name="zfp Template Matching"
#SBATCH --mem=9GB
#SBATCH --time=30:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL

module load apps/python3/2020.02

source activate ny_env

for i in 1 19 # {15..17}
do
# python /u/st/by/aissah/bins/summer2022exp/Eventdetection/template_matching.py event_id first_channel last_channel batch batch_size compression_flag compression_type compression_function compression_levels
srun --exclusive --ntasks=1 python /u/st/by/aissah/scratch/summer2022exp/Eventdetection/template_matching.py 2201050 1000 5000 "$i" 300 1 zfp "general_funcs.compressReconstruct_zfp(data,mode=\"precision\", precision=compression_level)" 2,3,4,5 &
done
wait
echo "end of run"


