#!/bin/bash
#SBATCH --job-name="svd Template Matching"
#SBATCH --mem=12GB
#SBATCH --ntasks=19
#SBATCH --export=ALL

module load apps/python3/2020.02

source activate ny_env

for i in 8 14 19 # {1..19}
do
# python /u/st/by/aissah/bins/summer2022exp/Eventdetection/template_matching.py event_id first_channel last_channel batch batch_size compression_flag compression_type compression_function compression_levels
srun --exclusive --ntasks=1 python %~dp0../template_matching.py 2201050 1000 5000 "$i" 300 1 svd "general_funcs.randomized_SVD_comp_decomp(data, compression_factor=compression_level)" 5,10,20,50 &
done
wait
echo "end of run"


