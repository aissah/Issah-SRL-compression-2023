#!/bin/bash
#SBATCH --job-name="Template Matching"
#SBATCH --time=3:00:00
#SBATCH --ntasks=19
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL

module load apps/python3/2020.02

source activate ny_env

for i in {1..19}
do
# python /u/st/by/aissah/bins/summer2022exp/Eventdetection/template_matching.py event_id first_channel last_channel batch batch_size compression_flag compression_type compression_function compression_levels
srun --exclusive --ntasks=1 python %~dp0../template_matching.py 2201051 1000 5000 "$i" 300 0 &
done
wait
echo "end of run"


