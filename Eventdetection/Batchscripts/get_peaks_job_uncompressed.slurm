#!/bin/bash
#SBATCH --job-name="Get Peaks"
#SBATCH --time=00:05:00
#SBATCH --export=ALL

module load apps/python3/2020.02

source activate ny_env

# python /u/st/by/aissah/bins/summer2022exp/Eventdetection/peaks_and_detection_significance.py event_id compression_type detection_significance_threshold

python %~dp0../peaks_and_detection_significance.py 2201051 N/A 9

python %~dp0../makeplots.py 2201051 N/A