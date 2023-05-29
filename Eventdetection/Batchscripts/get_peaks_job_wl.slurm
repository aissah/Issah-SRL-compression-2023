#!/bin/bash
#SBATCH --job-name="Get Peaks"
#SBATCH --time=10:00:00
#SBATCH --export=ALL

module load apps/python3/2020.02

source activate ny_env

# python /u/st/by/aissah/bins/summer2022exp/Eventdetection/peaks_and_detection_significance.py event_id compression_type detection_significance_threshold

#python /u/st/by/aissah/scratch/summer2022exp/Eventdetection/peaks_and_detection_significance.py 2201050 wl 9

python /u/st/by/aissah/scratch/summer2022exp/Eventdetection/makeplots.py 2201050 wl

python /u/st/by/aissah/scratch/summer2022exp/Eventdetection/makeplots_clusters.py 2201050 wl