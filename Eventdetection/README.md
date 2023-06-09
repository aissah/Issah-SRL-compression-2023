Each test in this subdirectory can be run using the SLURM job submission scripts
in the Batchscripts subdirectory. 

If you wish to use these on another computing system or from a different user,
be sure to change:
* the path to the *.py scripts referred to by each *.slurm file
* you will need to set up the appropriate python environment and specify your python module in each *.slurm file
* sys.path.import(), data_basepath, save_location, file in template_matching.py
* cc_location, save_location in peaks_and_detection_significance.py
* peaks_location, save_location in makeplots.py
* peaks_location, save_location in makeplots_clusters.py
