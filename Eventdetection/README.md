These tests were performed to assess the effect of compression on detectability and location of microsiesmic events. The files are meant to be ran from the command line; they take inputs from command line. This can be changed by editing *argv* inputs in the files with the necessary inputs.

Each test in this subdirectory can be run using the SLURM job submission scripts
in the Batchscripts subdirectory.

If you wish to use these on another computing system or from a different user,
be sure to change:

* you will need to set up the appropriate python environment and specify your python module in each *.slurm file
* data_basepath, save_location in template_matching.py
* cc_location, save_location in peaks_and_detection_significance.py
* peaks_location, save_location in makeplots.py
* peaks_location, save_location in makeplots_clusters.py
