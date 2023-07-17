These tests were performed to assess the norm of the noise introduced by compression with respect to compression rate. The files are meant to be ran from the command line; they take inputs from command line. This can be changed by editing *argv* inputs in the files with the necessary inputs.

All scripts to run these examples are in batch_scripts/ and can be run like
sbatch/myscript.slurm.

For anyone wishing to run these codes on other computing systems, the locations of 
changes that will need to be made are:

* The python module name in each batch_scripts/*.slurm file
* The srun line's path preceding the .py specification in each bach_scripts/*.slurm file
* sdata_basepath, saveLocation and checkpointFilename in accuracyTests_BH.py
* basepath, saveLocation and checkpointFilename in accuracyTests_foresee.py
* errorsfolder, savefolder in plot_error_norm.py