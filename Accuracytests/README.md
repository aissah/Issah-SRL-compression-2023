All scripts to run these examples are in batch_scripts/ and can be run like
sbatch/myscript.slurm.

For anyone wishing to run these codes on other computing systems, the locations of 
changes that will need to be made are:

* The python module name in each batch_scripts/*.slurm file
* The srun line's path preceding the .py specification in each bach_scripts/*.slurm file
* sys.path.insert(), data_basepath, saveLocation and checkpointFilename in accuracyTests_BH.py
* sys.path.insert(), basepath, saveLocation and checkpointFilename in accuracyTests_foresee.py
* errorsfolder, savefolder in plot_error_norm.py