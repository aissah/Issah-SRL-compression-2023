# Issah-SRL-compression-2023

## Overview

This repo contains the codes for the tests and visualizations made for this paper intended for submission to SRL in 2023. There are individual directories for the different classes of tests done and can be ran independent of each other.

Packages used:

- python 3.10.9
- h5py 3.7.0
- pywavelets 1.4.1
- zfpy 0.5.5

## Accuracytests

These tests were performed to assess the norm of the noise introduced by compression with respect to compression rate. The files are meant to be ran from the command line; they take inputs from command line. This can be changed by editing *argv* inputs in the files with the necessary inputs. The *batch_scripts* directory has the scripts used to ran tests for the paper.

### accuracyTests_BH.py

Tests ran using the Brady's Hotspring data.

- ***Code dependencies***: *Functions/ATFuncs.py*, *Functions/general_funcs.py*
- ***Required data***: Brady's Hotspring data (can be downloaded with instructions in Datadownload directory).
- ***Output***: Pickle file with arrays for compression ratios and errors produced. Depending the compression type, there may be multple arrays for error and/or compression ratios. These are handled in the code provided the right input is used.
- ***Code adjustments before running***: *data_basepath*, *saveLocation*, and *checkpointFilename* variables need to be changed. These repesent the location of the data, location to save the output pickle files and location to save checkpointing files respectively.

### accuracyTests_foresee.py

Tests ran using data from the FORESSEE urban experiment.

- ***Code dependencies***: *Functions/ATFuncs.py*, *Functions/general_funcs.py*
- ***Required data***: FORESEE urban experiment data (can be downloaded with instructions in Datadownload directory).
- ***Output***: Pickle file with arrays for compression ratios and errors produced. Depending the compression type, there may be multple arrays for error and/or compression ratios. These are handled in the code provided the right input I used.
- ***Code adjustments before running***: *data_basepath*, *saveLocation*, and *checkpointFilename* variables need to be changed. These repesent the location of the data, location to save the output pickle files and location to save checkpointing files respectively.

### plot_error_norm.py

Plot results of tests.

- ***Code dependencies***: *Functions/ATFuncs.py*
- ***Required data***: Output files from *accuracyTests_foresee.py* and/or *accuracyTests_BH.py*.
- ***Output***: Various images showing the relationship between compression rates and errors.
- ***Code adjustments before running***: *errorsfolder* and *savefolder* variables need to be changed. These repesent the location of the output from any of the 2 other files and location to save the images created here.

## Eventdetection

These tests were performed to assess the effect of compression on detectability and location of microsiesmic events. The files are meant to be ran from the command line; they take inputs from command line. This can be changed by editing *argv* inputs in the files with the necessary inputs. The *batch_scripts* directory has the scripts used to ran tests with parameters for the paper.

### template_matching.py

- ***Code dependencies***: *Functions/eventDTFuncs.py*, *Functions/general_funcs.py*
- ***Required data***: Brady's Hotspring data (can be downloaded with instructions in Datadownload directory).
- ***Output***: Pickle file which contains array for normalised cross correlation of between template and all specified files and dictionary of metadata.
- ***Code adjustments before running***: *data_basepath* and *save_location* variables need to be changed. These repesent the location of the data and location to save output.

### peaks_and_detection_significance.py

Get events from results of normalised cross-correlation.

- ***Code dependencies***: *Functions/eventDTFuncs.py*
- ***Required data***: output from *template_matching.py*.
- ***Output***: Pickle file with a ditionary for amplitudes of events detected, and other dictionaries with location of event in samples and time, detection significance, and compression rates at which these were computed. These are handled in the code provided the right input is used.
- ***Code adjustments before running***: *cc_location* and *save_location* variables need to be changed. These repesent the location of the output from *template_matching.py* and location to save the output pickle files for this file.

### makeplots.py and makeplots_clusters.py

Make plots with events detected in *peaks_and_detection_significance.py*

- ***Code dependencies***: -
- ***Required data***: Output files from *peaks_and_detection_significance.py*.
- ***Output***: Various images showing the relationship between compression rates and errors.
- ***Code adjustments before running***: *peaks_location* and *save_location* variables need to be changed. These repesent the location of the output from *peaks_and_detection_significance.py* and location to save the images created here.

### Jupyter notebooks

These show plots of 3 events to get a closer look at the errors present.

- ***Code dependencies***: *Functions/eventDTFuncs.py*, *Functions/general_funcs.py*
- ***Required data***: The files used (PoroTomo_iDAS16043_160314083848.h5 and PoroTomo_iDAS16043_160314083918.h5) from the Brady's Hotspring data.
- ***Output***: Images showing various errors.
- ***Code adjustments before running***: *file* should point to the 2 files required

## Frequencytests

These tests were performed to assess the effect on frequency content introduced by compression with respect to compression rate. This has test parameters hardwired and hence does not require commandline inputs.

### frequencyTests_hdf5.py

- ***Code dependencies***: *Functions/FTFuncs.py*, *Functions/general_funcs.py*
- ***Required data***: FORESEE urban experiment data (can be downloaded with instructions in Datadownload directory).
- ***Output***: Pickle file with arrays and various images used in the paper. 
- ***Code adjustments before running***: *basepath* which repesents the location of the data.
