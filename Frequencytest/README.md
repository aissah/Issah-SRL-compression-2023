These tests were performed to assess the effect on frequency content introduced by compression with respect to compression rate. This has test parameters hardwired and hence does not require commandline inputs.

If running these codes on another system, you will need to change:

* frequencyTests_hdf5.py: basepath, this is the path to the directory containing the data from FORESEE urban experiment downloaded through globus. Each file contains 10 minutes of data and hence the code processes 3 files (variable count) for 20 minutes.
