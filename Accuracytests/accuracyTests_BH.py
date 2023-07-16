"""
Get the norm of noise at various levels of compression for multiple files.

The output is a file containing the norm of noise at various levels of compression
for each file. To make sure this output fit in memory, we parallelize by batching
the files outside this module

The setup of the files here is a directory containing other directories that contain
the files

Created on Tue Nov  8 11:53:24 2022

@author: issah
"""
import os
import pickle
import random
import sys

import numpy as np

# import random

# sys.path.insert(0, "/u/st/by/aissah/scratch/summer2022exp/Accuracytests")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
    # This line does not work when ran in an interactive IDE. Instead, make sure the
    # directory containing "Functions" folder is in the python path
except NameError:
    pass

from Functions import ATFuncs, general_funcs

selectionsize = int(sys.argv[1])  # number of files for this batch to analyze
groups = int(sys.argv[2])  # number of groups
batch = int(
    sys.argv[3]
)  # batch ID (e.g. if 4 tasks, job sub. script sets this to 1, 2, 3, or 4)
compression_type = sys.argv[4]  # "wavelet", "zfp", "svd"
flag = int(sys.argv[5])  # 1=from beginning, 0=start from checkpoint

data_basepath = "/beegfs/projects/martin/BradyHotspring"  # directory containing data
# make list of all files in directory
data_files = []
for dir_path, dir_names, file_names in os.walk(data_basepath):
    dir_names.sort()
    file_names.sort()
    data_files.extend(
        [
            os.path.join(dir_path, file_name)
            for file_name in file_names
            if ".h5" in file_name
        ]
    )
numberoffiles = len(data_files)
batch_size = np.ceil(numberoffiles / groups)
end = int(batch_size * batch)
start = int(end - batch_size)
if end > numberoffiles:
    end = numberoffiles
# select random subset of file indices for this batch to analyze
selection = random.sample(range(start, end), int(selectionsize / groups))
# selection = range(start, end)

saveLocation = "/u/st/by/aissah/scratch/BradyHotspringResults/accuracyTests/"
checkpointFilename = (
    "/u/st/by/aissah/scratch/summer2022exp/Accuracytests/checkpoints/checkpoint"
    + str(batch)
    + ".pkl"
)

if flag == 0:
    with open(checkpointFilename, "rb") as f:
        checkpoint = pickle.load(f)
    flag = 1
else:
    checkpoint = selection[0]
count = selection[0]


thresholds = list(range(5, 95, 5)) + list(range(92, 100, 2))
compression_factors_wl = [100 / (100 - a) for a in thresholds]
comp_factors_svd = list(range(5, 51, 5))

for b in selection:
    print(b, flush=True)
    count += 1
    if count > checkpoint:
        fname = os.path.join(data_basepath, data_files[b])
        # chosenfiles.append(files[b])
        data, _ = general_funcs.loadBradyHShdf5(fname, normalize="no")

        if len(data) % 2 == 1:
            data = data[:-1]

        if compression_type == "wavelet":  # run 1D then 2D wavelet tests
            error1d, _ = ATFuncs.accuracyTest_wavelet(
                data, mode="1d", threshold_percentiles=thresholds
            )
            error2d, _ = ATFuncs.accuracyTest_wavelet(
                data, mode="2d", threshold_percentiles=thresholds
            )
            if flag == 1:
                errors_1dw = np.array([error1d])
                errors_2dw = np.array([error2d])
                flag = 0
            else:
                errors_1dw = np.append(errors_1dw, np.array([error1d]), axis=0)
                errors_2dw = np.append(errors_2dw, np.array([error2d]), axis=0)
        elif compression_type == "zfp":  # run zfp tests with 3 types of accuracy
            errort, compressionfactort = ATFuncs.accuracyTest_zfp(
                data, mode="tolerance"
            )
            errorp, compressionfactorp = ATFuncs.accuracyTest_zfp(
                data, mode="precision"
            )
            errorb, compressionfactorb = ATFuncs.accuracyTest_zfp(data, mode="bitrate")
            if flag == 1:
                errors_tolerance = np.array([errort])
                errors_precision = np.array([errorp])
                errors_bitrate = np.array([errorb])
                compressionfactors_tolerance = np.array([compressionfactort])
                compressionfactors_precision = np.array([compressionfactorp])
                compressionfactors_bitrate = np.array([compressionfactorb])
                flag = 0
            else:
                errors_tolerance = np.append(
                    errors_tolerance, np.array([errort]), axis=0
                )
                errors_precision = np.append(
                    errors_precision, np.array([errorp]), axis=0
                )
                errors_bitrate = np.append(errors_bitrate, np.array([errorb]), axis=0)
                compressionfactors_tolerance = np.append(
                    compressionfactors_tolerance, np.array([compressionfactort]), axis=0
                )
                compressionfactors_precision = np.append(
                    compressionfactors_precision, np.array([compressionfactorp]), axis=0
                )
                compressionfactors_bitrate = np.append(
                    compressionfactors_bitrate, np.array([compressionfactorb]), axis=0
                )
        elif compression_type == "svd":  # run SVD tests (with randomized SVD)
            errorSVD = ATFuncs.normalised_errors_SVD(
                data, comp_factors_svd, mode="randomized"
            )
            if flag == 1:
                errorsSVD = np.array([errorSVD])
                flag = 0
            else:
                errorsSVD = np.append(errorsSVD, np.array([errorSVD]), axis=0)
        else:
            raise Exception("Unrecognised compression type")

        # record results every 200 files
        # with errors recorded as arrays according to the compression types' options
        if (count - start) % 200 == 0:
            errorsFilename = (
                saveLocation
                + "errors/"
                + compression_type
                + "_error_norm_files"
                + str(checkpoint + 1)
                + "to"
                + str(b + 1)
                + ".pkl"
            )

            checkpoint = b + 1

            with open(checkpointFilename, "wb") as f:
                pickle.dump(checkpoint, f)

            if compression_type == "wavelet":
                with open(errorsFilename, "wb") as f:
                    pickle.dump(
                        [
                            errors_1dw,
                            errors_2dw,
                            compression_factors_wl,
                        ],
                        f,
                    )
            elif compression_type == "zfp":
                with open(errorsFilename, "wb") as f:
                    pickle.dump(
                        [
                            errors_tolerance,
                            errors_precision,
                            errors_bitrate,
                            compressionfactors_tolerance,
                            compressionfactors_precision,
                            compressionfactors_bitrate,
                        ],
                        f,
                    )
            elif compression_type == "svd":
                with open(errorsFilename, "wb") as f:
                    pickle.dump(
                        [
                            errorsSVD,
                            comp_factors_svd,
                        ],
                        f,
                    )
            else:
                raise Exception("Unrecognised compression type")

            flag = 1

if count != checkpoint:
    errorsFilename = (
        saveLocation
        + "errors/"
        + compression_type
        + "_error_norm_files"
        + str(checkpoint + 1)
        + "to"
        + str(b + 1)
        + ".pkl"
    )

    if compression_type == "wavelet":
        with open(errorsFilename, "wb") as f:
            pickle.dump(
                [
                    errors_1dw,
                    errors_2dw,
                    compression_factors_wl,
                ],
                f,
            )
    elif compression_type == "zfp":
        with open(errorsFilename, "wb") as f:
            pickle.dump(
                [
                    errors_tolerance,
                    errors_precision,
                    errors_bitrate,
                    compressionfactors_tolerance,
                    compressionfactors_precision,
                    compressionfactors_bitrate,
                ],
                f,
            )
    elif compression_type == "svd":
        with open(errorsFilename, "wb") as f:
            pickle.dump(
                [
                    errorsSVD,
                    comp_factors_svd,
                ],
                f,
            )
    else:
        raise Exception("Unrecognised compression type")

print("End of run")
os.remove(checkpointFilename)
