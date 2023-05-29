"""
Get the norm of noise at various levels of compression for multiple files.

The output is a file containing the norm of noise at various levels of compression
for each file. To make sure this output fit in memory, we parallelize by batching
the files outside this module

The setup of the files here is just one directory containing all the files

Created on Tue Aug  9 10:58:46 2022

@author: issah
"""
import sys

sys.path.insert(0, "/u/st/by/aissah/scratch/summer2022exp/Accuracytests")
import os
import pickle

import ATFuncs
import numpy as np

basepath = "/beegfs/projects/martin/foresee/apr10_20"  # directory containing data
# files
files = os.listdir(basepath)
selectionsize = int(sys.argv[1])
groups = int(sys.argv[2])
batch = int(sys.argv[3])
compression_type = sys.argv[4]  # "wavelet"
flag = int(sys.argv[5])
numberoffiles = len(files)
batch_size = np.ceil(numberoffiles / groups)
end = int(batch_size * batch)
start = int(end - batch_size)
if end > numberoffiles:
    end = numberoffiles
# selection=random.sample(range(start,end),int(selectionsize/groups))
selection = range(start, end)


saveLocation = "/u/st/by/aissah/scratch/foreseeDataResults/accuracyTests/"
checkpointFilename = (
    "/u/st/by/aissah/scratch/summer2022exp/Accuracytests/checkpoints/checkpoint_"
    + compression_type
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
        fname = os.path.join(basepath, files[b])

        data = ATFuncs.loadFORESEEhdf5(fname, normalize="no")

        if len(data) % 2 == 1:
            data = data[:-1]

        if compression_type == "wavelet":
            error1d, _ = ATFuncs.accracyTest_wavelet(
                data, mode="1d", threshold_percentiles=thresholds
            )
            error2d, _ = ATFuncs.accracyTest_wavelet(
                data, mode="2d", threshold_percentiles=thresholds
            )
            if flag == 1:
                errors_1dw = np.array([error1d])
                errors_2dw = np.array([error2d])
                flag = 0
            else:
                errors_1dw = np.append(errors_1dw, np.array([error1d]), axis=0)
                errors_2dw = np.append(errors_2dw, np.array([error2d]), axis=0)
        elif compression_type == "zfp":
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
        elif compression_type == "svd":
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
