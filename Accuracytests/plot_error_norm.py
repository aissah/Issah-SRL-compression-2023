"""
Make plots for outputs of accuracyTests_BH.py and accuracyTests_foresee.py

Created on Tue Sep 27 09:36:12 2022
@author: issah
"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
    # This line does not work when ran in an interactive IDE. Instead, make sure the
    # directory containing "Functions" folder is in the python path
except NameError:
    pass
from Functions import ATFuncs

thresholds = list(range(5, 95, 5)) + list(range(92, 100, 2))

# folder containing the outputs
errorsfolder = "/u/st/by/aissah/scratch/foreseeDataResults/accuracyTests/errors/"
savefolder = "/u/st/by/aissah/scratch/foreseeDataResults/accuracyTests/images/"
errorsfiles = os.listdir(errorsfolder)
# indices of files to consider
# filestoload=list(range(10))
filestoload = list(range(len(errorsfiles)))


# load the rest of the files and append them to the first loaded file
# for a in filestoload[1:]:
for a in filestoload:
    errorsfile = os.path.join(errorsfolder, errorsfiles[filestoload[a]])
    if "errors" in errorsfile and "wavelet" in errorsfile:
        with open(errorsfile, "rb") as f:
            (
                errors_1dw1,
                errors_2dw1,
                compFactorsWavelets,
            ) = pickle.load(f)
            # e, n, te, rs_1dw,rs_2dw = pickle.load(f)
            try:
                errors_1dw = np.append(errors_1dw, errors_1dw1, axis=0)
                errors_2dw = np.append(errors_2dw, errors_2dw1, axis=0)
            except:
                errors_1dw = errors_1dw1
                errors_2dw = errors_2dw1
    elif "errors" in errorsfile and "zfp" in errorsfile:
        with open(errorsfile, "rb") as f:
            (
                errors_tolerance1,
                errors_precision1,
                errors_bitrate1,
                compressionfactors_tolerance1,
                compressionfactors_precision1,
                compressionfactors_bitrate1,
            ) = pickle.load(f)
            # e, n, te, rs_1dw,rs_2dw = pickle.load(f)
            try:
                errors_tolerance = np.append(
                    errors_tolerance, errors_tolerance1, axis=0
                )
                errors_precision = np.append(
                    errors_precision, errors_precision1, axis=0
                )
                errors_bitrate = np.append(errors_bitrate, errors_bitrate1, axis=0)
                compressionfactors_tolerance = np.append(
                    compressionfactors_tolerance, compressionfactors_tolerance1, axis=0
                )
                compressionfactors_precision = np.append(
                    compressionfactors_precision, compressionfactors_precision1, axis=0
                )
                compressionfactors_bitrate = np.append(
                    compressionfactors_bitrate, compressionfactors_bitrate1, axis=0
                )
            except:
                errors_tolerance = errors_tolerance1
                errors_precision = errors_precision1
                errors_bitrate = errors_bitrate1
                compressionfactors_tolerance = compressionfactors_tolerance1
                compressionfactors_precision = compressionfactors_precision1
                compressionfactors_bitrate = compressionfactors_bitrate1
    elif "errors" in errorsfile and "svd" in errorsfile:
        try:
            with open(errorsfile, "rb") as f:
                (
                    errors_tolerance1,
                    errors_precision1,
                    errors_bitrate1,
                    errors_1dw1,
                    errors_2dw1,
                    errors_1dw,
                    errors_2dw,
                    compression_factors_wl,
                ) = pickle.load(f)
                # e, n, te, rs_1dw,rs_2dw = pickle.load(f)
                try:
                    errors_tolerance = np.append(
                        errors_tolerance, errors_tolerance1, axis=0
                    )
                    errors_precision = np.append(
                        errors_precision, errors_precision1, axis=0
                    )
                    errors_bitrate = np.append(errors_bitrate, errors_bitrate1, axis=0)
                    errors_1dw = np.append(errors_1dw, errors_1dw1, axis=0)
                    errors_2dw = np.append(errors_2dw, errors_2dw1, axis=0)
                except:
                    errors_tolerance = errors_tolerance1
                    errors_precision = errors_precision1
                    errors_bitrate = errors_bitrate1
                    errors_1dw = errors_1dw1
                    errors_2dw = errors_2dw1
        except:
            with open(errorsfile, "rb") as f:
                (errorsSVD1, compFactorsSVD1) = pickle.load(f)
                try:
                    errorsSVD = np.append(errorsSVD, errorsSVD1, axis=0)
                    compFactorsSVD = np.append(compFactorsSVD, compFactorsSVD1, axis=0)
                except:
                    errorsSVD = errorsSVD1
                    compFactorsSVD = compFactorsSVD1


fig = ATFuncs.plotfill_stats(
    errors_tolerance,
    middle="median",
    x_data=np.mean(compressionfactors_tolerance, axis=0),
)
fsize = 15
plt.xlabel("Compression Factor", fontsize=fsize)
plt.ylabel("Normalised L2 Error", fontsize=fsize)
plt.title(
    "Error against Compression factor for zfp with fixed accuracy parameter",
    fontsize=fsize,
)
plt.yticks(fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.legend(fontsize=fsize)
plt.savefig(
    savefolder + "tolerance.png",
    bbox_inches="tight",
)  # save location

plt.clf()
fig = ATFuncs.plotfill_stats(
    errors_precision,
    middle="median",
    x_data=np.mean(compressionfactors_precision, axis=0),
)
plt.xlabel("Compression Factor", fontsize=fsize)
plt.ylabel("Normalised L2 Error", fontsize=fsize)
plt.title(
    "Error against Compression factor for zfp with fixed precision parameter",
    fontsize=fsize,
)
plt.yticks(fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.legend(fontsize=fsize)
plt.savefig(
    savefolder + "precision.png",
    bbox_inches="tight",
)  # save location

plt.clf()
fig = ATFuncs.plotfill_stats(
    errors_bitrate, middle="median", x_data=np.mean(compressionfactors_bitrate, axis=0)
)
plt.xlabel("Compression Factor", fontsize=fsize)
plt.ylabel("Normalised L2 Error", fontsize=fsize)
plt.title(
    "Error against Compression factor for zfp with fixed rate parameter", fontsize=fsize
)
plt.yticks(fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.legend(fontsize=fsize)
plt.savefig(
    savefolder + "bitrate.png",
    bbox_inches="tight",
)  # save location

plt.clf()
compressionfactors_wavelets = [100 / (100 - a) for a in thresholds]
fig = ATFuncs.plotfill_stats(
    errors_1dw, middle="median", x_data=compressionfactors_wavelets
)
plt.xlabel("Compression Factor", fontsize=fsize)
plt.ylabel("Normalised L2 Error", fontsize=fsize)
plt.title("Error against Compression factor for 1d wavelet compression", fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.legend(fontsize=fsize)
plt.savefig(
    savefolder + "1dwavelet.png",
    bbox_inches="tight",
)  # save location

plt.clf()
fig = ATFuncs.plotfill_stats(
    errors_2dw, middle="median", x_data=compressionfactors_wavelets
)
plt.xlabel("Compression Factor", fontsize=fsize)
plt.ylabel("Normalised L2 Error", fontsize=fsize)
plt.title("Error against Compression factor for 2d wavelet compression", fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.legend(fontsize=fsize)
plt.savefig(
    savefolder + "2dwavelets.png",
    bbox_inches="tight",
)  # save location

plt.clf()
compFactorsSVD = list(range(5, 51, 5))
fig = ATFuncs.plotfill_stats(errorsSVD, middle="average", x_data=compFactorsSVD)
plt.xlabel("Compression Factor", fontsize=fsize)
plt.ylabel("Normalised L2 Error", fontsize=fsize)
plt.title(
    "Error against Compression factor for low rank approximation with SVD",
    fontsize=fsize,
)
plt.yticks(fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.legend(fontsize=fsize)
plt.savefig(
    savefolder + "SVD.png",
    bbox_inches="tight",
)  # save location

plt.clf()
plt.plot(
    np.mean(compressionfactors_tolerance, axis=0)[1:],
    np.mean(errors_tolerance, axis=0)[1:],
    "-o",
    label="fixed accuracy",
)
plt.plot(
    np.mean(compressionfactors_precision, axis=0),
    np.mean(errors_precision, axis=0),
    "-o",
    label="fixed precision",
)
plt.plot(
    np.mean(compressionfactors_bitrate, axis=0)[1:],
    np.mean(errors_bitrate, axis=0)[1:],
    "-o",
    label="fixed rate",
)
plt.plot(
    compressionfactors_wavelets, np.mean(errors_1dw, axis=0), "-o", label="1d wavelet"
)
plt.plot(
    compressionfactors_wavelets, np.mean(errors_2dw, axis=0), "-o", label="2d wavelet"
)
plt.plot(compFactorsSVD, np.mean(errorsSVD, axis=0), "-o", label="SVD")
plt.xlabel("Compression Factor", fontsize=fsize)
plt.ylabel("Normalised Error", fontsize=fsize)
plt.title(
    "Error against Compression factor for different compression schemes", fontsize=fsize
)
plt.yticks(fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.legend(fontsize=fsize)
plt.savefig(
    savefolder + "All.png",
    bbox_inches="tight",
)  # save location

plt.clf()
plt.semilogx(
    np.mean(compressionfactors_tolerance, axis=0),
    np.mean(errors_tolerance, axis=0),
    "-o",
    label="fixed accuracy",
)
plt.semilogx(
    np.mean(compressionfactors_precision, axis=0),
    np.mean(errors_precision, axis=0),
    "-o",
    label="fixed precision",
)
plt.semilogx(
    np.mean(compressionfactors_bitrate, axis=0),
    np.mean(errors_bitrate, axis=0),
    "-o",
    label="fixed rate",
)
plt.semilogx(
    compressionfactors_wavelets, np.mean(errors_1dw, axis=0), "-o", label="1d wavelet"
)
plt.semilogx(
    compressionfactors_wavelets, np.mean(errors_2dw, axis=0), "-o", label="2d wavelet"
)
plt.semilogx(compFactorsSVD, np.mean(errorsSVD, axis=0), "-o", label="SVD")
plt.xlabel("Compression Factor", fontsize=fsize)
plt.ylabel("Normalised Error", fontsize=fsize)
# plt.title("Error against Compression factor for different compression schemes", fontsize=fsize)
plt.title("Noise Introduced by Lossy Compression", fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.legend(fontsize=12)
plt.savefig(
    savefolder + "Allsemilog.png",
    bbox_inches="tight",
)  # save location
