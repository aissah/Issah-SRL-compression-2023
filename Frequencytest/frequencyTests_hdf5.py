"""
Create a cube populated by the (compressed data power spectrum)/(original data
                                                                 power spectrum)
This is done in windows of 5 second lengths. The cube has dimensions in time,
channels and frequency. 

Files are saves for the power spectra at the various time windows for each of
the 3 compression types used here (wavelet, zfp, svd). These also include the
norm of errors introduced for each time window.

Images are plotted and save in the current workspace to visualize the change in
this ratio with respect to the 3 axes.

An image is plotted for the data and the norm of errors for each compression 
type too.

Created on Tue Aug  9 10:58:46 2022

@author: issah
"""
import sys

import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import zfpy

sys.path.insert(
    0, r"D:\CSM\Mines_Research\Summer_2022_paper\summer2022exp\Frequencytest"
)  # ATFuncs location
import os
import pickle

from FTFuncs import (
    loadFORESEEhdf5,
    multweigthedAverageRatio,
    plotsaveimshow,
    plt,
    randomized_SVD_comp_decomp,
    soft_comp_decomp1d,
    soft_comp_decomp2d,
    stackInWindows,
    windowedNormalisedErrors,
    windowedPowerSpectrum,
)

basepath = "D:\\CSM\\Mines_Research\\Test_data\\FORESEE_Aug"  # data location
# entries = os.listdir("D:\\CSM\\Mines_Research\\Test_data\\FORESEE_Aug")  # data location

window = 5
tolerance = 0.2         # tolerance for zfp compression
precision = 4           # precision for zfp compression
bitrate = 4             # bitrate for zfp compression
threshold1 = 95         # percentile for 1D wavelet thresholding
threshold2 = 95         # percentile for 2D wavelet thresholding
svd_comp_factor = 20    # compression factor for SVD
samplingFrequency = 125 # sampling frequency of data 

flag = 1
count = 0

files = os.listdir(basepath)
for b in files:
    if count < 3:
        count += 1
        fname = os.path.join(basepath, b)
        data, timestamps = loadFORESEEhdf5(fname, normalize="no")
        # time = datetime.utcfromtimestamp(int(timestamps[0]))
        # stack = stackInWindows(data, samplingFrequency, 1)
        if len(data) % 2 == 1:
            data = data[:-1]

        if flag == 1:
            stackedData = stackInWindows(data, samplingFrequency, 1)
            windowedpowerspectrum, frequencies = windowedPowerSpectrum(
                data, samplingFrequency, windowlength=window
            )
            windowedpowerspectra_original = windowedpowerspectrum

            # compresseddata_lossy = zfpy.compress_numpy(data, tolerance=tolerance)
            # decompresseddata_lossy = zfpy.decompress_numpy(compresseddata_lossy)
            # windowedpowerspectrum, _ = windowedPowerSpectrum(
            #     decompresseddata_lossy, samplingFrequency, windowlength=window
            # )
            # windowedpowerspectra_tolerance = windowedpowerspectrum
            # windowedNormalisedError_tolerance = windowedNormalisedErrors(
            #     data, decompresseddata_lossy, samplingFrequency, window, axis=1
            # )
            # windowedNormalisedErrors_tolerance = windowedNormalisedError_tolerance

            # zfp error analysis
            compresseddata_lossy = zfpy.compress_numpy(data, precision=precision)
            decompresseddata_lossy = zfpy.decompress_numpy(compresseddata_lossy)
            windowedpowerspectrum, _ = windowedPowerSpectrum(
                decompresseddata_lossy, samplingFrequency, windowlength=window
            )
            windowedpowerspectra_precision = windowedpowerspectrum
            windowedNormalisedError_precision = windowedNormalisedErrors(
                data, decompresseddata_lossy, samplingFrequency, window, axis=1
            )
            windowedNormalisedErrors_precision = windowedNormalisedError_precision

            # compresseddata_lossy = zfpy.compress_numpy(data, rate=bitrate)
            # decompresseddata_lossy = zfpy.decompress_numpy(compresseddata_lossy)
            # windowedpowerspectrum, _ = windowedPowerSpectrum(
            #     decompresseddata_lossy, samplingFrequency, windowlength=window
            # )
            # windowedpowerspectra_bitrate = windowedpowerspectrum
            # windowedNormalisedError_bitrate = windowedNormalisedErrors(
            #     data, decompresseddata_lossy, samplingFrequency, window, axis=1
            # )
            # windowedNormalisedErrors_bitrate = windowedNormalisedError_bitrate

            # 1D wavelet error analysis
            decompresseddata = soft_comp_decomp1d(
                data, lvl=5, comp_ratio=threshold1
            )
            windowedpowerspectrum, _ = windowedPowerSpectrum(
                decompresseddata, samplingFrequency, windowlength=window
            )
            windowedpowerspectra_1dw = windowedpowerspectrum
            windowedNormalisedError_1dw = windowedNormalisedErrors(
                data, decompresseddata, samplingFrequency, window, axis=1
            ) 
            windowedNormalisedErrors_1dw = windowedNormalisedError_1dw

            # 2D wavelet error analysis
            decompresseddata = soft_comp_decomp2d(
                data, lvl=5, comp_ratio=threshold2
            )
            windowedpowerspectrum, _ = windowedPowerSpectrum(
                decompresseddata, samplingFrequency, windowlength=window
            )
            windowedpowerspectra_2dw = windowedpowerspectrum
            windowedNormalisedError_2dw = windowedNormalisedErrors(
                data, decompresseddata, samplingFrequency, window, axis=1
            )
            windowedNormalisedErrors_2dw = windowedNormalisedError_2dw

            # randomized SVD error analysis
            decompresseddata = randomized_SVD_comp_decomp(data, svd_comp_factor)
            windowedpowerspectrum, _ = windowedPowerSpectrum(
                decompresseddata, samplingFrequency, windowlength=window
            )
            windowedpowerspectra_svd = windowedpowerspectrum
            windowedNormalisedError_svd = windowedNormalisedErrors(
                data, decompresseddata, samplingFrequency, window, axis=1
            )
            windowedNormalisedErrors_svd = windowedNormalisedError_svd

            flag = 0
            starttime = datetime.utcfromtimestamp(int(timestamps[0]))
        else:
            stack = stackInWindows(data, samplingFrequency, 1)
            stackedData = np.append(stackedData, stack, axis=1)
            windowedpowerspectrum, _ = windowedPowerSpectrum(
                data, samplingFrequency, windowlength=window
            )
            windowedpowerspectra_original = np.append(
                windowedpowerspectra_original, windowedpowerspectrum, axis=0
            )

            # compresseddata_lossy = zfpy.compress_numpy(data, tolerance=tolerance)
            # decompresseddata_lossy = zfpy.decompress_numpy(compresseddata_lossy)
            # windowedpowerspectrum, _ = windowedPowerSpectrum(
            #     decompresseddata_lossy, samplingFrequency, windowlength=window
            # )
            # windowedpowerspectra_tolerance = np.append(
            #     windowedpowerspectra_tolerance, windowedpowerspectrum, axis=0
            # )
            # windowedNormalisedError_tolerance = windowedNormalisedErrors(
            #     data, decompresseddata_lossy, samplingFrequency, window, axis=1
            # )
            # windowedNormalisedErrors_tolerance = np.append(
            #     windowedNormalisedErrors_tolerance,
            #     windowedNormalisedError_tolerance,
            #     axis=1,
            # )

            # zfp compression analysis of errors in windowed spectrum
            compresseddata_lossy = zfpy.compress_numpy(data, precision=precision)
            decompresseddata_lossy = zfpy.decompress_numpy(compresseddata_lossy)
            windowedpowerspectrum, _ = windowedPowerSpectrum(
                decompresseddata_lossy, samplingFrequency, windowlength=window
            )
            windowedpowerspectra_precision = np.append(
                windowedpowerspectra_precision, windowedpowerspectrum, axis=0
            )
            windowedNormalisedError_precision = windowedNormalisedErrors(
                data, decompresseddata_lossy, samplingFrequency, window, axis=1
            )
            windowedNormalisedErrors_precision = np.append(
                windowedNormalisedErrors_precision,
                windowedNormalisedError_precision,
                axis=1,
            )

            # compresseddata_lossy = zfpy.compress_numpy(data, rate=bitrate)
            # decompresseddata_lossy = zfpy.decompress_numpy(compresseddata_lossy)
            # windowedpowerspectrum, _ = windowedPowerSpectrum(
            #     decompresseddata_lossy, samplingFrequency, windowlength=window
            # )
            # windowedpowerspectra_bitrate = np.append(
            #     windowedpowerspectra_bitrate, windowedpowerspectrum, axis=0
            # )
            # windowedNormalisedError_bitrate = windowedNormalisedErrors(
            #     data, decompresseddata_lossy, samplingFrequency, window, axis=1
            # )
            # windowedNormalisedErrors_bitrate = np.append(
            #     windowedNormalisedErrors_bitrate,
            #     windowedNormalisedError_bitrate,
            #     axis=1,
            # )

            # 1d wavelet compression analysis of errors in windowed spectrum
            decompresseddata = soft_comp_decomp1d(
                data, lvl=5, comp_ratio=threshold1
            )
            windowedpowerspectrum, _ = windowedPowerSpectrum(
                decompresseddata, samplingFrequency, windowlength=window
            )
            windowedpowerspectra_1dw = np.append(
                windowedpowerspectra_1dw, windowedpowerspectrum, axis=0
            )
            windowedNormalisedError_1dw = windowedNormalisedErrors(
                data, decompresseddata, samplingFrequency, window, axis=1
            )
            windowedNormalisedErrors_1dw = np.append(
                windowedNormalisedErrors_1dw, windowedNormalisedError_1dw, axis=1
            )

            # 2d wavelet compression analysis of errors in windowed spectrum
            decompresseddata = soft_comp_decomp2d(
                data, lvl=5, comp_ratio=threshold2
            )
            windowedpowerspectrum, _ = windowedPowerSpectrum(
                decompresseddata, samplingFrequency, windowlength=window
            )
            windowedpowerspectra_2dw = np.append(
                windowedpowerspectra_2dw, windowedpowerspectrum, axis=0
            )
            windowedNormalisedError_2dw = windowedNormalisedErrors(
                data, decompresseddata, samplingFrequency, window, axis=1
            )
            windowedNormalisedErrors_2dw = np.append(
                windowedNormalisedErrors_2dw, windowedNormalisedError_2dw, axis=1
            )

            # SVD compression analysis of errors in windowed spectrum
            decompresseddata = randomized_SVD_comp_decomp(data, svd_comp_factor)
            windowedpowerspectrum, _ = windowedPowerSpectrum(
                decompresseddata, samplingFrequency, windowlength=window
            )
            windowedpowerspectra_svd = np.append(
                windowedpowerspectra_svd, windowedpowerspectrum, axis=0
            )
            windowedNormalisedError_svd = windowedNormalisedErrors(
                data, decompresseddata, samplingFrequency, window, axis=1
            )
            windowedNormalisedErrors_svd = np.append(
                windowedNormalisedErrors_svd, windowedNormalisedError_svd, axis=1
            )

            endtime = datetime.utcfromtimestamp(int(timestamps[-1]))
            # endtime = np.datetime64(endtime) + np.timedelta64(60, "s")

# write results of power spectra, frequencies, and errors for each compression scheme
with open(
    "windowedpowerspectra_originalFrequencies.pkl", "wb"
) as f:  # Python 3: open(..., 'wb')
    pickle.dump([stackedData, windowedpowerspectra_original, frequencies], f)

with open("windowedpowerspectra_precision.pkl", "wb") as f:  # Python 3: open(..., 'wb')
    pickle.dump([windowedpowerspectra_precision, windowedNormalisedErrors_precision], f)

with open("windowedpowerspectra_1dw.pkl", "wb") as f:  # Python 3: open(..., 'wb')
    pickle.dump([windowedpowerspectra_1dw, windowedNormalisedErrors_1dw], f)

with open("windowedpowerspectra_2dw.pkl", "wb") as f:  # Python 3: open(..., 'wb')
    pickle.dump([windowedpowerspectra_2dw, windowedNormalisedErrors_2dw], f)

with open("windowedpowerspectra_svd.pkl", "wb") as f:  # Python 3: open(..., 'wb')
    pickle.dump([windowedpowerspectra_svd, windowedNormalisedErrors_svd], f)

# with open('errorsTolPrecBit1d2d.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     errors_tolerance, errors_precision, errors_bitrate, errors_1dw,errors_2dw = pickle.load(f)
#     #e, n, te, rs_1dw,rs_2dw = pickle.load(f)


# read results and make plots of errors
if True:
    with open("windowedpowerspectra_originalFrequencies.pkl", "rb") as f:
        [stackedData, windowedpowerspectra_original, frequencies] = pickle.load(f)

    with open(
        "windowedpowerspectra_precision.pkl", "rb"
    ) as f:  # Python 3: open(..., 'wb')
        [
            windowedpowerspectra_precision,
            windowedNormalisedErrors_precision,
        ] = pickle.load(f)

    with open("windowedpowerspectra_1dw.pkl", "rb") as f:  # Python 3: open(..., 'wb')
        [windowedpowerspectra_1dw, windowedNormalisedErrors_1dw] = pickle.load(f)

    with open("windowedpowerspectra_2dw.pkl", "rb") as f:  # Python 3: open(..., 'wb')
        [windowedpowerspectra_2dw, windowedNormalisedErrors_2dw] = pickle.load(f)

    with open("windowedpowerspectra_svd.pkl", "rb") as f:  # Python 3: open(..., 'wb')
        [windowedpowerspectra_svd, windowedNormalisedErrors_svd] = pickle.load(f)



shape = np.shape(windowedpowerspectra_precision)
label_size = 15


# # Plot stacked data
plt.figure(figsize=(11, 7))
plt.imshow(
    np.transpose(stackedData),
    plt.cm.RdBu,
    extent=(0, shape[1] * 2.0419, mdates.date2num(starttime), mdates.date2num(endtime)),
    aspect="auto",
    vmin=-np.percentile(abs(stackedData), 90),
    vmax=np.percentile(abs(stackedData), 90),
)
plt.xlabel("Distance along fiber (m)", fontsize=label_size)
plt.ylabel("Time (Days H:m)", fontsize=label_size)
plt.title("Strain Rate Data", fontsize=label_size)
plt.yticks(fontsize=label_size)
plt.xticks(fontsize=label_size)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=label_size)
ax = plt.gca()
ax.yaxis_date()
plt.savefig("Stacked_data.png", bbox_inches="tight")


# Calculate averages of spectral ratios across frequencies
weightedaverageratios_freqp = multweigthedAverageRatio(
    windowedpowerspectra_original, windowedpowerspectra_precision, axis=2
) # zfp
weightedaverageratios_freq1dw = multweigthedAverageRatio(
    windowedpowerspectra_original, windowedpowerspectra_1dw, axis=2
) # 1d wavelet
weightedaverageratios_freq2dw = multweigthedAverageRatio(
    windowedpowerspectra_original, windowedpowerspectra_2dw, axis=2
) # 2d wavelet
weightedaverageratios_freqsvd = multweigthedAverageRatio(
    windowedpowerspectra_original, windowedpowerspectra_svd, axis=2
) # SVD

# Plot averages across frequencies
mini = 1.1
maxi = 1.6
plotsaveimshow(
    weightedaverageratios_freqp,
    [0, shape[1] * 2.0419],
    [mdates.date2num(starttime), mdates.date2num(endtime)],
    mini,
    maxi,
    "Distance along fiber (m)",
    "Time (Days H:m)",
    "Proportion of spectral density after compression with zfp",
    plt.cm.cividis,
    label_size,
    "weightedaverageratios_freqp.png",
    dateaxis="y",
)
mini = 0.2
maxi = 1
plotsaveimshow(
    weightedaverageratios_freq1dw,
    [0, shape[1] * 2.0419],
    [mdates.date2num(starttime), mdates.date2num(endtime)],
    mini,
    maxi,
    "Distance along fiber (m)",
    "Time (Days H:m)",
    "Proportion of spectral density after compression with 1d wavelets (averaged across frequencies)",
    plt.cm.cividis,
    label_size,
    "weightedaverageratios_freq1dw.png",
    dateaxis="y",
)
mini = 0
maxi = 1
plotsaveimshow(
    weightedaverageratios_freqsvd,
    [0, shape[1] * 2.0419],
    [mdates.date2num(starttime), mdates.date2num(endtime)],
    mini,
    maxi,
    "Distance along fiber (m)",
    "Time (Days H:m)",
    "Proportion of spectral density after compression with svd",  # (averaged across frequencies)",
    plt.cm.cividis,
    label_size,
    "weightedaverageratios_freqsvd.png",
    dateaxis="y",
)
mini = 0.2
maxi = 1
plotsaveimshow(
    weightedaverageratios_freq2dw,
    [0, shape[1] * 2.0419],
    [mdates.date2num(starttime), mdates.date2num(endtime)],
    mini,
    maxi,
    "Distance along fiber (m)",
    "Time (Days H:m)",
    "Proportion of spectral density after wavelet compression",  # (averaged across frequencies)",
    plt.cm.cividis,
    label_size,
    "weightedaverageratios_freq2dw.png",
    dateaxis="y",
)


# Calculate averages across time windows
weightedaverageratios_twinp = multweigthedAverageRatio(
    windowedpowerspectra_original, windowedpowerspectra_precision, axis=0
) # zfp
weightedaverageratios_twin1dw = multweigthedAverageRatio(
    windowedpowerspectra_original, windowedpowerspectra_1dw, axis=0
) # 1d wavelet
weightedaverageratios_twin2dw = multweigthedAverageRatio(
    windowedpowerspectra_original, windowedpowerspectra_2dw, axis=0
) # 2d wavelet
weightedaverageratios_twinsvd = multweigthedAverageRatio(
    windowedpowerspectra_original, windowedpowerspectra_svd, axis=0
) # SVD


# Calculate averages across time windows
mini = 0.8
maxi = 1.7
plotsaveimshow(
    weightedaverageratios_twinp,
    [0, shape[1] * 2.0419],
    [frequencies[0], frequencies[-1]],
    mini,
    maxi,
    "Distance along fiber (m)",
    "Frequency (Hz)",
    "Proportion of spectral density after compression with zfp",
    plt.cm.cividis,
    label_size,
    "weightedaverageratios_twinp.png",
)

plotsaveimshow(
    weightedaverageratios_twin1dw,
    [0, shape[1] * 2.0419],
    [frequencies[0], frequencies[-1]],
    mini,
    maxi,
    "Distance along fiber (m)",
    "Frequency (Hz)",
    "Weighted average ratio of frequencies across time windows for 1d wavelets",
    plt.cm.cividis,
    label_size,
    "weightedaverageratios_twin1dw.png",
)
mini = 0
maxi = 1
plotsaveimshow(
    weightedaverageratios_twin2dw,
    [0, shape[1] * 2.0419],
    [frequencies[0], frequencies[-1]],
    mini,
    maxi,
    "Distance along fiber (m)",
    "Frequency (Hz)",
    "Proportion of spectral density after wavelet compression",
    plt.cm.cividis,
    label_size,
    "weightedaverageratios_twin2dw.png",
)
mini = 0.5
maxi = 1.5
plotsaveimshow(
    weightedaverageratios_twinsvd,
    [0, shape[1] * 2.0419],
    [frequencies[0], frequencies[-1]],
    mini,
    maxi,
    "Distance along fiber (m)",
    "Frequency (Hz)",
    "Proportion of spectral density after compression with svd",
    plt.cm.cividis,
    label_size,
    "weightedaverageratios_twinsvd.png",
)


# Calculate averages across channels
weightedaverageratios_channelsp = multweigthedAverageRatio(
    windowedpowerspectra_original, windowedpowerspectra_precision, axis=1
) # zfp
weightedaverageratios_channels1dw = multweigthedAverageRatio(
    windowedpowerspectra_original, windowedpowerspectra_1dw, axis=1
) # 1d wavelet
weightedaverageratios_channels2dw = multweigthedAverageRatio(
    windowedpowerspectra_original, windowedpowerspectra_2dw, axis=1
) # 2d wavelet
weightedaverageratios_channelssvd = multweigthedAverageRatio(
    windowedpowerspectra_original, windowedpowerspectra_svd, axis=1
) # SVD


# Plot averages across channels
mini = 0.9
maxi = 1.9
plotsaveimshow(
    weightedaverageratios_channelsp,
    [frequencies[0], frequencies[-1]],
    [mdates.date2num(starttime), mdates.date2num(endtime)],
    mini,
    maxi,
    "Frequency (Hz)",
    "Time (Days H:m)",
    "Proportion of spectral density after compression with zfp",
    plt.cm.cividis,
    label_size,
    "weightedaverageratios_channelsp.png",
    dateaxis="y",
)
plotsaveimshow(
    weightedaverageratios_channels1dw,
    [frequencies[0], frequencies[-1]],
    [mdates.date2num(starttime), mdates.date2num(endtime)],
    mini,
    maxi,
    "Frequency (Hz)",
    "Time (Days H:m)",
    "Weighted average ratio of frequencies across channels for 1d wavelets",
    plt.cm.inferno,
    label_size,
    "weightedaverageratios_channels1dw.png",
    dateaxis="y",
)
mini = 0.2
maxi = 1
plotsaveimshow(
    weightedaverageratios_channels2dw,
    [frequencies[0], frequencies[-1]],
    [mdates.date2num(starttime), mdates.date2num(endtime)],
    mini,
    maxi,
    "Frequency (Hz)",
    "Time (Days H:m)",
    "Proportion of spectral density after wavelet compression",
    plt.cm.cividis,
    label_size,
    "weightedaverageratios_channels2dw.png",
    dateaxis="y",
)
mini = 0.5
maxi = 1
plotsaveimshow(
    weightedaverageratios_channelssvd,
    [frequencies[0], frequencies[-1]],
    [mdates.date2num(starttime), mdates.date2num(endtime)],
    mini,
    maxi,
    "Frequency (Hz)",
    "Time (Days H:m)",
    "Proportion of spectral density after compression with svd",
    plt.cm.cividis,
    label_size,
    "weightedaverageratios_channelssvd.png",
    dateaxis="y",
)


mini = 0.35
maxi = 0.8
plotsaveimshow(
    windowedNormalisedErrors_precision.T,
    [0, shape[1] * 2.0419],
    [mdates.date2num(starttime), mdates.date2num(endtime)],
    mini,
    maxi,
    "Distance along fiber (m)",
    "Time (Days H:m)",
    "Normalised error after compression with zfp",
    plt.cm.Oranges,
    label_size,
    "normalised_errors_precision.png",
    dateaxis="y",
)
mini = 0.35
maxi = 1
plotsaveimshow(
    windowedNormalisedErrors_2dw.T,
    [0, shape[1] * 2.0419],
    [mdates.date2num(starttime), mdates.date2num(endtime)],
    mini,
    maxi,
    "Distance along fiber (m)",
    "Time (Days H:m)",
    "Normalised error after wavelet compression",
    plt.cm.Oranges,
    label_size,
    "normalised_errors_wl.png",
    dateaxis="y",
)
mini = 0.5
maxi = 1
plotsaveimshow(
    windowedNormalisedErrors_svd.T,
    [0, shape[1] * 2.0419],
    [mdates.date2num(starttime), mdates.date2num(endtime)],
    mini,
    maxi,
    "Distance along fiber (m)",
    "Time (Days H:m)",
    "Normalised error after compression with svd",
    plt.cm.OrRd,
    label_size,
    "normalised_errors_svd.png",
    dateaxis="y",
)
