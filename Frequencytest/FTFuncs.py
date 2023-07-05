"""
Created on Fri Aug 12 08:51:28 2022

@author: issah
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pywt


def loadFORESEEhdf5(file, normalize="yes"):
    """

    Parameters
    ----------
    file : str
        path to foresee h5py data file
    normalize : str, optional
        "yes" or "no". Indicates whether or not to remove laser drift and
        normalize. The default is 'yes'.

    Returns
    -------
    data : np array
        channel by samples numpy array of data
    timestamp_arr : numpy array
        array of the timestamps corresponding to the various samples in the
        data.

    """
    with h5py.File(file, "r") as open_file:
        dataset = open_file["raw"]
        time = open_file["timestamp"]
        data = np.float32(dataset)
        timestamp_arr = np.float32(time)
    if normalize == "yes":
        nSamples = np.shape(data)[1]
        # get rid of laser drift
        med = np.median(data, axis=0)
        for i in range(nSamples):
            data[:, i] = data[:, i] - med[i]
        # L1 normalized rows
        max_of_rows = abs(data[:, :]).sum(axis=1)
        data = data / max_of_rows[:, np.newaxis]
    return data, timestamp_arr


def loadBradyHShdf5(file, normalize="yes"):
    """

    Parameters
    ----------
    file : str
        path to brady hotspring h5py data file
    normalize : str, optional
        "yes" or "no". Indicates whether or not to remove laser drift and
        normalize. The default is 'yes'.

    Returns
    -------
    data : np array
        channel by samples numpy array of data
    timestamp_arr : numpy array
        array of the timestamps corresponding to the various samples in the
        data. Timestamps for brady hotspring data are with respect to the
        beginning time of the survey.

    """

    with h5py.File(file, "r") as open_file:
        dataset = open_file["das"]
        time = open_file["t"]
        data = np.array(dataset)
        timestamp_arr = np.array(time)
    data = np.transpose(data)
    if normalize == "yes":
        nSamples = np.shape(data)[1]
        # get rid of laser drift and normalize
        med = np.median(data, axis=0)
        for i in range(nSamples):
            data[:, i] = data[:, i] - med[i]
        max_of_rows = abs(data[:, :]).sum(axis=1)
        data = data / max_of_rows[:, np.newaxis]
    return data, timestamp_arr


def soft_comp_decomp1d(data_inuse, lvl=5, comp_ratio=80):
    """
    1d-Wavelet compress data using soft thresholding. Then return reconstructed
    data.

    Parameters
    ----------
    data_inuse : 2-dimensional numpy array
        Data to be compressed. (channels by time samples)
    lvl : int, optional
        Levels of multiresolution wavelet decomposition. The default is 5.
    comp_ratio : int/float, optional
        Compression ratio to comress data to. The default is 80.

    Returns
    -------
    dc_data :  2-dimensional numpy array
        Reconstructed data after compression.

    """
    # wavelet decomposition
    coeffs = pywt.wavedec(data_inuse, "db5", level=lvl)
    # threshold wavelet coefficients
    thresheld_coeffs = soft_threshold(coeffs, comp_ratio, mode="1d")
    # reconstruct data using thresheld coefficients
    dc_data = pywt.waverec(thresheld_coeffs, "db5")

    return dc_data


def soft_comp_decomp2d(data_inuse, lvl=5, comp_ratio=96):
    """
    2d-Wavelet compress data using soft thresholding. Then return reconstructed
    data.

    Parameters
    ----------
    data_inuse : 2-dimensional numpy array
        Data to be compressed. (channels by time samples)
    lvl : int, optional
        Levels of multiresolution wavelet decomposition. The default is 5.
    comp_ratio : int/float, optional
        Compression ratio to comress data to. The default is 96.

    Returns
    -------
    dc_data2d : 2-dimensional numpy array
        Reconstructed data after compression.

    """
    # wavelet decomposition
    coeffs2d = pywt.wavedec2(data_inuse, "db5", level=lvl)
    # threshold wavelet coefficients
    thresheld_coeffs = soft_threshold(coeffs2d, comp_ratio, mode="2d")
    # reconstruct data using thresheld coefficients
    dc_data2d = pywt.waverec2(thresheld_coeffs, "db5")

    return dc_data2d


def soft_threshold(wavelet_coeffs, threshold_percentile, mode="1d"):
    """
    Soft thresholding operator on wavelet_coeffs.

    Parameters
    ----------
    wavelet_coeffs : tuple
        wavelet coefficients obtained by using pywavelets to decompose data.
    threshold_percentile : int
        percentile of coefficients to set to zero.
    mode : string optional
        2d or 1d to indicate 2d or 1d wavelet decomposition coefficients in
        wavelet_coeffs respectively. The default is "1d".

    Returns
    -------
    thresheld_coeffs : tuple
        Coefficients after thrsholding.

    """
    if mode == "1d":
        if wavelet_coeffs[1].ndim == 2:
            # threshold each level and channel separately
            thresheld_coeffs = []
            for level_coeffs in wavelet_coeffs:
                thresholds = np.percentile(
                    abs(level_coeffs), threshold_percentile, axis=1
                )
                thresheld_coeffs.append(
                    np.sign(level_coeffs)
                    * np.maximum(np.abs(level_coeffs) - thresholds[:, np.newaxis], 0)
                )

            # threshold just channels separately
            # all_coeffs = wavelet_coeffs[0]
            # for a in range(1, len(wavelet_coeffs)):
            #     all_coeffs = np.append(all_coeffs, wavelet_coeffs[a], axis=1)
            # thresholds = np.percentile(abs(all_coeffs), threshold_percentile, axis = 1)
            # thresheld_coeffs = [np.sign(level_coeffs) * np.maximum(np.abs(level_coeffs) - thresholds[:,np.newaxis], 0) for level_coeffs in wavelet_coeffs]
        elif wavelet_coeffs[1].ndim == 1:
            # threshold each level and channel separately
            thresheld_coeffs = []
            for level_coeffs in wavelet_coeffs:
                threshold = np.percentile(abs(level_coeffs), threshold_percentile)
                thresheld_coeffs.append(
                    np.sign(level_coeffs)
                    * np.maximum(np.abs(level_coeffs) - threshold, 0)
                )

            # threshold just channels separately
            # all_coeffs = wavelet_coeffs[0]
            # for a in range(1, len(wavelet_coeffs)):
            #     all_coeffs = np.append(all_coeffs, wavelet_coeffs[a])
            # threshold = np.percentile(abs(all_coeffs), threshold_percentile)
            # thresheld_coeffs = [np.sign(level_coeffs) * np.maximum(np.abs(level_coeffs) - threshold, 0) for level_coeffs in wavelet_coeffs]
    elif mode == "2d":
        if len(wavelet_coeffs[0]) == 1:
            thresheld_coeffs = [wavelet_coeffs[0], wavelet_coeffs[1]]
            next_level = 2
        elif len(wavelet_coeffs[0]) == 2:
            thresheld_coeffs = [wavelet_coeffs[0], wavelet_coeffs[1]]
            next_level = 1
        else:
            threshold = np.percentile(abs(wavelet_coeffs[0]), threshold_percentile)
            thresheld_coeffs = [
                np.sign(wavelet_coeffs[0])
                * np.maximum(np.abs(wavelet_coeffs[0]) - threshold, 0)
            ]
            next_level = 1
        for level_coeffs in wavelet_coeffs[next_level:]:
            threshold = np.percentile(
                abs(np.concatenate(level_coeffs)), threshold_percentile
            )
            thresheld_level = tuple(
                [
                    np.sign(orient_coeffs)
                    * np.maximum(np.abs(orient_coeffs) - threshold, 0)
                    for orient_coeffs in level_coeffs
                ]
            )
            thresheld_coeffs.append(thresheld_level)

    return thresheld_coeffs


def randomized_SVD_comp_decomp(data, compFactor):
    """
    Compress data with randomized SVD by compression factor and return
    reconstructed data.

    Parameters
    ----------
    data : 2-dimensional numpy array
        Data to be compressed.
    compFactor : int/float
        Compression factor.

    Returns
    -------
    recon : 2-dimensional numpy array
        Reconstructed data after compression.

    """
    from sklearn.utils.extmath import randomized_svd

    rows, columns = data.shape
    approxRank = int((rows * columns) / (compFactor * (rows + columns)))
    # U, S, Vt = la.svd(data)
    U, S, Vt = randomized_svd(data, n_components=approxRank)
    recon = U @ np.diag(S) @ Vt
    # sv = np.dot(np.diag(S), Vt)
    # recon = np.dot(U, sv)
    return recon


def powerSpectrum(data, samplingFrequency):
    """
    Calculate the power spectrum and its frequencies for each channel of data.

    Parameters
    ----------
    data : 1 or 2-dimensional numpy array
        Data to be analyzed for frequency content.
    samplingFrequency : float
        Number of samples per second for each sensor.

    Returns
    -------
    powerspectrum : 1 or 2-dimensional numpy array
        Power spectrum of each channel
    frequencies : 1-dimensional numpy array
        Array of frequencies (in Hz) represented by power spectrum

    """
    dimensions = np.ndim(data)

    fouriertransform = np.fft.rfft(data, norm="forward")
    absfouriertransform = np.abs(fouriertransform)
    powerspectrum = np.square(absfouriertransform)

    if dimensions == 1:
        frequencies = np.linspace(0, samplingFrequency / 2, len(powerspectrum))
    elif dimensions == 2:
        frequencies = np.linspace(0, samplingFrequency / 2, len(powerspectrum[1]))

    return powerspectrum, frequencies


def windowedPowerSpectrum(data, samplingFrequency, windowlength=5):
    """
    Calculate power spectrum of a time series in "windowlength" seconds windows.
    If the data is 2 dimensional, the second axis is taken as the time axis

    Parameters
    ----------
    data : 1 or 2-dimensional numpy array
        Data to be analyzed for frequency content.
    samplingFrequency : int/float
        Sampling frequency of time series.
    windowlength : int/float, optional
       Length of windows to use in seconds. The default is 5.

    Returns
    -------
    windowedpowerspectrum : 1 or 2-dimensional numpy array
        Power spectra of the data.
    frequencies : 1-dimensional numpy array
        Frequencies corresponding to the power spectra.

    """
    dimensions = np.ndim(data)

    if dimensions == 1:
        totaltime = len(data) / samplingFrequency
        intervals = np.arange(windowlength, totaltime, windowlength, dtype=int) * int(
            samplingFrequency
        )  # break time series into windowed intervals
        win_start = 0
        win_data = data[win_start : intervals[0]]
        powerspectrum, frequencies = powerSpectrum(win_data, samplingFrequency)
        windowedpowerspectrum = powerspectrum[np.newaxis]
        win_start = intervals[0]
        for win_end in intervals[
            1:
        ]:  # for each interval, calculate and record its spectrum
            win_data = data[win_start:win_end]
            powerspectrum, _ = powerSpectrum(win_data, samplingFrequency)
            windowedpowerspectrum = np.append(
                windowedpowerspectrum, powerspectrum[np.newaxis], axis=0
            )
            win_start = win_end
    elif dimensions == 2:
        totaltime = len(data[0]) / samplingFrequency
        intervals = np.arange(windowlength, totaltime, windowlength, dtype=int) * int(
            samplingFrequency
        )  # break time series into windowed intervals
        win_start = 0
        win_data = data[:, win_start : intervals[0]]
        powerspectrum, frequencies = powerSpectrum(win_data, samplingFrequency)
        windowedpowerspectrum = powerspectrum[np.newaxis]
        win_start = intervals[0]
        for win_end in intervals[
            1:
        ]:  # for each interval, calculate and record its spectrum
            win_data = data[:, win_start:win_end]
            powerspectrum, _ = powerSpectrum(win_data, samplingFrequency)
            windowedpowerspectrum = np.append(
                windowedpowerspectrum, powerspectrum[np.newaxis], axis=0
            )
            win_start = win_end

    return windowedpowerspectrum, frequencies


def weigthedAverageRatio(data1, data2):
    """
    Weighted average of pointwise ratio of elements in data2 to elements in data1.
    Weights are proportions of magnitudes of the respective elements in data1.

    Parameters
    ----------
    data1 : 2-dimensional numpy array
        Denominators.
    data2 : 2-dimensional numpy array
        Numerators.

    Returns
    -------
    weightedaverageratio : 1-dimensional numpy array
        weighted average ratios same length as that of the first axis.

    """
    weights = data1 / np.sum(data1, axis=1)[:, np.newaxis]
    ratios = data2 / data1
    weightedaverageratio = np.nansum(weights * ratios, axis=1)

    return weightedaverageratio


def multweigthedAverageRatio(data1, data2, axis=1):
    """
    Uses weigthedAverageRatio to get the weighted average of pointwise ratio
    of elements in data2 to elements in data1. The data here are 3-dimensional.
    Weights are proportions of magnitudes of the respective elements in data1.

    Parameters
    ----------
    data1 : 3-dimensional numpy array
        Denominators.
    data2 : 3-dimensional numpy array
        Numerators.
    axis : int, optional
        Axis to average along. The default is 1.

    Returns
    -------
    weightedaverageratio : 2-dimensional numpy array
        weighted average ratios same shape as the other axis apart from the
        axis in input as that of the first axis.

    """
    shape = np.shape(data1)
    # average of ratios across all frequencies
    if axis == 0:
        weightedaverageratio = weigthedAverageRatio(
            np.transpose(data1[:, :, 0]), np.transpose(data2[:, :, 0])
        )
        weightedaverageratios = weightedaverageratio[np.newaxis]
        for a in range(1, shape[2]):
            weightedaverageratio = weigthedAverageRatio(
                np.transpose(data1[:, :, a]), np.transpose(data2[:, :, a])
            )
            weightedaverageratios = np.append(
                weightedaverageratios, weightedaverageratio[np.newaxis], axis=0
            )
    elif axis == 1:
        weightedaverageratio = weigthedAverageRatio(
            np.transpose(data1[0, :, :]), np.transpose(data2[0, :, :])
        )
        weightedaverageratios = weightedaverageratio[np.newaxis]
        for a in range(1, shape[0]):
            weightedaverageratio = weigthedAverageRatio(
                np.transpose(data1[a, :, :]), np.transpose(data2[a, :, :])
            )
            weightedaverageratios = np.append(
                weightedaverageratios, weightedaverageratio[np.newaxis], axis=0
            )
    elif axis == 2:
        weightedaverageratio = weigthedAverageRatio(data1[0, :, :], data2[0, :, :])
        weightedaverageratios = weightedaverageratio[np.newaxis]
        for a in range(1, shape[0]):
            weightedaverageratio = weigthedAverageRatio(data1[a, :, :], data2[a, :, :])
            weightedaverageratios = np.append(
                weightedaverageratios, weightedaverageratio[np.newaxis], axis=0
            )

    return weightedaverageratios


def windowedNormalisedErrors(data1, data2, samplingFrequency, windowinterval, axis=1):
    """
    Find the l2-norm of error between data1 and data2 normalized by the norm
    of data1. This is done in windows of length samplingFrequency * windowinterval

    Parameters
    ----------
    data1 : 1d or 2d numpy array
        One of the arrays to be compared, typically the true data.
    data2 : 1d or 2d numpy array
        The other array; an estimation of data1. This should have the same shape
        as data1.
    samplingFrequency : int
        Sampling frequency of axis in which windowing is going to be done.
    windowinterval : int/float
        Length of the window in seconds or whatever unit is recognized by the
        axis along which windowing is going to be done.
    axis : int, optional
        1 0r 0;the axis along which windowing is going to be done. The default
        is 1.

    Returns
    -------
    windowednormalisederrors : 1d or 2d numpy array
        Array containing norm of errors. The axis the corresponds to the
        windowing axis has length divided by samplingFrequency * windowinterval.
        The other axis is the same as that of data1 and data2.

    """
    windowinterval = samplingFrequency * windowinterval
    dimensions = np.ndim(data1)

    if dimensions == 1:
        datalength = len(data1)
        intervals = np.arange(windowinterval, datalength, windowinterval, dtype=int)
        win_start = 0
        win_data1 = data1[win_start : intervals[0]]
        win_data2 = data2[win_start : intervals[0]]
        normalisederrors = np.linalg.norm(win_data1 - win_data2) / np.linalg.norm(
            win_data1
        )
        windowednormalisederrors = [normalisederrors]
        win_start = intervals[0]
        # for each window, check normalized error across frequencies
        for win_end in intervals[1:]:
            win_data1 = data1[win_start:win_end]
            win_data2 = data2[win_start:win_end]
            normalisederrors = np.linalg.norm(win_data1 - win_data2) / np.linalg.norm(
                win_data1
            )
            windowednormalisederrors.append(normalisederrors)
            win_start = win_end
    elif dimensions == 2:
        datalength = np.shape(data1)[axis]
        intervals = np.arange(windowinterval, datalength, windowinterval, dtype=int)
        win_start = 0
        win_data1 = data1[:, win_start : intervals[0]]
        win_data2 = data2[:, win_start : intervals[0]]
        normalisederrors = np.linalg.norm(
            (win_data1 - win_data2), axis=axis
        ) / np.linalg.norm(win_data1, axis=axis)
        windowednormalisederrors = normalisederrors[:, np.newaxis]
        win_start = intervals[0]
        # for each window, check normalized error across frequencies
        for win_end in intervals[1:]:
            win_data1 = data1[:, win_start:win_end]
            win_data2 = data2[:, win_start:win_end]
            normalisederrors = np.linalg.norm(
                (win_data1 - win_data2), axis=axis
            ) / np.linalg.norm(win_data1, axis=axis)
            windowednormalisederrors = np.append(
                windowednormalisederrors, normalisederrors[:, np.newaxis], axis=axis
            )
            win_start = win_end
    return windowednormalisederrors


def plotsaveimshow(
    data,
    xextent,
    yextent,
    mini,
    maxi,
    xlabel,
    ylabel,
    title,
    color_scheme,
    label_size,
    filename,
    dateaxis=None,
):
    """
    Plotting function: very specialized and definitely not optimal way of doing
    this.

    Parameters
    ----------
    data : 2-dimensional numpy array
        Data.
    xextent : list/tuple
        2-value list indicating the values at the beginning and end of x-axis.
    yextent : list/tuple
        2-value list indicating the values at the beginning and end of y-axis.
    mini : float
        minimum value to clip colorbar to.
    maxi : int/float
        maximum value to clip colorbar to.
    xlabel : str
        x-axis label.
    ylabel : str
        label of y-axis.
    title : str
        titile of plot.
    color_scheme : cmap options
        color scheme to use.
    label_size : TYPE
        size of plot lavels.
    filename : str
        name of saved image of plot.
    dateaxis : int, optional
        Indicate if either axis is a datetime axis. The default is None.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(11, 7))
    if mini is None:
        plt.imshow(
            data,
            cmap=color_scheme,
            aspect="auto",
            extent=(xextent[0], xextent[1], yextent[0], yextent[1]),
        )
    else:
        plt.imshow(
            data,
            vmin=mini,
            vmax=maxi,
            cmap=color_scheme,
            aspect="auto",
            extent=(xextent[0], xextent[1], yextent[0], yextent[1]),
        )
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    plt.title(title, fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.xticks(fontsize=label_size)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=label_size)
    if dateaxis == "x":
        ax = plt.gca()
        ax.xaxis_date()
    elif dateaxis == "y":
        ax = plt.gca()
        ax.yaxis_date()

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def stackInWindows(data, samplingFrequency, windowlength=5):
    """
    Stack/average data in windows of windowlength seconds so that length along
    the time axis of the data is divided by 5.

    Parameters
    ----------
    data : 1d or 2d numpy array
        Data to be stacked.
    samplingFrequency : int/float
        Samples per second of the input data.
    windowlength : int/float, optional
        length in seconds of windows to stack data in. The default is 5.

    Returns
    -------
    stacks : 1d or 2d numpy array
        stacked data. Same number of dimensions as the input data with the time
        axis length divided by 5.

    """
    dimensions = np.ndim(data)

    if dimensions == 1:
        totaltime = len(data) / samplingFrequency
        intervals = np.arange(windowlength, totaltime, windowlength, dtype=int) * int(
            samplingFrequency
        )
        win_start = 0
        win_data = data[win_start : intervals[0]]
        stack = np.mean(win_data, axis=1)
        stacks = stack[:, np.newaxis]
        win_start = intervals[0]
        for win_end in intervals[1:]:
            win_data = data[win_start:win_end]
            stack = np.mean(win_data, axis=1)
            stacks = np.append(stacks, stack[:, np.newaxis], axis=1)
            win_start = win_end
    elif dimensions == 2:
        totaltime = len(data[0]) / samplingFrequency
        intervals = np.arange(windowlength, totaltime, windowlength, dtype=int) * int(
            samplingFrequency
        )
        win_start = 0
        win_data = data[:, win_start : intervals[0]]
        stack = np.mean(win_data, axis=1)
        stacks = stack[:, np.newaxis]
        win_start = intervals[0]
        for win_end in intervals[1:]:
            win_data = data[:, win_start:win_end]
            stack = np.mean(win_data, axis=1)
            stacks = np.append(stacks, stack[:, np.newaxis], axis=1)
            win_start = win_end

    return stacks
