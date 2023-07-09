"""


"""
import h5py
import numpy as np
import pywt
import scipy.signal as ss
import zfpy


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
        # get rid of laser drift
        med = np.median(data, axis=0)
        for i in range(nSamples):
            data[:, i] = data[:, i] - med[i]
        # normalize each row by its L1 norm
        max_of_rows = abs(data[:, :]).sum(axis=1)
        data = data / max_of_rows[:, np.newaxis]
    return data, timestamp_arr


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
            # for 2-d data i.e. multiple channels
            # threshold channels separately
            all_coeffs = wavelet_coeffs[0]
            for a in range(1, len(wavelet_coeffs)):
                all_coeffs = np.append(all_coeffs, wavelet_coeffs[a], axis=1)
            thresholds = np.percentile(abs(all_coeffs), threshold_percentile, axis=1)
            thresheld_coeffs = [
                np.sign(level_coeffs)
                * np.maximum(np.abs(level_coeffs) - thresholds[:, np.newaxis], 0)
                for level_coeffs in wavelet_coeffs
            ]
        elif wavelet_coeffs[1].ndim == 1:
            # for a single time series. i.e. one channel
            # put all coefficients in a single array
            all_coeffs = wavelet_coeffs[0]
            for a in range(1, len(wavelet_coeffs)):
                all_coeffs = np.append(all_coeffs, wavelet_coeffs[a])
            # Calculate threshold as a percentile of all coefficients
            threshold = np.percentile(abs(all_coeffs), threshold_percentile)
            # Thresholf coefficients
            thresheld_coeffs = [
                np.sign(level_coeffs) * np.maximum(np.abs(level_coeffs) - threshold, 0)
                for level_coeffs in wavelet_coeffs
            ]
    elif mode == "2d":
        # cofficients from 2-d dimensional wavelet decomposition
        # put all coefficients in a single array
        allcoeffs = wavelet_coeffs[0]
        for b in wavelet_coeffs[1:]:
            allcoeffs = np.append(allcoeffs, np.ravel(np.concatenate(b)))
        # Calculate threshold as a percentile of all coefficients
        threshold = np.percentile(abs(allcoeffs), threshold_percentile)
        # Threshold coefficients. Start by handling approximation coefficients
        thresheld_coeffs = [
            np.sign(wavelet_coeffs[0])
            * np.maximum(np.abs(wavelet_coeffs[0]) - threshold, 0)
        ]
        # Threshold detail coefficients
        for level_coeffs in wavelet_coeffs[1:]:
            thresheld_level = tuple(
                [
                    np.sign(orient_coeffs)
                    * np.maximum(np.abs(orient_coeffs) - threshold, 0)
                    for orient_coeffs in level_coeffs
                ]
            )
            thresheld_coeffs.append(thresheld_level)

    return thresheld_coeffs


def compDecompSVD(data, compFactor):
    """
    Compress data with SVD by compression factor and return reconstructed data.

    Parameters
    ----------
    data : 2-dimensional numpy array
        Data to be compressed.(channels by time samples)
    compFactor : int/float
        Compression factor.

    Returns
    -------
    recon : 2-dimensional numpy array
        Reconstructed data after compression.

    """
    import scipy.linalg as la

    # factorization
    rows, columns = data.shape
    approxRank = int((rows * columns) / (compFactor * (rows + columns)))
    U, S, Vt = la.svd(data)
    # reconstruction with low rank approximation
    sv = np.dot(np.diag(S[:approxRank]), Vt[:approxRank, :])
    recon = np.dot(U[:, :approxRank], sv)
    return recon


def partition_compDecompSVD(data, compFactor, min_ncols):
    """
    Partition and compress data with SVD by compression factor and return
    reconstructed data. Partitioning avoids memory issues.

    Parameters
    ----------
    data : 2-dimensional numpy array
        Data to be compressed. (channels by time samples)
    compFactor : int/float
        Compression factor.
    min_ncols : int/float
        Number of columns to partition data into. Columns here corresponds to
        time samples.

    Returns
    -------
    reconstructed_data : 2-dimensional numpy array
        Reconstructed data after compression.

    """
    flag = 1
    ncols = len(data[0])
    partition_start = 0
    partition_end = min(min_ncols, ncols)
    if (partition_end + min_ncols) > ncols:  # just 1 patch of data
        reconstructed_data = compDecompSVD(data[:, partition_start:], compFactor)
        flag = 0
    else:  # grab a smaller subset of the data
        reconstructed_data = compDecompSVD(
            data[:, partition_start:partition_end], compFactor
        )
    while (
        flag == 1
    ):  # loop through smaller subsets of the data until all patches decomposed
        partition_start = partition_end
        partition_end = min(partition_end + min_ncols, ncols)
        if (partition_end + min_ncols) > ncols:
            reconstructed_data = np.append(
                reconstructed_data,
                compDecompSVD(data[:, partition_start:], compFactor),
                axis=1,
            )
            flag = 0
        else:
            reconstructed_data = np.append(
                reconstructed_data,
                compDecompSVD(data[:, partition_start:partition_end], compFactor),
                axis=1,
            )

    return reconstructed_data


def randomized_SVD_comp_decomp(data, compression_factor):
    """
    Compress data with randomized SVD by compression factor and return
    reconstructed data.

    Parameters
    ----------
    data : 2-dimensional numpy array
        Data to be compressed.
    compression_factor : int/float
        Compression factor.

    Returns
    -------
    recon : 2-dimensional numpy array
        Reconstructed data after compression.
    compression_factor : float/int
        Same as input compression_factor

    """
    from sklearn.utils.extmath import randomized_svd

    rows, columns = data.shape
    approxRank = int((rows * columns) / (compression_factor * (rows + columns)))
    # calculate randomized SVD and reconstruct
    U, S, Vt = randomized_svd(data, n_components=approxRank)
    recon = U @ np.diag(S) @ Vt

    return recon, compression_factor


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


def compressReconstruct_wavelets(
    data, mode="1D", wavelet="db5", lvl=5, compressionFactor=5, partition="yes"
):
    """

    Parameters
    ----------
    data : numpy array
        2 (space by time) dimensional data to be compressed
    mode : string, optional
        1D or 2D for 1d or 2d wavelet transform respectively. The default is "1D".
    wavelet : string, optional
        Type of wavelet to use. The default is "db5".
    lvl : int, optional
        level of multiscale decomposition. The default is 5.
    compressionFactor : float or int, optional
        This is the preferred (size of original data)/(size of compressed data)
        The default is 5.

    Returns
    -------
    reconstructedData : numpy array
        Compressed data after reconstruction

    """

    if partition == "yes":
        halfpoint = int(data.shape[1] / 2)
        return (
            np.append(
                compressReconstruct_wavelets(
                    data[:, :halfpoint],
                    mode=mode,
                    wavelet=wavelet,
                    lvl=lvl,
                    compressionFactor=compressionFactor,
                    partition="no",
                )[0],
                compressReconstruct_wavelets(
                    data[:, halfpoint:],
                    mode=mode,
                    wavelet=wavelet,
                    lvl=lvl,
                    compressionFactor=compressionFactor,
                    partition="no",
                )[0],
                axis=1,
            ),
            compressionFactor,
        )
    thresholdPercentile = 100 - (100 / compressionFactor)
    if mode == "1D":
        # nSamples = len(data[1])
        # reconstructedData = np.array([range(nSamples)])
        reconstructedData = np.array([])
        coeffs = pywt.wavedec(data, wavelet, level=lvl)
        thresheld_coeffs = soft_threshold(coeffs, thresholdPercentile, mode="1d")
        reconstructedData = pywt.waverec(thresheld_coeffs, wavelet)
        reconstructedData = reconstructedData[: len(data), : len(data[0])]

        return reconstructedData, compressionFactor

    elif mode == "2D":
        waveletCoefficients = pywt.wavedec2(data, wavelet, level=lvl)
        # thresheldCoefficients = thresholdWaveletCoefficients(
        #    waveletCoefficients, "2D", thresholdPercentile
        # )
        thresheldCoefficients = soft_threshold(
            waveletCoefficients, thresholdPercentile, mode="2d"
        )
        reconstructedData = pywt.waverec2(thresheldCoefficients, wavelet)
        reconstructedData = reconstructedData[: len(data), : len(data[0])]

        return reconstructedData, compressionFactor


def compressReconstruct_zfp(
    data,
    compressionFactor=None,
    mode=None,
    tolerance=None,
    precision=None,
    bitrate=None,
):
    """
    Compress data with zfp and return reconstructed data.

    Parameters
    ----------
    data : numpy array
        n dimensional data to be compressed
    compressionFactor : int or float, optional (when provided, all the other
                                            optional parameters are not needed)
        This is the preferred (size of original data)/(size of compressed data)
        The default is None.
    mode : string, optional
        four modes;lossless, tolerance, precision,bitrate. The default is None.
    tolerance : float, optional
        The maximum error that can be tolerated in the compressed data. eg 0.1.
        Should be provided if mode is tolerance. The default is None.
    precision : int, optional
        The amount of bitplanes to be encoded in compressed data. Should be
        provided if mode is precision. The default is None.
    bitrate : int, optional
        This amount of bits to assign to each number in the compressed data.
        Should be provided if mode is bitrate. The default is None.

    Returns
    -------
    reconstructedData : numpy array
        Compressed data after reconstruction
    compressionFactor : float
        Compression factor. This is modified sometimes even when provided to
        fit the method better.

    """
    if mode is None:
        dataSize = data.nbytes
        itemSize = dataSize / data.size
        bitrate = (itemSize * 8) / compressionFactor
        bitrate = round(bitrate)
        compressedData = zfpy.compress_numpy(data, rate=bitrate)
        compressedSize = len(compressedData)
        reconstructedData = zfpy.decompress_numpy(compressedData)
        dataSize = data.nbytes
        compressionFactor = dataSize / compressedSize
        return reconstructedData, compressionFactor
    elif mode == "lossless":
        compressedData = zfpy.compress_numpy(data)
        compressedSize = len(compressedData)
        reconstructedData = zfpy.decompress_numpy(compressedData)
        dataSize = data.nbytes
        compressionFactor = dataSize / compressedSize
        return reconstructedData, compressionFactor
    elif mode == "tolerance":
        compressedData = zfpy.compress_numpy(data, tolerance=tolerance)
        compressedSize = len(compressedData)
        reconstructedData = zfpy.decompress_numpy(compressedData)
        dataSize = data.nbytes
        compressionFactor = dataSize / compressedSize
        return reconstructedData, compressionFactor
    elif mode == "precision":
        compressedData = zfpy.compress_numpy(data, precision=precision)
        compressedSize = len(compressedData)
        reconstructedData = zfpy.decompress_numpy(compressedData)
        dataSize = data.nbytes
        compressionFactor = dataSize / compressedSize
        return reconstructedData, compressionFactor

    elif mode == "bitrate":
        compressedData = zfpy.compress_numpy(data, rate=bitrate)
        compressedSize = len(compressedData)
        reconstructedData = zfpy.decompress_numpy(compressedData)
        dataSize = data.nbytes
        compressionFactor = dataSize / compressedSize
        return reconstructedData, compressionFactor


def crosscorrelate_channels(signal, template, lagmax, stacked="yes"):
    """
    Do a channel by channel normalized cross-correlation of template with a
    signal up to a lag of lagmax

    Parameters
    ----------
    signal : 2-dimensional numpy array
        (channels by time_samples) array of signal.
    template : 2-dimensional numpy array
        (channels by time_samples) array of template. Number of channels should
        be the same as that of signal and number of time samples should be at
        least that of signal.
    lagmax : int
        Number of lags to do normalized cross-correlation at.
    stacked : string, optional
        Return the average across channels if yes and return a 2-dimensional
        array of (channels by cross-correlation) if no. The default is "yes".

    Returns
    -------
    cross_correlations : numpy array
        channel by channel cross-correlation(2d) or its stack(1-d).

    """
    # lag_zero = len(template[1]) - 1
    number_of_channels, len_template = template.shape
    cross_correlations = np.empty((number_of_channels, lagmax + 1))
    # parts of normalization factors
    template_autocc = np.sum(template * template, axis=1)
    signal_squared = signal**2  #
    template_ones = np.ones(len_template)
    for a in range(number_of_channels):
        one_channel_cc = ss.correlate(signal[a], template[a], mode="valid")
        signal_autocc = ss.correlate(signal_squared[a], template_ones, mode="valid")
        normalization = np.sqrt(template_autocc[a] * signal_autocc)
        # one_channel_cc = one_channel_cc[lag_zero : lag_zero + lagmax + 1]
        cross_correlations[a] = one_channel_cc / normalization
    if stacked == "yes":
        cross_correlations = np.mean(cross_correlations, axis=0)
    return cross_correlations


def thresholdWaveletCoefficients(waveletCoefficients, mode, thresholdPercentile):
    """
    Soft thresholding operator on wavelet_coeffs. Initial version.

    Parameters
    ----------
    waveletCoefficients : tuple
        wavelet coefficients obtained by using pywavelets to decompose data.
    mode : string
        2d or 1d to indicate 2d or 1d wavelet decomposition coefficients in
        wavelet_coeffs respectively. The default is "1d".
    threshold_percentile : int
        percentile of coefficients to set to zero.

    Returns
    -------
    thresheldCoefficients : tuple
        Coefficients after thrsholding.

    """
    if mode == "1D":
        for a in range(len(waveletCoefficients[1])):
            allcoeffs = waveletCoefficients[0][a].ravel()
            for b in range(1, len(waveletCoefficients)):
                allcoeffs = np.append(allcoeffs, waveletCoefficients[b][a].ravel())
            threshold = np.percentile(np.absolute(allcoeffs), thresholdPercentile)
            for b in range(len(waveletCoefficients)):
                c = waveletCoefficients[b][a]
                thresheldCoefficients = waveletCoefficients.copy()
                thresheldCoefficients[b][a][np.absolute(c) <= threshold] = 0
                thresheldCoefficients[b][a][c > 0] -= threshold
                thresheldCoefficients[b][a][c < 0] += threshold
        return thresheldCoefficients
    if mode == "2D":
        allcoeffs2d = waveletCoefficients[0].ravel()
        for a in range(1, len(waveletCoefficients)):
            for b in waveletCoefficients[a]:
                allcoeffs2d = np.append(allcoeffs2d, b.ravel())

        threshold = np.percentile(np.absolute(allcoeffs2d), thresholdPercentile)
        thresheldCoefficients = waveletCoefficients.copy()
        thresheldCoefficients[0][np.absolute(thresheldCoefficients[0]) <= threshold] = 0
        thresheldCoefficients[0][thresheldCoefficients[0] > threshold] -= threshold
        thresheldCoefficients[0][thresheldCoefficients[0] < threshold] += threshold
        for a in range(1, len(thresheldCoefficients)):
            for b in range(len(thresheldCoefficients[a])):
                c = thresheldCoefficients[a][b]
                thresheldCoefficients[a][b][np.absolute(c) <= threshold] = 0
                thresheldCoefficients[a][b][c > 0] -= threshold
                thresheldCoefficients[a][b][c < 0] += threshold
        return thresheldCoefficients


def rm_laserdrift(data):
    """
    Remove the effect of laser drift along the channels

    Parameters
    ----------
    data : 2-dimensional numpy array
        2-d array DAS data (channels by time samples)).

    Returns
    -------
    data : 2-dimensional numpy array
        2-d array DAS data (channels by time samples)) after removing effects
    of laser drift.

    """
    # get rid of laser drift
    med = np.median(data, axis=0)
    data = data - med[np.newaxis, :]

    return data


def normalize_data(data):
    """Normalize each channel of data (channel by time sample) by the sum at
    all time samples"""
    max_of_rows = abs(data).sum(axis=1)
    # max_of_rows = abs(data).max(axis=1)
    data = data / (max_of_rows[:, np.newaxis])

    return data


def preprocess(data):
    """Remove laser drift and normalize data by sum across time samples"""
    nSamples = np.shape(data)[1]
    # get rid of laser drift
    med = np.median(data, axis=0)
    # med = np.mean(data, axis=0)
    for i in range(nSamples):
        data[:, i] = data[:, i] - med[i]

    max_of_rows = abs(data).sum(axis=1)
    # max_of_rows = abs(data).max(axis=1)
    data = data / (max_of_rows[:, np.newaxis])

    return data


def frequency_filter(data, frequency_range, mode, order, sampling_frequency):
    """
    Butterworth filter of data.

    Parameters
    ----------
    data : array
        1d or 2d array.
    frequency_range : int/sequence
        int if mode is lowpass or high pass. Sequence of 2 frequencies if mode
        is bandpass
    mode : str
        lowpass, highpass or bandpass.
    order : int
        Order of the filter.
    sampling_frequency : int
        sampling frequency.

    Returns
    -------
    filtered_data : array
        Frequency filtered data.

    """

    from scipy.signal import butter, sosfiltfilt

    sos = butter(
        order, frequency_range, btype=mode, output="sos", fs=sampling_frequency
    )
    filtered_data = sosfiltfilt(sos, data)

    return filtered_data
