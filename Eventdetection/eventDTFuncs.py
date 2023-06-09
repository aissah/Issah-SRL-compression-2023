"""
Created on Fri Oct  7 11:52:51 2022

@author: issah
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
        data. Timestamps for brady hotspring data are with respect to the
        beginning time of the survey.

    """

    with h5py.File(file, "r") as open_file:
        dataset = open_file["raw"]
        time = open_file["timestamp"]
        data = np.array(dataset)
        timestamp_arr = np.array(time)
    if normalize == "yes":
        nSamples = np.shape(data)[1]
        # get rid of laser drift
        # med = np.median(data, axis=0)
        med = np.mean(data, axis=0)
        for i in range(nSamples):
            data[:, i] = data[:, i] - med[i]

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
        data.

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
        # med = np.mean(data, axis=0)
        for i in range(nSamples):
            data[:, i] = data[:, i] - med[i]

        # max_of_rows = abs(data[:, :]).sum(axis=1)
        # data = data / max_of_rows[:, np.newaxis]
    return data, timestamp_arr


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
            # thresheld_coeffs = []
            # for level_coeffs in wavelet_coeffs:
            #     thresholds = np.percentile(
            #         abs(level_coeffs), threshold_percentile, axis=1
            #     )
            #     thresheld_coeffs.append(
            #         np.sign(level_coeffs)
            #         * np.maximum(np.abs(level_coeffs) - thresholds[:, np.newaxis], 0)
            #     )

            # threshold just channels separately
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
            # threshold each level and channel separately
            # thresheld_coeffs = []
            # for level_coeffs in wavelet_coeffs:
            #     threshold = np.percentile(abs(level_coeffs), threshold_percentile)
            #     thresheld_coeffs.append(
            #         np.sign(level_coeffs)
            #         * np.maximum(np.abs(level_coeffs) - threshold, 0)
            #     )

            # threshold just channels separately
            all_coeffs = wavelet_coeffs[0]
            for a in range(1, len(wavelet_coeffs)):
                all_coeffs = np.append(all_coeffs, wavelet_coeffs[a])
            threshold = np.percentile(abs(all_coeffs), threshold_percentile)
            thresheld_coeffs = [
                np.sign(level_coeffs) * np.maximum(np.abs(level_coeffs) - threshold, 0)
                for level_coeffs in wavelet_coeffs
            ]
    elif mode == "2d":
        allcoeffs = wavelet_coeffs[0]
        for b in wavelet_coeffs[1:]:
            allcoeffs = np.append(allcoeffs, np.ravel(np.concatenate(b)))

        threshold = np.percentile(abs(allcoeffs), threshold_percentile)
        thresheld_coeffs = [
            np.sign(wavelet_coeffs[0])
            * np.maximum(np.abs(wavelet_coeffs[0]) - threshold, 0)
        ]

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


def randomized_SVD_comp_decomp(data, compression_factor):
    """
    Compress data with svd and return reconstructed data.

    Parameters
    ----------
    data : 2-dimensional numpy array
        Data to compress
    compression_factor : float/int
        compression factor to compress data to

    Returns
    -------
    recon : 2-dimensional numpy array
        Reconstructed data after compressing to a factor of compression_factor
    compression_factor : float/int
        Same as input compression_factor

    """

    from sklearn.utils.extmath import randomized_svd

    rows, columns = data.shape
    approxRank = int((rows * columns) / (compression_factor * (rows + columns)))
    # U, S, Vt = la.svd(data)
    U, S, Vt = randomized_svd(data, n_components=approxRank)
    recon = U @ np.diag(S) @ Vt
    # sv = np.dot(np.diag(S), Vt)
    # recon = np.dot(U, sv)
    return recon, compression_factor


def get_peaks(trace, threshold=9, min_distance_btn_peaks=3000):
    """
    Get peaks with detection significance above a threshold.

    Parameters
    ----------
    trace : 1D numpy array
        Stacked cross-correlation across channels. Could be used for other data
        with similar requirements at own discretion.
    threshold : float/int, optional
        This code calculates the detection significance and finds peaks which
        have detection significance above the detection siginificance.
        The default is 9 (based on some literature).
    max_distance_btn_peaks : int, optional
        Minimum number of indices between peaks. This helps avoid picking
        rise in amplitude leading to peak as separate events. The default is 3000.

    Returns
    -------
    peaks : list
        Amplitude at the various peaks.
    peak_locations : list
        Indices of the peaks detected.
    peak_dectection_significance : list
        Detection significance of the various peaks.
    peak_neighbors : list of list
        Amplitudes of the area around the peaks.
    peak_neighbors_locations : list of list
        corresponding indices of the neighborhood of the peaks.

    """

    peaks = []
    peak_locations = []
    peak_dectection_significance = []
    peak_neighbors = []
    peak_neighbors_locations = []
    detection_significance = find_detection_significance(trace)
    a = 1
    while a < len(trace):
        if trace[a] < trace[a - 1]:
            a += 1
        else:
            neighbors = []
            neighbor_locations = []
            try:
                while detection_significance[a] < threshold: # if below threshold, iterate until you get to an event
                    a += 1
                # now you should be at a potential event
                while trace[a] > trace[a - 1]: # track number of time samples increasing in significance
                    neighbors.append(trace[a]) 
                    neighbor_locations.append(a)
                    a += 1
                # if first peak event or long time since last event, record this event
                if len(peaks) == 0 or a - peak_locations[-1] > min_distance_btn_peaks:
                    peaks.append(trace[a - 1])
                    peak_locations.append(a - 1)
                    peak_dectection_significance.append(detection_significance[a - 1])
                    flag = 1
                elif peaks[-1] < trace[a - 1]:
                    peaks[-1] = trace[a - 1]
                    peak_locations[-1] = a - 1
                    peak_dectection_significance[-1] = detection_significance[a - 1]
                    peak_neighbors = peak_neighbors[:-1]
                    peak_neighbors_locations = peak_neighbors_locations[:-1]
                    flag = 1
                else:
                    flag = 0
            except IndexError:
                return (
                    peaks,
                    peak_locations,
                    peak_dectection_significance,
                    peak_neighbors,
                    peak_neighbors_locations,
                )
            if flag == 1:
                try:
                    while (
                        detection_significance[a] > threshold
                        and trace[a] < trace[a - 1]
                    ):
                        neighbors.append(trace[a])
                        neighbor_locations.append(a)
                        a += 1
                    peak_neighbors.append(neighbors)
                    peak_neighbors_locations.append(neighbor_locations)
                except IndexError:
                    peak_neighbors.append(neighbors)
                    peak_neighbors_locations.append(neighbor_locations)
                    return (
                        peaks,
                        peak_locations,
                        peak_dectection_significance,
                        peak_neighbors,
                        peak_neighbors_locations,
                    )
    return (
        peaks,
        peak_locations,
        peak_dectection_significance,
        peak_neighbors,
        peak_neighbors_locations,
    )


def find_detection_significance(trace, peak_locations=[], whole_trace="yes"):
    """
    Find detection significance of either some points (peak_locations) or
    all points on a trace.

    Parameters
    ----------
    trace : numpy array
        Made for stacked cross correlation across channels but could be used
        for any appropriate siesmic trace
    peak_locations : list, optional
        The indices of locations in trace to get the detection significance of.
        The default is [].
    whole_trace : str, optional
        Indicate whether to calculate the detection significance of the whole
        trace or not. If no, a list of locations(indices) must be given in
        peak_locations. The default is "yes".

    Returns
    -------
    numpy array
        Detection significance in of the peaks in the locations provided or
        of every entry of the trace if whole_trace is yes.

    """

    median = np.median(trace)
    median_absolute_dev = np.median(abs(trace - median))
    if whole_trace == "yes": # use whole trace for detection significance
        dectection_significance = (trace - median) / median_absolute_dev
        return dectection_significance
    else: # just use indices of peak_locations for detection significance
        detection_significance = []
        for a in peak_locations:
            detection_significance.append((trace[a] - median) / median_absolute_dev)
        return np.array(detection_significance)


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
    signal_squared = signal**2 # 
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


def brady_preprocess(data):
    """Do preprocessing done in original Li and Zhan i.e. remove laser drift"""
    nSamples = np.shape(data)[1]
    # get rid of laser drift
    med = np.median(data, axis=0)
    # med = np.mean(data, axis=0)
    for i in range(nSamples):
        data[:, i] = data[:, i] - med[i]

    # mean_of_rows = np.mean(data, axis=1)
    # max_of_rows = abs(data).sum(axis=1)
    # max_of_rows = abs(data).max(axis=1)
    # data = data - (mean_of_rows[:, np.newaxis])

    return data


def section_and_get_peaks(
    trace, section_length=300000, threshold=9, min_distance_btn_peaks=3000
):
    """
    divide data into sections and find peaks using:
        get_peaks(trace, threshold=9, min_distance_btn_peaks=3000)

    Parameters
    ----------
    trace : 1D numpy array
        Stacked cross-correlation across channels. Could be used for other data
        with similar requirements at own discretion.
    section_length : TYPE, optional
        Number of time samples to put in each section. The default is 300000.
    threshold : float/int, optional
        This code calculates the detection significance and finds peaks which
        have detection significance above the detection siginificance.
        The default is 9 (based on some literature).
    max_distance_btn_peaks : int, optional
        Minimum number of indices between peaks. This helps avoid picking
        rise in amplitude leading to peak as separate events. The default is 3000.

    Returns
    -------
    all_peaks : list
        Amplitude at the various peaks.
    all_peak_locations : list
        Indices of the peaks detected.
    all_peak_dectection_significance : list
        Detection significance of the various peaks.
    all_peak_neighbors : list of list
        Amplitudes of the area around the peaks.
    all_peak_neighbors_locations : list of list
        corresponding indices of the neighborhood of the peaks.

    """
    trace_length = len(trace)
    section_start = 0
    section_end = min(section_length, trace_length)
    (
        all_peaks,
        all_peak_locations,
        all_detection_significances,
        all_peak_neighbors,
        all_peak_neighbors_locations,
    ) = get_peaks(trace[section_start:section_end], threshold=threshold)
    while section_end != trace_length:
        section_start = section_end
        section_end = min(section_end + section_length, trace_length)
        (
            peaks,
            peak_locations,
            detection_significances,
            peak_neighbors,
            peak_neighbors_locations,
        ) = get_peaks(trace[section_start:section_end], threshold=threshold)
        peak_locations = [b + section_start for b in peak_locations]
        peak_neighbors_lags = []
        for b in peak_neighbors_locations:
            peak_neighbors_lags.append([c + section_start for c in b])
        all_peaks.extend(peaks)
        all_peak_locations.extend(peak_locations)
        all_detection_significances.extend(detection_significances)
        all_peak_neighbors.extend(peak_neighbors)
        all_peak_neighbors_locations.extend(peak_neighbors_lags)

    return (
        all_peaks,
        all_peak_locations,
        all_detection_significances,
        all_peak_neighbors,
        all_peak_neighbors_locations,
    )


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
