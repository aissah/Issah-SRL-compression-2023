"""
Created on Mon Aug  8 11:40:15 2022

@author: issah
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pywt
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
        data = np.float32(dataset)
    if normalize == "yes":
        max_of_rows = abs(data[:, :]).max(axis=1)
        data = data / max_of_rows[:, np.newaxis]
    return data


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

        max_of_rows = abs(data[:, :]).sum(axis=1)
        data = data / max_of_rows[:, np.newaxis]
    return data, timestamp_arr


def accuracyTest_zfp(data, mode):
    """
    Calculates frobenius norm of noise introduced by compression and various
    levels (hardwired) of compression.

    Parameters
    ----------
    data : 2-dimensional numpy array
        Data to be compressed. (channels by time samples)
    mode : string
        tolerance, precision or bitrate to indicate the mode of zfp compression
        to use.

    Returns
    -------
    errors : list
        List of norm of errors atintroduced at each level of compression.
    compressionfactors : list
        Compression factor at each level of compression.

    """
    compressed_data_lossless = zfpy.compress_numpy(data)
    lossless_size = len(compressed_data_lossless)
    decompressed_array_lossless = zfpy.decompress_numpy(compressed_data_lossless)
    lossless_error = np.linalg.norm(
        data - decompressed_array_lossless
    ) / np.linalg.norm(data)
    datasize = data.nbytes
    datanorm = np.linalg.norm(data)
    sizes = []
    errors = []
    if mode == "tolerance":
        tolerances = np.logspace(3, -5, 20)
        for tol in tolerances:
            compressed_data_lossy = zfpy.compress_numpy(data, tolerance=tol)
            size = len(compressed_data_lossy)
            decompressed_array_lossy = zfpy.decompress_numpy(compressed_data_lossy)
            error = np.linalg.norm(data - decompressed_array_lossy) / datanorm
            sizes.append(size)
            errors.append(error)
        sizes.append(lossless_size)
        errors.append(lossless_error)
        compressionfactors = [datasize / a for a in sizes]
    elif mode == "precision":
        precisions = np.linspace(3, 16, 14)
        for prec in precisions:
            compressed_data_lossy = zfpy.compress_numpy(data, precision=prec)
            size = len(compressed_data_lossy)
            decompressed_array_lossy = zfpy.decompress_numpy(compressed_data_lossy)
            error = np.linalg.norm(data - decompressed_array_lossy) / datanorm
            sizes.append(size)
            errors.append(error)
        sizes.append(lossless_size)
        errors.append(lossless_error)
        compressionfactors = [datasize / a for a in sizes]

    elif mode == "bitrate":
        bitrates = np.linspace(1, 16, 16)
        for bitrate in bitrates:
            compressed_data_lossy = zfpy.compress_numpy(data, rate=bitrate)
            size = len(compressed_data_lossy)
            decompressed_array_lossy = zfpy.decompress_numpy(compressed_data_lossy)
            error = np.linalg.norm(data - decompressed_array_lossy) / datanorm
            sizes.append(size)
            errors.append(error)
        sizes.append(lossless_size)
        errors.append(lossless_error)
        compressionfactors = [datasize / a for a in sizes]
    return errors, compressionfactors


def accracyTest_wavelet(data, mode, threshold_percentiles=list(range(5, 95, 5))):
    """
    Calculates frobenius norm of noise introduced by compression and various
    levels of compression.

    Parameters
    ----------
    data : 2-dimensional numpy array
        Data to be compressed. (channels by time samples)
    mode : string
        1d or 2d to indicate 1d or 2d wavelet compression respectively.
    threshold_percentiles : int/float, optional
        percentiles of coefficients to set to zero. The default is list(range(5, 95, 5)).

    Returns
    -------
    errors : list
        List of norm of errors atintroduced at each level of compression.
    threshold_percentiles : list
        Same as the input threshold percentiles for uniformity with zfp case.

    """
    level = 5
    errors = []
    datanorm = np.linalg.norm(data)
    if mode == "1d":
        coeffs = pywt.wavedec(data, "db5", level=level)
        for threshold in threshold_percentiles:
            thresheld_coeffs = soft_threshold(coeffs, threshold, mode="1d")
            decompresseddata = pywt.waverec(thresheld_coeffs, "db5")
            # decompresseddata = soft_comp_decomp1d(data, lvl=5, comp_ratio=threshold)
            error = np.linalg.norm(data - decompresseddata) / datanorm
            errors.append(error)
    elif mode == "2d":
        coeffs = pywt.wavedec2(data, "db5", level=level)
        for threshold in threshold_percentiles:
            thresheld_coeffs = soft_threshold(coeffs, threshold, mode="2d")
            decompresseddata = pywt.waverec2(thresheld_coeffs, "db5")
            # decompresseddata = soft_comp_decomp2d(data, lvl=5, comp_ratio=threshold)
            error = np.linalg.norm(data - decompresseddata) / datanorm
            errors.append(error)
    return errors, threshold_percentiles


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
            thresholds = np.percentile(abs(all_coeffs), threshold_percentile, axis = 1)
            thresheld_coeffs = [np.sign(level_coeffs) * np.maximum(np.abs(level_coeffs) - thresholds[:,np.newaxis], 0) for level_coeffs in wavelet_coeffs]
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
            thresheld_coeffs = [np.sign(level_coeffs) * np.maximum(np.abs(level_coeffs) - threshold, 0) for level_coeffs in wavelet_coeffs]
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

    rows, columns = data.shape
    approxRank = int((rows * columns) / (compFactor * (rows + columns)))
    U, S, Vt = la.svd(data)
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
    reconstructed_data : TYPE
        DESCRIPTION.

    """
    flag = 1
    ncols = len(data[0])
    partition_start = 0
    partition_end = min(min_ncols, ncols)
    if (partition_end + min_ncols) > ncols:
        reconstructed_data = compDecompSVD(data[:, partition_start:], compFactor)
        flag = 0
    else:
        reconstructed_data = compDecompSVD(
            data[:, partition_start:partition_end], compFactor
        )
    while flag == 1:
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


def normalisedErrorsSVD(data, compFactors):
    """
    Calculates frobenius norm of noise introduced by compression and various
    levels of compression.

    Parameters
    ----------
    data : 2-dimensional numpy array
        Data to be compressed.
    compFactors : list
        Compression factors to evaluate norm of errors at.

    Returns
    -------
    normalisedErrors : list
        List of norm of errors at introduced at each level of compression.

    """
    import scipy.linalg as la

    rows, columns = data.shape
    U, S, Vt = la.svd(data)
    datanorm = np.linalg.norm(data)
    normalisedErrors = []
    for cf in compFactors:
        approxRank = int((rows * columns) / (cf * (rows + columns)))
        sv = np.dot(np.diag(S[:approxRank]), Vt[:approxRank, :])
        recon = np.dot(U[:, :approxRank], sv)
        normalisedErrors.append(np.linalg.norm(data - recon) / datanorm)
    return normalisedErrors


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


def normalised_errors_SVD(data, compFactors, mode="randomized"):
    """
    Calculates frobenius norm of noise introduced by compression and various
    levels of compression.

    Parameters
    ----------
    data : 2-dimensional numpy array
        Data to be compressed. (channels by time samples)
    compFactors : list
        Compression factors to evaluate the norm of noise at.
    mode : string, optional
        randomized to indicate randomized svd or regular svd otherwise. The
        default is "randomized".

    Returns
    -------
    normalisedErrors : list
        List of norm of errors at introduced at each level of compression.

    """
    import scipy.linalg as la
    from sklearn.utils.extmath import randomized_svd

    rows, columns = data.shape
    approx_ranks = [
        int((rows * columns) / (compFactor * (rows + columns)))
        for compFactor in compFactors
    ]
    datanorm = np.linalg.norm(data)
    normalisedErrors = []
    if mode == "randomized":
        U, S, Vt = randomized_svd(data, n_components=5 + max(approx_ranks))
        for r in approx_ranks:
            # recon = randomized_SVD_comp_decomp(data, cf)
            recon = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
            normalisedErrors.append(np.linalg.norm(data - recon) / datanorm)
    else:
        rows, columns = data.shape
        U, S, Vt = la.svd(data)

        for cf in compFactors:
            approxRank = int((rows * columns) / (cf * (rows + columns)))
            sv = np.dot(np.diag(S[:approxRank]), Vt[:approxRank, :])
            recon = np.dot(U[:, :approxRank], sv)
            normalisedErrors.append(np.linalg.norm(data - recon) / datanorm)
    return normalisedErrors


def plotfill_stats(data, middle="average", x_data=None):
    if middle == "average":
        stats = np.percentile(data, [0, 10, 50, 90, 100], axis=0)
        middle_data = np.mean(data, axis=0)
    elif middle == "median":
        stats = np.percentile(data, [0, 10, 50, 90, 100], axis=0)
        middle_data = stats[2]
    if x_data is None:
        x_data = list(range(len(data[1])))
    fig = plt.figure()
    plt.plot(x_data, middle_data, "r-o", alpha=0.7, label=middle)
    plt.plot(x_data, stats[1], "b--", alpha=0.3, label="10th percentile")
    plt.plot(x_data, stats[3], "b--", alpha=0.3, label="90th percentile")

    plt.fill_between(x_data, middle_data, stats[1], color="blue", alpha=0.5)
    plt.fill_between(x_data, middle_data, stats[3], color="blue", alpha=0.5)
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('STALTA statistics over channels')
    plt.legend()
    return fig
