"""
Created on Mon Aug  8 11:40:15 2022

@author: issah
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pywt
import zfpy

from general_funcs import soft_threshold

def accuracyTest_zfp(data, mode):
    """
    Calculates Frobenius norm of noise introduced by compression and various
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
    # Lossless compression check
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
        tolerances = np.logspace(
            3, -5, 20
        )  # multiple tolerance factors to check relative error
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
        precisions = np.linspace(
            3, 16, 14
        )  # multiple precision levels to check relative error
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
        bitrates = np.linspace(
            1, 16, 16
        )  # multiple bitrate levels to check relative errors
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


def accuracyTest_wavelet(data, mode, threshold_percentiles=list(range(5, 95, 5))):
    """
    Calculates Frobenius norm of noise introduced by compression and various
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
            # sparse wavelet compressed version at threshold, then check reconstruction error
            thresheld_coeffs = soft_threshold(coeffs, threshold, mode="1d")
            decompresseddata = pywt.waverec(thresheld_coeffs, "db5")
            # decompresseddata = soft_comp_decomp1d(data, lvl=5, comp_ratio=threshold)
            error = np.linalg.norm(data - decompresseddata) / datanorm
            errors.append(error)
    elif mode == "2d":
        coeffs = pywt.wavedec2(data, "db5", level=level)
        for threshold in threshold_percentiles:
            # sparse wavelet compressed version at threshold, then check reconstruction error
            thresheld_coeffs = soft_threshold(coeffs, threshold, mode="2d")
            decompresseddata = pywt.waverec2(thresheld_coeffs, "db5")
            # decompresseddata = soft_comp_decomp2d(data, lvl=5, comp_ratio=threshold)
            error = np.linalg.norm(data - decompresseddata) / datanorm
            errors.append(error)
    return errors, threshold_percentiles


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

    # approximate rank from dimension and desired compression factor
    rows, columns = data.shape
    approx_ranks = [
        int((rows * columns) / (compFactor * (rows + columns)))
        for compFactor in compFactors
    ]

    datanorm = np.linalg.norm(data)
    normalisedErrors = []
    if mode == "randomized":  # randomized approximate SVD
        U, S, Vt = randomized_svd(data, n_components=5 + max(approx_ranks))
        for r in approx_ranks:
            # recon = randomized_SVD_comp_decomp(data, cf)
            recon = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
            normalisedErrors.append(np.linalg.norm(data - recon) / datanorm)
    else:  # true low rank SVD
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
