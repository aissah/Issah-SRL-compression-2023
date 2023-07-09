"""
Created on Fri Oct  7 11:52:51 2022

@author: issah
"""

import h5py
import numpy as np
import pywt
import scipy.signal as ss
import zfpy



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
                while (
                    detection_significance[a] < threshold
                ):  # if below threshold, iterate until you get to an event
                    a += 1
                # now you should be at a potential event
                while (
                    trace[a] > trace[a - 1]
                ):  # track number of time samples increasing in significance
                    neighbors.append(trace[a])
                    neighbor_locations.append(a)
                    a += 1
                # if first peak event or long time since last event, record this event
                if len(peaks) == 0 or a - peak_locations[-1] > min_distance_btn_peaks:
                    peaks.append(trace[a - 1])
                    peak_locations.append(a - 1)
                    peak_dectection_significance.append(detection_significance[a - 1])
                    flag = 1
                # if peak is with minimum distance of another peak, only take the bigger peak
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
    if whole_trace == "yes":  # use whole trace for detection significance
        dectection_significance = (trace - median) / median_absolute_dev
        return dectection_significance
    else:  # just use indices of peak_locations for detection significance
        detection_significance = []
        for a in peak_locations:
            detection_significance.append((trace[a] - median) / median_absolute_dev)
        return np.array(detection_significance)


def brady_preprocess(data):
    """Do preprocessing done in original Li and Zhan i.e. remove laser drift"""
    nSamples = np.shape(data)[1]
    # get rid of laser drift
    med = np.median(data, axis=0)
    # med = np.mean(data, axis=0)
    for i in range(nSamples):
        data[:, i] = data[:, i] - med[i]

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
