"""
For getting the peaks and subsequently calculating the detection significance
and classifying peaks as detections or not

template_matching.py must have been ran before this.

Takes as input event_id, threshold and compression_type and searches through 
the directory "save_location" for files with both event_id and compression_type
in the name.
compression_type should be N/A if the files correspond to uncompressed data 

For each file, we get the peaks all_peaks) with detection significance above 
threshold, location of peaks in relation to  the particular data in the file's
index (all_peak_locations), detection significance of the events
(all_detection_significances) and the start times of the file the event was 
found in (data_times).

Each of these output variables are dictionaries with keys "original_data" if it
is for uncompressed data or "compression_rate + .." for every .. compression rate 


Created on Wed Feb  8 12:18:57 2023

@author: issah
"""
import datetime
import os
import pickle
import sys
from pathlib import Path

import numpy as np

import eventDTFuncs

# location of files produced by runnig template_matching.py
cc_location = Path("/u/st/by/aissah/scratch/event_detection/template_matching")
save_location = Path("/u/st/by/aissah/scratch/event_detection/peaks")
event_id = sys.argv[1]
compression_type = sys.argv[2]  # N/A if original data
threshold = int(sys.argv[3])

all_files = os.listdir(cc_location)
all_files.sort()
data_files = []

if compression_type == "N/A":
    for a in all_files:
        if event_id in a and "uncompressed" in a:
            data_files.append(a)
    all_peaks = {"original_data": []}
    all_peak_locations = {"original_data": []}
    all_detection_significances = {"original_data": []}
    data_time = {"original_data": []}
else:
    for a in all_files:
        if event_id in a and compression_type in a:
            data_files.append(a)
    all_peaks = {}
    all_peak_locations = {}
    all_detection_significances = {}
    data_time = {}

data_time["detection_time"] = []
all_compression_rates = None

for a in data_files:
    with open(cc_location / a, "rb") as f:
        (mean_ccs_acrossfiles, metadata) = pickle.load(f)
        data_time["start_lag"] = metadata["start_lag"]
        data_time["start_time"] = metadata[
            "start_time"
        ]  # np.datetime64(first_file_time) - np.timedelta64(6, "s")
        # print(data_time["start_time"], flush=True)
    if mean_ccs_acrossfiles.ndim == 1:
        (
            peaks,
            peak_locations,
            detection_significances,
            peak_neighbors,
            peak_neighbors_locations,
        ) = eventDTFuncs.get_peaks(mean_ccs_acrossfiles, threshold=threshold)
        # data_time["detection_time"].extend([data_time["start_time"] + np.timedelta64(int(b/1000), "s") for b in peak_locations])
        data_time["detection_time"].extend(
            [
                data_time["start_time"] + datetime.timedelta(milliseconds=b)
                for b in peak_locations
            ]
        )
        peak_neighbors_lags = []
        for b in peak_neighbors_locations:
            peak_neighbors_lags.append([c + metadata["start_lag"] for c in b])
        all_peaks["original_data"].extend(peaks)
        all_peak_locations["original_data"].extend(peak_locations)
        all_detection_significances["original_data"].extend(detection_significances)
        data_time["original_data"].extend(
            [
                data_time["start_time"] + datetime.timedelta(milliseconds=b)
                for b in peak_locations
            ]
        )
        # What is this next line doing for me?
        peak_locations = [b + metadata["start_lag"] for b in peak_locations]
        # data_time["original_data"].extend(len(peaks) * [metadata["files"][0]])
    else:
        peaks = []
        peak_locations = []
        detection_significances = []
        i = 0
        if all_compression_rates is None:
            all_compression_rates = np.array(metadata["compression_rates"])
        else:
            all_compression_rates = np.append(
                all_compression_rates, np.array(metadata["compression_rates"]), axis=0
            )
        for b in mean_ccs_acrossfiles:
            b = np.nan_to_num(b, nan=0.0)
            if np.isnan(np.mean(b)):
                print(np.mean(b))
                print(np.max(b), flush=True)
                print(np.sum(b), flush=True)
                nan_count = np.isnan(b).sum()
                print(a, "\n with ", nan_count, "nans", " out of ", len(b))
            else:
                (
                    peaks,
                    peak_locations,
                    detection_significances,
                    peak_neighbors,
                    peak_neighbors_locations,
                ) = eventDTFuncs.get_peaks(
                    b, threshold=threshold
                )  # section_and_get_peaks(b, threshold=threshold)
                # peak_locations = [b + metadata["start_lag"] for b in peak_locations]
                all_peaks.setdefault(
                    "compression_rate" + str(i),
                    [],  # str(int(metadata["compression_rates"][0][i])), []
                ).extend(peaks)
                all_peak_locations.setdefault(
                    "compression_rate" + str(i),
                    [],  # str(int(metadata["compression_rates"][0][i])), []
                ).extend(peak_locations)
                all_detection_significances.setdefault(
                    "compression_rate" + str(i),
                    [],  # str(int(metadata["compression_rates"][0][i])), []
                ).extend(detection_significances)
                data_time.setdefault(
                    "compression_rate" + str(i),
                    [],  # str(int(metadata["compression_rates"][0][i])), []
                ).extend(
                    [
                        data_time["start_time"] + datetime.timedelta(milliseconds=b)
                        for b in peak_locations
                    ]
                )
            i += 1

compression_levels = metadata["compression_rates"]

if metadata["compression_type"] == "N/A":
    savefile_name = save_location / ("peaks_" + str(event_id))
else:
    savefile_name = save_location / (
        "peaks_"
        + str(event_id)
        + "_"
        + compression_type
        + "_"
        + "_".join([str(int(a)) for a in metadata["compression_rates"][0][:4]])
    )

with open(savefile_name, "wb") as f:
    pickle.dump(
        [
            all_peaks,
            all_peak_locations,
            all_detection_significances,
            data_time,
            all_compression_rates,
        ],
        f,
    )