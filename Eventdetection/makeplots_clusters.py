"""
Plot the peaks saved in files made by running peaks_and_detection_significance.py

This module plots detection significance at various levels of compression against
that of uncompressed data. This can be ran for compessed data and 
peaks_and_detection_significance.py must have been ran for both compressed and 
uncompressed data and output save in the same file.

"""
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

event_id = sys.argv[1]
compression_type = sys.argv[2]
start_time = np.datetime64("2016-03-13T00:00:18")

peaks_location = Path("/u/st/by/aissah/scratch/event_detection/peaks")
save_location = Path("/u/st/by/aissah/scratch/event_detection/figures")

if compression_type == "N/A":
    peak_file = peaks_location / ("peaks_" + str(event_id))
    compression_type = "uncompressed"
else:
    peak_file_original = peaks_location / ("peaks_" + str(event_id))

    all_files = os.listdir(peaks_location)
    for a in all_files:
        if str(event_id) in a and compression_type in a:
            peak_file = peaks_location / a

with open(peak_file, "rb") as f:
    (
        all_peaks,
        all_peak_locations,
        all_detection_significances,
        data_time,
        all_compression_rates,
    ) = pickle.load(f)

with open(peak_file_original, "rb") as f:
    (
        all_peaks_original,
        all_peak_locations_original,
        all_detection_significances_original,
        data_time_original,
        all_compression_rates_original,
    ) = pickle.load(f)

if all_compression_rates is not None:
    average_compression_rates = np.mean(all_compression_rates, axis=0)
    print(average_compression_rates, flush=True)

# peak amplitudes against peak lag
for a in data_time_original:
    original = a
original = "original_data"
range_threshold = np.timedelta64(1, "s")  # 0.1
i = 0

all_keys = all_peaks.keys()
if compression_type == "zfp":
    all_keys = reversed(list(all_keys))
    average_compression_rates = np.flip(average_compression_rates)


for a in all_keys:  # all_peaks:
    peaks = all_peaks[a]
    peak_lags = all_peak_locations[a]

    peak_times = data_time[a]

    abs_diff = np.abs(
        (
            np.array(data_time_original[original])[:, np.newaxis] - np.array(peak_times)
        ).astype("timedelta64[ms]")
    )  # total_seconds())
    abs_diff1 = np.min(abs_diff, axis=1)
    indices_comp = np.argmin(abs_diff, axis=1)
    print(indices_comp)
    indices_uniq, order = np.unique(indices_comp, return_index=True)

    indices = [c for c in range(len(abs_diff1)) if abs_diff1[c] <= range_threshold]
    abs_diff1 = np.min(abs_diff, axis=0)
    indices_comp = [
        c for c in indices_comp[sorted(order)] if abs_diff1[c] <= range_threshold
    ]

    print(indices_comp)

    if all_compression_rates is None:
        label = a
    else:
        label = (
            str(int(average_compression_rates[i]))
            + "x ("
            + str(len(peaks))
            + " events)"
        )
        label = (
            str(int(average_compression_rates[i]))
            + "x ("
            + str(len(indices_comp))
            + " events)"
        )
    corresponding_orig = [all_peaks_original[original][i] for i in indices]
    peaks_plot = [peaks[i] for i in indices_comp]
    print(len(corresponding_orig))
    print(len(peaks_plot))
    if i == 0:
        label_orig = "1x (" + str(len(all_peaks_original[original])) + " events)"
        plt.scatter(
            all_peaks_original[original], all_peaks_original[original], label=label_orig
        )
    plt.scatter(
        corresponding_orig, peaks_plot, label=label
    )  # a + " (" + str(len(peaks)) + " events)")
    i += 1

fsize = 15
plt.xlabel("Amplitude (Original)", fontsize=fsize)
plt.ylabel("Amplitude (Compressed)", fontsize=fsize)
plt.title(
    "Peak amplitudes variation with " + compression_type + " compression",
    fontsize=fsize,
)
plt.yticks(fontsize=fsize)
plt.xticks(fontsize=fsize)  # , rotation=45)
plt.legend(fontsize=fsize)

savefile_name = save_location / (
    "Amplitudes_var" + str(event_id) + "_" + compression_type + ".png"
)
plt.savefig(savefile_name)

# detection significance against peak lag
fig, ax = plt.subplots()
i = 0

all_keys = all_peaks.keys()
if compression_type == "zfp":
    all_keys = reversed(list(all_keys))

for a in all_keys:  # all_peaks:
    detection_sigs = all_detection_significances[a]
    peak_lags = all_peak_locations[a]

    peak_times = data_time[a]

    abs_diff = np.abs(
        (
            np.array(data_time_original[original])[:, np.newaxis] - np.array(peak_times)
        ).astype("timedelta64[ms]")
    )  # total_seconds())
    # print(np.min(abs_diff, axis=0))
    abs_diff1 = np.min(abs_diff, axis=1)
    indices_comp = np.argmin(abs_diff, axis=1)
    indices_uniq, order = np.unique(indices_comp, return_index=True)

    indices = [c for c in range(len(abs_diff1)) if abs_diff1[c] <= range_threshold]
    abs_diff1 = np.min(abs_diff, axis=0)
    indices_comp = [
        c for c in indices_comp[sorted(order)] if abs_diff1[c] <= range_threshold
    ]
    # indices_comp = [c for c in range(len(abs_diff1)) if abs_diff1[c] <= range_threshold]

    if all_compression_rates is None:
        label = a
    else:
        label = (
            str(int(average_compression_rates[i]))
            + "x ("
            + str(len(indices_comp))
            + " events)"
        )
    corresponding_orig = [
        all_detection_significances_original[original][i] for i in indices
    ]
    detection_sigs_plot = [detection_sigs[i] for i in indices_comp]
    if i == 0:
        label_orig = (
            "1x ("
            + str(len(all_detection_significances_original[original]))
            + " events)"
        )
        plt.scatter(
            all_detection_significances_original[original],
            all_detection_significances_original[original],
            label=label_orig,
        )
    plt.scatter(
        corresponding_orig, detection_sigs_plot, label=label
    )  # a + " (" + str(len(peaks)) + " events)")
    i += 1

fsize = 15
plt.xlabel("Detection significance (Original)", fontsize=fsize)
plt.ylabel("Detection significance (Compressed)", fontsize=fsize)
plt.title(
    "Detection significance variation with " + compression_type + " compression",
    fontsize=fsize,
)
plt.yticks(fontsize=fsize)
plt.xticks(fontsize=fsize)  # , rotation=45)
plt.legend(fontsize=fsize)
savefile_name = save_location / (
    "detection_sigs_var_" + str(event_id) + "_" + compression_type + ".png"
)
plt.savefig(savefile_name)
