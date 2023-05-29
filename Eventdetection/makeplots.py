"""
Plot the peaks saved in files made by running peaks_and_detection_significance.py

This module plots peaks against time and detection significance against time 
showing the different events over time. For compressed data at different levels,
event detected at those levels are all plotted

"""
import os
import pickle
import sys
from pathlib import Path

import matplotlib.dates as mdates
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

if all_compression_rates is not None:
    average_compression_rates = np.mean(all_compression_rates, axis=0)
# peak amplitudes against peak lag
fig, ax = plt.subplots()
i = 0

all_keys = all_peaks.keys()
if compression_type == "zfp":
    all_keys = reversed(list(all_keys))
    average_compression_rates = np.flip(average_compression_rates)

for a in all_keys:  # all_peaks:
    peaks = all_peaks[a]
    peak_lags = all_peak_locations[a]

    # peak_lags = [b + data_time["start_lag"] for b in peak_lags]
    # peak_times = [mdates.date2num(start_time + timedelta(seconds=b)) for b in peak_lags]
    # peak_times = data_time["detection_time"] #[start_time + np.timedelta64(int(b/1000), "s") for b in peak_lags]
    peak_times = data_time[a]
    if all_compression_rates is None:
        label = a
    else:
        label = (
            str(int(average_compression_rates[i]))
            + "x ("
            + str(len(peaks))
            + " events)"
        )
    plt.scatter(
        peak_times, peaks, label=label
    )  # a + " (" + str(len(peaks)) + " events)")
    print(len(peaks), flush=True)
    print(sum(1 for b in all_detection_significances[a] if b < 12), flush=True)
    i += 1

fsize = 15
plt.ylabel("Amplitude", fontsize=fsize)
plt.xlabel("Time (s)", fontsize=fsize)
plt.title("Amplitudes of peaks detected", fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xticks(fontsize=fsize, rotation=45)
plt.legend(fontsize=fsize)
# ax = plt.gca()
# ax.xaxis_date()
fig.autofmt_xdate()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
savefile_name = save_location / (
    "peak_amplitudes_" + str(event_id) + "_" + compression_type + ".png"
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
    # peak_lags = [b + data_time["start_lag"] for b in peak_lags]
    # peak_times = [mdates.date2num(start_time + timedelta(seconds=b)) for b in peak_lags]
    # peak_times = data_time["detection_time"] #[start_time + np.timedelta64(int(b/1000), "s") for b in peak_lags]
    peak_times = data_time[a]
    if all_compression_rates is None:
        label = a
    else:
        # label = "compression_rate " + str(int(average_compression_rates[i])) + " (" + str(len(peak_times)) + " events)"
        label = (
            str(int(average_compression_rates[i]))
            + "x ("
            + str(len(peak_times))
            + " events)"
        )
    plt.scatter(
        peak_times, detection_sigs, label=label
    )  # a + " (" + str(len(peaks)) + " events)")
    i += 1

fsize = 15
plt.ylabel("Detection significance", fontsize=fsize)
plt.xlabel("Time (s)", fontsize=fsize)
plt.title("Detection significance of peaks detected", fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xticks(fontsize=fsize, rotation=45)
plt.legend(fontsize=fsize)
fig.autofmt_xdate()
# ax = plt.gca()
# ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
savefile_name = save_location / (
    "detection_sigs_" + str(event_id) + "_" + compression_type + ".png"
)
plt.savefig(savefile_name)
