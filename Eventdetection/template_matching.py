"""
Parameters
----------
event_id : int
    This is an identifier of which event template to use. A number of event_ids
    are "registered" here which basically means the instrutions on how/where to
    get the associated template are defined here. 
first_channel : int 
    First channel in the range of channels to use.
last_channel : int
    Last channel in the range of channels to use.
batch : 1
    If the files in the directory are going to be processed in batches, which
    batch does this run correspond to. If no batches we could just say 1 for 
    this and the "total number of files in directory" for batch_size
batch_size : int
    Size of the batches of files to be processed
compression_flag : int
    1 or 0 to indicate whether we want to compress the data before template 
    matching or not
compression_type : str
    The type of compression being used. This is solely used in naming the 
    output files.
compression_func : str
    The compression function exactly how it should be run. The first input 
    should be "data" and if level of compression is required by function, 
    "compression_level" should be used
    eg: eventDTFuncs.compressReconstruct_wavelets(data, compressionFactor=compression_level)
compression_levels : int, optional
    The level(s) of compression to test. compression_levels is used as opposed
    to compression rate because some compression scheme like zfp takes precision,
    tolerance... as opposed to compression rate
Returns
-------
Creates files with the stacked template matching of the template across all the 
files.Files are named event_id + "_" + compression_type + compression_levels
+ "_batch" + str(batch)+ "_" + first_file + "_" + last_file

Created on Thu Jan  5 10:21:17 2023

@author: issah
"""
import os
import pickle
import sys
from datetime import datetime

import eventDTFuncs
import numpy as np

# add path that contains eventDTFuncs to scope of search
sys.path.insert(0, "/u/st/by/aissah/scratch/summer2022exp/Eventdetection")

# compression type dict
COMPRESSION_FUNCTIONS = {
    "wavelet": 'eventDTFuncs.compressReconstruct_wavelets(data,mode="2D", compressionFactor=compression_level)',
    "wl": 'eventDTFuncs.compressReconstruct_wavelets(data,mode="2D", compressionFactor=compression_level)',
    "zfp": 'eventDTFuncs.compressReconstruct_zfp(data,mode="precision", precision=compression_level)',
    "svd": "eventDTFuncs.randomized_SVD_comp_decomp(data, compression_factor=compression_level)",
}
COMPRESSION_LEVELS = {
    "wavelet": "5,10,20,50",
    "wl": "5,10,20,50",
    "zfp": "2,3,4,5",
    "svd": "5,10,20,50",
}

event_id = int(sys.argv[1])  # Used to select the event template. Used: 2201050
first_channel = int(sys.argv[2])  # First channel in range of channels used. Used: 1000
last_channel = int(sys.argv[3])  # Last channel in range of channels used: Used:5000
batch = int(
    sys.argv[4]
)  # Batch of files assuming jobs are run in parallel for files in batches. Should be one if that is not the case.
batch_size = int(
    sys.argv[5]
)  # Number of files in batch. Should be number of files being considered if job is not done in batches
compression_flag = int(
    sys.argv[6]
)  # 1 if compressed data is used otherwise uncompressed data is used

sampling_frequency = 1000
metadata = {}
metadata["event_id"] = event_id

if compression_flag == 1:
    compression_type = sys.argv[
        7
    ]  #  COMPRESSION_FUNCTIONS shows compression types as keys. wl and wavelet are the same
    compression_func = sys.argv[
        8
    ]  # String of the exact function to run to compress and reconstruct data. THe output of this should be formatted as (reconstructed_data, compression_factor).
    # COMPRESSION_FUNCTIONS shows those used for our compression types.
    compression_levels = sys.argv[
        9
    ]  # compression levels to be used separated by commas(,). Could be compression factors or whatever indicates level of compression for said compression type.
    # COMPRESSION_LEVELS shows those used for our compression types.

    compression_levels = compression_levels.split(",")
    compression_levels = [float(a) for a in compression_levels]
    metadata["compression_levels"] = compression_levels
    metadata["compression_type"] = compression_type
else:
    # compression_type = "N/A"
    compression_levels = None
    # compression_rate = "N/A"
    metadata["compression_rates"] = "N/A"  # [compression_rate]
    metadata["compression_levels"] = "N/A"  # compression_levels
    metadata["compression_type"] = "N/A"  # compression_type

data_basepath = "/beegfs/projects/martin/BradyHotspring"  # "D:/CSM/Mines_Research/Test_data/Brady Hotspring"
# files = os.listdir(data_basepath)
save_location = "/u/st/by/aissah/scratch/event_detection/template_matching"  # "D:/CSM/Mines_Research/Test_data/"

# build up collection of templates across channels
if event_id == 2201050:
    template, _ = eventDTFuncs.loadBradyHShdf5(
        data_basepath + "/03_14_2016/PoroTomo_iDAS16043_160314083848.h5",
        normalize="no",
    )
    template = template[first_channel:last_channel, 17240:23240]
elif event_id == 2201051:
    template, _ = eventDTFuncs.loadBradyHShdf5(
        data_basepath + "/03_14_2016/PoroTomo_iDAS16043_160314121048.h5",
        normalize="no",
    )
    template = template[first_channel:last_channel, 21730:27730]
elif event_id == 2201052:
    template, _ = eventDTFuncs.loadBradyHShdf5(
        data_basepath + "/03_14_2016/PoroTomo_iDAS16043_160316220248.h5",
        normalize="no",
    )
    template = template[first_channel:last_channel, 22250:28250]
elif event_id == "test":
    file = r"D:\CSM\Mines_Research\Test_data\Brady Hotspring\PoroTomo_iDAS16043_160314083848.h5"
    template, _ = eventDTFuncs.loadBradyHShdf5(file, normalize="yes")
    template = template[first_channel:last_channel, 17240:23240]
else:
    raise Exception("Unregistered event_id")


def _get_mean_ccs(
    data,
    template,
    lagmax,
    compression_flag,
    compression_levels,
    mean_ccs_acrossfiles=None,
    metadata=None,
):
    """
    Internal function used to cross-correlate and find mean across channels either for just the uncompressed data or different
    levels of compression for a compression type. Intended for data from one of several files.

    Parameters
    ----------
    data : 2-dimensional numpy array
        uncompressed data.
    template : 2-dimensional numpy array
        template to cross-correlate with data. Should have same channels as data.
    lagmax : int
        Maximum lag for cross-correlation given number of time samples in template and data.
    compression_flag : int
        1:reconstructed data, other: uncompressed data.
    compression_levels : list of ints or N/A if uncompressed
        Compression levels to compress data at and find mean of cc.
    mean_ccs_acrossfiles : numpy array (1d for no compression, 2d for different levels of compression), optional
        mean of cross-correlations across multiple files if previous files have been processed already. The default is None.
    metadata : TYPE, optional
        Metadata of previously processed files. The default is None.

    Returns
    -------
    mean_ccs_acrossfiles : numpy array (1d for no compression, 2d for different levels of compression)
        mean of cross-correlations across multiple files up to the current file.
    metadata : dict
        metadata of file processed.

    """
    if compression_flag == 1:
        mean_ccs = np.array([range(lagmax + 1)])
        file_comp_rates = []
        for compression_level in compression_levels:
            # compression flag is 1 we consider the various compression levels provided
            decompressed_data, compression_rate = eval(compression_func)
            file_comp_rates.append(compression_rate)
            decompressed_template, _ = eval(
                compression_func.replace("data", "template")
            )
            decompressed_template = eventDTFuncs.brady_preprocess(decompressed_template)
            decompressed_data = eventDTFuncs.brady_preprocess(decompressed_data)
            decompressed_template = eventDTFuncs.frequency_filter(
                decompressed_template, [1, 15], "bandpass", 5, sampling_frequency
            )
            decompressed_data = eventDTFuncs.frequency_filter(
                decompressed_data, [1, 15], "bandpass", 5, sampling_frequency
            )

            normalizer = np.max(decompressed_data, axis=1)
            decompressed_data = decompressed_data / normalizer[:, np.newaxis]

            normalizer = np.max(decompressed_template, axis=1)
            decompressed_template = decompressed_template / normalizer[:, np.newaxis]

            lagmax = len(decompressed_data[0]) - len(decompressed_template[0])
            # decompressed_data=decompressed_data/abs(decompressed_template).max(axis=1)[:,np.newaxis]
            mean_cc = eventDTFuncs.crosscorrelate_channels(
                decompressed_data, decompressed_template, lagmax, stacked="yes"
            )
            mean_ccs = np.append(mean_ccs, [mean_cc], axis=0)
            # averageAcrossChannels[b]=TM
            # peaks,time= eventDTFuncs.get_peaks(mean_cc, detection_significance_threshold)
        if mean_ccs_acrossfiles is None:
            mean_ccs_acrossfiles = mean_ccs[1:]
            metadata["compression_rates"] = [file_comp_rates]
        else:
            metadata["compression_rates"].append(file_comp_rates)
            mean_ccs_acrossfiles = np.append(mean_ccs_acrossfiles, mean_ccs[1:], axis=1)
    elif compression_flag == 0:
        # compression flag is 0 we only consider the original data
        data = eventDTFuncs.brady_preprocess(data)
        data = eventDTFuncs.frequency_filter(
            data, [1, 15], "bandpass", 5, sampling_frequency
        )
        normalizer = np.max(data, axis=1)
        data = data / normalizer[:, np.newaxis]
        normalizer = np.max(template, axis=1)
        template = template / normalizer[:, np.newaxis]
        mean_cc = eventDTFuncs.crosscorrelate_channels(
            data, template, lagmax, stacked="yes"
        )
        if mean_ccs_acrossfiles is None:
            mean_ccs_acrossfiles = mean_cc
        else:
            mean_ccs_acrossfiles = np.append(mean_ccs_acrossfiles, mean_cc)

    return mean_ccs_acrossfiles, metadata


if compression_flag == 0:
    template = eventDTFuncs.brady_preprocess(template)
    template = eventDTFuncs.frequency_filter(
        template, [1, 15], "bandpass", 5, sampling_frequency
    )
    # template = template/abs(template).max(axis=1)[:,np.newaxis]

# Get the file names of the data files by going through the folders contained
# in the base path and putting together the paths to files ending in .h5
data_files = []
for dir_path, dir_names, file_names in os.walk(data_basepath):
    dir_names.sort()
    file_names.sort()
    data_files.extend(
        [
            os.path.join(dir_path, file_name)
            for file_name in file_names
            if ".h5" in file_name
        ]
    )

# metadata["files"] = [a[-15:-3] for a in data_files]
# metadata["compression_rates"] = [compression_rate]

# Process the first file in the list of files. This is done separately for batches
# that start with the earlier file recorded and the rest. For later batches,
# we append bits of the preceding file to make sure template matching is
# continuous across files
start_time = datetime.now()
if batch == 1:
    metadata["start_lag"] = 0
    first_file_time = data_files[0][-15:-3]
    # start_time = "20" + first_file_time[:2] + "-" + first_file_time[2:4] + "-" + first_file_time[4:6] + "T" +
    # first_file_time[6:8] + ":" + first_file_time[8:10] + ":" + first_file_time[10:]
    # metadata["start_time"] = np.datetime64(first_file_time)
    # datetime.datetime is able to handle milliseconds unlike np.datetime64
    metadata["start_time"] = datetime(
        2000 + int(first_file_time[:2]),
        int(first_file_time[2:4]),
        int(first_file_time[4:6]),
        int(first_file_time[6:8]),
        int(first_file_time[8:10]),
        int(first_file_time[10:]),
    )
    data_files = data_files[:batch_size]
    metadata["files"] = [a[-15:-3] for a in data_files]
    data, _ = eventDTFuncs.loadBradyHShdf5(data_files[0], normalize="no")
    data = data[first_channel:last_channel]
    # lagmax = len(data[0]) - 2 * len(template[0]) + 2 previously for some reason
    # Changed now but cannot remember the thinking behind initial the implementation
    # Something for future Hafiz to resolve
    lagmax = len(data[0]) - len(template[0])

    mean_ccs_acrossfiles, metadata = _get_mean_ccs(
        data,
        template,
        lagmax,
        compression_flag,
        compression_levels,
        mean_ccs_acrossfiles=None,
        metadata=metadata,
    )

else:  # with more batches, append end of previous file for continuity
    try:
        data_files = data_files[(batch - 1) * batch_size - 1 : batch * batch_size]
        metadata["files"] = [a[-15:-3] for a in data_files]
    except IndexError:
        data_files = data_files[(batch - 1) * batch_size - 1 :]
        metadata["files"] = [a[-15:-3] for a in data_files]
    data, _ = eventDTFuncs.loadBradyHShdf5(data_files[1], normalize="no")
    metadata["start_lag"] = (
        (batch - 1) * batch_size * len(data[0]) - len(template[0]) + 1
    )
    first_file_time = metadata["files"][1]
    metadata["start_time"] = datetime(
        2000 + int(first_file_time[:2]),
        int(first_file_time[2:4]),
        int(first_file_time[4:6]),
        int(first_file_time[6:8]),
        int(first_file_time[8:10]),
        int(first_file_time[10:]),
    )
    preceding_data, _ = eventDTFuncs.loadBradyHShdf5(data_files[0], normalize="no")
    data = np.append(
        preceding_data[first_channel:last_channel, -len(template[1]) + 1 :],
        data[first_channel:last_channel],
        axis=1,
    )

    # lagmax = len(data[0]) - 2 * len(template[0]) + 1 previously for some reason
    # Changed now but cannot remember the thinking behind initial the implementation
    # Something for future Hafiz to resolve
    lagmax = len(data[0]) - len(template[0])

    mean_ccs_acrossfiles, metadata = _get_mean_ccs(
        data,
        template,
        lagmax,
        compression_flag,
        compression_levels,
        mean_ccs_acrossfiles=None,
        metadata=metadata,
    )

# work on files after first file in batch. This works exactly as we handled the
# beginning of later batches. Then we keep appending to the variables set up for
# first file of the batch above

end_time = datetime.now()
print(f"Duration: {end_time - start_time}", flush=True)

for a in data_files[1:]:
    preceding_data = data[:, -len(template[1]) + 1 :]
    data, _ = eventDTFuncs.loadBradyHShdf5(a, normalize="no")
    data = np.append(preceding_data, data[first_channel:last_channel], axis=1)
    # lagmax = len(data[0]) - 2 * len(template[0]) + 1 previously for some reason
    # Changed now but cannot remember the thinking behind initial the implementation
    # Something for future Hafiz to resolve
    lagmax = len(data[0]) - len(template[0])

    mean_ccs_acrossfiles, metadata = _get_mean_ccs(
        data,
        template,
        lagmax,
        compression_flag,
        compression_levels,
        mean_ccs_acrossfiles,
        metadata,
    )

if compression_flag == 0:
    savefile_name = (
        save_location
        + "/"
        + str(event_id)
        + "uncompressed_batch"
        + str(batch)
        + "_"
        + metadata["files"][0]
        + "_"
        + metadata["files"][-1]
    )
else:
    savefile_name = (
        save_location
        + "/"
        + str(event_id)
        + "_"
        + compression_type
        + "_"
        + "_".join([str(int(a)) for a in metadata["compression_rates"][0]])
        + "_batch"
        + str(batch)
        + "_"
        + metadata["files"][0]
        + "_"
        + metadata["files"][-1]
    )

with open(savefile_name, "wb") as f:
    pickle.dump([mean_ccs_acrossfiles, metadata], f)

end_time = datetime.now()
print(f"Total duration: {end_time - start_time}", flush=True)
