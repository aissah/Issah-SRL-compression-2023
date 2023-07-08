import os
import sys
from datetime import datetime

import matplotlib.dates as mdates
import numpy as np
import zfpy

# sys.path.insert(
#     0, r"D:\CSM\Mines_Research\Summer_2022_paper\summer2022exp\Frequencytest"
# )  

import pickle
print(os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('..'))
print(os.path.dirname(os.path.abspath(__file__)) + "/..")
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from Functions.FTFuncs import (
    loadFORESEEhdf5,
    multweigthedAverageRatio,
    plotsaveimshow,
    plt,
    randomized_SVD_comp_decomp,
    soft_comp_decomp1d,
    soft_comp_decomp2d,
    stackInWindows,
    windowedNormalisedErrors,
    windowedPowerSpectrum,
)
# from Functions import FTFuncs
print("import successful")