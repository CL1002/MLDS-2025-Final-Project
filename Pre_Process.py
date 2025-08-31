# This code is going to do signal pre-process
# Author : Chong Liu (02542904)

#import libraries
import os
import re
import glob
import math
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import os # Check folders
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

import wfdb # specific required for MIMIC-III
from scipy.signal import resample_poly

# Function to filter signal
def band_pass_filter(signal, fs=125, low=0.5, high=40, order=3):
    nyquist = fs / 2
    low = low/nyquist
    high = high/nyquist
    b,a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def flatline(segment,fs=125, win = 10, eps = 1e-6):
    has_nan = np.isnan(segment).any()
    if has_nan:
        return False
    if len(segment) < win * fs:
        return False
    ave = np.mean(segment)
    buffer = 0
    # Use Var to find flatline
    for n in range(len(segment)):
        buffer += (segment[n] - ave)**2
    var = buffer/len(segment)

    if var < eps:
        return False
    else:
        return True



def qrs_complex(segment, fs=125, win = 10, N = 0.2, threshold = 0.01):
    ecg_bp = band_pass_filter(segment,fs=fs, low = 0.5, high=40)
    # differentiation
    has_nan = np.isnan(segment).any()
    if has_nan:
        return False
    if len(segment) < win * fs:
        return False
    diff = []
    for n in range(len(ecg_bp) - 1):
        diff.append((ecg_bp[n+1] - ecg_bp[n])**2)



    # Moving Window Integration
    emwi = []
    N = N * fs
    R = 0
    for n in range(25, len(diff), 25):
        emwi.append(np.mean(np.array(diff[n-25:n], dtype=float)))
    for n in emwi:
        if n > threshold:
            R += 1

    if R >= 5:
        if R <= 25:
            return True

    return False

def pulsatile(segment, fs=125, win=10, d_threshold = 0.05, a_threshold = 10):
    has_nan = np.isnan(segment).any()
    if has_nan:
        return False
    if len(segment) < win * fs:
        return False
    peaks, property = find_peaks(segment, distance=0.5 * fs, prominence= 0.05)
    troughs, property2 = find_peaks(-1 * segment, distance=0.5 * fs, prominence=0.05)
    # check each pulse difference
    peak_diff = []
    for n in range(len(peaks) - 1):
        peak_diff.append(peaks[n+1] - peaks[n])

    A = np.sort(peaks)
    B = np.sort(troughs)

    # Find indices in peaks which is the next peak for the troughs
    ind = np.searchsorted(A, B, side='right')

    # Find the valid indices in B
    valid = ind < len(A)

    #set a numpy array to record diff
    match = np.full_like(B, np.nan, dtype=float)
    diff = np.full_like(B, np.nan, dtype=float)


    match[valid] = A[ind[valid]]


    n = min(match.shape[0], A.shape[0], B.shape[0])
    match = match[:n]
    A = A[:n]
    diff = diff[:n]
    valid = valid[:n]
    #count
    diff[valid] = match[valid] - A[valid]
    if (len(diff) >= 5):
        if len(diff) <= 25:
            return True
    return False




if __name__ == "__main__":
    import argparse
    Load_path = "C:/Users/Chong/PycharmProjects/pythonProject/MLDS_Final/converted_npy"
    fs_out = 125  # Hz
    win = 10
    eps = 1e-6
    np.save('index.npy', '0')
    # Read all npy files
    npy_ecg = [file for file in os.listdir(Load_path) if file.endswith("ECG.npy")]
    npy_ecg.sort()

    # Quality check ( if any of them not qualified, removed all of them) Good = 0 unqualified > 0


    for file in npy_ecg:
        ecg_qual = []
        ppg_qual = []
        abp_qual = []
        filepath = os.path.join(Load_path, file)
        data = np.load(filepath)
        ecg = data[:,0]
        for n in range(0, len(ecg)-1250, 1250):
            ecg_q = 0
            segment = ecg[n : n+1250]
            # check flatline
            if flatline(segment):
                ecg_q += 0
            else:
                ecg_q += 1

            if qrs_complex(segment):
                ecg_q += 0
            else:
                ecg_q += 1


            ecg_qual.append(ecg_q)


        # Find address for PPG
        file2 = file.replace('ECG', 'PPG')
        filepath = os.path.join(Load_path, file2)
        ppg = np.load(filepath)
        for n in range(0, len(ppg), 1250):
            segment = ppg[n:n + 1250]
            ppg_q = 0
            # check flatline
            if flatline(segment):
                ppg_q += 0
            else:
                ppg_q += 1

            if pulsatile(segment):
                ppg_q += 0
            else:
                ppg_q += 1

            ppg_qual.append(ppg_q)


        # Find the address for abp
        file3 = file.replace('ECG', 'ABP')
        filepath = os.path.join(Load_path, file3)

        abp = np.load(filepath)
        for n in range(0, len(abp), 1250):
            abp_q = 0
            segment = abp[n:n + 1250]
            # check flatline
            if flatline(segment):
                abp_q += 0
            else:
                abp_q += 1

            if pulsatile(segment * -1):
                abp_q += 0
            else:
                abp_q += 1

            abp_qual.append(abp_q)
        for n in range(len(ecg_qual)):
            # If segments are qualified
            if (ecg_qual[n] == 0) & (ppg_qual[n] == 0) & (abp_qual[n] == 0):
                ind = int(np.load('index.npy'))
                np.save('D:/MLDS/ecg/' + str(ind) + 'ecg.npy', ecg[n * 1250: (n + 1) * 1250])
                np.save('D:/MLDS/ppg/' + str(ind) + 'ppg.npy', ppg[n * 1250: (n + 1) * 1250])
                np.save('D:/MLDS/abp/' + str(ind) + 'abp.npy', abp[n * 1250: (n + 1) * 1250])

                ind = ind + 1
                print(str(ind) + ' segments')
                np.save('index.npy', ind)











