# Author : Chong Liu (02542904)

## This code is going to convert ecg, ppg segments in averaged cycle beats


from tqdm import tqdm
import os
import numpy as np
from scipy.signal import find_peaks
import torch
import torch.nn.functional as F
import os

# For signal and PPG segments, it is going to find peaks over the period and then extracts components and resample them.

Load = "D:/MLDS/"


ind = 0

def resample_to_1200(arr):
    n = arr.shape[0]
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, 1200)
    return np.interp(x_new, x_old, arr)

def average_cycle(signal,path, ind = 0, ep = 'ecg'):
    y = signal
    y2 = []
    peaks, property = find_peaks(y, distance=0.5 * 125, prominence=0.5)
    for m in range(len(peaks) - 1):
        buffer = resample_to_1200(y[peaks[m]:peaks[m + 1]])
        y2.append(buffer)

    y2 = np.array(y2, dtype=np.float32)
    y2 = y2.T
    y2 = np.mean(y2, axis=-1)

    if ep == 'ecg':
        np.save(path + 'ecg_cycling/' + str(ind) + '.npy', y2)

    else:
        np.save(path + 'ppg_cycling/' + str(ind) + '.npy', y2)

    ind = ind + 1


    return ind


if __name__ == "__main__":
    torch.manual_seed(42)
    Load_file = "D:/MLDS/"
    length = 597288
    fs = 125
    e_ind = 0
    p_ind = 0
    for n in range(length):
        ecg_segment = np.load(Load_file + 'ecg/' + str(n) + "ecg.npy")
        ppg_segment = np.load(Load_file + 'ppg/' + str(n) + "ppg.npy")
        average_cycle(ecg_segment, path = Load_file, ind=n, ep='ecg')
        average_cycle(ppg_segment, path=Load_file,ind=n, ep='ppg')
        print(n)




