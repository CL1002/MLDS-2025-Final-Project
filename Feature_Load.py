
# Author : Chong Liu (02542904)

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GraphNorm
from sklearn.model_selection import StratifiedShuffleSplit
import os


def feature_load(Load_file1 = "D:/MLDS/cnn_features/", Load_file2 = "D:/MLDS/labels/",length=597288):
    F = []
    L = []
    for n in range(length):
        path = Load_file1 + str(n) + '.npy'
        path2 = Load_file2 + str(n) + '.npy'
        if os.path.exists(path):
            if os.path.exists(path2):
                if np.isnan(np.load(path)).any() or np.isnan(np.load(path2)).any():
                    a = 1
                else:
                    F.append(np.load(path).tolist())
                    L.append(np.load(path2))
        print(n)
#
    F = np.array(F)
    L = np.array(L)
    np.save("D:/MLDS/cnn_features/cnn_overall.npy", F)
    np.save("D:/MLDS/labels/cnn_overall.npy", L)
    return F, L

feature_load()