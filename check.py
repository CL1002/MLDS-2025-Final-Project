
# This code use scalar ECG and PPG features to train three different types of GNN models - GCN, RGCN and H2G2
# Author : Chong Liu (02542904)

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import StratifiedShuffleSplit
import os

#length=597288
###############################################
# GCN Model

def feature_load(Load_file1 = "D:/MLDS/features/", Load_file2 = "D:/MLDS/labels/",length=50000):
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

# F = np.load("D:/MLDS/features_2/overall.npy")
x = np.load("D:/MLDS/cnn_features/cnn_overall.npy")
print(x.shape)
y = np.load("D:/MLDS/labels/cnn_overall.npy")
print(y.shape)

y = np.asarray(y,dtype=np.float32)
print(y.dtype)
y_int = np.round(y).astype(np.int64)

classes, counts = np.unique(y_int, return_counts=True)
for c, n in zip(classes, counts):
    print(f"class {int(c)}: {n}")
print("total:", counts.sum())