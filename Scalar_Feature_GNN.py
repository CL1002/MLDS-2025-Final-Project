# This code use scalar ECG and PPG features to train three different types of GNN models - GCN, RGCN and H2G2
# Author : Chong Liu (02542904)

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GraphNorm
from sklearn.model_selection import StratifiedShuffleSplit
import os

#length=597288
###############################################
# GCN Model

def feature_load(Load_file1 = "D:/MLDS/features/", Load_file2 = "D:/MLDS/labels/",length=597288):
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
    np.save("D:/MLDS/features_2/overall.npy", F)
    np.save("D:/MLDS/labels_2/overall.npy", L)
    return F, L





class GCNmodel(nn.Module):
    def __init__(self, input_ch = 1, hidden_ch =192, num_classes=4):
        super().__init__()
        self.conv1 = GCNConv(input_ch, hidden_ch)
        self.norm1 = GraphNorm(hidden_ch)
        self.drop1 = nn.Dropout(0.2)
        self.conv2 = GCNConv(hidden_ch, hidden_ch)
        self.norm2 = GraphNorm(hidden_ch)
        self.drop2 = nn.Dropout(0.2)
        self.clas= nn.Sequential(
            nn.Linear(hidden_ch, hidden_ch // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_ch // 2, num_classes),
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x_norm1 = self.norm1(x, batch).relu()
        x_drop1 = self.drop1(x_norm1)

        x_2 = self.conv2(x_drop1, edge_index)
        x_norm2 = self.norm2(x_2, batch).relu()
        x_drop2 = self.drop2(x_norm2)

        x = global_mean_pool(x_drop2, batch)  # [B, hidden]
        return self.clas(x)

def fully_connected_edge(num_classes=4):
    edge_list = [(i, j) for i in range(num_classes) for j in range(num_classes) if i != j] # There is no type difference for edges
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # [2, E]

def feature_to_graph(F, label, edge_index, num_classes=4):
    data = []

    for n in range(len(F)):
        x = torch.from_numpy(F[n].astype(np.float32))[:, None]
        data.append(Data(x = x, edge_index=edge_index, y=torch.tensor(int(label[n]))))

    return data

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    #features, label = feature_load()
    #print(features.shape)
    #x = np.stack(features, axis=1).astype(np.float32)

    x = np.load("D:/MLDS/features_2/overall.npy")
    y = np.load("D:/MLDS/labels_2/overall.npy")

    y = np.asarray(y, dtype=np.float32)
    y = np.round(y).astype(np.int64)


    mask = y != 8
    x = x[mask]
    y = y[mask]

    y[y==1] = 0

    y[y==1] = 0
    y[y==2] = 1
    y[y==3] = 1
    y[y==4] = 1
    y[y==6] = 2
    y[y==5] = 3
    y[y==7] = 3




    ind = np.arange(len(x))
    s_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    trainval_ind, test_ind = next(s_test.split(x,y))
    s_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42) # 25% of 0.8 is 0.2

    t_sub, val_sub = next(s_val.split(trainval_ind, y[trainval_ind]))
    train_ind = trainval_ind[t_sub]
    val_ind = trainval_ind[val_sub]
    # 60% - train, 20% - validate, 20% - test
    x_train, y_train = x[train_ind], y[train_ind]
    x_val, y_val = x[val_ind], y[val_ind]
    x_test, y_test = x[test_ind], y[test_ind]

    # Standard score:
    m = np.nanmean(x_train, axis=0, keepdims=True)
    std = np.nanstd(x_train, axis=0, keepdims=True)
    x_train = (x_train - m) / (std + 1e-6)
    x_val = (x_val - m) / (std + 1e-6)
    x_test = (x_test - m) / (std + 1e-6)

    # fully connected edge
    edge_index = fully_connected_edge(num_classes=4)

    train_set = feature_to_graph(x_train, y_train, edge_index)
    val_set = feature_to_graph(x_val, y_val, edge_index)
    test_set = feature_to_graph(x_test, y_test, edge_index)

    # Each batch contains features from 512 segments
    batch_size = 512
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = GCNmodel(input_ch=1, hidden_ch=64, num_classes=4).to(DEVICE)

    cls_counts = np.bincount(y_train.astype(int), minlength=4)
    cls_weights = 1.0 / (cls_counts + 1e-6)
    cls_weights = cls_weights * (4 / cls_weights.sum())
    class_weights_t = torch.tensor(cls_weights, dtype=torch.float32, device=DEVICE)

    def add_weight_decay(m, weight_decay=1e-2, skip=("bias", "norm", "bn")):
        decay, no_decay = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if any(s in n.lower() for s in skip):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    opt = torch.optim.AdamW(add_weight_decay(model, 1e-2), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)

    epochs = 100 # if it is too small, then increase to 20
    best_val_acc = -1
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        train_loss_sum, train_correct, train_total = 0 ,0 ,0
        for data in train_loader:
            data = data.to(DEVICE)
            y_out = model(data.x, data.edge_index, data.batch)
            loss = loss_fn(y_out, data.y)

            opt.zero_grad()
            loss.backward() # backward
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)

            opt.step()


            # Record the loss or accuracy]

            train_loss_sum += float(loss.item())
            pred = y_out.argmax(dim=1)
            train_correct += int((pred == data.y).sum().item())
            train_total += int(data.y.numel())

        train_loss = train_loss_sum / max(1, len(train_loader)) # if there is no train data, then use number 1
        train_acc = train_correct / train_total if train_total else 0.0  # If there is no data, then 0

        # Now do evaluate - validate
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val in val_loader:
                val = val.to(DEVICE)

                y_out = model(val.x, val.edge_index, val.batch)
                loss = loss_fn(y_out, val.y)
                val_loss_sum += float(loss.item())
                pred = y_out.argmax(dim=1)
                val_correct += int((pred == val.y).sum().item())
                val_total += int(val.y.numel())
        val_loss = val_loss_sum / max(1, len(val_loader))
        val_acc = val_correct / val_total if val_total else 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:02d} | train {train_loss:.4f}/{train_acc:.4f} | "
              f"val {val_loss:.4f}/{val_acc:.4f}")

        #Finally Test

    model.eval()
    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for test in test_loader:
            test = data.to(DEVICE)
            y_out = model(test.x, test.edge_index, test.batch)
            loss = loss_fn(y_out, test.y)
            test_loss_sum += float(loss.item())
            pred = y_out.argmax(dim=1)
            test_correct += int((pred == test.y).sum().item())
            test_total += int(data.y.numel())
    test_loss = test_loss_sum / max(1, len(test_loader))
    test_acc = test_correct / test_total if test_total else 0.0
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

























