# This code use scalar ECG and PPG features to train RGCN model - the edges between different nodes are different
# Author : Chong Liu (02542904)

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool,GraphNorm
from sklearn.model_selection import StratifiedShuffleSplit
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# build relational edges
def relational_edge():
    A = {0,1,2,3}

    B = {4,5,6,7}
    C = {8}
    send = []
    receive = []
    etypes = []
    for n in range(9):
        for m in range(9):
            if n == m:
                continue
            if (n in A) & (m in A):
                buffer = 0
            elif (n in B) & (m in B):
                buffer = 1
            else:
                buffer = 2

            send.append(n)
            receive.append(m)
            etypes.append(buffer)

    edge_index = torch.tensor([send, receive], dtype=torch.long)
    edge_type = torch.tensor(etypes, dtype=torch.long)
    return edge_index, edge_type

def feature_to_graph(F, label, edge_index, edge_type):
    x = np.asarray(F, dtype=np.float32)
    y = np.asarray(label)

    y = torch.as_tensor(y, dtype=torch.long)
    data = []
    for n in range(x.shape[0]):
        x_n = torch.from_numpy(x[n][:,None])
        y_n = y[n]

        data.append(Data(x =x_n, edge_index=edge_index, edge_type=edge_type,y=y_n))
    return data

class RGCNmodel(nn.Module):
    def __init__(self, input_ch, hidden=192, num_relations = 3, num_classes=4):
        super().__init__()
        self.rgcn1 = RGCNConv(input_ch, hidden, num_relations=num_relations, num_bases=3)
        self.norm1 = GraphNorm(hidden)
        self.drop1 = nn.Dropout(0.2)
        self.rgcn2 = RGCNConv(hidden, hidden, num_relations=num_relations, num_bases=3)
        self.norm2 = GraphNorm(hidden)
        self.drop2 = nn.Dropout(0.2)
        self.clas = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(hidden // 2, num_classes),
        )
    def forward(self, x, edge_index, edge_type, batch):
        # batch size here should be 1. The activation is set as relu
        x_rgcn1 = self.rgcn1(x, edge_index, edge_type)
        x_norm1 = self.norm1(x_rgcn1,batch).relu()
        x_drop1 = self.drop1(x_norm1)
        x_rgcn2 = self.rgcn2(x_drop1, edge_index, edge_type)
        x_norm2 = self.norm2(x_rgcn2,batch).relu()
        x_drop2 = self.drop2(x_norm2)
        x = global_mean_pool(x_drop2, batch)  # [B, hidden]
        return self.clas(x)

if __name__ == "__main__":
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


    # split it 6/2/2
    ind = []
    split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state = 42)
    trainval_ind, test_ind = next(split_1.split(x,y))
    # 25% of 80% of data is 20% of the data
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25,random_state=42)
    t_sub, val_sub = next(split_2.split(x[trainval_ind], y[trainval_ind]))

    train_ind = trainval_ind[t_sub] # 75% of 80% of data
    val_ind = trainval_ind[val_sub]

    x_train, y_train = x[train_ind],y[train_ind]
    x_val, y_val = x[val_ind], y[val_ind]
    x_test, y_test = x[test_ind], y[test_ind]

    # standard score
    m = np.nanmean(x_train, axis=0, keepdims=True)
    std = np.nanstd(x_train, axis=0, keepdims=True)
    x_train = (x_train - m) / (std + 1e-6)
    x_val = (x_val - m) / (std+1e-6)
    x_test=(x_test - m) / (std+1e-6)

    edge_index, edge_type = relational_edge()
    # Convert feature to graph structure
    train_set = feature_to_graph(x_train, y_train, edge_index,edge_type)
    val_set = feature_to_graph(x_val, y_val, edge_index,edge_type)
    test_set =  feature_to_graph(x_test, y_test, edge_index,edge_type)

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=512, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False)

    model = RGCNmodel(input_ch=1, hidden=64, num_relations=3, num_classes=4).to(DEVICE)


    def add_weight_decay(m, weight_decay=1e-2, skip=("bias", "norm", "bn")):
        decay, no_decay = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if any(s in n.lower() for s in skip):
                no_decay.append(p)
            else:
                decay.append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]


    opt = torch.optim.AdamW(add_weight_decay(model, 1e-2), lr=1e-3)
    # Class weights (optional)
    cls_counts = np.bincount(y_train.astype(int), minlength=4)
    weights = 1.0 / (cls_counts + 1e-6);
    weights = weights * (4 / weights.sum())
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))

    best_val, best_state = -1.0, None
    epochs = 100
    for epoch in range(1, epochs+1):
        model.train()
        train_loss_sum, train_correct, train_total = 0, 0, 0
        for data in train_loader:
            data = data.to(DEVICE)
            y_out = model(data.x, data.edge_index,data.edge_type, data.batch)
            loss = loss_fn(y_out, data.y)

            opt.zero_grad()
            loss.backward()  # backward
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            opt.step()

            # Record the loss or accuracy]

            train_loss_sum += float(loss.item())
            pred = y_out.argmax(dim=1)
            train_correct += int((pred == data.y).sum().item())
            train_total += int(data.y.numel())

        train_loss = train_loss_sum / max(1, len(train_loader))  # if there is no train data, then use number 1
        train_acc = train_correct / train_total if train_total else 0.0  # If there is no data, then 0

        # Now do evaluate - validate
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val in val_loader:
                val = val.to(DEVICE)

                y_out = model(val.x, val.edge_index,val.edge_type, val.batch)
                loss = loss_fn(y_out, val.y)
                val_loss_sum += float(loss.item())
                pred = y_out.argmax(dim=1)
                val_correct += int((pred == val.y).sum().item())
                val_total += int(val.y.numel())
        val_loss = val_loss_sum / max(1, len(val_loader))
        val_acc = val_correct / val_total if val_total else 0.0

        best_val, best_state = -1.0, None

        if val_acc > best_val:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:02d} | train {train_loss:.4f}/{train_acc:.4f} | "
              f"val {val_loss:.4f}/{val_acc:.4f}")

        # Finally Test

    model.eval()
    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for test in test_loader:
            test = data.to(DEVICE)
            y_out = model(test.x, test.edge_index,test.edge_type, test.batch)
            loss = loss_fn(y_out, test.y)
            test_loss_sum += float(loss.item())
            pred = y_out.argmax(dim=1)
            test_correct += int((pred == test.y).sum().item())
            test_total += int(data.y.numel())
    test_loss = test_loss_sum / max(1, len(test_loader))
    test_acc = test_correct / test_total if test_total else 0.0
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")
















