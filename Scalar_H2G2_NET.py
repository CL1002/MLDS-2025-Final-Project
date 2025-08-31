# This code use scalar ECG and PPG features to train H2G2 Net model
# Author : Chong Liu (02542904)

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
from sklearn.model_selection import StratifiedShuffleSplit

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Spatial (3 types of edges) + temporal edges

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tempo    = 1       # +1 temporal relation

def relational_edge():
    A = {0, 1, 2, 3}

    B = {4, 5, 6, 7}
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

def temporal_edge(timesteps):
    edge_i_sp, edge_t_sp = relational_edge()
    send = []
    receive = []
    etype = []
    for t in range(timesteps):
        off_line = t * 9
        send.extend((edge_i_sp[0] + off_line).tolist()) # t
        receive.extend((edge_i_sp[1] + off_line).tolist()) # t + 1
        etype.extend(edge_t_sp.tolist())

    # from (t,i) -> (t+1, i), new type 3 (not spatial 0, 1, 2)
    for t in range(timesteps-1):
        off_line_t = t * 9
        off_line_t1 = (t+1) * 9
        for n in range(9):
            send.append(off_line_t + n)
            receive.append(off_line_t1 + n)
            etype.append(3)

    edge_index = torch.tensor([send, receive], dtype =torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    return edge_index, edge_type

def sequence_build(x, y, T=5, stride=5):  # The sequence size is 5
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    sequence = []
    for st in range(0, len(x) - T + 1, stride):
        x_win = x[st:st + T]
        y_win = y[st:st+T]
        label = int(y_win[-1])
        sequence.append((x_win, label))
    return sequence

def sequence_to_graph (sequence, label, edge_index, edge_type): # Both spacial and temporal
    timestep = sequence.shape[0]
    x = torch.from_numpy(sequence.reshape(timestep*9,1).astype(np.float32))
    y = torch.tensor(int(label), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)


class H2G2_Net(nn.Module):
    def __init__(self, input_ch=1, hidden=64, num_rel=4, num_classes=4): # H2G2_RGCN
        super().__init__()
        self.rgcn1 = RGCNConv(input_ch, hidden, num_relations=num_rel, num_bases=num_rel)
        self.rgcn2 = RGCNConv(hidden, hidden, num_relations=num_rel, num_bases=num_rel)
        self.clas = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, edge_type, batch):
        x = self.rgcn1(x, edge_index, edge_type).relu()
        x = self.rgcn2(x, edge_index, edge_type).relu()
        x = global_mean_pool(x, batch)  # pool over all space–time nodes per sequence
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


    split_1 =  StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    ind = np.arange(len(x))


    trainval_ind, test_ind = next(split_1.split(ind, y))
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)#
    t_sub, val_sub = next(split_2.split(trainval_ind, y[trainval_ind]))

    train_ind = trainval_ind[t_sub]
    val_ind = trainval_ind[val_sub]
    x_train, y_train = x[train_ind], y[train_ind]
    x_val, y_val = x[val_ind], y[val_ind]
    x_test, y_test = x[test_ind], y[test_ind]

    # standard score
    m = np.nanmean(x_train, axis=0, keepdims=True)
    std = np.nanstd(x_train, axis=0, keepdims=True)
    x_train = (x_train - m) / (std + 1e-6)
    x_val = (x_val - m) / (std + 1e-6)
    x_test = (x_test - m) / (std + 1e-6)

    # Make sequence of graphs - dynamic graphs
    timestep=5
    stride =5

    train_seqs = sequence_build(x_train, y_train, T=timestep, stride=stride)
    val_seqs = sequence_build(x_val, y_val, T=timestep, stride=stride)
    test_seqs = sequence_build(x_test, y_test, T=timestep, stride=stride)
    edge_index, edge_type =temporal_edge(timestep)
    train_set = [sequence_to_graph(x, y, edge_index, edge_type) for x, y in train_seqs]
    val_set = [sequence_to_graph(x, y, edge_index, edge_type) for x, y in val_seqs]
    test_set = [sequence_to_graph(x, y, edge_index, edge_type) for x, y in test_seqs]
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # Train/validate and test model
    model=H2G2_Net(input_ch=1).to(DEVICE)

    # class weights from *sequence* labels
    y_tr_seq = np.array([lab for _, lab in train_seqs], dtype=np.int64)
    cls_counts = np.bincount(y_tr_seq, minlength=4)
    weights = 1.0 / (cls_counts + 1e-6);
    weights = weights * (4 / weights.sum())
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))


    def add_weight_decay(m, weight_decay=1e-2, skip=("bias", "norm", "bn")):
        decay, no_decay = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad: continue
            (no_decay if any(s in n.lower() for s in skip) else decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]


    opt = torch.optim.AdamW(add_weight_decay(model, 1e-2), lr=1e-3)

    # (8) Train / validate
    best_val, best_state = -1.0, None
    epochs = 100

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss= 0
        train_hit = 0
        train_tot = 0
        for data in train_loader:
            data = data.to(DEVICE)
            pred = model(data.x, data.edge_index, data.edge_type, data.batch)
            loss = loss_fn(pred, data.y)
            opt.zero_grad();
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += float(loss.item())
            train_hit += int((pred.argmax(1) == data.y).sum().item())
            train_tot += int(data.y.numel())
        train_loss /= max(1, len(train_loader))
        tr_acc = train_hit / train_tot if train_tot else 0.0

        model.eval()
        val_loss= 0
        val_hit=0
        val_tot=0

        with torch.no_grad():
            for val in val_loader:
                val= val.to(DEVICE)
                pred = model(val.x, val.edge_index,val.edge_type, val.batch)
                loss = loss_fn(pred, val.y)
                val_loss += float(loss.item())
                val_hit += int((pred.argmax(1) == val.y).sum().item())
                val_tot += int(val.y.numel())
        val_loss /= max(1, len(val_loader))
        va_acc = val_hit / val_tot if val_tot else 0

        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:02d} | train {train_loss:.4f}/{tr_acc:.4f} | val {val_loss:.4f}/{va_acc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    test_loss = 0
    test_hit = 0
    test_tot = 0
    with torch.no_grad():
        for test in test_loader:
            test = test.to(DEVICE)
            pred = model(test.x, test.edge_index,test.edge_type, test.batch)
            loss = loss_fn(pred, test.y)
            test_loss += float(loss.item())
            test_hit += int((pred.argmax(1) == test.y).sum().item())
            test_tot += int(test.y.numel())
    test_loss /= max(1, len(test_loader))
    test_acc = test_hit / test_tot if test_tot else 0
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")
















