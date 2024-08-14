import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
import tensorflow as tf
import networkx as nx
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
import torch.nn as nn
data_csv = pd.read_csv("data.csv")
users = {}
labels = {}
counter = 0
edge_list = {}
for i in range(len(data_csv)):
    row = data_csv.loc[i]
    fuser = row["from"]
    tuser = row["to"]
    
    if(not(fuser in users)):
        users[fuser] = counter
        counter += 1
    if(not(tuser in users)):
        users[tuser] = counter
        counter += 1
num_nodes = len(users)

#There are six node features
# Min sent amount
# Max sent amount
# Transaction count while sending
# Average sent amount
# Min received amount
# Max received amount
# Transaction count while receiving
# Average received amount

print("Number of nodes are ", len(users))
inf = 1e9
node_features = np.zeros((num_nodes, 8), dtype=float)
edge_list = [[], []]
edge_attr = []
for i in range(len(users)):
    node_features[i][0] = inf
    node_features[i][1] = 0
    node_features[i][3]= 0
    node_features[i][4] = inf
    node_features[i][5] = 0
for txn in range(len(data_csv)):
    row = data_csv.loc[txn]
    fuser = row["from"]
    tuser = row["to"]
    fid = users[fuser]
    tid = users[tuser]
    fip = row["fip"]
    tip = row["tip"]
    labels[fid] = fip
    labels[tid] = tip
    edge_list[0].append(fid)
    edge_list[1].append(tid)
    node_features[fid][2] += 1
    node_features[tid][6] += 1
    amount = float(row["amount"])
    f = [amount, row["time"]]
    edge_attr.append(f)
    node_features[fid][3] += amount
    node_features[tid][7] += amount
    #Updating the node_features

    node_features[fid][0] = min(node_features[fid][0], amount)
    node_features[fid][1] = max(node_features[fid][1], amount)
    node_features[tid][4] = min(node_features[tid][4], amount)
    node_features[tid][5] = max(node_features[tid][5], amount)

#Setting up train_mask and test_mask for the nodes 
fraud = []
non_fraud=  []
train_mask = [False] * len(users)
test_mask = [False] * len(users)

for i in range(len(users)):
    if(labels[i] == 1):
        fraud.append(i)
    else:
        non_fraud.append(i)

train_ratio = 0.8
test_ratio = 0.3
train_count_fraud = int(len(fraud) * train_ratio)
np.random.shuffle(fraud)
np.random.shuffle(non_fraud)
for i in range(int(train_ratio * len(fraud))):
    train_mask[fraud[i]] = True

for i in range(int(train_ratio * len(non_fraud))):
    train_mask[non_fraud[i]] = True

for i in range(len(users)):
    test_mask[i] = not train_mask[i]

ftrain = 0
nftrain = 0
ftest = 0
nftest = 0
for i in range(len(users)):
    if(train_mask[i]):
        if(labels[i] == 1):
            ftrain += 1
        else:
            nftrain += 1
    else:
        if(labels[i] == 1):
            ftest += 1
        else:
            nftest += 1
for i in range(len(users)):
    if(node_features[i][0] == inf):
        node_features[i][0] = 0
    if(node_features[i][4] == inf):
        node_features[i][4] = 0
        
for i in range(len(users)):
    if(node_features[i][2] != 0):
        node_features[i][3] /= node_features[i][2]
    if(node_features[i][6] != 0):
        node_features[i][7] /= node_features[i][6]

#Converting arrays to tensors 
node_features = torch.tensor(node_features, dtype=torch.float32)
edge_list = torch.tensor(edge_list)
edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
train_mask = torch.tensor(train_mask)
test_mask = torch.tensor(test_mask)
y = [0] * len(users)
for i in range(len(users)):
    y[i] = labels[i]
    
labels = y
labels= torch.tensor(labels)
graph = Data(x=node_features, y = labels, edge_index=edge_list, edge_attr=edge_attr, train_mask=train_mask, test_mask= test_mask)

class GCRNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(8, 12)
        self.rnn = nn.RNN(12, 16, batch_first=True)
        self.conv2 = GCNConv(16, 2)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x, _ = self.rnn(x.unsqueeze(0))
        x = x.squeeze(0)
        output = self.conv2(x, edge_index)
        return output

def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200):
    for epoch in range(1, n_epochs + 1):
        model.train()
        print("Epoch = ", epoch)
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out[graph.train_mask], graph.y[train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:
            acc = evaluate(model, graph, graph.test_mask)
            print(f'Epoch : {epoch:03d}, Train los : {loss:.3f}, Acc : {acc:.3f}')
    return model

def evaluate(model, graph, mask):
    model.eval()
    pred = model(graph).argmax(dim = 1)
    correct_fraud = 0
    correct_nfraud = 0
    for i in range(len(users)):
        if(labels[i] == 1):
            if(labels[i] == pred[i]):
                correct_fraud += 1
        else:
            if(labels[i] == pred[i]):
                correct_nfraud += 1
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())
    return acc

gcrnn = GCRNN()
optimizer = torch.optim.Adam(gcrnn.parameters())
criterion = torch.nn.CrossEntropyLoss()
print("Model created")
gcn = train_node_classifier(gcrnn, graph, optimizer, criterion)

























        
        

