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
import matplotlib.pyplot as plt
data_csv = pd.read_csv("data.csv")
users = {}
labels = {}
counter = 0
edge_list = set()
for i in range(1000):
    row = data_csv.loc[i]
    fuser = row["from"]
    tuser = row["to"]
    if(not(fuser in users)):
        users[fuser] = counter
        counter += 1
    if(not(tuser in users)):
        users[tuser] = counter
        counter += 1
    edge_list.add((users[fuser], users[tuser]))
num_nodes = len(users)
edges = []
for edge in edge_list:
    edges.append(edge)

G = nx.Graph()
nodes = [i+1 for i in range(len(users))]
G.add_nodes_from(nodes)
G.add_edges_from(edges)
pos = nx.spring_layout(G)
scale_factor = 2.0
pos_scaled = {node : (x * scale_factor, y * scale_factor) for node, (x, y) in pos.items()}
nx.draw(G, pos=pos_scaled, node_size=1, edge_color='lightblue')
plt.show()



















        
        

