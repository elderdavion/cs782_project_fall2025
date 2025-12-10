import torch
import numpy as np
from torch_geometric.data import HeteroData
from collections import defaultdict

def per_user_leave_one_out(data, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    edge_index = data['user', 'contributes_to', 'repo'].edge_index
    num_edges = edge_index.size(1)
    
    user_ids = edge_index[0].tolist()
    repo_ids = edge_index[1].tolist()
    
    # Group edges by user
    user_to_edges = defaultdict(list)
    for eid, u in enumerate(user_ids):
        user_to_edges[u].append(eid)
    
    train_edges = []
    val_edges = []
    test_edges = []
    
    for u, edges in user_to_edges.items():
        edges = list(edges)
        deg = len(edges)
        
        if deg == 1:
            # cannot evaluate this user
            train_edges.extend(edges)
        
        elif deg == 2:
            # 1 for test, 1 for train
            np.random.shuffle(edges)
            test_edges.append(edges[0])
            train_edges.append(edges[1])
        
        else:
            # ≥3 edges: train / val / test
            np.random.shuffle(edges)
            test_edges.append(edges[0])
            val_edges.append(edges[1])
            train_edges.extend(edges[2:])
    
    return train_edges, val_edges, test_edges

def build_split(data, edge_ids):
    edge_ids = torch.tensor(edge_ids)
    new_data = HeteroData()
    
    new_data['user'].x = data['user'].x
    new_data['repo'].x = data['repo'].x
    
    full_ei = data['user', 'contributes_to', 'repo'].edge_index
    full_ea = data['user', 'contributes_to', 'repo'].edge_attr
    
    new_data['user', 'contributes_to', 'repo'].edge_index = full_ei[:, edge_ids]
    new_data['user', 'contributes_to', 'repo'].edge_attr = full_ea[edge_ids]
    
    return new_data

data = torch.load("/Users/hnasrolahi/Desktop/cs782project/hetero_graph.pt", weights_only=False)

train_ids, val_ids, test_ids = per_user_leave_one_out(data, seed=42)

train_data = build_split(data, train_ids)
val_data = build_split(data, val_ids)
test_data = build_split(data, test_ids)

torch.save(train_data, "/Users/hnasrolahi/Desktop/cs782project/train_data.pt")
torch.save(val_data, "/Users/hnasrolahi/Desktop/cs782project/val_data.pt")
torch.save(test_data, "/Users/hnasrolahi/Desktop/cs782project/test_data.pt")

print("Train edges:", len(train_ids))
print("Val edges:", len(val_ids))
print("Test edges:", len(test_ids))
