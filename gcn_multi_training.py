import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from torch_geometric.data import HeteroData, Data, DataLoader
from torch_geometric.loader import HGTLoader

from torch_geometric.nn import GCNConv
from tqdm import tqdm
import torch.nn.functional as F
import utils
from utils import load_hubmap_data_CL_SB, load_hubmap_data_CL_CL

# Custom module functions (ensure these are properly defined in your 'data_processing' module)
from data_processing import (
    load_csv,
    encode_cell_types,
    construct_similarity_adjacency,
    construct_spatial_adjacency
)

dist_threshold = 30
neighborhood_size_threshold = 5
sample_rate = 1
train_X, train_y, test_X, test_y, labeled_spatial_edges, unlabeled_spatial_edges, labeled_similarity_edges, unlabeled_similarity_edges, inverse_dict = load_hubmap_data_CL_CL('/content/B004_training_dryad.csv', dist_threshold, neighborhood_size_threshold, sample_rate)


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Convert them into torch.tensor
train_X = torch.tensor(train_X, device=device, dtype = torch.float)
train_y = torch.tensor(train_y, device=device, dtype = torch.long)
test_X = torch.tensor(test_X, device=device, dtype = torch.float)
# test_y = torch.tensor(test_y, device=device, dtype = torch.long)
labeled_spatial_edges = torch.tensor(labeled_spatial_edges, device=device, dtype = torch.long).T
unlabeled_spatial_edges = torch.tensor(unlabeled_spatial_edges, device=device, dtype = torch.long).T
labelled_similarity_edges = torch.tensor(labeled_similarity_edges, device=device, dtype = torch.long).T
unlabelled_similarity_edges = torch.tensor(unlabeled_similarity_edges, device=device, dtype = torch.long).T

# Build the heterogeneous graph
train_edge_index_dict = {
    ('cell', 'spatially_close_to', 'cell'): {'edge_index': labeled_spatial_edges},
    ('cell', 'similar_to', 'cell'): {'edge_index' : labelled_similarity_edges}
}

test_edge_index_dict = {
    ('cell', 'spatially_close_to', 'cell'): {'edge_index': unlabeled_spatial_edges},
    ('cell', 'similar_to', 'cell'): {'edge_index' :unlabelled_similarity_edges}
}

# Define training dataset
train_hetero_data = HeteroData(train_edge_index_dict)
train_hetero_data['cell'].x = train_X
train_hetero_data['cell'].y = train_y

# Define test dataset
test_hetero_data = HeteroData(test_edge_index_dict)
test_hetero_data['cell'].x = test_X


''' Training HAN model'''
# Assign train and validation masks
num_nodes = train_hetero_data['cell'].num_nodes
train_ratio = 0.8
num_train = int(num_nodes * train_ratio)

perm = torch.randperm(num_nodes)
train_idx = perm[:num_train]
val_idx = perm[num_train:]

train_hetero_data['cell'].train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_hetero_data['cell'].train_mask[train_idx] = True

train_hetero_data['cell'].test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_hetero_data['cell'].test_mask[val_idx] = True

for edge_type in train_hetero_data.edge_types:
    edge_index = train_hetero_data[edge_type].edge_index
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        print(f"Fixing edge index for edge type: {edge_type}")
        train_hetero_data[edge_type].edge_index = edge_index.T if edge_index.shape[1] == 2 else edge_index

train_loader = HGTLoader(train_hetero_data, num_samples=[100]*2, shuffle=True, batch_size = 128,
                             input_nodes=('cell', train_hetero_data['cell'].train_mask))
val_loader = HGTLoader(train_hetero_data, num_samples=[100]*2, batch_size = 128,
                           input_nodes=('cell', train_hetero_data['cell'].test_mask))



# Define a GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x

# Prepare the graph data for GCN
edge_index = torch.cat((labeled_spatial_edges.T, labelled_similarity_edges), dim=1)

train_gcn_data = Data(
    x=train_X,
    y=train_y,
    edge_index=edge_index,
    train_mask=train_hetero_data['cell'].train_mask,
    test_mask=train_hetero_data['cell'].test_mask
).to(device)

test_gcn_data = Data(
    x=test_X,
    edge_index=torch.cat([unlabeled_spatial_edges.T, unlabelled_similarity_edges], dim=1)
).to(device)

# Instantiate the GCN model
in_channels = train_gcn_data.x.size(1)
num_classes = train_gcn_data.y.max().item() + 1
gcn_model = GCN(in_channels, num_classes, hidden_channels=256).to(device)

# Define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=0.0001)

# Training loop
train_losses = []
test_accuracies = []
num_epochs = 200

for epoch in tqdm(range(1, num_epochs + 1), desc='Epoch'):
    gcn_model.train()
    optimizer.zero_grad()

    # Training step
    out = gcn_model(train_gcn_data.x, train_gcn_data.edge_index)
    loss = criterion(out[train_gcn_data.train_mask], train_gcn_data.y[train_gcn_data.train_mask])
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Validation step
    gcn_model.eval()
    with torch.no_grad():
        logits = gcn_model(train_gcn_data.x, train_gcn_data.edge_index)
        pred = logits[train_gcn_data.test_mask].argmax(dim=-1)
        correct = (pred == train_gcn_data.y[train_gcn_data.test_mask]).sum()
        acc = int(correct) / int(train_gcn_data.test_mask.sum())
        test_accuracies.append(acc)

    print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Test Accuracy: {acc:.4f}')

with torch.no_grad():
    logits = gcn_model(test_gcn_data.x, test_gcn_data.edge_index)
    pred = logits.argmax(dim=1)
    y_pred = pred.cpu().numpy()

from sklearn.metrics import accuracy_score, f1_score

# Assuming y_pred and test_y are numpy arrays or lists
accuracy = accuracy_score(test_y, y_pred)

print(f"Accuracy: {accuracy}")

np.save("gcn_multi_y_pred_cl_sb.npy", y_pred)
