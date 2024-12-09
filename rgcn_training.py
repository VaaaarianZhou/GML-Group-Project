import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import HeteroConv, GCNConv
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


# Define the Heterogeneous RGCN model
class HeteroRGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HeteroConv({
            ('cell', 'spatially_close_to', 'cell'): GCNConv(in_channels, hidden_channels),
            ('cell', 'similar_to', 'cell'): GCNConv(in_channels, hidden_channels),
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('cell', 'spatially_close_to', 'cell'): GCNConv(hidden_channels, out_channels),
            ('cell', 'similar_to', 'cell'): GCNConv(hidden_channels, out_channels),
        }, aggr='mean')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

in_channels = train_hetero_data['cell'].x.size(1)
hidden_channels = 256
out_channels = train_hetero_data['cell'].y.max().item() + 1

model = HeteroRGCN(in_channels, hidden_channels, out_channels).to(device)

# Define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Training loop
train_losses = []
test_accuracies = []
num_epochs = 100

# print(hetero_data.edge_index_dict)

for epoch in tqdm(range(1, num_epochs + 1), desc='Epoch'):
    model.train()
    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device, 'edge_index')
        batch_size = batch['cell'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)['cell']
        y = batch['cell'].y
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
    train_losses.append(total_loss)


    # Validation
    model.eval()
    total_examples = total_correct = 0
    for batch in val_loader:
        batch = batch.to(device, 'edge_index')
        batch_size = batch['cell'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)['cell']
        pred = out.argmax(dim=-1)
        labels = batch['cell'].y
        total_examples += labels.size(0)
        total_correct += int((pred == labels).sum())

    val_acc = total_correct / total_examples
    test_accuracies.append(val_acc)
    print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Test Accuracy: {val_acc:.4f}')

model_path = "RGCN_model_cl_sb.pth"  # Choose a suitable path and filename
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Evaluate the model
model.eval()
with torch.no_grad():
    out = model(test_hetero_data.x_dict, test_hetero_data.edge_index_dict)['cell']
    pred = out.argmax(dim=1)
    y_pred = pred.cpu().numpy()

accuracy = accuracy_score(y_pred, test_y)
print(f"Accuracy: {accuracy}")