import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
from utils import EarlyStopping, write_result_to_csv
from torch_geometric.utils import to_dense_adj

project_dir = os.environ.get('PROJECT', os.curdir)
project_dir += '/GML-Group-Project'
local_dir = os.environ.get('LOCAL', os.curdir)
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

MODEL_NAME = 'GTN'
ANNOTATED_DATA = os.path.join(project_dir, 'data/B004_training_dryad.csv')
UNANNOTATED_DATA = os.path.join(project_dir, 'data/B0056_unnanotated_dryad.csv')
MODEL_PATH = os.path.join(project_dir, 'model')
IMAGE_PATH = os.path.join(project_dir, 'image')
RESULT_PATH = os.path.join(project_dir, 'result')
CHECK_POINT_PATH = os.path.join(local_dir, 'checkpoints')

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
if not os.path.exists(IMAGE_PATH):
    os.mkdir(IMAGE_PATH)
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)
if not os.path.exists(CHECK_POINT_PATH):
    os.mkdir(CHECK_POINT_PATH)

dist_thresholds = {10, 20, 30, 40, 50}
neighborhood_size_threshold = 10 * (task_id + 1)
sample_rate = 1

#Read training data
train_df = pd.read_csv(ANNOTATED_DATA, low_memory=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#############################################################
# Define the GTN layers and model
# Code adapted from the official GTN implementation:
# https://github.com/seongjunyun/Graph_Transformer_Network
#############################################################

class GTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_A, num_channels, dropout=0.0):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_A = num_A
        self.num_channels = num_channels
        self.is_train = True
        self.weight = nn.Parameter(torch.zeros(num_channels, in_channels, out_channels))
        self.att_weight = nn.Parameter(torch.zeros(num_channels, in_channels, num_A))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att_weight)
        self.dropout = dropout

    def forward(self, As, H):
        # As: list of adjacency matrices (num_A)
        # H: node embeddings [N, in_channels]
        # Output: new node embeddings [N, out_channels], meta adjacency selection
        # For simplicity, let's assume a single layer of GTN which chooses adjacency combos.

        # Compute attention scores for each channel
        # att: [num_channels, in_channels, num_A]
        att = self.att_weight
        # We want to select adjacency matrices from As based on att:
        # Expand H: [N, in_channels] -> broadcasting over channels
        # We'll first compute a linear transform: H' = H W  (for each channel)
        # shape: H: [N, in_channels], weight: [num_channels, in_channels, out_channels]
        # After matrix multiplication: H': [num_channels, N, out_channels]

        H_prime = []
        for c in range(self.num_channels):
            Hc = H @ self.weight[c]  # [N, out_channels]
            # Now combine adjacency matrices based on attention
            # att[c] shape: [in_channels, num_A], but we need scalar weights for A selection

            att_c = att[c].mean(dim=0)  # [num_A]
            score = F.softmax(att_c, dim=0)  # distribution over As

            # Combine adjacency:
            A_combined = torch.sparse_coo_tensor([[],[]], [], size = As[0].shape, device=device, dtype=torch.float)
            for i, A in enumerate(As):
                A_combined += score[i] * A

            # Now propagate Hc through A_combined
            Hc = torch.sparse.mm(A_combined, Hc)
            H_prime.append(Hc)

        H_prime = torch.stack(H_prime, dim=0)  # [num_channels, N, out_channels]
        # Mean pooling over channels
        H_out = H_prime.mean(dim=0)  # [N, out_channels]

        if self.dropout > 0:
            H_out = F.dropout(H_out, p=self.dropout, training=self.training)
        return H_out


class GTN(nn.Module):
    def __init__(self, num_edge_types, num_channels, in_features, hidden_features, out_features, num_layers=1,
                 dropout=0.5):
        super(GTN, self).__init__()
        self.num_edge_types = num_edge_types
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.dropout = dropout

        layers = []
        in_dim = in_features
        for l in range(num_layers):
            out_dim = hidden_features if l < num_layers - 1 else out_features
            layers.append(GTLayer(in_dim, out_dim, num_edge_types, num_channels, dropout=dropout))
            in_dim = out_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, As, X):
        H = X
        for i, layer in enumerate(self.layers):
            H = layer(As, H)
            if i < self.num_layers - 1:
                H = F.relu(H)
        return torch.squeeze(H)

for dist_threshold in dist_thresholds:
    # Load train and test data
    X, Y, spatial_edges, similarity_edges, inverse_dict = utils.load_hubmap_data(train_df, dist_threshold, neighborhood_size_threshold, sample_rate)

    num_nodes = X.shape[0]
    X = torch.tensor(X, device=device, dtype = torch.float)
    Y = torch.tensor(Y, device=device, dtype = torch.long)
    spatial_edges = torch.tensor(spatial_edges, device=device, dtype=torch.long).T
    similarity_edges = torch.tensor(similarity_edges, device=device, dtype=torch.long).T

    A_spatial_coo = torch.sparse_coo_tensor(spatial_edges, values=torch.ones(spatial_edges.shape[1]) ,size=(num_nodes, num_nodes), device=device, dtype=torch.float)
    A_similar_coo = torch.sparse_coo_tensor(similarity_edges, values=torch.ones(similarity_edges.shape[1]), size=(num_nodes, num_nodes), device=device, dtype=torch.float)
    As = [A_spatial_coo, A_similar_coo]  # list of adjacency matrices

    # Assign train and validation masks
    train_ratio = 0.7
    val_ratio = 0.1
    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    perm = torch.randperm(num_nodes)
    train_idx = perm[:num_train]
    val_idx = perm[num_train:num_train + num_val]
    test_idx = perm[num_train + num_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True

    num_classes = Y.max().item() + 1
    #############################################################
    # Initialize and train the GTN model
    #############################################################

    in_channels = X.shape[1]
    model = GTN(num_edge_types=len(As),
                num_channels=2,  # can be tuned
                in_features=in_channels,
                hidden_features=128,
                out_features=num_classes,
                num_layers=2,
                dropout=0.5).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 200
    early_stopping = EarlyStopping(patience=10, path=os.path.join(CHECK_POINT_PATH, f'{MODEL_NAME}_best_model.pt'))

    for epoch in tqdm(range(1, num_epochs + 1), desc='Epoch'):
        model.train()
        optimizer.zero_grad()
        out = model(As, X)
        loss = criterion(out[train_mask], Y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = out[val_mask]
            val_pred = val_out.argmax(dim=-1)
            val_correct = (val_pred == Y[val_mask]).sum().item()
            val_loss = criterion(out[val_mask], Y[val_mask]).item()
            val_acc = val_correct / val_mask.sum().item()
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        if epoch % 20 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
    model.eval()
    with torch.no_grad():
        test_out = out[test_mask]
        test_pred = test_out.argmax(dim=-1)
        test_correct = (test_pred == Y[test_mask]).sum().item()
        test_acc = test_correct / test_mask.sum().item()
        test_loss = criterion(out[test_mask], Y[test_mask]).item()


    print(f'Model: {MODEL_NAME}, Threshold Distance: {dist_threshold}, TopK: {neighborhood_size_threshold}, Epoch: {epoch}, Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    write_result_to_csv(RESULT_PATH, MODEL_NAME, dist_threshold, neighborhood_size_threshold, test_loss, test_acc)

    # Save model
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, f'{MODEL_NAME}_model_{dist_threshold}_{neighborhood_size_threshold}_weights.pth'))
