import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import HANConv
from tqdm import tqdm
import torch.nn.functional as F
from utils import load_hubmap_data, visualize_predictions, write_result_to_csv, EarlyStopping, load_tonsilbe_data, load_hubmap_data_CL_SB

project_dir = os.environ.get('PROJECT', os.curdir)
project_dir += '/GML-Group-Project'
local_dir = os.environ.get('LOCAL', os.curdir)
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

MODEL_NAME = 'HAN'
ANNOTATED_DATA = os.path.join(project_dir, 'data/B004_training_dryad.csv')
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

dist_thresholds = [20,30,40,50]
neighborhood_size_thresholds = [1,3,5,10]
sample_rate = .5

#Read the data DataFrame
# Read training data
df = pd.read_csv(ANNOTATED_DATA, low_memory=False)

for neighborhood_size_threshold in neighborhood_size_thresholds:
    for dist_threshold in dist_thresholds:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        torch.cuda.empty_cache()
    
        # Load train and test data
        train_X, train_y, test_X, test_y, labeled_spatial_edges, unlabeled_spatial_edges, labeled_similarity_edges, unlabeled_similarity_edges, inverse_dict = load_hubmap_data_CL_SB(
            df,
            dist_threshold,
            neighborhood_size_threshold,
            sample_rate)
    
        # Convert them into torch.tensor
        train_X = torch.tensor(train_X, device=device, dtype=torch.float)
        train_y = torch.tensor(train_y, device=device, dtype=torch.long)
        labeled_spatial_edges = torch.tensor(labeled_spatial_edges, device=device, dtype=torch.long).T
        labelled_similarity_edges = torch.tensor(labeled_similarity_edges, device=device, dtype=torch.long).T
    
        # Build the heterogeneous graphs
        train_edge_index_dict = {
            ('cell', 'spatially_close_to', 'cell'): {'edge_index': labeled_spatial_edges},
            ('cell', 'similar_to', 'cell'): {'edge_index': labelled_similarity_edges}
        }
    
        # Define training dataset
        train_hetero_data = HeteroData(train_edge_index_dict)
        train_hetero_data['cell'].x = train_X
        train_hetero_data['cell'].y = train_y
    
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
    
        train_hetero_data['cell'].val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_hetero_data['cell'].val_mask[val_idx] = True
    
        train_loader = HGTLoader(train_hetero_data, num_samples=[1024] * 2, shuffle=True, batch_size=128,
                                 input_nodes=('cell', train_hetero_data['cell'].train_mask))
        val_loader = HGTLoader(train_hetero_data, num_samples=[1024] * 2, batch_size=128,
                               input_nodes=('cell', train_hetero_data['cell'].val_mask))
        # Get metadata
        metadata = train_hetero_data.metadata()
    
        # Define the HAN model
        class HAN(torch.nn.Module):
            def __init__(self, metadata, in_channels, out_channels, hidden_channels=128, heads=2):
                super().__init__()
                self.conv1 = HANConv(in_channels, hidden_channels, heads=heads, dropout=0.1, metadata=metadata)
                self.conv2 = HANConv(hidden_channels, hidden_channels, heads=heads, dropout=0.1, metadata=metadata)
                self.lin = nn.Linear(hidden_channels, out_channels)
    
            def forward(self, x_dict, edge_index_dict):
                out = self.conv1(x_dict, edge_index_dict)
                out = {key: x.relu() for key, x in out.items()}
                out = self.conv2(out , edge_index_dict)
                out = {key: x.relu() for key, x in out.items()}
                out = self.lin(out['cell'])
                return out
    
        in_channels = train_hetero_data['cell'].x.size(1)
        num_classes = train_hetero_data['cell'].y.max().item() + 1
    
        model = HAN(metadata, in_channels, num_classes, hidden_channels=256, heads=4).to(device)
    
        # Define optimizer and loss function
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    
        num_epochs = 200
        early_stopping = EarlyStopping(patience=10, path=os.path.join(CHECK_POINT_PATH, f'{MODEL_NAME}_best_model.pt'))
    
        # print(hetero_data.edge_index_dict)
        for epoch in tqdm(range(1, num_epochs + 1), desc='Epoch'):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                batch = batch.to(device, 'edge_index')
                batch_size = batch['cell'].batch_size
                out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
                loss = F.cross_entropy(out, batch['cell'].y[:batch_size])
                loss.backward()
                optimizer.step()
    
            # Validation
            model.eval()
            total_examples = total_correct = 0
            total_loss = 0
            for batch in val_loader:
                batch = batch.to(device, 'edge_index')
                batch_size = batch['cell'].batch_size
                out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
                pred = out.argmax(dim=-1)
                loss = F.cross_entropy(out, batch['cell'].y[:batch_size])
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size
                total_correct += int((pred == batch['cell'].y[:batch_size]).sum())
    
            val_acc = total_correct / total_examples
            val_loss = total_loss / total_examples
            # Call early stopping with the current validation loss and model
            early_stopping(val_loss, model)
    
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
            # test_accuracies.append(val_acc)
            total_examples = total_correct = 0
    
        test_X = torch.tensor(test_X, device=device, dtype=torch.float)
        test_y = torch.tensor(test_y, device=device, dtype=torch.long)
        unlabeled_spatial_edges = torch.tensor(unlabeled_spatial_edges, device=device, dtype=torch.long).T
        unlabeled_similarity_edges = torch.tensor(unlabeled_similarity_edges, device=device, dtype=torch.long).T
    
        test_edge_index_dict = {
            ('cell', 'spatially_close_to', 'cell'): {'edge_index': unlabeled_spatial_edges},
            ('cell', 'similar_to', 'cell'): {'edge_index': unlabeled_similarity_edges}
        }
        # Define test dataset
        test_hetero_data = HeteroData(test_edge_index_dict)
        test_hetero_data['cell'].x = test_X
        test_hetero_data['cell'].y = test_y
        test_loader = HGTLoader(test_hetero_data, num_samples=[1024] * 2, batch_size=128, input_nodes=('cell', None))
    
        for batch in test_loader:
            batch = batch.to(device, 'edge_index')
            batch_size = batch['cell'].batch_size
            out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
            pred = out.argmax(dim=-1)
            loss = F.cross_entropy(out, batch['cell'].y[:batch_size])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            total_correct += int((pred == batch['cell'].y[:batch_size]).sum())
    
        test_acc = total_correct / total_examples
        test_loss = total_loss / total_examples
    
        print(
            f'Model: {MODEL_NAME}, Threshold Distance: {dist_threshold}, TopK: {neighborhood_size_threshold}, Epoch: {epoch}, Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        write_result_to_csv(RESULT_PATH, MODEL_NAME, dist_threshold, neighborhood_size_threshold, test_loss, test_acc)
    
        # Load the model's state dictionary on CPU
        torch.save(model.state_dict(),
                   os.path.join(MODEL_PATH,
                                f'{MODEL_NAME}_model_{dist_threshold}_{neighborhood_size_threshold}_weights.pth'))
