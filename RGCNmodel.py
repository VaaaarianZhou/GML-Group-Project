import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, HGTLoader
from torch_geometric.nn import HeteroConv, GCNConv
from tqdm import tqdm
import os
from utils import load_hubmap_data, visualize_predictions

project_dir = os.environ.get('PROJECT', os.curdir)

ANNOTATED_DATA = os.path.join(project_dir, 'data/B004_training_dryad.csv')
UNANNOTATED_DATA = os.path.join(project_dir, 'data/B0056_unnanotated_dryad.csv')
MODEL_PATH = os.path.join(project_dir, 'model')
IMAGE_PATH = os.path.join(project_dir, 'image')
RESULT_PATH = os.path.join(project_dir, 'result')

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
if not os.path.exists(IMAGE_PATH):
    os.mkdir(IMAGE_PATH)
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

dist_threshold = 30
neighborhood_size_threshold = 10
sample_rate = .01

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Load train and test data
train_X, train_y, test_X, labeled_spatial_edges, unlabeled_spatial_edges, labeled_similarity_edges, unlabeled_similarity_edges, inverse_dict = load_hubmap_data(ANNOTATED_DATA, UNANNOTATED_DATA, dist_threshold, neighborhood_size_threshold, sample_rate)

# Convert them into torch.tensor
train_X = torch.tensor(train_X, device=device, dtype = torch.float)
train_y = torch.tensor(train_y, device=device, dtype = torch.long)
test_X = torch.tensor(test_X, device=device, dtype = torch.float)
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

''' Training RGCN model'''
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

train_loader = HGTLoader(train_hetero_data, num_samples=[64], shuffle=True, batch_size = 128,
                             input_nodes=('cell', train_hetero_data['cell'].train_mask))
val_loader = HGTLoader(train_hetero_data, num_samples=[64], batch_size = 128,
                           input_nodes=('cell', train_hetero_data['cell'].test_mask))

# Get metadata
metadata = train_hetero_data.metadata()
# Define the Heterogeneous RGCN model
class HeteroRGCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HeteroConv({
            ('cell', 'spatially_close_to', 'cell'): GCNConv(-1, hidden_channels),
            ('cell', 'similar_to', 'cell'): GCNConv(-1, hidden_channels),
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

hidden_channels = 64
out_channels = train_hetero_data['cell'].y.max().item() + 1

model = HeteroRGCN(hidden_channels, out_channels).to(device)

# Define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

# Training loop
train_losses = []
test_accuracies = []
num_epochs = 20

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        out = out['cell']
        y = batch['cell'].y
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            preds = out['cell'].argmax(dim=1)
            labels = batch['cell'].y
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    test_accuracies.append(val_acc)

    print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}, Test Accuracy: {val_acc:.4f}')

# Plot the training loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# Plot the test accuracy over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.show()

# Load the model's state dictionary on CPU
torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'HAN_model_weights.pth'))


# Evaluate the model
model.eval()
with torch.no_grad():
    out = model(test_hetero_data.x_dict, test_hetero_data.edge_index_dict)
    pred = out.argmax(dim=1)
    y_pred = pred.cpu().numpy()
    visualize_predictions(test_X, y_pred, inverse_dict, os.path.join(IMAGE_PATH, 'UMAP_HAN_Modelpwd.png'))
    np.save(os.path.join(RESULT_PATH, 'pred.npy'), y_pred)
    # Save dictionary to a file
    with open(os.path.join(IMAGE_PATH, 'inverse_dict.pkl'), 'wb') as fp:
        pickle.dump(inverse_dict, fp)
    print('Dictionary saved successfully to file')
