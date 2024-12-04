import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import HANConv
from tqdm import tqdm
import torch.nn.functional as F
from utils import load_hubmap_data, visualize_predictions

# Custom module functions (ensure these are properly defined in your 'data_processing' module)
from data_processing import (
    load_csv,
    encode_cell_types,
    construct_similarity_adjacency,
    construct_spatial_adjacency
)

project_dir = os.environ.get('PROJECT', os.curdir)

ANNOTATED_DATA = os.path.join(project_dir, 'data/B004_training_dryad.csv')
UNANNOTATED_DATA = os.path.join(project_dir, 'data/B0056_unnanotated_dryad.csv')
MODEL_PATH = os.path.join(project_dir, 'model')
IMAGE_PATH = os.path.join(project_dir, 'image')

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
if os.path.exists(MODEL_PATH):
    os.mkdir(IMAGE_PATH)

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

train_loader = HGTLoader(train_hetero_data, num_samples=[64], shuffle=True, batch_size = 128,
                             input_nodes=('cell', train_hetero_data['cell'].train_mask))
val_loader = HGTLoader(train_hetero_data, num_samples=[64], batch_size = 128,
                           input_nodes=('cell', train_hetero_data['cell'].test_mask))

# Get metadata
metadata = train_hetero_data.metadata()

# Define the HAN model
class HAN(torch.nn.Module):
    def __init__(self, metadata, in_channels, out_channels, hidden_channels=128, heads=2):
        super().__init__()
        self.conv1 = HANConv(in_channels, hidden_channels, heads=heads, dropout=0.6, metadata=metadata)
        self.conv2 = HANConv(hidden_channels, hidden_channels, heads=heads, dropout=0.6, metadata=metadata)
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

model = HAN(metadata, in_channels, num_classes, hidden_channels=128, heads=2).to(device)


# Define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

# Training loop
train_losses = []
test_accuracies = []
num_epochs = 200

# print(hetero_data.edge_index_dict)

for epoch in tqdm(range(1, num_epochs + 1), desc='Epoch'):
    model.train()
    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device, 'edge_index')
        batch_size = batch['cell'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]

        loss = F.cross_entropy(out, batch['cell'].y[:batch_size])
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
        out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
        pred = out.argmax(dim=-1)

        total_examples += batch_size
        total_correct += int((pred == batch['cell'].y[:batch_size]).sum())

    val_acc = total_correct / total_examples
    test_accuracies.append(val_acc)

print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Test Accuracy: {val_acc:.4f}')

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
    visualize_predictions(test_X, y_pred, inverse_dict, os.path.join(IMAGE_PATH, 'UMAP_HAN_Model.png'))
    # y_true = test_hetero_data['cell'].y[test_hetero_data['cell'].test_mask].cpu().numpy()

# Compute and plot the confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(12, 10))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.show()
#
# # Print classification report
# print(classification_report(y_true, y_pred))
