import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv

# Custom module functions (ensure these are properly defined in your 'data_processing' module)
from data_processing import (
    load_csv,
    encode_cell_types,
    construct_similarity_adjacency,
    construct_spatial_adjacency
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
TRAINING_DATA_PATH = 'data/B004_training_dryad.csv'
df = load_csv(TRAINING_DATA_PATH)

# List of genes
genes = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1',
         'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3',
         'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68',
         'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CK7', 'CD117',
         'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

df.rename(columns={'Unnamed: 0': 'cell_id'}, inplace=True)

# Randomly select a subset of data
random_indices = torch.randint(low=0, high=df.shape[0], size=(10000,)).tolist()
X = df.loc[random_indices, genes].values
y, _ = encode_cell_types(df.loc[random_indices, 'cell_type_A'].values.flatten())
coordinates = df.loc[random_indices, ['x', 'y']].values

# Construct adjacency matrices
sim_edge_index, _ = construct_similarity_adjacency(X)
spatial_edge_index, _ = construct_spatial_adjacency(coordinates)

# Build the heterogeneous graph
edge_index_dict = {
    ('cell', 'spatially_close_to', 'cell'): {'edge_index': spatial_edge_index.to(device)},
    ('cell', 'similar_to', 'cell'): {'edge_index' :sim_edge_index.to(device)}
}

hetero_data = HeteroData(edge_index_dict)
hetero_data['cell'].x = torch.tensor(X, device=device, dtype=torch.float)
hetero_data['cell'].y = torch.tensor(y, device=device, dtype=torch.long)
# hetero_data.edge_index_dict = edge_index_dict

# Assign train and test masks
num_nodes = hetero_data['cell'].num_nodes
train_ratio = 0.8
num_train = int(num_nodes * train_ratio)

perm = torch.randperm(num_nodes)
train_idx = perm[:num_train]
test_idx = perm[num_train:]

hetero_data['cell'].train_mask = torch.zeros(num_nodes, dtype=torch.bool)
hetero_data['cell'].train_mask[train_idx] = True

hetero_data['cell'].test_mask = torch.zeros(num_nodes, dtype=torch.bool)
hetero_data['cell'].test_mask[test_idx] = True

# Get metadata
metadata = hetero_data.metadata()
# print(metadata)

# Define the HAN model
class HAN(torch.nn.Module):
    def __init__(self, metadata, in_channels, out_channels, hidden_channels=64, heads=8):
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

in_channels = hetero_data['cell'].x.size(1)
num_classes = hetero_data['cell'].y.max().item() + 1

model = HAN(metadata, in_channels, num_classes, hidden_channels=64, heads=8).to(device)

# Define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

# Training loop
train_losses = []
test_accuracies = []
num_epochs = 200

print(hetero_data.edge_index_dict)

for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(hetero_data.x_dict, hetero_data.edge_index_dict)
    loss = criterion(
        out[hetero_data['cell'].train_mask],
        hetero_data['cell'].y[hetero_data['cell'].train_mask]
    )
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        out = model(hetero_data.x_dict, hetero_data.edge_index_dict)
        pred = out.argmax(dim=1)
        test_correct = pred[hetero_data['cell'].test_mask] == hetero_data['cell'].y[hetero_data['cell'].test_mask]
        test_acc = int(test_correct.sum()) / int(hetero_data['cell'].test_mask.sum())
        test_accuracies.append(test_acc)

    print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Test Accuracy: {test_acc:.4f}')

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

# Evaluate the model
model.eval()
with torch.no_grad():
    out = model(hetero_data.x_dict, hetero_data.edge_index_dict)
    pred = out.argmax(dim=1)
    y_pred = pred[hetero_data['cell'].test_mask].cpu().numpy()
    y_true = hetero_data['cell'].y[hetero_data['cell'].test_mask].cpu().numpy()

# Compute and plot the confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(y_true, y_pred))
