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

# Custom module functions
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
# random_indices = np.random.choice(df.shape[0], size=10000, replace=False)
X = df[genes].values
y, _ = encode_cell_types(df['cell_type_A'].values.flatten())
coordinates = df[['x', 'y']].values

# Construct adjacency matrices
sim_edge_index, _ = construct_similarity_adjacency(X)
spatial_edge_index, _ = construct_spatial_adjacency(coordinates)

print(sim_edge_index)

# Build the heterogeneous graph
edge_index_dict = {
    ('cell', 'spatially_close_to', 'cell'): {'edge_index': spatial_edge_index},
    ('cell', 'similar_to', 'cell'): {'edge_index': sim_edge_index}
}

hetero_data = HeteroData(edge_index_dict)
hetero_data['cell'].x = torch.tensor(X, dtype=torch.float)
hetero_data['cell'].y = torch.tensor(y, dtype=torch.long)


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

train_loader = HGTLoader(hetero_data, num_samples=[64], shuffle=True, batch_size = 128,
                         input_nodes=('cell', hetero_data['cell'].train_mask))
val_loader = HGTLoader(hetero_data, num_samples=[64], batch_size = 128,
                           input_nodes=('cell', hetero_data['cell'].test_mask))

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
out_channels = hetero_data['cell'].y.max().item() + 1

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

# Final Evaluation on Test Set
test_loader = NeighborLoader(
    hetero_data,
    num_neighbors={key: [-1] for key in hetero_data.edge_types},
    batch_size=128,
    input_nodes=('cell', hetero_data['cell'].test_mask),
)

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)
        preds = out['cell'].argmax(dim=1)
        labels = batch['cell'].y
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

y_pred = torch.cat(all_preds).numpy()
y_true = torch.cat(all_labels).numpy()

# Compute and plot the confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(y_true, y_pred))
