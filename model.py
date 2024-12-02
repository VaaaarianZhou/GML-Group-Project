import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from torch_geometric.data import HeteroData
from torch_geometric.nn.models import MetaPath2Vec

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
meta_path = [('cell', 'spatially_close_to', 'cell'), ('cell', 'similar_to', 'cell')]
edge_dict = {
    ('cell', 'spatially_close_to', 'cell'): {'edge_index': spatial_edge_index.to(device)},
    ('cell', 'similar_to', 'cell'): {'edge_index': sim_edge_index.to(device)}
}

hetero_data = HeteroData(edge_dict)
hetero_data['cell'].x = torch.tensor(X, device=device, dtype=torch.float)
hetero_data['cell'].y = torch.tensor(y, device=device, dtype=torch.long)

# Initialize the MetaPath2Vec model
HIDDEN_DIM = 128
metapath2vec_model = MetaPath2Vec(
    edge_index_dict=hetero_data.edge_index_dict,
    embedding_dim=HIDDEN_DIM,
    metapath=meta_path,
    walk_length=2,
    context_size=2
).to(device)

loader = metapath2vec_model.loader(batch_size=128, shuffle=True, num_workers=4)
optimizer = optim.Adam(metapath2vec_model.parameters(), lr=0.01)

def train(epoch, log_steps=100):
    metapath2vec_model.train()
    total_loss = 0
    losses = []
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = metapath2vec_model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        losses.append(loss.item())
        if (i + 1) % log_steps == 0:
            avg_loss = total_loss / log_steps
            print(f'Epoch: {epoch}, Step: {i + 1}/{len(loader)}, Loss: {avg_loss:.4f}')
            total_loss = 0
    return losses

@torch.no_grad()
def test():
    metapath2vec_model.eval()
    z = metapath2vec_model('cell')
    y = hetero_data['cell'].y

    perm = torch.randperm(z.size(0))
    train_perm = perm[:int(z.size(0) * 0.1)]
    test_perm = perm[int(z.size(0) * 0.1):]

    return metapath2vec_model.test(
        z[train_perm], y[train_perm],
        z[test_perm], y[test_perm],
        max_iter=150
    )

# Training the MetaPath2Vec model
train_losses = []
test_accuracies = []

for epoch in range(1, 10):
    epoch_losses = train(epoch)
    train_losses.extend(epoch_losses)
    acc = test()
    print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
    test_accuracies.append(acc)

# Plot the MetaPath2Vec training loss curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.title('MetaPath2Vec Training Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

# Plot the test accuracy over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, 10), test_accuracies)
plt.title('MetaPath2Vec Test Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Compute the cell embeddings
cell_embeddings = metapath2vec_model('cell')
labels = hetero_data['cell'].y

# Convert to NumPy arrays and split
X = cell_embeddings.cpu().detach().numpy()
y = labels.cpu().detach().numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert back to tensors
X_train = torch.tensor(X_train, device=device, dtype=torch.float)
X_test = torch.tensor(X_test, device=device, dtype=torch.float)
y_train = torch.tensor(y_train, device=device, dtype=torch.long)
y_test = torch.tensor(y_test, device=device, dtype=torch.long)

# Define the MLP classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

input_dim = cell_embeddings.size(1)
hidden_dim = 64
output_dim = labels.max().item() + 1

model = MLPClassifier(input_dim, hidden_dim, output_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from torch.utils.data import TensorDataset, DataLoader

batch_size = 128

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

def train_model(model, criterion, optimizer, train_loader, num_epochs=20):
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        epoch_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return epoch_losses

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
    return all_preds, all_labels

# Train the MLP classifier
num_epochs = 20
mlp_train_losses = train_model(model, criterion, optimizer, train_loader, num_epochs)

# Plot the MLP training loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), mlp_train_losses)
plt.title('MLP Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Evaluate the model
preds, labels = evaluate_model(model, test_loader)
print(classification_report(labels, preds))

# Compute and plot the confusion matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
