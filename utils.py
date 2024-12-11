import pip
def install_if_not_exist(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])


install_if_not_exist('umap')
import torch
import numpy as np
from builtins import range
from torch_geometric.data import InMemoryDataset, Data
from sklearn.metrics import pairwise_distances
import pandas as pd
from sklearn.preprocessing import normalize, LabelEncoder
import umap
import matplotlib.pyplot as plt
import csv
import os
import torch
import numpy as np



genes = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f',
       'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR',
       'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90',
       'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34',
       'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CK7',
       'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a',
       'CD163', 'CD161']

def get_hubmap_edge_index(X, pos, regions, distance_thres, neighborhood_size_thres):
    # construct edge indexes when there is region information
    spatial_edge_list = []

    # For similarity, src and dst
    src = []
    dst = []
    regions_unique = np.unique(regions)
    index = np.arange(X.shape[0])
    for reg in regions_unique:
        # Build spatial edges
        locs = np.where(regions == reg)[0]
        pos_region = pos[locs, :]
        dists = pairwise_distances(pos_region)
        dists_mask = dists < distance_thres
        np.fill_diagonal(dists_mask, 0)
        region_edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
        for (i, j) in region_edge_list:
            spatial_edge_list.append([locs[i], locs[j]])
        # Build molecular similarity edges

        X_region = X[locs, :]
        indices = index[locs]
        for idx, row in enumerate(X_region):
            scores = row @ X_region.T
            neighbors = scores.argsort()[-neighborhood_size_thres:]
            # src.append(np.full((neighborhood_size_thres,), idx))
            src.extend([indices[idx]]*neighborhood_size_thres)
            dst.extend(indices[neighbors])
        # similarity_edge_index = np.stack([np.concatenate(src), np.concatenate(dst)])
        # edge_weights = torch.concat(edge_weights)
    similarity_edge_list = [[u,v] for (u,v) in zip(src, dst)]
    return spatial_edge_list, similarity_edge_list

def get_tonsilbe_edge_index(pos, distance_thres):
    # construct edge indexes in one region
    edge_list = []
    dists = pairwise_distances(pos)
    dists_mask = dists < distance_thres
    np.fill_diagonal(dists_mask, 0)
    edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
    return edge_list

# def load_hubmap_data(labeled_file, unlabeled_file, distance_thres, neighborhood_size_thres, sample_rate):

def load_hubmap_data(train_df, distance_thres, neighborhood_size_thres, sample_rate):
    # test_df = pd.read_csv(unlabeled_file)
    train_df = train_df.sample(n=round(sample_rate*len(train_df)), random_state=1)
    # test_df = test_df.sample(n=round(sample_rate*len(test_df)), random_state=1)
    train_df = train_df[(train_df['tissue'] == 'CL') & (train_df['donor'] == 'B004')]
    # test_df = test_df[(test_df['tissue'] == 'CL') & (test_df['donor'] == 'B005')]
    train_X = train_df.iloc[:, 1:49].to_numpy() # node features, indexes depend on specific datasets
    # train_X = normalize(train_X)
    # test_X = test_df.iloc[:, 1:49].to_numpy()
    # test_X = normalize(test_X)
    labeled_pos = train_df.iloc[:, -6:-4].values # x,y coordinates, indexes depend on specific datasets
    # unlabeled_pos = test_df.iloc[:, -5:-3].values
    labeled_regions = train_df['unique_region']
    # unlabeled_regions = test_df['unique_region']
    train_y = train_df['cell_type_A'] # class information
    cell_types = np.sort(list(set(train_df['cell_type_A'].values))).tolist()
    # we here map class in texts to categorical numbers and also save an inverse_dict to map the numbers back to texts
    cell_type_dict = {}
    inverse_dict = {}
    for i, cell_type in enumerate(cell_types):
        cell_type_dict[cell_type] = i
        inverse_dict[i] = cell_type
    train_y = np.array([cell_type_dict[x] for x in train_y])
    labeled_spatial_edges, labeled_similarity_edges = get_hubmap_edge_index(train_X, labeled_pos, labeled_regions, distance_thres, neighborhood_size_thres)
    # unlabeled_spatial_edges, unlabeled_similarity_edges = get_hubmap_edge_index(test_X, unlabeled_pos, unlabeled_regions, distance_thres, neighborhood_size_thres)
    # return train_X, train_y, test_X, labeled_spatial_edges, unlabeled_spatial_edges, labeled_similarity_edges, unlabeled_similarity_edges, inverse_dict
    return train_X, train_y, labeled_spatial_edges, labeled_similarity_edges, inverse_dict


def load_tonsilbe_data(filename, distance_thres, sample_rate):
    df = pd.read_csv(filename)
    train_df = df.loc[df['sample_name'] == 'tonsil']
    train_df = train_df.sample(n=round(sample_rate*len(train_df)), random_state=1)
    test_df = df.loc[df['sample_name'] == 'Barretts Esophagus']
    train_X = train_df.iloc[:, 1:-4].values
    test_X = test_df.iloc[:, 1:-4].values
    train_y = train_df['cell_type'].str.lower()
    labeled_pos = train_df.iloc[:, -4:-2].values
    unlabeled_pos = test_df.iloc[:, -4:-2].values
    cell_types = np.sort(list(set(train_y))).tolist()
    cell_type_dict = {}
    inverse_dict = {}
    for i, cell_type in enumerate(cell_types):
        cell_type_dict[cell_type] = i
        inverse_dict[i] = cell_type
    train_y = np.array([cell_type_dict[x] for x in train_y])
    labeled_spatial_edges, labeled_similarity_edges = get_tonsilbe_edge_index(labeled_pos, distance_thres)
    unlabeled_spatial_edges, unlabeled_similarity_edges = get_tonsilbe_edge_index(unlabeled_pos, distance_thres)
    return train_X, train_y, test_X, labeled_spatial_edges, unlabeled_spatial_edges, labeled_similarity_edges, unlabeled_similarity_edges, inverse_dict


def visualize_predictions(X, predicted_labels, inverse_dict, save_location):
    # Map numerical labels to their string annotations
    predicted_annotations = [inverse_dict[label] for label in predicted_labels]

    # Compute UMAP coordinates
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_coordinates = umap_model.fit_transform(X)

    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'UMAP1': umap_coordinates[:, 0],
        'UMAP2': umap_coordinates[:, 1],
        'Cell Type': predicted_annotations
    })

    # Plot the UMAP with cell type annotations
    plt.figure(figsize=(10, 8))
    for cell_type in df['Cell Type'].unique():
        subset = df[df['Cell Type'] == cell_type]
        plt.scatter(subset['UMAP1'], subset['UMAP2'], label=cell_type, alpha=0.7, s=20)

    plt.title('UMAP Visualization of Predicted Cell Types')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend(title='Cell Type', loc='best', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.tight_layout()
    plt.savefig(save_location)
    # plt.show()

def write_result_to_csv(result_path, model_name, threshold_distance, topk, val_loss, validation_acc):
    with open(os.path.join(result_path, 'result.csv'), 'a', newline='') as csvfile:
        fieldnames = ['Model', 'Threshold_Distance', 'TopK', 'Validation_Loss', 'Validation_Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'Model': model_name, 'Threshold_Distance': threshold_distance, 'TopK': topk, 'Validation_Loss':val_loss, 'Validation_Accuracy': validation_acc})


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0.0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0.0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss  # Since we want to minimize loss, a smaller val_loss is better. Hence, invert sign.

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class GraphDataset(InMemoryDataset):

    def __init__(self, labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges, transform=None,):
        self.root = '.'
        super(GraphDataset, self).__init__(self.root, transform)
        self.labeled_data = Data(x=torch.FloatTensor(labeled_X), edge_index=torch.LongTensor(labeled_edges).T, y=torch.LongTensor(labeled_y))
        self.unlabeled_data = Data(x=torch.FloatTensor(unlabeled_X), edge_index=torch.LongTensor(unlabeled_edges).T)

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return self.labeled_data, self.unlabeled_data
