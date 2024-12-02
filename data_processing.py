import pandas as pd
from sklearn.preprocessing import normalize, LabelEncoder
from scipy.spatial import Delaunay
import torch
import numpy as np

TRAINING_DATA_PATH = 'data/B004_training_dryad.csv'

def load_csv(filepath):
    df = pd.read_csv(filepath)
    return df

def encode_cell_types(cell_type_list):
    """Encode cell types into numerical labels."""
    le = LabelEncoder()
    cell_types = le.fit_transform(cell_type_list)
    class_values = np.unique(cell_types)
    return cell_types, class_values

def construct_similarity_adjacency(count_matrix, neighborhood_size = 10, threshold = .5):
    # Normalize each row first
    X = normalize(count_matrix)
    X = torch.from_numpy(X)
    src = []
    dst = []
    edge_weights = []
    for idx, row in enumerate(X):
        scores = row @ X.T
        neighbors = scores.argsort(descending=True)[:neighborhood_size]
        src.append(torch.full((neighborhood_size,), idx))
        dst.append(neighbors)
        edge_weights.append(scores[neighbors])
    edge_index = torch.stack([torch.concat(src), torch.concat(dst)])
    edge_weights = torch.concat(edge_weights)

    return edge_index.to(torch.long), edge_weights

def construct_spatial_adjacency(coordinates):
    """
    Computes the Delaunay triangulation of a set of points and returns an adjacency matrix as a TensorFlow sparse tensor.

    Parameters:
    coordinates (numpy array): 2D numpy array where each row represents a point in 2D space.

    Returns:
    adj_matrix (tf.sparse.SparseTensor): Adjacency matrix representing the graph.
    """

    delaunay = Delaunay(coordinates)
    simplices = delaunay.simplices

    # Create adjacency list
    edges = set()
    for simplex in simplices:
        # Create edges between all pairs of vertices in each simplex
        for i in range(3):
            for j in range(i + 1, 3):
                # Sort the tuple so that (i, j) is the same as (j, i)
                edge = tuple(sorted((simplex[i], simplex[j])))
                edges.add(edge)

    # Convert edges to numpy arrays (adjacency matrix format)
    edges = torch.tensor(list(edges))
    row, col = edges[:, 0], edges[:, 1]
    edge_index = np.stack([row, col])
    edge_weights = torch.ones(len(edges), dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return edge_index, edge_weights


def main():
    df = load_csv(TRAINING_DATA_PATH)
    print(df.head())
    print(df.columns)
    genes = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f',
       'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR',
       'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90',
       'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34',
       'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CK7',
       'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a',
       'CD163', 'CD161']
    df.rename(columns={'Unnamed: 0': 'cell_id'}, inplace=True)
    X = df[genes][:1000].to_numpy()
    y = encode_cell_types(df[:1000])
    coordinates = df[['x', 'y']][:1000].to_numpy()
    # Construct similarity adjacency
    sim_edge_index, sim_edge_weights = construct_similarity_adjacency(X)
    spatial_edge_index, spatial_edge_weights = construct_spatial_adjacency(coordinates)


if __name__ == '__main__':
    main()