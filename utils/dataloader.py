from torch_geometric.datasets import Planetoid
import torch
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse import diags
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import identity

def load_data_cite(dataset_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if(dataset_name == "citeseer"):
        dataset_citeseer = Planetoid(root='./data/citeseer/',name='Citeseer')
    if (dataset_name == "cora"):
        dataset_citeseer = Planetoid(root='./data/cora/', name='Cora')
    print(dataset_citeseer)

    data_citeseer = dataset_citeseer[0].to(device)
    # print(dataset_citeseer.num_classes)
    # print(dataset_citeseer.num_node_features)
    # print(len(dataset_citeseer))
    # print(data_citeseer)

    # -------------------
    # need to get adj, features(in vstack format), labels(in 3327,6-hot shape); idx_train, idx_val, idx_test are index
    num_nodes = dataset_citeseer.x.shape[0]
    num_edges = data_citeseer.num_edges
    edges_reindexed = data_citeseer.edge_index

    # 1. get features
    features = data_citeseer.x

    # 2. get adj
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    identity_matrix = identity(num_nodes)
    for i in range(edges_reindexed.shape[1]):
        row = edges_reindexed[0][i].item()
        col = edges_reindexed[1][i].item()
        adjacency_matrix[row][col] = 1
    # res = (adjacency_matrix==adjacency_matrix.T).all()
    adjacency_matrix = adjacency_matrix + identity_matrix
    A = torch.from_numpy(adjacency_matrix)
    # Check your adjacency matrix by using the sum as proxy
    print(f"The number of connections, {int(A.sum())}, must equal the number of edges, {num_edges}," 
          f" plus the number of nodes, {num_nodes}")
    A.to_dense()[:10,:10]

    # 3 get labels
    labels = torch.zeros(num_nodes, np.unique(data_citeseer.y).shape[0])
    # Iterate over each row and set the corresponding indices to 1
    for i, indices in enumerate(data_citeseer.y.tolist()):
        labels[i, indices] = 1

    # 4 get idx_train
    idx_train = data_citeseer.train_mask
    idx_val = data_citeseer.val_mask
    idx_test = data_citeseer.test_mask
    return A, features, labels, idx_train, idx_val, idx_test

def get_A_hat_torch(A):
    # A_hat = DAD
    A_tilde = coo_matrix(A, dtype=float)
    degrees = A_tilde.sum(axis=1).flatten().A
    Diag_matrix = diags(degrees, list(range(len(degrees))), dtype=float)
    A_hat = (Diag_matrix.power(-0.5) @ A_tilde @ Diag_matrix.power(-0.5)).tocoo()

    # A as sparse PyTorch tensor
    indices = np.vstack((A_hat.col, A_hat.row))
    A_hat_torch = torch.sparse_coo_tensor(indices, A_hat.data, dtype=torch.float)
    return A_hat_torch

def normalize_features(features):
    # remornalized the featrure matrix
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    # 添加异常值处理，除以0的问题
    rowsum[rowsum == 0] = 1e-10
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = csr_matrix(r_mat_inv.dot(features)).todense()
    features = torch.FloatTensor(features[np.newaxis])
    return features

# A, features, labels, idx_train, idx_val, idx_test = load_data_cite(dataset_name)
# adj = get_A_hat_torch(A)
# features = normalize_features(features)
#
# labels = torch.FloatTensor(labels[np.newaxis])

# no need
# x_train = torch.LongTensor(idx_train)
# train = idx_train.type(torch.int)
# indices_train = torch.nonzero(train).squeeze()
# range_train = range(len(indices_train))
#
# # idx_val = torch.LongTensor(idx_val)
# val = idx_val.type(torch.int)
# indices_val = torch.nonzero(val).squeeze()
# range_val = range(len(indices_train), len(indices_train) + len(indices_val))
#
#
# test = idx_test.type(torch.int)
# indices_test = torch.nonzero(test).squeeze()
# idx_test = torch.LongTensor(idx_test)



