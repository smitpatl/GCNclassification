import scipy.sparse as sp
import networkx as nx
import numpy as np
import sys

def sparse_to_tuple(sparse_matrix):
    def to_tuple(matrix):
        if not sp.isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        coords = np.vstack((matrix.row, matrix.col)).transpose()
        values = matrix.data
        shape = matrix.shape
        return coords, values, shape

    if isinstance(sparse_matrix, list):
        for i in range(len(sparse_matrix)):
            sparse_matrix[i] = to_tuple(sparse_matrix[i])
    else:
        sparse_matrix = to_tuple(sparse_matrix)

    return sparse_matrix

def preprocess_adj(adj):
    adj_hat = adj + np.identity(n=adj.shape[0])
    d_hat_diag = np.squeeze(np.sum(np.array(adj_hat), axis=1))
    d_hat_inv_sqrt_diag = np.power(d_hat_diag, -1/2)
    d_hat_inv_sqrt = np.diag(d_hat_inv_sqrt_diag)
    adj_norm = np.dot(np.dot(d_hat_inv_sqrt, adj_hat), d_hat_inv_sqrt)
    adj_norm_tuple = sparse_to_tuple(sp.coo_matrix(adj_norm))
    return adj_norm_tuple

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = sp.coo_matrix(features)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def build_label(Graph):
    g = Graph
    G = g.G
    nodes = g.node_list
    look_up = g.look_up
    labels = []
    label_dict = {}
    label_id = 0
    for node in nodes:
        labels.append((node,G.nodes[node]['label']))
        for l in G.nodes[node]['label']:
            if l not in label_dict:
                label_dict[l] = label_id
                label_id += 1
    label_mat = np.zeros((len(labels),label_id))
    for node,l in labels:
        node_id = look_up[node]
        for ll in l:
            l_id = label_dict[ll]
            label_mat[node_id][l_id] = 1
    return label_mat,label_dict

def preprocess_labels(Graph,labels,train_ratio):
    train_percent = train_ratio
    g = Graph
    node_size = g.node_size
    look_up = g.look_up
    training_size = int(train_percent * node_size)
    state = np.random.get_state()
    np.random.seed(0)
    shuffle_indices = np.random.permutation(np.arange(node_size))
    np.random.set_state(state)
    def sample_mask(begin,end):
        mask = np.zeros(node_size)
        for i in range(begin, end):
            mask[shuffle_indices[i]] = 1
        return mask

    train_mask = sample_mask(0,training_size - 5)
    val_mask = sample_mask(training_size - 5, training_size+20)
    test_mask = sample_mask(training_size+20, node_size)
    return train_mask,val_mask,test_mask

def preprocess_data(Graph,train_ratio,has_features=True):
    g = Graph
    G = g.G
    nodes = g.node_list
    adj = nx.to_numpy_matrix(G)
    adj = preprocess_adj(adj)
    if has_features == True:
        features = np.vstack([G.nodes[i]['feature']
                for i in g.node_list])
        features = preprocess_features(features)
    else:
        features = g.features
    labels,label_dict = build_label(g)
    train_mask,val_mask,test_mask = preprocess_labels(g,labels,train_ratio)
    return adj,labels,features,train_mask,val_mask,test_mask
