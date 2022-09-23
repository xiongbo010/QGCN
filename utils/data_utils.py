"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from .preprocessing import *
import random

def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    elif args.task == 'md':
        data = load_data_md(args.dataset, False, datapath)
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    # print(data['features'].max())
    # print(data['adj_train_norm'])
    if args.task == 'md':
        data['adj_train_norm']=sparse_mx_to_torch_sparse_tensor(preprocess_graph(data['adj_train']))
    # print(data['adj_train_norm'])
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])

    # feature = data['features']
    # norm_feature = F.normalize(feature)
    # norm_feature[:,0] = 1 + norm_feature[:,0]
    # print(norm_feature.max().item(),norm_feature.min().item())
    # ones = torch.ones(norm_feature.shape[0],1)
    # new_feature = torch.cat((ones,norm_feature),1)
    # new_feature[] = torch.cat((ones,norm_feature),1)
    # data['features'] = norm_feature
    # k = 2/(feature.max()-feature.min()) 
    # new_feature = -1+k*(feature-feature.min())
    # data['features'] = norm_feature

    return data


# ############### FEATURES PROCESSING ####################################

def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################
def load_data_md(dataset, use_feats, data_path):
    if dataset in ['disease_md','grid','tree','tree_cycle','tree_grid','cycle_tree','sphere','cycle', 'cs_phd', 'power', 'facebook', 'random', 'club','nips','bio-diseasome','bio-wormnet','california','grqc','road-m','web-edu']:
        adj, features,labels, G = load_synthetic_md_data(dataset, False, data_path)[:4]
    elif dataset in ['toroidal','spherical','uniform_tree','random_geometric_graph','ring_of_tree','erdos_graph','tree_with_random_cycle']:
        adj, features,labels, G = load_synthetic_rec_data(dataset, False, data_path)[:4]
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    # if adj_dict == None:
    data = {'adj_train': adj, 'features': features,'labels': labels, 'G': G}
    # else:
    #     data = {'adj_train': adj, 'features': features,'labels': labels, "adj_dict":adj_dict}
    return data

def load_synthetic_rec_data(dataset_str, use_feats, data_path):
    if dataset_str == 'toroidal':
        g = ToroidalGraph(num_nodes=1000, R=0.01)
    elif dataset_str == 'spherical':
        g = SphericalGraph(num_nodes=1000, R=0.2)
    elif dataset_str == "uniform_tree":
        g = UniformTree(depth=5, branching_factor=4)
    elif dataset_str == 'random_geometric_graph':
        g = nx.random_geometric_graph(1000, 0.125)
    elif dataset_str == 'erdos_graph':
        g = ErdosGraph(num_nodes=1000, p=0.005)
    elif dataset_str == 'cycle':
        g = Cycle(1000)
    elif dataset_str == 'ring_of_tree':
        g = RingOfTrees(order=3, branching_factor=4)
    elif dataset_str == "tree_with_random_cycle":
        g = RandomTree(depth=5, branching_factor=4, num=500)
        # G = nx.Graph(g.edge_matrix)
        # g = random_edge(G, del_orig=True, num=100)
    elif dataset_str == 'club':
        g = nx.karate_club_graph()
    
    adj = g.edge_matrix
    # print(adj[0])
    labels = g.get_weight_matrix()
    features = np.eye(g.get_num_nodes())
    # features = adj
    # features[:,0] = 1
    G = nx.Graph(g.edge_matrix)
    # edges = G.edges()
    # positive_pairs = set(list(edges))
    # all_pairs = set([(i,j) for j in range(G.number_of_nodes()) for i in range(G.number_of_nodes())])
    # negative_pairs = all_pairs - positive_pairs
    # positive_pairs = torch.Tensor(list(positive_pairs)).long()
    # negative_pairs = torch.Tensor(list(negative_pairs)).long()
    return adj, features, labels, G


from multiprocessing import Pool
import scipy.sparse.csgraph as csg

def build_distance(G):
    length = dict(nx.all_pairs_shortest_path_length(G))
    R = np.array([[length.get(m, {}).get(n, 0) for m in G.nodes] for n in G.nodes], dtype=np.int32)
    return R

def load_synthetic_md_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    adj_dict = {}
    for i,j in edges:
        adj_dict[i] = set()
    for i, j in edges:
        adj_dict[i].add(j)
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0]) 
        rand_feature = np.random.uniform(low=-0.02, high=0.02, size=(adj.shape[0],adj.shape[0]))
        features = features + sp.csr_matrix(rand_feature)
        # print(features)
        # features = 
    G = nx.from_numpy_matrix(adj)
    labels = build_distance(G)
    labels = labels
    # G = nx.Graph(G)
    # print(features.shape, adj.shape)
    return sp.csr_matrix(adj), features, labels,G

def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed','citeseer']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset in ['disease_lp']:
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################
def load_data_nc_md(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed','citeseer']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset in ['disease_nc','tree_cycle','tree_grid','ba_shape']:
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.0, 0.0
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        elif dataset == 'deezer':
            dj, features, labels = load_json_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    # labels = torch.LongTensor(labels)
    # data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    adj = nx.to_scipy_sparse_matrix(adj)
    G = nx.from_numpy_matrix(adj)
    labels = build_distance(G)
    labels = labels
    # G = nx.Graph(G)
    # print(features.shape, adj.shape)
    return sp.csr_matrix(adj), features, labels,G


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed','citeseer']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset in ['disease_nc','tree_cycle','tree_grid','ba_shape']:
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        elif dataset == 'deezer':
            dj, features, labels = load_json_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    rand_feature = np.random.uniform(low=-0.01, high=0.01, size=(adj.shape[0],features.shape[1]))
    features = features + sp.csr_matrix(rand_feature)
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    if dataset_str in ['tree_cycle','tree_grid','ba_shape']:
        import pandas as pd
        labels = pd.read_csv(os.path.join(data_path, "{}.label.csv".format(dataset_str))).values[:,0]
        print(labels.shape)
    else:
        labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels

def load_json_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features

def load_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.node[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features


import networkx as nx
import scipy.sparse.csgraph as graph
from scipy import sparse
import numpy as np


class ToroidalGraph():
    def __init__(self, R, num_nodes, num_classes=None):
        self.R = R
        self.num_nodes = num_nodes
        self.num_classes = num_classes

        self.samples = self.get_samples()
        self.edge_matrix = self.construct_adjacency()
        self.graph = self.construct_graph()
        self.weight_matrix = self.get_weight_matrix()
        if num_classes is not None:
            self.labels = self.random_labeling(self.num_classes)

    def torus_distance(self, x, y):
        dx = np.abs(x[0] - y[0])
        dy = np.abs(x[1] - y[1])

        if dx > 0.5:
            dx = 1 - dx

        if dy > 0.5:
            dy = 1 - dy

        return dx**2 + dy**2

    def get_num_nodes(self):
        return self.num_nodes

    def get_samples(self):
        samples = np.random.uniform(low=-1, high=1, size=(self.num_nodes,2))

        return samples

    def construct_adjacency(self):
        edge_matrix = np.zeros(shape=(self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                x = self.samples[i, :]
                y = self.samples[j, :]
                dist = self.torus_distance(x, y)
                if dist < self.R and i != j:
                    edge_matrix[i, j] = 1

        edge_matrix = edge_matrix + np.transpose(edge_matrix)
        return sparse.csr_matrix(edge_matrix)

    def construct_graph(self):
        graph = nx.from_scipy_sparse_matrix(self.edge_matrix)
        if not nx.is_connected(graph):
            print('Graph is not connected')
        else:
            print('Everything fine')
            return graph

    def random_labeling(self, num_classes):
        labels = np.ones(shape=self.num_nodes) * 1000
        A = np.array(self.edge_matrix.todense(), dtype=np.float64)
        # let's evaluate the degree matrix D
        D = np.diag(np.sum(A, axis=0))
        length = 3 * self.num_nodes // num_classes
        for k in range(num_classes):
            source = np.random.randint(low=0, high=self.num_nodes, size=1)
            labels[source] = k
            visited = list()
            for _ in range(length):
                # evaluate the next state vector
                count = 0
                nn = np.random.randint(0, D[source, source], 1)
                for i in range(self.num_nodes):
                    if count == nn:
                        source = i
                        visited.append(i)
                        if labels[i] == 1000:
                            labels[i] = k
                        else:
                            p = np.random.uniform(low=0, high=1, size=1)
                            if p > 0.5:
                                labels[i] = k
                        break
                    if A[source, i] == 1:
                        count += 1
        labels[labels == 1000] = 0
        return labels

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)

        return weight_matrix

    def get_features(self, dim):
        ni

import networkx as nx
import scipy.sparse.csgraph as graph
from scipy import sparse
import numpy as np


import networkx as nx
import scipy.sparse.csgraph as graph
from scipy import sparse
import numpy as np


class SphericalGraph():
    def __init__(self, R, num_nodes):
        self.R = R
        self.num_nodes = num_nodes
        self.samples = self.get_samples()
        self.edge_matrix = self.construct_adjacency()
        self.graph = self.construct_graph()
        self.weight_matrix = self.get_weight_matrix()

    def stereo_distance(self, x, y):
        eps = 1e-10
        arg = np.clip(np.sum(x*y), a_max=1-eps, a_min=-1+eps)
        d = np.arccos(arg)

        return d

    def get_num_nodes(self):
        return self.num_nodes

    def get_samples(self):
        phi = np.random.uniform(low=0, high=2*np.pi, size=self.num_nodes)
        psi = np.random.uniform(low=0, high=2 * np.pi, size=self.num_nodes)
        samples = np.concatenate([np.reshape(np.sin(phi)*np.cos(psi), (-1, 1)), np.reshape(np.sin(phi)*np.sin(psi), (-1, 1)), np.reshape(np.cos(phi), (-1, 1))], axis=1)

        return samples

    def construct_adjacency(self):
        edge_matrix = np.zeros(shape=(self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                x = self.samples[i, :]
                y = self.samples[j, :]
                dist = self.stereo_distance(x, y)
                if dist < self.R and i != j:
                    edge_matrix[i, j] = 1

        edge_matrix = edge_matrix + np.transpose(edge_matrix)
        return sparse.csr_matrix(edge_matrix)

    def construct_graph(self):
        graph = nx.from_scipy_sparse_matrix(self.edge_matrix)
        if not nx.is_connected(graph):
            print('Graph is not connected')
        else:
            print('Everything fine')
            return graph

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)
        return weight_matrix


import networkx as nx
import scipy.sparse.csgraph as graph
from scipy import sparse
import numpy as np


class HyperbolicGraph():
    def __init__(self, R, num_nodes):
        self.R = R
        self.num_nodes = num_nodes

        self.samples = self.get_samples()
        self.edge_matrix = self.construct_adjacency()
        self.graph = self.construct_graph()
        self.weight_matrix = self.get_weight_matrix()

    def minkowski(self, x, y):
        a = x[-1] * y[-1] - np.sum(x[0:-1] * y[0:-1])

        return a

    def lorentz_distance(self, x, y):
        d = np.arccosh(self.minkowski(x, y))

        return d

    def get_num_nodes(self):
        return self.num_nodes

    def get_samples(self):
        phi = np.random.uniform(low=0, high=1, size=self.num_nodes)
        psi = np.random.uniform(low=0, high=2 * np.pi, size=self.num_nodes)
        samples = np.concatenate([np.reshape(np.cosh(phi)*np.cos(psi), (-1, 1)), np.reshape(np.cosh(phi)*np.sin(psi), (-1, 1)), np.reshape(np.sinh(phi), (-1, 1))], axis=1)

        return samples

    def construct_adjacency(self):
        edge_matrix = np.zeros(shape=(self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                x = self.samples[i, :]
                y = self.samples[j, :]
                dist = self.lorentz_distance(x, y)
                if dist < self.R and i != j:
                    edge_matrix[i, j] = 1
         
        edge_matrix = edge_matrix + np.transpose(edge_matrix)

        return sparse.csr_matrix(edge_matrix)

    def construct_graph(self):
        graph = nx.from_scipy_sparse_matrix(self.edge_matrix)
        if not nx.is_connected(graph):
            print('Graph is not connected')
        else:
            print('Everything fine')
            return graph

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)

        return weight_matrix


import numpy as np
import scipy.sparse.csgraph as graph
from scipy import sparse
import networkx as nx


class UniformTree:

    def __init__(self, depth, branching_factor):
        self.depth = depth
        self.b_factor = branching_factor

        self.num_nodes = self.get_num_nodes()
        self.edge_matrix = self.get_edge_matrix()
        self.weight_matrix = self.get_weight_matrix()
        self.graph = nx.from_scipy_sparse_matrix(self.edge_matrix)

    def get_num_nodes(self):
        num_nodes = int((self.b_factor**(self.depth+1)-1)/(self.b_factor-1))

        return num_nodes

    def get_edge_matrix(self):
        edge_matrix = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes-self.b_factor**self.depth):
            edge_matrix[i, self.b_factor*i+1:self.b_factor*(i+1)+1] = 1

        edge_matrix = edge_matrix + np.transpose(edge_matrix)

        return sparse.csr_matrix(edge_matrix)

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)

        return weight_matrix

    def get_batch_distances(self, indices):
        batch = []
        for index in indices:
            batch.append(self.weight_matrix[index[0], index[1]])
        return batch

class RandomTree:

    def __init__(self, depth, branching_factor, num=100):
        self.depth = depth
        self.b_factor = branching_factor
        self.num = num
        self.num_nodes = self.get_num_nodes()
        self.samples = self.get_samples()
        self.edge_matrix = self.get_edge_matrix()
        self.weight_matrix = self.get_weight_matrix()
        self.graph = nx.from_scipy_sparse_matrix(self.edge_matrix)
        

    def get_samples(self):
        phi = np.random.uniform(low=0, high=2*np.pi, size=self.num_nodes)
        psi = np.random.uniform(low=0, high=2 * np.pi, size=self.num_nodes)
        samples = np.concatenate([np.reshape(np.sin(phi)*np.cos(psi), (-1, 1)), np.reshape(np.sin(phi)*np.sin(psi), (-1, 1)), np.reshape(np.cos(phi), (-1, 1))], axis=1)

        return samples

    def stereo_distance(self, x, y):
        eps = 1e-10
        arg = np.clip(np.sum(x*y), a_max=1-eps, a_min=-1+eps)
        d = np.arccos(arg)
        return d

    def get_num_nodes(self):
        num_nodes = int((self.b_factor**(self.depth+1)-1)/(self.b_factor-1))
        return num_nodes

    def get_edge_matrix(self):
        edge_matrix = np.zeros((self.num_nodes, self.num_nodes))
        
        for i in range(self.num_nodes-self.b_factor**self.depth):
            # tree_edge.append( (i, self.b_factor*i+1:self.b_factor*(i+1)+1)) 
            edge_matrix[i, self.b_factor*i+1:self.b_factor*(i+1)+1] = 1

        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                x = self.samples[i, :]
                y = self.samples[j, :]
                dist = self.stereo_distance(x, y)
                if dist < 0.2 and i != j:
                    edge_matrix[i, j] = 1

        # count = 0 
        # tree_edge = edge_matrix.copy()

        # while count< self.num:
        #     a = random.choice(range(self.num_nodes-self.b_factor**self.depth))
        #     b = random.choice(range(0, self.num_nodes))
        #     if a!=b and edge_matrix[a,b]==0:
        #         edge_matrix[a,b] = 1
        #         count=count+1
                # drop = random.choice(range(self.b_factor*a+1, self.b_factor*(a+1)+2))
                # print(a,b,drop)
                # edge_matrix[a, drop] = 0
                # edge_matrix[drop_tree_edge[0], drop_tree_edge[1]] = 0
            
        edge_matrix = edge_matrix + np.transpose(edge_matrix)
        return sparse.csr_matrix(edge_matrix)

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)

        return weight_matrix

    def get_batch_distances(self, indices):
        batch = []
        for index in indices:
            batch.append(self.weight_matrix[index[0], index[1]])
        return batch

class Cycle:

    def __init__(self, order):
        self.order = order

        self.num_nodes = self.get_num_nodes()
        self.edge_matrix = self.get_edge_matrix()
        self.weight_matrix = self.get_weight_matrix()
        self.graph = nx.from_scipy_sparse_matrix(self.edge_matrix)

    def get_num_nodes(self):
        num_nodes = self.order

        return num_nodes

    def get_edge_matrix(self):
        edge_matrix = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes):
            left = i - 1
            right = i + 1
            if left < 0:
                left = self.num_nodes - 1
            if right == self.num_nodes:
                right = 0
            edge_matrix[i, left] = 1
            edge_matrix[i, right] = 1

        return sparse.csr_matrix(edge_matrix)

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)

        return weight_matrix

class RingOfTrees:

    def __init__(self, order, branching_factor):
        self.order = order
        self.b = branching_factor

        self.num_nodes = self.get_num_nodes()
        self.edge_matrix = self.get_edge_matrix()
        self.weight_matrix = self.get_weight_matrix()
        self.graph = nx.from_scipy_sparse_matrix(self.edge_matrix)

    def get_num_nodes(self):
        num_nodes = 2 * self.order + self.b * self.order

        return num_nodes

    def get_edge_matrix(self):
        edge_matrix = np.zeros((self.num_nodes, self.num_nodes))
        cycle_matrix = np.zeros((self.order, self.order))

        for i in range(self.order):
            left = i - 1
            right = i + 1
            if left < 0:
                left = self.order - 1
            if right == self.order:
                right = 0
            cycle_matrix[i, left] = 1
            cycle_matrix[i, right] = 1

        edge_matrix[0:self.order, 0:self.order] = cycle_matrix

        for i in range(self.order):
            edge_matrix[i, self.order+i] = 1
            edge_matrix[self.order + i, i] = 1

        for i in range(self.order):
            for j in range(2 * self.order + i * self.b, 2 * self.order + (i+1) * self.b):
                edge_matrix[i + self.order, j] = 1
                edge_matrix[j, i + self.order] = 1

        return sparse.csr_matrix(edge_matrix)

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)

        return weight_matrix

class ErdosGraph():
    def __init__(self, p, num_nodes):
        self.p = p
        self.num_nodes = num_nodes
        self.setting()
        self.edge_matrix = self.construct_adjacency()
        self.graph = self.construct_graph()
        self.weight_matrix = self.get_weight_matrix()

    def get_num_nodes(self):
        return self.num_nodes

    def setting(self):
        if self.p >= (np.log(self.num_nodes)/self.num_nodes) ** (1/3):
            print('Spherical setting')
        if self.p > (np.log(self.num_nodes)/self.num_nodes ** 2) ** (1/3) and self.p < 1 /np.sqrt(self.num_nodes):
            print('Hyperbolic setting')

    def construct_adjacency(self):
        edge_matrix = np.random.binomial(2, self.p, size=(self.num_nodes, self.num_nodes))
        return sparse.csr_matrix(edge_matrix)

    def construct_graph(self):
        G = nx.from_scipy_sparse_matrix(self.edge_matrix)
        if not nx.is_connected(G):
            print('Graph is not connected')
            Gs = nx.connected_component_subgraphs(G, copy=True)
        else:
            print('Everything fine')
            return G

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)
        return weight_matrix
        
#H = HyperbolicGraph(num_nodes=1000, R=1)
def random_edge(graph, del_orig=True, num=100):
    '''
    Create a new random edge and delete one of its current edge if del_orig is True.
    :param graph: networkx graph
    :param del_orig: bool
    :return: networkx graph
    '''
    for i in range(num):
        edges = list(graph.edges)
        nonedges = list(nx.non_edges(graph))
        # random edge choice
        chosen_edge = random.choice(edges)
        chosen_nonedge = random.choice([x for x in nonedges if chosen_edge[0] == x[0]])
        if del_orig:
            # delete chosen edge
            graph.remove_edge(chosen_edge[0], chosen_edge[1])
        # add new edge
        graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])
    return graph