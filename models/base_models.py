"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
# import geoopt as manifolds
import models.encoders as encoders
from models.decoders import model2decoder,MDDecoder
from utils.eval_utils import acc_f1
import utils.distortions as dis
import scipy
import networkx as nx
import scipy.sparse as sp
import math

import numpy as np


def average_distortion(g_pdists, m_pdists):
    r"""The average distortion used to measure the quality of the embedding.
    See, e.g., [1].
    Parameters
    ----------
    g_pdists : numpy.ndarray
        Pairwise distances on the graph, as an (n*(n-1)//2,)-shaped array.
    m_pdists : numpy.ndarray
        Pairwise distances on the manifold, as an (n*(n-1)//2,)-shaped array.
    """
    return np.mean(np.abs(m_pdists - g_pdists) / g_pdists)


def mean_average_precision(g, m_pdists):
    r"""The MAP as defined in [1]. The complexity is squared in the number of
    nodes.
    """
    n = m_pdists.shape[0]
    assert n == g.number_of_nodes()

    ap_scores = []
    for u in g.nodes():
        sorted_nodes = np.argsort(m_pdists[u])
        neighs = set(g.neighbors(u))
        n_correct = 0.0
        precisions = []
        for i in range(1, n):
            if sorted_nodes[i] in neighs:
                n_correct += 1
                precisions.append(n_correct / i)

        ap_scores.append(np.mean(precisions))

    return np.mean(ap_scores)

class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c],requires_grad=True)
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            if args.manifold in ['Hyperboloid', 'PoincareBall']:
                self.c = nn.Parameter(torch.Tensor([1.]))
            else:
                self.c = nn.Parameter(torch.Tensor([-1.]))
        
        self.manifold = getattr(manifolds, self.manifold_name)(space_dim=args.space_dim, time_dim=args.time_dim)
        if self.manifold.name in ['Hyperboloid', 'PseudoHyperboloid']:
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name in ['Hyperboloid','PseudoHyperboloid']:
            o = torch.zeros_like(x)
            o[:, 0:1] = 1.0
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

class MDModel(BaseModel):
    """
    Base model for minimizing distrotion task.
    """

    def __init__(self, args):
        super(MDModel, self).__init__(args)
        self.decoder = MDDecoder(self.c, self.manifold, args)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return output

    def compute_metrics(self, embeddings, data, split):
        device = embeddings.get_device()
        x, emb_dist, loss, max_dist,imax, imin = self.decode(embeddings,data,None)
        # print(data['labels'].shape)
        G = data['G']
        n = G.order()
        G = nx.to_scipy_sparse_matrix(G, nodelist=list(range(G.order())))
        true_dist = (torch.Tensor(data['labels'])).to(device)
        # true_dist = (true_dist/true_dist.max())*math.pi

        mask = np.array(np.triu(np.ones((true_dist.shape[0],true_dist.shape[0]))) - np.eye(true_dist.shape[0], true_dist.shape[0]), dtype=bool)
        # mc, me, avg_dist, nan_elements = dis.distortion(true_dist.cpu().detach().numpy(), emb_dist.cpu().detach().numpy(), n, 16)
        # wc_dist = me*mc
        mapscore = dis.map_score(sp.csr_matrix.todense(G).A, emb_dist.cpu().detach().numpy(), n, 16)
        # mapscore = mean_average_precision(G, emb_dist.cpu().detach().numpy())
        # distortion = average_distortion(true_dist,emb_dist.cpu().detach().numpy())
        # np.savetxt('true_dist.txt',true_dist.cpu().detach().numpy())

        true_dist = true_dist[mask] 
        emb_dist = emb_dist[mask]
        
        # emb_dist = F.normalize(emb_dist)
        distortion = (((emb_dist)/(true_dist)-1)**2).mean()
        
        metrics = {'loss': loss, 'distortion':distortion, 'mapscore':mapscore,'c': self.c.item(),'max_dist':max_dist, 'imax':imax, 'imin':imin}
        return metrics

    def init_metric_dict(self):
        return {'distortion':1, 'mapscore': -1,'c': -1,'max_dist':0}

    def has_improved(self, m1, m2):
        return m1["mapscore"] < m2["mapscore"]


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.manifold, self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

