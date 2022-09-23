"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
from layers.att_layers import GraphAttentionLayer
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
import utils.math_utils as pmath


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

class GCN(Encoder):
    """
    Graph Convolution Networks.
    """
    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True
        self.skip_connect = True 
        self.task = args.task

    def encode(self, x, adj):
        if self.skip_connect and self.task == 'md':
            hidden = self.layers[0].linear(x)
            output = super(GCN, self).encode(x,adj)
            output = (hidden + output)/2
        else:
            output = super(GCN, self).encode(x,adj)
        return output


class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """
    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)(space_dim=args.space_dim, time_dim=args.time_dim )
        # self.time_dim_ratio = (self.manifold.time_dim/self.manifold.dim)
        # args.time_dim_ratio = self.time_dim_ratio
        assert args.num_layers > 1
        # dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                    hyp_layers.HNNLayer(
                            self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias)
            )
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        if self.manifold.time_dim<self.manifold.dim:
            time_dim = self.manifold.time_dim
        else:
            time_dim = int((self.manifold.time_dim/self.manifold.dim)*x.shape[1])
        x_tan = self.manifold.proj_tan0(x, self.c, time_dim=time_dim)
        x_hyp = self.manifold.expmap0(x_tan, self.c, time_dim=time_dim)
        x_hyp = self.manifold.proj(x_hyp, self.c,time_dim=time_dim)
        return super(HNN, self).encode(x_hyp, adj)
        
class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """
    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)(space_dim=args.space_dim, time_dim=args.time_dim)
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.skip_connect = True 
        self.task = args.task

    def encode(self, x, adj):
        # print(x.max().item(),)
        # time_dim = self.manifold.time_dim
        if self.manifold.time_dim<=self.manifold.dim:
            time_dim = self.manifold.time_dim
        else:
            time_dim = int((self.manifold.time_dim/self.manifold.dim)*x.shape[1])

        x_tan = self.manifold.proj_tan0(x, self.c, time_dim=time_dim)
        # print(x_tan)
        assert not torch.isnan( x_tan ).any()
        # print(x.max().item(), x_tan.max().item())
        # assert self.manifold._check_vector_on_tangent0(x_tan, self.c,time_dim=time_dim)
        x_hyp = self.manifold.expmap0(x_tan, self.c, time_dim=time_dim)
        x_hyp = self.manifold.proj(x_hyp, self.c,time_dim=time_dim)
        # inn = self.manifold.inner(x_hyp,x_hyp,time_dim=time_dim)
        # print(self.c.item(), inn.max().item(), inn.min().item())
        # assert self.manifold._check_point_on_manifold(x_hyp, self.c,time_dim=time_dim)
        if self.skip_connect and self.task == 'md':
            hidden = self.layers[0].linear(x_hyp)
            output = super(HGCN, self).encode(x_hyp, adj)
            output = self.manifold.expmap0( self.manifold.proj_tan0( (self.manifold.logmap0(hidden,self.c) + self.manifold.logmap0(output, self.c) )/2, self.c), self.c)
        else:
            output = super(HGCN, self).encode(x_hyp, adj)
        return output

class GAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True


class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        weights = torch.Tensor(args.n_nodes, args.dim)
        if not args.pretrained_embeddings:
            weights = self.manifold.init_weights(weights, self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + weights.shape[1]
            else:
                dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        if self.use_feats:
            h = torch.cat((h, x), 1)
        return super(Shallow, self).encode(h, adj)
