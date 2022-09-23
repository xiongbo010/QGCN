"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt, GeoAtt


import random

def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        if args.manifold in ['Hyperboloid', 'PoincareBall']:
            curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
        else:
            curvatures = [nn.Parameter(torch.Tensor([-1.]))  for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c], requires_grad=True) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.linear = HypLinear(manifold, in_features, out_features, self.c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, self.c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, self.c_in, self.c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0.0001)

    def forward(self, x):
        assert not torch.isnan(x).any()
        # time_dim = self.manifold.time_dim
        if self.manifold.time_dim<self.manifold.dim:
            time_dim = self.manifold.time_dim
        else:
            time_dim = int((self.manifold.time_dim/self.manifold.dim)*x.shape[1])
        
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.manifold.mobius_matvec(drop_weight, x, self.c, time_dim=time_dim)
        res = self.manifold.proj(res, self.c)
        assert not torch.isnan(res).any()
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            # assert not torch.isnan(hyp_bias).any()
            res = self.manifold.mobius_add(res, hyp_bias, self.c)
            res = self.manifold.proj(res, self.c)
        # assert self.manifold._check_point_on_manifold(res,self.c)
        assert not torch.isnan(res).any()
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        assert not torch.isnan(x).any()
        x_tangent = self.manifold.logmap0(x, self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, self.c), self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.clamp(torch.matmul(adj_att, x_tangent), max=self.manifold.max_norm)
        else:
            support_t = torch.clamp(torch.spmm(adj, x_tangent), max=self.manifold.max_norm)

        assert not torch.isnan(x_tangent).any()
        assert not torch.isnan(support_t).any()
        res = self.manifold.proj_tan0(support_t,self.c)
        res = self.manifold.expmap0(res, self.c)
        output = self.manifold.proj(res,self.c)
        # assert self.manifold._check_point_on_manifold(output,self.c)
        # output = self.manifold.perform_rescaling_beta(output, self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = torch.clamp(self.act(self.manifold.logmap0(x, self.c_in)), max=self.manifold.max_norm)
        xt = self.manifold.proj_tan0(xt, self.c_out)
        output = self.manifold.expmap0(xt, self.c_out)
        # assert self.manifold._check_point_on_manifold(output, self.c_out)
        # output = self.manifold.perform_rescaling_beta(output, self.c_out)
        return output

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
