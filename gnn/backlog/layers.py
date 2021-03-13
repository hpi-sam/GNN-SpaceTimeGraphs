import math
import torch
import torch.nn as nn

from gnn.utils import generate_knn_ids
from gnn.layers import TimeBlock, GlobalSLC, BatchNorm
from torch.nn.parameter import Parameter
from torch.functional import F


# Has only been used when studying the "Structure Learning Convolution" and will be replaced by GlobalSLC and LocalSLC
@DeprecationWarning
class SLConv(nn.Module):
    def __init__(self, c_in, c_out, act_func=None):
        super(SLConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.weight = Parameter(torch.rand(c_in, c_out))
        self.reset_parameters()
        self.act_func = act_func

    def reset_parameters(self):
        """ Applies z-score normalization"""
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)

    def forward(self, x, adj, S):
        x = torch.matmul(x, self.weight)  # (1,num_nodes,c_out)
        weighting = torch.mul(S, adj)  # (num_nodes,num_nodes)
        output = torch.matmul(weighting, x)
        if self.act_func:
            output = self.act_func(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.c_in) + ' -> ' \
               + str(self.c_out) + ')'


class LocalSLC(nn.Module):
    def __init__(self, adj, c_in, c_out, num_nodes, k, g=None, act_func=None):
        super(LocalSLC, self).__init__()
        self.adj = adj
        self.c_in = c_in
        self.c_out = c_out
        self.num_nodes = num_nodes
        self.k = k
        self.act_func = act_func

        # learnable parameters and functions
        self.bs = Parameter(torch.randn(num_nodes, self.k))
        self.ws = Parameter(torch.randn(self.k, c_in, c_out))
        self.wd = Parameter(torch.randn(self.k, c_in, c_out))
        self.nu = Parameter(torch.randn(c_in))
        # TODO: implement dynamical component of local convolution
        self.param_list = [self.bs, self.ws]
        self.knn_ids = generate_knn_ids(self.adj, self.k)

        self.reset_parameters()

    def reset_parameters(self):
        for parameter in self.param_list:
            std = .1 / math.sqrt(parameter.size(1))
            parameter.data.uniform_(-std, std)

    def dynamical_part(self, x):
        bd = torch.matmul(x, self.nu)
        return torch.einsum("bnk,kio,bnki->bno", bd, self.wd, x)

    def static_part(self, x):
        return torch.einsum("nk,kio,bnki->bno", self.bs, self.ws, x)

    def forward(self, x):
        x = x[:, self.knn_ids, :]  # (batch_size, n, k, c_in)
        y = self.static_part(x)  #+ self.dynamical_part(x)
        if self.act_func:
            y = self.act_func(y)
        return y


class SLGRUCell(nn.Module):
    def __init__(self, num_units, adj, num_nodes, input_dim,
                 hidden_state_size, gconv=SLConv):
        """ GRU Cell that integrates Graph Convolutions into the gating mechanisms"""
        super().__init__()
        self._activation = torch.tanh
        self.adj = adj
        self._num_units = num_units
        self._num_nodes = num_nodes
        self._input_dim = input_dim
        self._hidden_state_size = hidden_state_size
        self.gc1 = gconv(input_dim + hidden_state_size, hidden_state_size)
        self.gc2 = gconv(input_dim + hidden_state_size, hidden_state_size)
        self.gc3 = gconv(input_dim + hidden_state_size, hidden_state_size)

    def forward(self, inputs, hx, S):
        x = torch.cat([inputs, hx], dim=2)  # (batch_size, num_nodes, num_features+num_hidden_features)
        u = torch.sigmoid(self.gc1(x, self.adj, S))
        r = torch.sigmoid(self.gc2(x, self.adj, S))
        x = torch.cat([inputs, r * hx], dim=2)
        c = self._activation(self.gc3(x, self.adj, S))

        new_state = u * hx + (1.0 - u) * c
        return new_state


class STGCNBlock(nn.Module):
    def __init__(self, args, in_channels, spatial_channels, out_channels,
                 num_nodes, adj):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=out_channels)
        self.spatial1 = GlobalSLC(adj, args, out_channels, spatial_channels, num_nodes, act_func=F.relu)
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = BatchNorm(num_nodes)

    def forward(self, x):
        t = self.temporal1(x)
        t2 = self.spatial1(t)
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)


