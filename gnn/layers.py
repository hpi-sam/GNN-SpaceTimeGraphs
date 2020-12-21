import torch
import math

import torch.nn as nn

from torch.nn.parameter import Parameter


class SLConv(nn.Module):
    def __init__(self, in_chanels, out_chanels, n):
        super(SLConv, self).__init__()
        self.in_chanels = in_chanels
        self.out_chanels = out_chanels
        self.weight = Parameter(torch.rand(in_chanels, out_chanels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, S):
        x = torch.matmul(x, self.weight)  # (1,N,out_chanels)
        weighting = torch.mul(S, adj)  # (N,N)
        output = torch.matmul(weighting, x)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_chanels) + ' -> ' \
               + str(self.out_chanels) + ')'


class SLGRUCell(nn.Module):
    def __init__(self, num_units, adj, num_nodes, input_dim, hidden_state_size):
        super().__init__()
        self._activation = torch.tanh
        self.adj = adj
        self._num_units = num_units
        self._num_nodes = num_nodes
        self._input_dim = input_dim
        self._hidden_state_size = hidden_state_size
        self.gc1 = SLConv(input_dim+hidden_state_size, num_units, num_nodes)
        self.gc2 = SLConv(input_dim+hidden_state_size, num_units, num_nodes)
        self.gc3 = SLConv(input_dim+hidden_state_size, num_units, num_nodes)

    def forward(self, inputs, hx, S):
        x = torch.cat([inputs, hx])  # (batch_size, num_nodes, num_features+num_hidden_features)
        u = torch.sigmoid(self.gc1(x, self.adj, S))
        r = torch.sigmoid(self.gc2(x, self.adj, S))
        x = torch.cat([inputs, r*hx])
        c = self._activation(self.gc3(x, self.adj, S))

        new_state = u * hx + (1.0 - u) * c
        return new_state
