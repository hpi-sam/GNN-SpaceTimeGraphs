import torch
import math

import torch.nn as nn

from torch.nn.parameter import Parameter


class SLConv(nn.Module):
    def __init__(self, in_chanels, out_chanels, act_func=None):
        super(SLConv, self).__init__()
        self.in_chanels = in_chanels
        self.out_chanels = out_chanels
        self.weight = Parameter(torch.rand(in_chanels, out_chanels))
        self.reset_parameters()
        self.act_func = act_func

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, S):
        x = torch.matmul(x, self.weight)  # (1,N,out_chanels)
        weighting = torch.mul(S, adj)  # (N,N)
        output = torch.matmul(weighting, x)
        if self.act_func:
            output = self.act_func(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_chanels) + ' -> ' \
               + str(self.out_chanels) + ')'


class GlobalSLC(nn.Module):
    def __init__(self, cin, cout, cs, cd, adj, act_func=None):
        super(GlobalSLC, self).__init__()
        self.cin = cin
        self.cout = cout
        self.adj = adj
        self.num_nodes = adj.shape[0]

        # convolution parameters
        self.ws = Parameter(torch.rand(self.num_nodes, self.num_nodes))
        self.wp = Parameter(torch.rand(cin, cout))
        self.ts = Parameter(torch.rand(cs, cin, cout))
        self.td = Parameter(torch.rand(cd, cin, cout))
        self.param_list = [self.ws, self.wp, self.ts, self, self.td]

        self.t0 = torch.eye(self.num_nodes, self.num_nodes)
        if act_func:
            self.act_func = act_func

    def reset_parameters(self):
        for parameter in self.param_list:
            stdv = 1. / math.sqrt(parameter.size(1))
            parameter.data_uniform_(-stdv, stdv)

    def forward(self, x):
        out = torch.mm(self.ts[0], torch.mm(self.t0, x))

        # computation of static graph structure convolution
        out_s = out + torch.mm(self.ts[1], torch.mm(self.ws, x))
        tk_prev = self.ws
        tk = 2.*torch.mm(self.ws, self.ws) - self.t0
        for k in range(2, self.cs):
            out_s = out_s + torch.mm(self.ts[k], torch.mm(tk, x))
            tk = 2.*torch.mm(self.ws, tk) - tk_prev

        # computation of dynamical graph structure convolution
        wd = torch.mm(x.T, torch.mm(self.wp, x))
        out_d = out + torch.mm(self.td[1], torch.mm(self.wd, x))
        tk_prev = wd
        tk = 2.*torch.mm(wd, wd) - self.t0
        for k in range(2, self.cd):
            out_s = out_s + torch.mm(self.td[k], torch.mm(tk, x))
            tk = 2.*torch.mm(wd, tk) - tk_prev

        if self.act_func:
            out_s = self.act_func(out_s)
            out_d = self.act_func(out_d)

        output = out_s + out_d
        return output


class SLGRUCell(nn.Module):
    def __init__(self, num_units, adj, num_nodes, input_dim, hidden_state_size):
        super().__init__()
        self._activation = torch.tanh
        self.adj = adj
        self._num_units = num_units
        self._num_nodes = num_nodes
        self._input_dim = input_dim
        self._hidden_state_size = hidden_state_size
        self.gc1 = SLConv(input_dim+hidden_state_size, hidden_state_size)
        self.gc2 = SLConv(input_dim+hidden_state_size, hidden_state_size)
        self.gc3 = SLConv(input_dim+hidden_state_size, hidden_state_size)

    def forward(self, inputs, hx, S):
        x = torch.cat([inputs, hx], dim=2)  # (batch_size, num_nodes, num_features+num_hidden_features)
        u = torch.sigmoid(self.gc1(x, self.adj, S))
        r = torch.sigmoid(self.gc2(x, self.adj, S))
        x = torch.cat([inputs, r*hx], dim=2)
        c = self._activation(self.gc3(x, self.adj, S))

        new_state = u * hx + (1.0 - u) * c
        return new_state
