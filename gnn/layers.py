import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from gnn.argparser import parse_arguments
from gnn.utils import generate_knn_ids

parser = parse_arguments()
args = parser.parse_args()

if args.gpu:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cpu")


class SLConv(nn.Module):
    def __init__(self, c_in, c_out, act_func=None):
        super(SLConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.weight = Parameter(torch.rand(c_in, c_out))
        self.reset_parameters()
        self.act_func = act_func

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

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


class GlobalSLC(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, cs=6, cd=6, act_func=None):
        super(GlobalSLC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.num_nodes = num_nodes
        self.cs = cs
        self.cd = cd

        # convolution parameters
        self.ws = Parameter(torch.rand(self.num_nodes, self.num_nodes))
        self.wp = Parameter(torch.rand(c_in, c_in))
        self.ts = Parameter(torch.rand((cs, c_in, c_out)))
        self.td = Parameter(torch.rand((cd, c_in, c_out)))
        self.param_list = [self.ws, self.wp, self.ts, self.td]

        self.t0 = torch.eye(self.num_nodes, self.num_nodes, device=DEVICE)
        self.act_func = act_func
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in self.param_list:
            stdv = .1 / math.sqrt(parameter.size(1))
            parameter.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Spatial Graph Convolution using the Global Structure Learning architecture.

        :param x: graph signal at time [t-time_steps + 1,...,t] (batch_size, time_steps, num_nodes, c_in)
        :return: convolved signal at time [t-time_steps + 1,...,t] (batch_size, time_steps, num_nodes, c_out)
        """
        # (num_nodes, num_nodes) x (batch_size, num_nodes, in_feat) x (c_in, c_out)
        out = torch.matmul(torch.matmul(self.t0, x), self.ts[0])  # (batch_size, num_nodes, c_out)

        # computation of static graph structure convolution
        out_s = out + torch.matmul(torch.matmul(self.ws, x), self.ts[1])
        tk_prev = self.ws
        tk = 2. * torch.matmul(self.ws, self.ws) - self.t0
        for k in range(2, self.cs):
            out_s = out_s + torch.matmul(torch.matmul(tk, x), self.ts[k])
            tk = 2. * torch.matmul(self.ws, tk) - tk_prev

        # computation of dynamical graph structure convolution
        # QUESTION: Why isnt it implemented as x^T W_p x as in the paper?
        wd = torch.matmul(x, torch.matmul(self.wp, torch.transpose(x, 1, 2)))
        # normalize wd -> to get rid of exploding gradients
        wd = wd + torch.min(wd)
        wd = wd / torch.max(wd) / self.num_nodes**2

        out_d = out + torch.matmul(torch.matmul(wd, x), self.td[1])
        tk_prev = wd
        tk = 2. * torch.matmul(wd, wd) - self.t0
        for k in range(2, self.cd):
            out_d = out_d + torch.matmul(torch.matmul(tk, x), self.td[k])
            tk = 2. * torch.matmul(wd, tk) - tk_prev

        if self.act_func:
            out_s = self.act_func(out_s)
            out_d = self.act_func(out_d)

        output = out_s + out_d
        return output


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
            stdv = .1 / math.sqrt(parameter.size(1))
            parameter.data.uniform_(-stdv, stdv)

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
                 hidden_state_size):
        super().__init__()
        self._activation = torch.tanh
        self.adj = adj
        self._num_units = num_units
        self._num_nodes = num_nodes
        self._input_dim = input_dim
        self._hidden_state_size = hidden_state_size
        self.gc1 = SLConv(input_dim + hidden_state_size, hidden_state_size)
        self.gc2 = SLConv(input_dim + hidden_state_size, hidden_state_size)
        self.gc3 = SLConv(input_dim + hidden_state_size, hidden_state_size)

    def forward(self, inputs, hx, S):
        x = torch.cat([inputs, hx], dim=2)  # (batch_size, num_nodes, num_features+num_hidden_features)
        u = torch.sigmoid(self.gc1(x, self.adj, S))
        r = torch.sigmoid(self.gc2(x, self.adj, S))
        x = torch.cat([inputs, r * hx], dim=2)
        c = self._activation(self.gc3(x, self.adj, S))

        new_state = u * hx + (1.0 - u) * c
        return new_state
