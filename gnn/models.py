import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.parameter import Parameter
from gnn.layers import SLConv, SLGRUCell, GlobalSLC, LocalSLC


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, N, nhid_multipliers=(1, 2), device=None):
        super(GCN, self).__init__()
        in_dim = nfeat
        self.layer_list = np.zeros_like(nhid_multipliers, dtype='object_')
        for idx, layer_multiplier in enumerate(nhid_multipliers):
            out_dim = nhid * layer_multiplier
            self.layer_list[idx] = SLConv(in_dim, out_dim, act_func=F.leaky_relu).to(device)
            in_dim = out_dim
        self.gc_last = SLConv(in_dim, nclass).to(device)

        self.S = Parameter(torch.ones(N, N, device=device))

    def forward(self, x, adj):
        for layer in self.layer_list:
            x = layer(x, adj, self.S)
        x = self.gc_last(x, adj, self.S)
        return x


class MultiTempGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, n, time_hops=(1, 12, 288), device=None):
        super(MultiTempGCN, self).__init__()
        self.gcn_blocks = np.zeros_like(time_hops, dtype='object_')
        for idx, time_hop in time_hops:
            self.gcn_blocks[idx] = self.gcn_block = GCN(nfeat=nfeat,
                                                        nhid=nhid,
                                                        nclass=nclass,
                                                        n=n, device=device)
        self.linear = torch.nn.Linear(len(time_hops), nclass)

    def forward(self, x, adj):
        for idx, gcn_block in enumerate(self.gcn_blocks):
            x[idx] = F.relu(gcn_block(x[idx], adj))
        x = self.linear(x)
        return x


class GCRNN(nn.Module):
    def __init__(self, adj, num_nodes, num_units, input_dim, nclass, hidden_state_size, seq_len):
        super(GCRNN, self).__init__()
        self.S = Parameter(torch.ones(num_nodes, num_nodes))
        self.hidden_state_size = hidden_state_size
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.adj = adj
        self.gru1 = SLGRUCell(num_units, adj, num_nodes, input_dim, hidden_state_size)
        self.gc1 = SLConv(hidden_state_size, 2*num_units, F.leaky_relu)
        self.gc2 = SLConv(2*num_units, num_units, F.leaky_relu)
        self.gc_last = SLConv(num_units, nclass)

    def forward(self, inputs, hidden_state=None):
        batch_size = inputs.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.num_nodes, self.hidden_state_size, device=torch.device("cuda:0"))

        for step in range(self.seq_len):
            hidden_state = self.gru1(inputs[:, step], hidden_state, self.S)

        x = self.gc1(hidden_state, self.adj, self.S)
        x = self.gc2(x, self.adj, self.S)
        output = self.gc_last(x, self.adj, self.S)
        return output


class SLGCN(nn.Module):
    def __init__(self, adj, nfeat, nhid, nclass, N, nhid_multipliers=(1, 2), device=None):
        super(SLGCN, self).__init__()
        self.adj = adj
        in_dim = nfeat
        self.g_layer_list = np.zeros_like(nhid_multipliers, dtype='object_')
        self.l_layer_list = np.zeros_like(nhid_multipliers, dtype='object_')
        for idx, layer_multiplier in enumerate(nhid_multipliers):
            out_dim = nhid * layer_multiplier
            self.g_layer_list[idx] = GlobalSLC(in_dim, out_dim, N, act_func=F.leaky_relu).to(device)
            self.l_layer_list[idx] = LocalSLC(in_dim, out_dim, N, act_func=F.leaky_relu).to(device)
            in_dim = out_dim
        self.g_last = GlobalSLC(in_dim, nclass, N).to(device)
        self.l_last = LocalSLC(in_dim, nclass, N).to(device)

    def forward(self, x):
        for g_layer, l_layer in zip(self.g_layer_list, self.l_layer_list):
            x = g_layer(x) + l_layer(x, self.adj)
        x = self.g_last(x) + self.l_last(x, self.adj)
        return x
