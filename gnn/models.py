import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.parameter import Parameter
from gnn.layers import SLConv, SLGRUCell
from gnn.dataset import TrafficDataset
from torch.utils.data import DataLoader


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, n, nhid_multipliers=(1, 2), device=None):
        super(GCN, self).__init__()
        in_dim = nfeat
        self.layer_list = np.zeros_like(nhid_multipliers, dtype='object_')
        for idx, layer_multiplier in enumerate(nhid_multipliers):
            out_dim = nhid * layer_multiplier
            self.layer_list[idx] = SLConv(in_dim, out_dim, F.leaky_relu).to(device)
            in_dim = out_dim
        self.gc_last = SLConv(in_dim, nclass).to(device)

        self.S = Parameter(torch.ones(n, n, device=device))

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
    def __init__(self, adj, num_nodes, num_units, input_dim, hidden_state_size, seq_len):
        super(GCRNN, self).__init__()
        self.S = Parameter(torch.ones(num_nodes, num_nodes))
        self.hidden_state_size = hidden_state_size
        self.seq_len = seq_len
        self.gru1 = SLGRUCell(num_units, adj, num_nodes, input_dim, hidden_state_size)

    def forward(self, inputs, hidden_state=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_state_size)

        hidden_states = []
        for step in range(self.seq_len):
            hidden_state = self.gru1(inputs[:, step], hidden_state, S)
            hidden_states.append(hidden_state)


