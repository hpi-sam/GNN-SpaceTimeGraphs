import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from gnn.layers import SLConv, SLGRUCell


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, n):
        super(GCN, self).__init__()

        self.gc1 = SLConv(nfeat, nhid, n)
        self.gc2 = SLConv(nhid, 2*nhid, n)
        self.gc3 = SLConv(2*nhid, nclass, n)

        self.S = Parameter(torch.ones(n, n))

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj, self.S))
        x = F.relu(self.gc2(x, adj, self.S))
        x = self.gc3(x, adj, self.S)
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


