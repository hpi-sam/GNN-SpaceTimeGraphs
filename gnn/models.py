import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.parameter import Parameter
from gnn.layers import SLConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, n, nhid_multipliers=(), device=None):
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
