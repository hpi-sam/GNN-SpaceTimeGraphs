import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from gnn.layers import SLConv


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
