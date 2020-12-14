import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from gnn.layers import SLConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, n):
        super(GCN, self).__init__()

        self.gc1 = SLConv(nfeat, nhid)
        self.gc2 = SLConv(nhid, nclass)

        self.S = Parameter(torch.FloatTensor(n, n)).data.uniform_(-1., 1.)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj, self.S))
        x = self.gc2(x, adj, self.S)
        return x
