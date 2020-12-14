import torch
import math

import torch.nn as nn

from torch.nn.parameter import Parameter


class SLConv(nn.Module):
    def __init__(self, in_chanels, out_chanels):
        super(SLConv, self).__init__()
        self.in_chanels = in_chanels
        self.out_chanels = out_chanels
        self.weight = Parameter(torch.FloatTensor(in_chanels, out_chanels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, S):
        x = torch.mm(x, self.weight)  # (N,out_chanels)
        weighting = torch.mul(S, adj)  # (N,N)
        output = torch.mm(weighting, x)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_chanels) + ' -> ' \
               + str(self.out_chanels) + ')'
