import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from gnn.layers import GlobalSLC, LocalSLC, SLConv, SLGRUCell
from gnn.utils import generate_knn_ids


# TODO: rename model parameters
class GCN(nn.Module):
    def __init__(self, adj, args, device=None):
        super(GCN, self).__init__()
        # model parameters
        in_dim = args.num_features
        nhid_multipliers = args.nhid_multipliers
        nhid = args.n_hid
        nclass = args.nclass
        N = args.num_nodes
        self.adj = adj

        self.layer_list = np.zeros_like(nhid_multipliers, dtype='object_')
        for idx, layer_multiplier in enumerate(nhid_multipliers):
            out_dim = nhid * layer_multiplier
            self.layer_list[idx] = SLConv(in_dim,
                                          out_dim,
                                          act_func=F.leaky_relu).to(device)
            in_dim = out_dim
        self.gc_last = SLConv(in_dim, nclass).to(device)

        self.S = Parameter(torch.ones(N, N, device=device))

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x, self.adj, self.S)
        x = self.gc_last(x, self.adj, self.S)
        return x


class MultiTempGCN(nn.Module):
    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 n,
                 time_hops=(1, 12, 288),
                 device=None):
        super(MultiTempGCN, self).__init__()
        self.gcn_blocks = np.zeros_like(time_hops, dtype='object_')
        for idx, time_hop in time_hops:
            self.gcn_blocks[idx] = self.gcn_block = GCN(nfeat=nfeat,
                                                        nhid=nhid,
                                                        nclass=nclass,
                                                        n=n,
                                                        device=device)
        self.linear = torch.nn.Linear(len(time_hops), nclass)

    def forward(self, x, adj):
        for idx, gcn_block in enumerate(self.gcn_blocks):
            x[idx] = F.relu(gcn_block(x[idx], adj))
        x = self.linear(x)
        return x


class GCRNN(nn.Module):
    def __init__(self, adj, args, device=None):
        super(GCRNN, self).__init__()
        # model parameters
        num_nodes = args.num_nodes
        num_units = args.num_units
        input_dim = args.num_features
        hidden_state_size = args.hidden_state_size
        nclass = args.nclass
        self.device = device

        self.S = Parameter(torch.ones(num_nodes, num_nodes, device=device))
        self.hidden_state_size = args.hidden_state_size
        self.seq_len = args.seq_len
        self.num_nodes = args.num_nodes
        self.adj = adj
        self.gru1 = SLGRUCell(num_units, adj, num_nodes, input_dim,
                              hidden_state_size).to(device)
        self.gc1 = SLConv(hidden_state_size, 2 * num_units,
                          F.leaky_relu).to(device)
        self.gc2 = SLConv(2 * num_units, num_units, F.leaky_relu).to(device)
        self.gc_last = SLConv(num_units, nclass).to(device)

    def forward(self, inputs, hidden_state=None):
        batch_size = inputs.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size,
                                       self.num_nodes,
                                       self.hidden_state_size,
                                       device=self.device)

        for step in range(self.seq_len):
            hidden_state = self.gru1(inputs[:, step], hidden_state, self.S)

        x = self.gc1(hidden_state, self.adj, self.S)
        x = self.gc2(x, self.adj, self.S)
        output = self.gc_last(x, self.adj, self.S)
        return output


class SLGCN(nn.Module):
    def __init__(self, adj, args, device=None):
        super(SLGCN, self).__init__()
        self.adj = adj
        # model parameters
        in_dim = args.num_features
        nhid_multipliers = args.nhid_multipliers
        nhid = args.n_hid
        nclass = args.nclass
        N = args.num_nodes
        k = args.k

        # layers
        self.g_layer_list = np.zeros_like(nhid_multipliers, dtype='object_')
        self.l_layer_list = np.zeros_like(nhid_multipliers, dtype='object_')
        for idx, layer_multiplier in enumerate(nhid_multipliers):
            out_dim = nhid * layer_multiplier
            self.g_layer_list[idx] = GlobalSLC(
                in_dim, out_dim, N, act_func=F.leaky_relu).to(device)
            self.l_layer_list[idx] = LocalSLC(adj,
                                              in_dim,
                                              out_dim,
                                              N,
                                              k,
                                              act_func=F.leaky_relu).to(device)
            in_dim = out_dim
        self.g_last = GlobalSLC(in_dim, nclass, N).to(device)
        self.l_last = LocalSLC(adj, in_dim, nclass, N, k).to(device)

    def forward(self, x):
        for g_layer, l_layer in zip(self.g_layer_list, self.l_layer_list):
            x = g_layer(x) + l_layer(x)
        x = self.g_last(x) + self.l_last(x)
        return x
