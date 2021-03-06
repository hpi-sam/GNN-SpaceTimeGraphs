import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from gnn.layers import (GlobalSLC, TimeBlock)
from gnn.backlog.layers import (LocalSLC, SLConv, SLGRUCell, STGCNBlock)


class GCN(nn.Module):
    def __init__(self, adj, args):
        super(GCN, self).__init__()
        # model parameters
        num_features = args.num_features
        nhid_multipliers = args.nhid_multipliers
        nhid = args.n_hid
        nclass = args.nclass
        num_nodes = args.num_nodes
        self.adj = adj

        self.layer_list = nn.ModuleList()
        for idx, layer_multiplier in enumerate(nhid_multipliers):
            out_dim = nhid * layer_multiplier
            self.layer_list.insert(idx, SLConv(num_features, out_dim, act_func=F.leaky_relu))
            num_features = out_dim
        self.gc_last = SLConv(num_features, nclass)

        self.S = Parameter(torch.ones(num_nodes, num_nodes))

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x, self.adj, self.S)
        x = self.gc_last(x, self.adj, self.S)
        return x


class GCRNN(nn.Module):
    def __init__(self, adj, args):
        super(GCRNN, self).__init__()
        # model parameters
        num_nodes = args.num_nodes
        num_units = args.num_units
        input_dim = args.num_features
        hidden_state_size = args.hidden_state_size
        nclass = args.nclass

        self.S = Parameter(torch.ones(num_nodes, num_nodes))
        self.hidden_state_size = args.hidden_state_size
        self.seq_len = args.seq_len
        self.num_nodes = args.num_nodes
        self.adj = adj
        self.gru1 = SLGRUCell(num_units, adj, num_nodes, input_dim, hidden_state_size)
        self.gc1 = SLConv(hidden_state_size, 2 * num_units, F.leaky_relu)
        self.gc2 = SLConv(2 * num_units, num_units, F.leaky_relu)
        self.gc_last = SLConv(num_units, nclass)

    def forward(self, inputs, hidden_state=None):
        batch_size = inputs.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.num_nodes, self.hidden_state_size, device=self.device)

        for step in range(self.seq_len):
            hidden_state = self.gru1(inputs[:, step], hidden_state, self.S)

        x = self.gc1(hidden_state, self.adj, self.S)
        x = self.gc2(x, self.adj, self.S)
        output = self.gc_last(x, self.adj, self.S)
        return output


class SLGCN(nn.Module):
    def __init__(self, adj, args):
        super(SLGCN, self).__init__()
        self.adj = adj
        # model parameters
        num_features = args.num_features
        nhid_multipliers = args.nhid_multipliers
        nhid = args.n_hid
        nclass = args.nclass
        num_nodes = args.num_nodes
        k = args.k

        # layers
        self.g_layer_list = nn.ModuleList()
        self.l_layer_list = nn.ModuleList()
        for idx, layer_multiplier in enumerate(nhid_multipliers):
            out_dim = nhid * layer_multiplier
            self.g_layer_list.insert(idx, GlobalSLC(adj, args, num_features, out_dim, num_nodes, act_func=F.leaky_relu))
            self.l_layer_list.insert(idx, LocalSLC(adj, num_features, out_dim, num_nodes, k, act_func=F.leaky_relu))
            num_features = out_dim
        self.g_last = GlobalSLC(adj, args, num_features, nclass, num_nodes)
        self.l_last = LocalSLC(adj, num_features, nclass, num_nodes, k)

    def forward(self, x):
        for g_layer, l_layer in zip(self.g_layer_list, self.l_layer_list):
            x = g_layer(x) + l_layer(x)
        x = self.g_last(x) + self.l_last(x)
        return x


class STGCN(nn.Module):
    def __init__(self, adj, args):
        super(STGCN, self).__init__()
        num_features = args.num_features
        nclass = args.nclass
        num_nodes = args.num_nodes

        num_timesteps = 12

        self.block1 = STGCNBlock(args, in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes, adj=adj)
        self.block2 = STGCNBlock(args, in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes, adj=adj)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps - 2 * 5) * 64, nclass)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[2], -1))) \
            .reshape(out3.shape[0], out3.shape[2], 1)
        return out4


