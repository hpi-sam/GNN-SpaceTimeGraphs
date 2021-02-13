import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from gnn.layers import (GlobalSLC, LocalSLC, SLConv, SLGRUCell, STGCNBlock, TimeBlock, P3DABlock, P3DBBlock, P3DCBlock,
                        Bottleneck)


class GCN(nn.Module):
    def __init__(self, adj, args, device=None):
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
            self.layer_list.insert(idx, SLConv(num_features, out_dim, act_func=F.leaky_relu).to(device))
            num_features = out_dim
        self.gc_last = SLConv(num_features, nclass).to(device)

        self.S = Parameter(torch.ones(num_nodes, num_nodes, device=device))

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x, self.adj, self.S)
        x = self.gc_last(x, self.adj, self.S)
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
        self.gru1 = SLGRUCell(num_units, adj, num_nodes, input_dim, hidden_state_size).to(device)
        self.gc1 = SLConv(hidden_state_size, 2 * num_units, F.leaky_relu).to(device)
        self.gc2 = SLConv(2 * num_units, num_units, F.leaky_relu).to(device)
        self.gc_last = SLConv(num_units, nclass).to(device)

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
    def __init__(self, adj, args, device=None):
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
            self.g_layer_list.insert(idx, GlobalSLC(num_features, out_dim, num_nodes, act_func=F.leaky_relu).to(device))
            self.l_layer_list.insert(idx, LocalSLC(adj, num_features, out_dim, num_nodes, k, act_func=F.leaky_relu).to(
                device))
            num_features = out_dim
        self.g_last = GlobalSLC(num_features, nclass, num_nodes).to(device)
        self.l_last = LocalSLC(adj, num_features, nclass, num_nodes, k).to(device)

    def forward(self, x):
        for g_layer, l_layer in zip(self.g_layer_list, self.l_layer_list):
            x = g_layer(x) + l_layer(x)
        x = self.g_last(x) + self.l_last(x)
        return x


class STGCN(nn.Module):
    def __init__(self, adj, args, device=None):
        super(STGCN, self).__init__()
        num_features = args.num_features
        nclass = args.nclass
        num_nodes = args.num_nodes

        num_timesteps = 12

        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes, adj=adj).to(device)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes, adj=adj).to(device)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64).to(device)
        self.fully = nn.Linear((num_timesteps - 2 * 5) * 64, nclass).to(device)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[2], -1)))\
            .reshape(out3.shape[0], out3.shape[2], 1)
        return out4


class P3D(nn.Module):
    def __init__(self, adj, args, device=None):
        super(P3D, self).__init__()
        self.num_out_steps = len(args.forecast_horizon)
        num_timesteps = 12
        num_features = args.num_features
        nclass = args.nclass
        num_nodes = args.num_nodes

        bottleneck_channels = args.bottleneck_channels
        spatial_channels = args.spatial_channels

        self.up_sample = Bottleneck(in_channels=num_features, out_channels=bottleneck_channels).to(device)
        self.block1 = P3DABlock(in_channels=bottleneck_channels, spatial_channels=spatial_channels,
                                out_channels=bottleneck_channels, num_nodes=num_nodes).to(device)
        self.block2 = P3DBBlock(in_channels=bottleneck_channels, spatial_channels=spatial_channels,
                                out_channels=bottleneck_channels, num_nodes=num_nodes).to(device)
        self.block3 = P3DCBlock(in_channels=bottleneck_channels, spatial_channels=spatial_channels,
                                out_channels=bottleneck_channels, num_nodes=num_nodes).to(device)

        self.fc = nn.Linear(num_timesteps * bottleneck_channels, nclass * self.num_out_steps).to(device)

    def forward(self, x):
        out1 = self.up_sample(x)
        out2 = F.relu(out1 + self.block1(out1))
        out3 = F.relu(out2 + self.block2(out2))
        out4 = F.relu(out3 + self.block3(out3))
        out = self.fc(out4.reshape((out4.shape[0], out4.shape[2], -1)))\
            .reshape(out4.shape[0], self.num_out_steps, out4.shape[2], 1)

        return out
