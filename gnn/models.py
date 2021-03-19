import torch.nn as nn
import torch.nn.functional as F

from gnn.layers import Bottleneck, P3DABlock, P3DBBlock, P3DCBlock


class P3D(nn.Module):
    def __init__(self, adj, args):
        super(P3D, self).__init__()
        self.num_out_steps = len(args.forecast_horizon)
        num_timesteps = 12
        num_features = args.num_features
        nclass = args.nclass
        num_nodes = args.num_nodes

        bottleneck_channels = args.bottleneck_channels
        spatial_channels = args.spatial_channels

        self.up_sample = Bottleneck(in_channels=num_features, out_channels=bottleneck_channels)
        self.block1 = P3DABlock(adj=adj, args=args, in_channels=bottleneck_channels, spatial_channels=spatial_channels,
                                out_channels=bottleneck_channels, num_nodes=num_nodes)
        self.block2 = P3DBBlock(adj=adj, args=args, in_channels=bottleneck_channels, spatial_channels=spatial_channels,
                                out_channels=bottleneck_channels, num_nodes=num_nodes)
        self.block3 = P3DCBlock(adj=adj, args=args, in_channels=bottleneck_channels, spatial_channels=spatial_channels,
                                out_channels=bottleneck_channels, num_nodes=num_nodes)

        self.fc = nn.Sequential(nn.Linear(num_timesteps * bottleneck_channels, num_timesteps * bottleneck_channels),
                                nn.LeakyReLU(),
                                nn.Linear(num_timesteps * bottleneck_channels, nclass * self.num_out_steps))
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, x):
        out1 = self.up_sample(x)
        out2 = F.relu(out1 + self.block1(out1))
        out3 = F.relu(out2 + self.block2(out2))
        out4 = F.relu(out3 + self.block3(out3))
        out = self.fc(out4.reshape((out4.shape[0], out4.shape[2], -1)))\
            .reshape(out4.shape[0], self.num_out_steps, out4.shape[2], 1)
        out = self.dropout(out)

        return out
