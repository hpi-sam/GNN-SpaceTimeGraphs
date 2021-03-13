import math
import torch
import torch.nn as nn

from gnn.utils import get_laplacian
from torch.nn.parameter import Parameter
from torch.functional import F


class GC(nn.Module):
    def __init__(self, adj, c_in, c_out):
        """ Graph Convolution operator that sums over the neighbors

        :param adj: Normalized adjacency matrix with added self connections
        :param c_in: number of input channels
        :param c_out: number of output channels
        """
        super(GC, self).__init__()
        self.adj = adj
        self.th = nn.Parameter(torch.FloatTensor(c_in, c_out))
        self.reset_parameters()

    def forward(self, x):
        out = torch.einsum("ij,kljm->kilm", self.adj, x)
        out = torch.matmul(out, self.th)
        return F.relu(out).permute(0, 2, 1, 3)

    def reset_parameters(self):
        """ Applies z-score normalization"""
        std = 1./math.sqrt(self.th.shape[1])
        self.th.data.uniform_(-std, std)


class SGC(nn.Module):
    def __init__(self, adj, args, c_in, c_out, num_nodes, act_func=None):
        super(SGC, self).__init__()
        self.cs = args.cs
        # convolution parameters
        if args.learnable_l:
            self.ws = Parameter(get_laplacian(adj))
        else:
            self.ws = get_laplacian(adj)
        self.ts = Parameter(torch.rand((self.cs, c_in, c_out)))
        self.param_list = [self.ws, self.ts]

        self.register_buffer('t0', torch.eye(num_nodes, num_nodes))
        self.act_func = act_func
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in self.param_list:
            std = .1 / math.sqrt(parameter.size(1))
            parameter.data.uniform_(-std, std)

    def forward(self, x):
        """
        Spectral Graph Convolution operator approximated with Chebyshev polynomials

        :param x: graph signal at time [t-time_steps + 1,...,t] (batch_size, time_steps, num_nodes, c_in)
        :return: convolved signal at time [t-time_steps + 1,...,t] (batch_size, time_steps, num_nodes, c_out)
        """
        # (num_nodes, num_nodes) x (batch_size, num_nodes, in_feat) x (c_in, c_out)
        out = torch.matmul(torch.matmul(self.t0, x), self.ts[0])  # (batch_size, num_nodes, c_out)

        # computation of static graph structure convolution
        out = out + torch.matmul(torch.matmul(self.ws, x), self.ts[1])
        tk_prev = self.ws
        tk = 2. * torch.matmul(self.ws, self.ws) - self.t0
        for k in range(2, self.cs):
            out = out + torch.matmul(torch.matmul(tk, x), self.ts[k])
            tk = 2. * torch.matmul(self.ws, tk) - tk_prev

        if self.act_func:
            out = self.act_func(out)
        return out


class GlobalSLC(nn.Module):
    def __init__(self, adj, args, c_in, c_out, num_nodes, act_func=None):
        super(GlobalSLC, self).__init__()
        self.cs = args.cs
        self.cd = args.cd
        self.num_nodes = num_nodes

        # convolution parameters
        self.wp = Parameter(torch.rand(c_in, c_in))
        self.td = Parameter(torch.rand((self.cd, c_in, c_out)))
        self.param_list = [self.wp, self.td]

        self.register_buffer('t0', torch.eye(num_nodes, num_nodes))
        self.act_func = act_func
        self.reset_parameters()

        # spectral graph convolution with static graph structure
        self.sgc = SGC(adj, args, c_in, c_out, num_nodes, act_func=act_func)

    def reset_parameters(self):
        for parameter in self.param_list:
            std = .1 / math.sqrt(parameter.size(1))
            parameter.data.uniform_(-std, std)

    def forward(self, x):
        """
        Spatial Graph Convolution using the Global Structure Learning architecture.

        :param x: graph signal at time [t-time_steps + 1,...,t] (batch_size, time_steps, num_nodes, c_in)
        :return: convolved signal at time [t-time_steps + 1,...,t] (batch_size, time_steps, num_nodes, c_out)
        """
        out_s = self.sgc(x)

        # (num_nodes, num_nodes) x (batch_size, num_nodes, in_feat) x (c_in, c_out)
        out = torch.matmul(torch.matmul(self.t0, x), self.td[0])  # (batch_size, num_nodes, c_out)

        # computation of dynamical graph structure convolution
        if len(x.shape) == 4:
            wd = torch.matmul(x, torch.matmul(self.wp, torch.transpose(x, 2, 3)))
        else:
            wd = torch.matmul(x, torch.matmul(self.wp, torch.transpose(x, 1, 2)))
        # normalize wd -> to get rid of exploding gradients
        wd = wd + torch.min(wd)
        wd = wd / torch.max(wd) / self.num_nodes**2

        out_d = out + torch.matmul(torch.matmul(wd, x), self.td[1])
        tk_prev = wd
        tk = 2. * torch.matmul(wd, wd) - self.t0
        for k in range(2, self.cd):
            out_d = out_d + torch.matmul(torch.matmul(tk, x), self.td[k])
            tk = 2. * torch.matmul(wd, tk) - tk_prev

        if self.act_func:
            out_s = self.act_func(out_s)
            out_d = self.act_func(out_d)

        output = out_s + out_d
        return output


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=(0, 0)):
        """ 1-D Convolution over time

        :param in_channels: number of time-channels in the input
        :param out_channels: number of time-channels in the output
        :param kernel_size: second dimension of the kernel size. Kernel will have shape (1, kernel_size)
        :param padding: optional padding.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)

    def forward(self, x):
        # Convert into NCHW format for pytorch to perform convolutions.
        x = x.permute(0, 3, 2, 1)
        temp = self.conv1(x) + torch.sigmoid(self.conv2(x))
        out = F.relu(temp + self.conv3(x))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 3, 2, 1)
        return out


class BatchNorm(nn.Module):
    def __init__(self, num_nodes):
        """BatchNorm2D over batch and node dimension.
         Needs additional reshaping to fit the shape of the data"""
        super(BatchNorm, self).__init__()
        self.norm = nn.BatchNorm2d(num_nodes)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        # Convert into NCHW format for pytorch to perform convolutions.
        x = x.permute(0, 3, 2, 1)
        out = self.conv(x)
        # Convert back from NCHW to NHWC
        out = out.permute(0, 3, 2, 1)
        return F.relu(out)


class P3DBlock(nn.Module):
    def __init__(self, adj, args, in_channels, spatial_channels, out_channels, num_nodes):
        super(P3DBlock, self).__init__()
        self.spatial = GlobalSLC(adj, args, spatial_channels, spatial_channels, num_nodes, act_func=F.relu)
        self.temporal = TimeBlock(in_channels=spatial_channels, out_channels=spatial_channels)
        self.down_sample = Bottleneck(in_channels=in_channels, out_channels=spatial_channels)
        self.up_sample = Bottleneck(in_channels=spatial_channels, out_channels=out_channels)
        self.temp_up_sample = nn.Conv2d(10, 12, (1, 1))
        self.batch_norm = BatchNorm(num_nodes)


class P3DABlock(P3DBlock):
    def __init__(self, adj, args, in_channels, spatial_channels, out_channels, num_nodes):
        P3DBlock.__init__(self, adj, args, in_channels, spatial_channels, out_channels, num_nodes)

    def forward(self, x):
        out = F.relu(self.down_sample(x))
        out1 = F.relu(self.spatial(out))
        out2 = F.relu(self.temporal(out1))
        out3 = self.up_sample(out2)
        out3 = self.temp_up_sample(out3)
        out4 = self.batch_norm(out3)

        return out4


class P3DBBlock(P3DBlock):
    def __init__(self, adj, args, in_channels, spatial_channels, out_channels, num_nodes):
        P3DBlock.__init__(self, adj, args, in_channels, spatial_channels, out_channels, num_nodes)

    def forward(self, x):
        out = F.relu(self.down_sample(x))
        out1 = self.temporal(out)
        out2 = self.temp_up_sample(out1)
        out3 = F.relu(self.spatial(out))
        out4 = F.relu(out3 + out2)
        out5 = self.up_sample(out4)
        out6 = self.batch_norm(out5)

        return out6


class P3DCBlock(P3DBlock):
    def __init__(self, adj, args, in_channels, spatial_channels, out_channels, num_nodes):
        P3DBlock.__init__(self, adj, args, in_channels, spatial_channels, out_channels, num_nodes)

    def forward(self, x):
        out = F.relu(self.down_sample(x))
        out1 = F.relu(self.spatial(out))
        out2 = self.temporal(out1)
        out3 = self.temp_up_sample(out2)
        out3 = F.relu(out1 + out3)
        out4 = self.up_sample(out3)
        out5 = self.batch_norm(out4)

        return out5
