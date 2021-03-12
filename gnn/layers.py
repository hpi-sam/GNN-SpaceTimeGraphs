import math

import torch
import torch.nn as nn
from torch.functional import F
from torch.nn.parameter import Parameter

from gnn.argparser import parse_arguments
from gnn.utils import generate_knn_ids, get_laplacian

parser = parse_arguments()
args = parser.parse_args()

if args.gpu:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cpu")


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
        stdv = 1. / math.sqrt(self.th.shape[1])
        self.th.data.uniform_(-stdv, stdv)


# Has only been used when studying the "Structure Learning Convolution" and will be replaced by GlobalSLC and LocalSLC
@DeprecationWarning
class SLConv(nn.Module):
    def __init__(self, c_in, c_out, act_func=None):
        super(SLConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.weight = Parameter(torch.rand(c_in, c_out))
        self.reset_parameters()
        self.act_func = act_func

    def reset_parameters(self):
        """ Applies z-score normalization"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, S):
        x = torch.matmul(x, self.weight)  # (1,num_nodes,c_out)
        weighting = torch.mul(S, adj)  # (num_nodes,num_nodes)
        output = torch.matmul(weighting, x)
        if self.act_func:
            output = self.act_func(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.c_in) + ' -> ' \
               + str(self.c_out) + ')'


class SGC(nn.Module):
    def __init__(self, adj, args, c_in, c_out, num_nodes, cs=6, act_func=None):
        super(SGC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.num_nodes = num_nodes
        self.cs = cs

        # convolution parameters
        if args.learnable_l:
            self.ws = Parameter(get_laplacian(adj))
        else:
            self.ws = get_laplacian(adj)
        self.ts = Parameter(torch.rand((cs, c_in, c_out)))
        self.param_list = [self.ws, self.ts]

        self.t0 = torch.eye(self.num_nodes, self.num_nodes, device=DEVICE)
        self.act_func = act_func
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in self.param_list:
            stdv = .1 / math.sqrt(parameter.size(1))
            parameter.data.uniform_(-stdv, stdv)

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
    def __init__(self, adj, args, c_in, c_out, num_nodes, cs=6, cd=6, act_func=None):
        super(GlobalSLC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.num_nodes = num_nodes
        self.cs = cs
        self.cd = cd

        # convolution parameters
        self.wp = Parameter(torch.rand(c_in, c_in))
        self.td = Parameter(torch.rand((cd, c_in, c_out)))
        self.param_list = [self.wp, self.td]

        self.t0 = torch.eye(self.num_nodes, self.num_nodes, device=DEVICE)
        self.act_func = act_func
        self.reset_parameters()

        # spectral graph convolution with static graph structure
        self.sgc = SGC(adj, args, c_in, c_out, num_nodes, cs, act_func=act_func)

    def reset_parameters(self):
        for parameter in self.param_list:
            stdv = .1 / math.sqrt(parameter.size(1))
            parameter.data.uniform_(-stdv, stdv)

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


class LocalSLC(nn.Module):
    def __init__(self, adj, c_in, c_out, num_nodes, k, g=None, act_func=None):
        super(LocalSLC, self).__init__()
        self.adj = adj
        self.c_in = c_in
        self.c_out = c_out
        self.num_nodes = num_nodes
        self.k = k
        self.act_func = act_func

        # learnable parameters and functions
        self.bs = Parameter(torch.randn(num_nodes, self.k))
        self.ws = Parameter(torch.randn(self.k, c_in, c_out))
        self.wd = Parameter(torch.randn(self.k, c_in, c_out))
        self.nu = Parameter(torch.randn(c_in))
        # TODO: implement dynamical component of local convolution
        self.param_list = [self.bs, self.ws]
        self.knn_ids = generate_knn_ids(self.adj, self.k)

        self.reset_parameters()

    def reset_parameters(self):
        for parameter in self.param_list:
            stdv = .1 / math.sqrt(parameter.size(1))
            parameter.data.uniform_(-stdv, stdv)

    def dynamical_part(self, x):
        bd = torch.matmul(x, self.nu)
        return torch.einsum("bnk,kio,bnki->bno", bd, self.wd, x)

    def static_part(self, x):
        return torch.einsum("nk,kio,bnki->bno", self.bs, self.ws, x)

    def forward(self, x):
        x = x[:, self.knn_ids, :]  # (batch_size, n, k, c_in)
        y = self.static_part(x)  #+ self.dynamical_part(x)
        if self.act_func:
            y = self.act_func(y)
        return y


class SLGRUCell(nn.Module):
    def __init__(self, num_units, adj, num_nodes, input_dim,
                 hidden_state_size, gconv=SLConv):
        """ GRU Cell that integrates Graph Convolutions into the gating mechanisms"""
        super().__init__()
        self._activation = torch.tanh
        self.adj = adj
        self._num_units = num_units
        self._num_nodes = num_nodes
        self._input_dim = input_dim
        self._hidden_state_size = hidden_state_size
        self.gc1 = gconv(input_dim + hidden_state_size, hidden_state_size)
        self.gc2 = gconv(input_dim + hidden_state_size, hidden_state_size)
        self.gc3 = gconv(input_dim + hidden_state_size, hidden_state_size)

    def forward(self, inputs, hx, S):
        x = torch.cat([inputs, hx], dim=2)  # (batch_size, num_nodes, num_features+num_hidden_features)
        u = torch.sigmoid(self.gc1(x, self.adj, S))
        r = torch.sigmoid(self.gc2(x, self.adj, S))
        x = torch.cat([inputs, r * hx], dim=2)
        c = self._activation(self.gc3(x, self.adj, S))

        new_state = u * hx + (1.0 - u) * c
        return new_state


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


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, adj):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=out_channels)
        self.spatial1 = GlobalSLC(out_channels, spatial_channels, num_nodes, act_func=F.relu)
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = BatchNorm(num_nodes)

    def forward(self, x):
        t = self.temporal1(x)
        t2 = self.spatial1(t)
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)


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
