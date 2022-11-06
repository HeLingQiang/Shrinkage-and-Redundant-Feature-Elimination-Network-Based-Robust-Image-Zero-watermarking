import time
import torch
from torch import nn, optim
import torch.nn.functional as F


def BN_Relu_Conv(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk

def BN_Leaky_Relu_Conv(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk

def Conv_BN_Relu(in_channels, out_channels):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                        )
    return blk

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(BN_Relu_Conv(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels
        self.dense_out=nn.Sequential(nn.BatchNorm2d(self.out_channels),
                        nn.ReLU(),
                        nn.Conv2d(self.out_channels, out_channels, kernel_size=3, padding=1),
                                     )
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        X=self.dense_out(X)
        return X

class denoise(nn.Module):
    def __init__(self, channel, gap_size):
        super(denoise, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x_raw = x
        x_abs= torch.abs(x_raw)
        x = self.gap(x_abs)
        average = torch.flatten(x, 1)
        # average = torch.mean(x, dim=1, keepdim=True)
        # average = x
        x = self.fc(average)
        x = torch.mul(average, x)
        x = x.unsqueeze(2).unsqueeze(2)

        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x

#Inter-feature weight learning
class Inter_feature(nn.Module):
    def __init__(self, channel, ratio=16):
        super(Inter_feature, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

#Intra_feature weight learning
class Intra_feature(nn.Module):
    def __init__(self):
        super(Intra_feature, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class REFM(nn.Module):
    def __init__(self, channel):
        super(REFM, self).__init__()
        self.inter_feature = Inter_feature(channel)
        self.intra_feature = Intra_feature()

    def forward(self, x):
        out = self.inter_feature(x)*x
        out = self.intra_feature(out) * out
        return out

