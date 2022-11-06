import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from model.my_dense import *
class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out                it has a watermark inserted into it, or not.
    """

    def conv7(self, in_channel, out_channel):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_channel,
                         stride=1,
                         kernel_size=7, padding=3)

    def conv3(self, in_channel, out_chanenl):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_chanenl,
                         stride=1,
                         kernel_size=3,
                         padding=1)
    def __init__(self, config: HiDDenConfiguration):
        super(Discriminator, self).__init__()
        self.conv_channels=config.discriminator_channels

        self.start_layer = nn.Sequential(
            self.conv3(3, self.conv_channels),
        )
        self.DenseBlock_1 = DenseBlock(2, self.conv_channels, self.conv_channels)

        self.attention = REFM(self.conv_channels)
        self.a1 = BN_Relu_Conv(self.conv_channels, self.conv_channels)
        self.a2 = BN_Relu_Conv(self.conv_channels * 2, self.conv_channels)
        self.a3 = BN_Relu_Conv(self.conv_channels * 3, self.conv_channels)
        self.a4 = BN_Relu_Conv(self.conv_channels * 4, self.conv_channels)

        self.b1 = BN_Relu_Conv(self.conv_channels, self.conv_channels)
        self.b2 = BN_Relu_Conv(self.conv_channels * 2, self.conv_channels)
        self.b3 = BN_Relu_Conv(self.conv_channels * 3, self.conv_channels)
        self.b4 = BN_Relu_Conv(self.conv_channels * 4, self.conv_channels)
        self.b_out = BN_Relu_Conv(self.conv_channels * 5, self.conv_channels)

        self.final_layer = BN_Leaky_Relu_Conv(self.conv_channels, config.message_length)

        self.pooling=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.linear = nn.Linear(config.message_length, 1)

    def forward(self, image):

        decoded_image = self.start_layer(image)  # (12,64,128,128)
        a0=b0=decoded_image = self.DenseBlock_1(decoded_image)
        # a0 = b0 = self.DenseBlock_1_out(decoded_image)

        a1 = self.a1(a0)
        a2 = self.a2(torch.cat([a0, a1], dim=1))
        a3 = self.a3(torch.cat([a0, a1, a2], dim=1))
        a4 = self.a4(torch.cat([a0, a1, a2, a3], dim=1))
        # attention
        A1 = self.attention(a1)
        A2 = self.attention(a2)
        A3 = self.attention(a3)
        A4 = self.attention(a4)

        # watermarking embeding
        b1 = self.b1(b0)
        b1 = A1 * b1
        b2 = self.b2(torch.cat([b0, b1], dim=1))
        b2 = A2 * b2
        b3 = self.b3(torch.cat([b0, b1, b2], dim=1))
        b3 = A3 * b3
        b4 = self.b4(torch.cat([b0, b1, b2, b3], dim=1))
        b4 = A4 * b4
        out = self.b_out(torch.cat([b0, b1, b2, b3, b4], dim=1))

        out = self.final_layer(out)
        out = self.pooling(out)  # (32,30,1,1)
        out.squeeze_(3).squeeze_(2)  # (32,30)
        watermark = self.linear(out)

        return watermark