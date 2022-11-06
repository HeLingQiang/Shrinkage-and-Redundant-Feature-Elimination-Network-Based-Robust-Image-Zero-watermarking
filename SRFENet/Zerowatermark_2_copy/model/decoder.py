
from options import HiDDenConfiguration
from model.my_dense import *
from torchvision import models
from model.formal_def import *

class Decoder(nn.Module):

    def __init__(self, config: HiDDenConfiguration):

        super(Decoder, self).__init__()
        self.conv_channels=64
        self.zw_size=32
        self.conv_num=8
        #######################  feature extracting
        self.cover_start=nn.Conv2d(3, self.conv_channels, kernel_size=3, padding=1)
        self.cover_dense_1=DenseBlock(num_convs=2,in_channels=self.conv_channels,out_channels=self.conv_channels)
        self.cover_maxpool_1=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.cover_dense_2=DenseBlock(num_convs=2,in_channels=self.conv_channels,out_channels=self.conv_channels)
        self.cover_maxpool_2=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.cover_conv_1 = Conv_BN_Relu(self.conv_channels, self.conv_channels)
        self.cover_conv_2 =Conv_BN_Relu(self.conv_channels, self.conv_channels)
        self.cover_conv_3=Conv_BN_Relu(self.conv_channels,self.conv_channels)
        self.cover_conv_4=Conv_BN_Relu(self.conv_channels,self.conv_channels)

        # self.cover_denoise_2=denoise(channel=self.conv_channels,gap_size=(1,1))
        self.cover_SM=denoise(channel=self.conv_channels,gap_size=(1,1))
        self.cover_REFM=REFM(self.conv_channels)
        self.noise_start = nn.Conv2d(3, self.conv_channels, kernel_size=3, padding=1)
        self.noise_dense_1 = DenseBlock(num_convs=2, in_channels=self.conv_channels, out_channels=self.conv_channels)
        self.noise_maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.noise_maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.noise_denoise_2 = denoise(channel=self.conv_channels, gap_size=(1, 1))
        self.noise_SM = denoise(channel=self.conv_channels, gap_size=(1, 1))
        self.noise_REFM=REFM(self.conv_channels)
        self.noise_conv_1 = Conv_BN_Relu(self.conv_channels, self.conv_channels)
        self.noise_conv_2 = Conv_BN_Relu(self.conv_channels, self.conv_channels)
        self.noise_conv_3=Conv_BN_Relu(self.conv_channels,self.conv_channels)
        self.noise_conv_4=Conv_BN_Relu(self.conv_channels, self.conv_channels)

    def forward(self, cover_image,noise_image):
        c0=self.cover_start(cover_image)
        c1=self.cover_dense_1(c0)
        c1=self.cover_conv_1(c1)
        c2=self.cover_maxpool_1(c1)

        c3=self.cover_conv_2(c2)
        c3=self.cover_conv_3(c3)
        c4=self.cover_maxpool_2(c3)

        # cover_out=c4=self.cover_denoise_2(c4)
        cover_out=c4=self.cover_SM(c4)
        cover_out=self.cover_REFM(cover_out)
        cover_feature=torch.mean(cover_out,dim=1)
        cover_feature = torch.unsqueeze(cover_feature, dim=1)

        n0=self.noise_start(noise_image)
        n1=self.noise_dense_1(n0)
        n1=self.noise_conv_1(n1)
        n2=self.noise_maxpool_1(n1)

        n3=self.noise_conv_2(n2)
        n3=self.noise_conv_3(n3)
        n4=self.noise_maxpool_2(n3)

        noise_out = n4 =self.noise_SM(n4)
        noise_out=self.noise_REFM(noise_out)
        noise_feature=torch.mean(noise_out,dim=1)
        noise_feature = torch.unsqueeze(noise_feature, dim=1)

        return cover_feature,noise_feature


