from options import HiDDenConfiguration
from model.my_dense import *

class Encoder(nn.Module):
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()

    def forward(self, image, message):
        return image