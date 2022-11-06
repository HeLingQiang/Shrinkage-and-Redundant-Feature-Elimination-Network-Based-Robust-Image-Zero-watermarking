import torch.nn as nn
import torch
from kornia.color.adjust import AdjustHue,AdjustSaturation,AdjustContrast,AdjustBrightness,AdjustGamma
import math
from torchvision.transforms import ToPILImage
class Adjust_hue(nn.Module):
    def __init__(self,factor):
        super(Adjust_hue, self).__init__()
        self.factor=factor

    def forward(self, noised_and_cover):
        encoded=((noised_and_cover[0]).clone())
        # encoded1=ToPILImage(encoded)
        # encoded=AdjustHue(hue_factor=0.4*math.pi)((encoded))
        # encoded=AdjustSaturation(saturation_factor=15.0)(encoded)
        encoded=AdjustBrightness(brightness_factor=1.1)(encoded)
        # encoded=AdjustContrast(contrast_factor=1.5)(encoded)
        # encoded=AdjustGamma(gamma=0.9)(encoded)
        noised_and_cover[0]=(encoded)
        return noised_and_cover