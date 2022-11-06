import torch.nn as nn
import torch
from kornia.color.adjust import AdjustHue, AdjustSaturation, AdjustContrast, AdjustBrightness, AdjustGamma
import math
from torchvision.transforms import ToPILImage


class Adjust_contrast(nn.Module):
    def __init__(self, factor):
        super(Adjust_contrast, self).__init__()
        self.factor = factor

    def forward(self, noised_and_cover):
        encoded = ((noised_and_cover[0]).clone())
        encoded = AdjustContrast(contrast_factor=self.factor)(encoded)
        noised_and_cover[0] = (encoded)
        return noised_and_cover


class Adjust_Brightness(nn.Module):
    def __init__(self, factor):
        super(Adjust_Brightness, self).__init__()
        self.factor = factor

    def forward(self, noised_and_cover):
        encoded = ((noised_and_cover[0]).clone())
        encoded = AdjustBrightness(brightness_factor=self.factor)(encoded)
        noised_and_cover[0] = (encoded)
        return noised_and_cover


class Adjust_Saturation(nn.Module):
    def __init__(self, factor):
        super(Adjust_Saturation, self).__init__()
        self.factor = factor

    def forward(self, noised_and_cover):
        encoded = ((noised_and_cover[0]).clone())
        encoded = AdjustSaturation(saturation_factor=self.factor)(encoded)  # 15.0
        noised_and_cover[0] = (encoded)
        return noised_and_cover


class Adjust_Hue(nn.Module):
    def __init__(self, factor):
        super(Adjust_Hue, self).__init__()
        self.factor = factor

    def forward(self, noised_and_cover):
        encoded = ((noised_and_cover[0]).clone())
        encoded = AdjustHue(hue_factor=0.4 * math.pi)((encoded))
        noised_and_cover[0] = (encoded)
        return noised_and_cover


class Adjust_Gamma(nn.Module):
    def __init__(self, factor):
        super(Adjust_Gamma, self).__init__()
        self.factor = factor

    def forward(self, noised_and_cover):
        encoded = ((noised_and_cover[0]).clone())
        encoded = AdjustGamma(gamma=self.factor)(encoded)  # 0.9
        noised_and_cover[0] = (encoded)
        return noised_and_cover
