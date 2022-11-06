import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser


class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):

        super(EncoderDecoder, self).__init__()
        # self.encoder = Encoder(config)
        self.noiser = noiser
        self.decoder = Decoder(config)

    def forward(self, cover_image, noise_image):
        # add random noise
        noise_and_cover = self.noiser([noise_image,cover_image])
        noise_image = noise_and_cover[0]

        #feature extracted from SRFENET
        cover_feature,noise_feature=self.decoder(cover_image,noise_image)
        return cover_feature, noise_feature