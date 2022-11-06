import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional
from options import HiDDenConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
# from vgg_loss import VGGLoss
from noise_layers.noiser import Noiser
from model.formal_def import *
# from tensorflow.python.ops.image_ops import ssim
import utils
import SSIM as loss_ssim
import pandas as pd
class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser, tb_logger):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param noiser: Object representing stacked noise layers.
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(Hidden, self).__init__()

        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)
        self.discriminator = Discriminator(configuration).to(device)
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters(),lr=0.0001)
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        self.config = configuration
        self.device = device
        self.ssim_loss=loss_ssim.SSIM()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)
        self.balance_loss=Balance_loss().to(device)

        self.cover_label = 1
        self.encoded_label = 0
        self.tb_logger = tb_logger
        if tb_logger is not None:

            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            encoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            decoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))
            discrim_final = self.discriminator._modules['linear']
            discrim_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/discrim_out'))

    def expand_message(self,image,message):
        B,C,H,W=image.size()
        message = message
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)
        expanded_message = expanded_message.expand(-1, -1, H,W)
        return expanded_message

    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        cover_image,noise_image=batch
        batch_size = cover_image.shape[0]
        self.encoder_decoder.train()
        self.discriminator.train()
        with torch.enable_grad():
            # ---------------- Train the discriminator -----------------------------
            self.optimizer_discrim.zero_grad()
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_on_cover = self.discriminator(cover_image)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())
            d_loss_on_cover.backward()
            # train on fake
            # cover_zw,noise_zw = self.encoder_decoder(cover_image,noise_image)
            cover_feature, noise_feature = self.encoder_decoder(cover_image, noise_image)
            d_on_encoded = self.discriminator(noise_image.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())
            d_loss_on_encoded.backward()
            self.optimizer_discrim.step()
            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            # target label for encoded images should be 'cover', because we want to fool the discriminator
            d_on_encoded_for_enc = self.discriminator(noise_image)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())
            self.optimizer_enc_dec.zero_grad()

            g_loss=self.balance_loss(cover_feature,noise_feature)
            g_loss.backward()
            self.optimizer_enc_dec.step()

        cover_feature_round=cover_feature.detach().cpu().numpy()
        noise_feature_round=noise_feature.detach().cpu().numpy()

        # generattion zerowatermark
        cover_zw_round=my_zw(cover_feature_round)
        noise_zw_round=my_zw(noise_feature_round)

        cover_zw_frame=pd.DataFrame(cover_zw_round[0].squeeze())
        noise_zw_frame=pd.DataFrame(noise_zw_round[0].squeeze())
        writer = pd.ExcelWriter('/home/dell/Documents/HLQ/Zerowatermark_2/Result/zw.xlsx', engine='xlsxwriter')
        cover_zw_frame.to_excel(writer,sheet_name='ZW',index=False,header=None,startrow=0,startcol=0)
        noise_zw_frame.to_excel(writer,sheet_name='ZW',index=False,header=None,startrow=0,startcol=cover_zw_round.shape[3]+1)
        compare_zw=pd.DataFrame(cover_zw_round[1].squeeze())
        compare_zw.to_excel(writer,sheet_name='ZW',index=False,header=None,startrow=cover_zw_round.shape[2]+1,startcol=0)
        writer.save()

        zw_row=cover_zw_round.shape[2]
        zw_col=cover_zw_round.shape[3]
        BER=np.sum(np.abs(cover_zw_round-noise_zw_round))/(batch_size*zw_row*zw_col)
        NCC=F_NCC(cover_zw_round,noise_zw_round)
        NC,NC_max,NC_avg=F_NC(cover_zw_round)

        losses = {
            'g_loss         ': g_loss.item(),
            'BER            ': BER,
            'NCC            ': NCC,
            'NC_max         ': NC_max,
            'NC_avg         ': NC_avg,
        }
        return losses,(cover_zw_round,noise_zw_round)

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        # if TensorboardX logging is enabled, save some of the tensors.
        if self.tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
            discrim_final = self.discriminator._modules['linear']
            self.tb_logger.add_tensor('weights/discrim_out', discrim_final.weight)

        cover_image,noise_image=batch
        batch_size=cover_image.shape[0]
        self.encoder_decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            cover_feature,noise_feature = self.encoder_decoder(cover_image,noise_image)
            g_loss=self.balance_loss(cover_feature,noise_feature)

        cover_feature_round=cover_feature.detach().cpu().numpy()
        noise_feature_round=noise_feature.detach().cpu().numpy()
        cover_zw_round = my_zw(cover_feature_round)
        noise_zw_round = my_zw(noise_feature_round)
        cover_zw_frame = pd.DataFrame(cover_zw_round[0].squeeze())
        noise_zw_frame = pd.DataFrame(noise_zw_round[0].squeeze())
        writer = pd.ExcelWriter('/home/dell/Documents/HLQ/Zerowatermark_2/Result/val_zw.xlsx', engine='xlsxwriter')
        cover_zw_frame.to_excel(writer, sheet_name='ZW', index=False, header=None, startrow=0, startcol=0)
        noise_zw_frame.to_excel(writer, sheet_name='ZW', index=False, header=None, startrow=0,
                                startcol=cover_zw_round.shape[3] + 1)
        writer.save()

        row=cover_zw_round.shape[2]
        col=cover_zw_round.shape[3]
        BER=np.sum(np.abs(cover_zw_round-noise_zw_round))/(batch_size*row*col)

        NCC=F_NCC(cover_zw_round,noise_zw_round)
        NC,NC_max,NC_avg=F_NC(cover_zw_round)

        losses = {
            'loss           ': g_loss.item(),
            'BER            ': BER,
            'NCC            ': NCC,
            'NC_max         ': NC_max,
            'NC_avg         ': NC_avg
        }
        return losses,(cover_zw_round,noise_zw_round)

    def to_stirng(self):
        return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))
