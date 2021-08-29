import logging

from . import network_new, base_function_new
from mmdet.models.losses.adverse_loss import *
from mmcv.runner import load_checkpoint

from ..registry import MASK_GAN


@MASK_GAN.register_module
class MASK_GAN(nn.Module):
    """This class implements the discriminator model for purely mask gan"""
    def __init__(self, label_nc=35,
                 output_nc=3,
                 ndf=64,
                 layers=4,
                 norm='spectral',
                 model='toy',
                 init_type='xavier',
                 init_variance=0.02,
                 loss_d=dict(type='GANloss', gan_mode='lsgan', loss_weight=1.0),
                 loss_g=dict(type='GANloss', gan_mode='lsgan', loss_weight=0.1),
                 ):
        super(MASK_GAN, self).__init__()
        self.init_type = init_type
        self.init_variance = init_variance

        self.net_D = network_new.define_d(label_nc, output_nc, ndf, layers, norm, model)


        if loss_d is not None:
            self.gan_mode = loss_d['gan_mode']
            self.loss_d_opt = loss_d
            self.loss_g_opt = loss_g
            self.GANLoss = GANLoss(self.gan_mode)
            self.optimizer_D = torch.optim.Adam([{'params': list(filter(lambda p: p.requires_grad,
                                            self.net_D.parameters()))}], lr=0.0001, betas=(0.0, 0.999))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            self.net_D.init_weights(self.init_type, self.init_variance)

    def divide_pred(self, pred):
        """take the prediction of fake and real images from the combined batch"""
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def d_loss_basic(self, netD, real, fake):
        """caculate GAN loss for the discriminator"""
        fake_and_real = torch.cat([fake.detach(), real], dim=0)
        discriminator_out = self.net_D(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        d_real_loss = self.GANLoss(pred_real, True, True, True)
        d_fake_loss = self.GANLoss(pred_fake, False, True, True)
        # loss for the discriminator
        D_loss = d_real_loss + d_fake_loss
        # gradient penalty for wgan-gp
        if self.gan_mode == 'wgangp':
            gradient_penalty, gradients = cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty

        D_loss = D_loss * self.loss_d_opt['loss_weight']

        return D_loss

    def optimizer_d(self, real, fake):
        """optimize the discriminator"""
        loss = dict()
        base_function_new._unfreeze(self.net_D)
        self.optimizer_D.zero_grad()
        D_loss = self.d_loss_basic(self.net_D, real, fake)
        D_loss.backward()
        self.optimizer_D.step()

        loss['loss_mask_d'] = D_loss

        return loss

    def g_loss(self, real, fake):
        """calculate training loss for the generator"""
        loss = dict()
        # generator adversarial loss
        base_function_new._freeze(self.net_D)
        pred_fake = self.net_D(fake)
        G_loss = self.GANLoss(pred_fake, True, False, True) * self.loss_g_opt['loss_weight']
        loss['loss_mask_g'] = G_loss

        return loss