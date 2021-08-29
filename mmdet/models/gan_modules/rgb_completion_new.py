import logging

from . import network_new
from . import task
from . import base_function_new
from mmdet.models.losses.adverse_loss import *
from mmcv.runner import load_checkpoint

from ..registry import RGB_COMPLETION

@RGB_COMPLETION.register_module
class RGB_COMPLETION(nn.Module):
    """This class implements the scene completion"""
    def __init__(self,
                 label_nc=0,
                 input_nc=4,
                 output_nc=3,
                 output_scale=5,
                 down_sampling=5,
                 dilation=3,
                 base_nc=64,
                 max_nc=256,
                 init_type='xavier',
                 loss_d=dict(
                     type='GANloss', gan_mode='lsgan', loss_weight=1.0),
                 loss_g=dict(
                     loss_g=dict(type='GANloss', gan_model='lsgan', loss_weight=1.0),
                     loss_rec=dict(type='l1loss', loss_weight=10.0),
                     loss_vgg=dict(type='l1loss', loss_weight=1.0),
                     loss_feat=dict(type='l1loss', loss_weight=1.0),
                 ),
                 ):
        super(RGB_COMPLETION, self).__init__()

        self.output_scale = output_scale
        self.input_nc = input_nc

        self.net_E = network_new.define_e(input_nc, base_nc, max_nc, down_sampling, dilation, norm='none',
                                          activation='LeakyReLU', use_spect=True)
        self.net_G = network_new.define_g(output_nc, base_nc, max_nc, down_sampling, norm='instance',
                                          activation='LeakyReLU', output_scale=output_scale, use_spect=True, use_attn=True)
        self.net_D_scene = network_new.define_d(label_nc, output_nc, base_nc, max_nc, down_sampling, norm='none',
                                                activation='LeakyReLU', use_spect=True, use_attn=True, model='resD')
        self.init_type = init_type

        # define the loss function for the completion network
        if loss_d is not None:
            self.gan_mode = loss_d['gan_mode']
            self.loss_d = loss_d
            self.loss_g  = loss_g
            self.GANLoss = GANLoss(self.gan_mode)
            self.VGGLoss = VGGLoss()
            self.smoothl1loss = nn.SmoothL1Loss()
            self.l1loss = nn.L1Loss()
            self.l2loss = nn.MSELoss()
            self.optimizer_D = torch.optim.Adam([{'params': list(filter(lambda p:p.requires_grad,
                            self.net_D_scene.parameters()))}], lr=0.0001, betas=(0.0, 0.999))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            network_new.init_weights(self.net_E, self.init_type)
            network_new.init_weights(self.net_G, self.init_type)
            network_new.init_weights(self.net_D_scene, self.init_type)

    def get_scale_img(self, input, scales):
        scale_input = task.scale_pyramid(input, scales)

        return scale_input

    def get_G_inputs(self, features, mask):
        f_in = features[-1]
        f_e = features[2]
        scale_mask = task.scale_img(mask, [f_e.size(2), f_e.size(3)])

        return f_in, f_e, scale_mask

    def forward(self, x, mask):
        """Run forward processing to get the outputs"""
        # encoder process
        b, n, h, w = x.size()
        if h != 512 or w != 512: # resize the image to fixed size for inpainting
            x = nn.functional.interpolate(x, (512, 512))
            mask = torch.nn.functional.interpolate(mask, (512, 512))
        features = self.net_E(x, mask=mask)
        # decoder process
        f_in, f_e, scale_mask = self.get_G_inputs(features, mask)
        g_img, attn_f = self.net_G(f_in, f_e, scale_mask)
        if h != 512 or w != 512:
            for i in range(len(g_img)):
                g_img[i] = nn.functional.interpolate(g_img[i], (h, w))
        return g_img

    def D_loss_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANLoss(D_real, True, True)
        # fake
        D_fake = netD(fake)
        D_fake_loss = self.GANLoss(D_fake, False, True)
        # loss fot the discriminator
        D_loss = D_real_loss + D_fake_loss
        # gradient penalty for wgan-gp
        if self.gan_mode == 'wgangp':
            gradient_penalty, gradients = cal_gradient_penalty(netD, real, fake)
            D_loss += gradient_penalty

        D_loss = D_loss * self.loss_d['loss_weight']

        return D_loss

    def optimizer_d(self, reals, fakes, labels=None):
        """optimize the discriminator"""
        losses = dict()
        fakes_, reals_ = [], []
        base_function_new._unfreeze(self.net_D_scene)
        for fake, real in zip(fakes, reals):
            if labels is not None and self.input_nc != fake.size(1):
                scale_label = F.interpolate(labels, size=[fake.size(2), fake.size(3)])
                fake = torch.cat([scale_label, fake], dim=1)
                real = torch.cat([scale_label, real], dim=1)
            fakes_.append(fake.detach())
            reals_.append(real)
        self.optimizer_D.zero_grad()
        D_loss = self.D_loss_basic(self.net_D_scene, reals_, fakes_)
        D_loss.backward()
        self.optimizer_D.step()

        losses['loss_mask_d'] = D_loss

        return losses

    def G_loss(self, reals, fakes, scale_masks, labels=None):
        """Calculate training loss for the generator"""
        losses = dict()
        for item in self.loss_g:
            # reconstruction loss
            if 'loss_rec' in item:
                loss_rec = 0
                for i, (fake, real, mask) in enumerate(zip(fakes, reals, scale_masks)):
                    in_weight = self.loss_g[item]['in_weight']
                    if self.loss_g[item]['type'].startswith('smooth'):
                        loss_rec += (in_weight * self.smoothl1loss(fake * (1-mask), real * (1-mask)) +
                                     self.smoothl1loss(fake * mask, real * mask))
                    if self.loss_g[item]['type'].startswith('l1'):
                        loss_rec += (in_weight * self.l1loss(fake * (1-mask), real * (1-mask)) +
                                     self.l1loss(fake * mask, real * mask))
                    if self.loss_g[item]['type'].startswith('l2'):
                        loss_rec += (in_weight * self.l2loss(fake * (1-mask), real * (1-mask)) +
                                     self.l2loss(fake * mask, real * mask))
                losses[item] = loss_rec * self.loss_g[item]['loss_weight']
            # adversarial loss
            if "loss_g" in item:
                base_function_new._freeze(self.net_D_scene)
                fakes_ = []
                for fake in fakes:
                    if labels is not None and self.input_nc != fake.size(1):
                        scale_label = F.interpolate(labels, size=[fake.size(2), fake.size(3)])
                        fake = torch.cat([scale_label, fake], dim=1)
                    fakes_.append(fake)
                losses[item] = self.GANLoss(self.net_D_scene(fakes_), True, False) * self.loss_g[item]['loss_weight']
            # content consistency loss
            if "loss_vgg" in item:
                loss_vgg = 0
                in_weight = self.loss_g[item]['in_weight']
                for i, (fake, real, mask) in enumerate(zip(fakes, reals, scale_masks)):
                    loss_vgg += (in_weight * self.VGGLoss(fake * (1-mask) + real * mask, real) +
                                 self.VGGLoss(fake * mask + real * (1-mask), real))
                losses[item] = loss_vgg * self.loss_g[item]['loss_weight']

        return losses