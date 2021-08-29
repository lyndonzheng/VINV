import logging

from . import network
from . import task
from . import base_function
from mmdet.models.losses.adverse_loss import *
from mmcv.runner import load_checkpoint

from ..registry import RGB_COMPLETION


@RGB_COMPLETION.register_module
class RGB_COMPLETION(nn.Module):
    """This Class implements the object completion"""
    def __init__(self,
                 input_nc=4,
                 output_nc=4,
                 output_scale=5,
                 down_sampling=5,
                 dilation=3,
                 base_nc=16,
                 max_nc=256,
                 init_type='xavier',
                 loss_d=dict(
                     type='GANloss', gan_mode='lsgan', loss_weight=1.0
                 ),
                 loss_g= dict(
                     loss_g=dict(type='GANloss', gan_model='lsgan', loss_weight=1.0),
                     loss_rec=dict(type='l1loss', loss_weight=10.0),
                     loss_vgg=dict(type='l1loss', loss_weight=1.0),
                     loss_feat=dict(type='l1loss', loss_weight=1.0),
                 )
                 ):
        super(RGB_COMPLETION, self).__init__()

        self.output_scale = output_scale
        self.input_nc = input_nc

        self.net_E = network.define_e(input_nc=input_nc, ngf=base_nc, img_f=max_nc, L=dilation, norm='none',
                                        layers=down_sampling, activation='LeakyReLU')
        self.net_G = network.define_g(output_nc=output_nc, ngf=int(base_nc/2), img_f=max_nc, layers=down_sampling,
                                         norm='instance', activation='LeakyReLU', output_scale=output_scale)
        self.net_D_sce = network.define_d(input_nc=input_nc, ndf=base_nc, img_f=max_nc, norm='none',
                                          layers=down_sampling, output_scale=2, use_attn=True)
        self.init_type=init_type

        # define the loss function for the completion network
        if loss_d is not None:
            self.gan_mode = loss_d['gan_mode']
            self.loss_d_weight = loss_d['loss_weight']
            self.GANLoss = GANLoss(self.gan_mode)
            self.VGGLoss = VGGLoss()
            self.l1loss = nn.L1Loss()
            self.l2loss = nn.MSELoss()
            self.loss_g = loss_g
            self.optimizer_D = torch.optim.Adam([#{'params': list(filter(lambda p: p.requires_grad, self.net_D.parameters())), 'lr': opt.lr * 0},
                 {'params': list(filter(lambda p: p.requires_grad, self.net_D_sce.parameters()))}], lr=0.0001, betas=(0.0, 0.999))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            network.init_weights(self.net_E, self.init_type)
            network.init_weights(self.net_G, self.init_type)
            network.init_weights(self.net_D_sce, self.init_type)

    def get_scale_img(self, input, scales):
        scale_input = task.scale_pyramid(input, scales)

        return scale_input

    # def get_G_inputs(self, feature, mask):
    #     """Process the encoder feature and mask for generation network"""
    #     f_in, f_e = feature[-1], feature[1]
    #     scale_mask = network.scale_img(mask, [f_e.size(2), f_e.size(3)])
    #     return f_in, f_e, scale_mask
    #
    # def forward(self, x, mask, x_c=None):
    #     # encoder process
    #     if self.input_nc != x.size(1):
    #         feature = self.net_E(torch.cat([x, mask], dim=1))
    #     else:
    #         feature = self.net_E(x, mask=mask)
    #     f_in, f_e, scale_mask = self.get_G_inputs(feature, mask)
    #     g_img, attn_f = self.net_G(f_in, f_e, scale_mask)
    #
    #     return g_img, attn_f

    def get_G_inputs(self, feature, f_m_c, mask):
        if feature[-1].size(0) != f_m_c.size(0):
            f_m, f_c = feature[-1].chunk(2)
            self.f_c = f_c
            self.f_m_c = f_m_c
            f_in = torch.cat([f_m_c, f_c], dim=0)
            f_e = torch.cat([feature[1].chunk(2)[0], feature[1].chunk(2)[0]], dim=0)
            mask = torch.cat([mask, mask], dim=0)
        else:
            self.f_c = None
            self.f_m_c = None
            f_in = f_m_c
            f_e = feature[1]
        scale_mask = network.scale_img(mask, [f_e.size(2), f_e.size(3)])

        return f_in, f_e, scale_mask

    # def forward(self, x, mask, x_c=None):
    #     """Run forward procesing to get the outputs"""
    #     # encoder process
    #     if self.input_nc!=x.size(1):
    #         if x_c is not None:
    #             x_c = torch.cat([x_c, torch.ones_like(mask)], dim=1)
    #         feature, f_m_c = self.net_E(torch.cat([x, mask], dim=1), img_c=x_c)
    #     else:
    #         feature, f_m_c = self.net_E(x, img_c=x_c, mask=mask)
    #     f_in, f_e, scale_mask = self.get_G_inputs(feature, f_m_c, mask)
    #     g_img, attn_f = self.net_G(f_in, f_e, scale_mask)
    #
    #     return g_img

    def forward(self, x, mask):
        """Run forward processing to get the outputs"""
        # encoder process
        b, n, h, w = x.size()
        if h != 256 or w != 256:  # resize the image to fixed size for inpainting
            x = nn.functional.interpolate(x, (256, 256))
            mask = torch.nn.functional.interpolate(mask, (256, 256))
        features, f_m_c = self.net_E(x, mask=mask)
        # decoder process
        f_in, f_e, scale_mask = self.get_G_inputs(features, f_m_c, mask)
        g_img, attn_f = self.net_G(f_in, f_e, scale_mask)
        if h != 256 or w != 256:
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
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.gan_mode == 'wgangp':
            gradient_penalty, gradients = base_function.cal_gradient_penalty(netD, real, fake)
            D_loss +=gradient_penalty

        D_loss = D_loss * self.loss_d_weight
        D_loss.backward()

        return D_loss

    def D_loss(self, reals, fakes, labels=None):
        """calculate the GAN loss for the discriminators"""
        fake_label, real_label = [], []
        base_function._unfreeze(self.net_D_sce)
        for fake, real in zip(fakes, reals):
            if labels is not None and self.input_nc!=fake.size(1):
                scale_label = F.interpolate(labels, size=[fake.size(2), fake.size(3)])
                fake = torch.cat([fake, scale_label], dim=1)
                real = torch.cat([real, scale_label], dim=1)
            fake_label.append(fake.detach())
            real_label.append(real)
        self.optimizer_D.zero_grad()
        loss_d_sce = self.D_loss_basic(self.net_D_sce, real_label, fake_label)
        self.optimizer_D.step()

        return loss_d_sce

    def G_loss(self, reals, fakes, scale_masks, labels=None):
        """Calculate training loss for the generator"""
        loss = dict()
        for item in self.loss_g:
            # generator adversarial loss
            if 'loss_g' in item:
                base_function._freeze(self.net_D_sce)
                fake_label = []
                for fake in fakes:
                    if labels is not None and self.input_nc != fake.size(1):
                        scale_label = F.interpolate(labels, size=[fake.size(2), fake.size(3)])
                        fake = torch.cat([fake, scale_label], dim=1)
                    fake_label.append(fake)
                loss[item] = self.GANLoss(self.net_D_sce(fake_label), True, False) * self.loss_g[item]['loss_weight']
                # loss[item] = self.l2loss(self.net_D_sce(fake), self.net_D_sce(real)) * self.loss_g[item]['loss_weight']
            # reconstruction loss
            if 'loss_rec' in item:
                loss_rec = 0
                for i, (g_img, real_img, mask) in enumerate(zip(fakes, reals, scale_masks)):
                    # g_img = torch.where(real_img == -1, -1 * torch.ones_like(real_img), g_img)
                    if 'l1' in self.loss_g[item]['type']:
                        loss_rec += (self.l1loss(g_img * mask, real_img * mask) +
                                     6 * self.l1loss(g_img * (1-mask), real_img * (1-mask))) * self.loss_g[item]['loss_weight']
                    if 'l2' in self.loss_g[item]['type']:
                        loss_rec += (self.l2loss(g_img * mask, real_img * mask) +
                                     6 * self.l2loss(g_img * (1-mask), real_img * (1-mask))) * self.loss_g[item]['loss_weight']
                loss[item] = loss_rec
            # content consistency loss
            if 'loss_vgg' in item:
                loss_vgg = 0
                for i, (g_img, real_img, mask) in enumerate(zip(fakes, reals, scale_masks)):
                    # g_img = torch.where(real_img == -1, -1 * torch.ones_like(real_img), g_img)
                    loss_vgg += (self.VGGLoss(g_img * mask + real_img * (1-mask), real_img) +
                                6 * self.VGGLoss(g_img * (1-mask) + real_img * mask, real_img)) * self.loss_g[item]['loss_weight']
                loss[item] = loss_vgg
            if 'loss_feat' in item and self.f_c is not None:
                if 'l1' in self.loss_g[item]['type']:
                    loss[item] = self.l1loss(self.f_m_c, self.f_c) * self.loss_g[item]['loss_weight']
                if 'l2' in self.loss_g[item]['type']:
                    loss[item] = self.l2loss(self.f_m_c, self.f_c) * self.loss_g[item]['loss_weight']
        return loss