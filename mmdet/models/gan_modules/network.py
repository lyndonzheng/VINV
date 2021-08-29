import numpy as np
import torchvision
from .base_function import *


##############################################################################################################
# Network function
##############################################################################################################
def define_e(input_nc=3, ngf=64, img_f=512, L=6, layers=5, norm='none', activation='ReLU', use_spect=True):

    net = ResEncoder(input_nc, ngf, img_f, L, layers, norm, activation, use_spect)

    return net


def define_g(output_nc=3, ngf=64, img_f=512, layers=5, norm='instance', activation='ReLU', output_scale=1,
             use_spect=True, use_attn=True):

    net = ResGenerator(output_nc, ngf, img_f, layers, norm, activation, output_scale, use_spect, use_attn)

    return net


def define_d(input_nc=3, ndf=64, img_f=512, layers=6, norm='none', activation='LeakyReLU', output_scale=1,
             use_spect=True, use_attn=True):

    net = ResDiscriminator(input_nc, ndf, img_f, layers, norm, activation, output_scale, use_spect, use_attn)

    # net = NLayerDiscriminator(input_nc, ndf, img_f, layers, norm, activation, output_scale, use_spect, use_attn)

    return net


#############################################################################################################
# Network structure
#############################################################################################################

# class ResEncoder(nn.Module):
#     """
#     ResNet Encoder Network
#     :param input_nc: number of channels in input
#     :param ngf: base filter channel
#     :param img_f: the largest feature channels
#     :param L: Number of refinements of density
#     :param layers: down and up sample layers
#     :param norm: normalization function 'instance, batch, group'
#     :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
#     """
#
#     def __init__(self, input_nc=3, ngf=64, img_f=512, L=6, layers=5, norm='none', activation='ReLU', use_spect=True):
#         super(ResEncoder, self).__init__()
#
#         self.layers = layers
#         self.L = L
#         self.down = nn.AvgPool2d(kernel_size=2, stride=2)
#
#         # encoder part
#         # self.block0 = spectral_norm_func(PartialConv2d(input_nc, ngf, kernel_size=7, stride=2, padding=3, return_mask=True), use_spect)
#         self.block0 = spectral_norm_func(nn.Conv2d(input_nc, ngf, kernel_size=7, stride=2, padding=3), use_spect)
#
#         mult = 1
#         for i in range(layers - 1):
#             mult_prev = mult
#             mult = min(2 ** (i + 1), img_f // ngf)
#             # block = PCResBlock(ngf * mult_prev, ngf * mult, norm, activation, use_spect)
#             block = ResBlock(ngf * mult_prev, ngf * mult, norm, activation, use_spect)
#             setattr(self, 'encoder' + str(i), block)
#
#         # dilation part
#         for i in range(self.L):
#             block = Block(ngf * mult, ngf * mult, 'none', activation, False, use_spect, 2**i)
#             setattr(self, 'infer'+str(i), block)
#
#     def forward(self, img, mask=None):
#         """
#         :param img_m: image with mask regions I_m
#         :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
#         """
#
#         # encoder part
#         # out, mask = self.block0(img, mask)
#         out = self.block0(img)
#         feature = [out]
#         for i in range(self.layers - 1):
#             model = getattr(self, 'encoder' + str(i))
#             # out, mask = model(self.down(out), self.down(mask))
#             out = model(self.down(out))
#             feature.append(out)
#
#         # dilation part
#         for i in range(self.L):
#             infer = getattr(self, 'infer' + str(i))
#             out = infer(out)
#             feature.append(out)
#
#         return feature

class ResEncoder(nn.Module):
    """
    ResNet Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """

    def __init__(self, input_nc=3, ngf=64, img_f=512, L=6, layers=5, norm='none', activation='ReLU', use_spect=True):
        super(ResEncoder, self).__init__()

        self.layers = layers
        self.L = L
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        # encoder part
        # self.block0 = spectral_norm_func(PartialConv2d(input_nc, ngf, kernel_size=7, stride=2, padding=3, return_mask=True), use_spect)
        self.block0 = spectral_norm_func(nn.Conv2d(input_nc, ngf, kernel_size=7, stride=2, padding=3), use_spect)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            # block = PCResBlock(ngf * mult_prev, ngf * mult, norm, activation, use_spect)
            block = ResBlock(ngf * mult_prev, ngf * mult, norm, activation, use_spect)
            setattr(self, 'encoder' + str(i), block)

        # dilation part
        for i in range(self.L):
            block = Block(ngf * mult, ngf * mult, 'none', activation, False, use_spect, 2**i)
            setattr(self, 'infer'+str(i), block)

    def forward(self, img_m, img_c=None, mask=None):
        """
        :param img_m: image with mask regions I_m
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        """

        # encoder part
        # out, mask = self.block0(img, mask)
        if img_c is not None:
            img = torch.cat([img_m, img_c], dim=0)
        else:
            img = img_m
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            # out, mask = model(self.down(out), self.down(mask))
            out = model(self.down(out))
            feature.append(out)

        # during the training, we have two paths, during the testing, we only have one paths
        if img_c is not None:
            f_m, f_c = out.chunk(2)
        else:
            f_m = out

        # dilation part
        f_m_c = f_m
        for i in range(self.L):
            infer = getattr(self, 'infer' + str(i))
            f_m_c = infer(f_m_c)

        return feature, f_m_c


class ResGenerator(nn.Module):
    """
    ResNet Generator Network
    :param output_nc: number of channels in output
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param output_scale: Different output scales
    """

    def __init__(self, output_nc=3, ngf=64, img_f=512, layers=5, norm='batch', activation='ReLU',
                 output_scale=1, use_spect=True, use_attn=True):
        super(ResGenerator, self).__init__()

        self.layers = layers
        self.output_scale = output_scale
        self.use_attn = use_attn
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # latent z to feature
        mult = min(2 ** (layers), img_f // ngf)
        self.generator = Block(ngf * mult, ngf * mult, norm, activation, True, use_spect)

        # decoder part
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers - i - 1), img_f // ngf)
            upconv = ResBlock(ngf * mult_prev, ngf * mult, norm, activation, True, use_spect)
            setattr(self, 'decoder' + str(i), upconv)
            # output part
            if i > layers - output_scale - 1:
                outconv = nn.Conv2d(ngf * mult, output_nc, 3, 1, 1, bias=True)
                setattr(self, 'out' + str(i), outconv)
            # short+long term attention part
            if i == (layers - 3) and use_attn:
                attn = Auto_Attn(ngf * mult, norm, activation, use_spect)
                setattr(self, 'attn' + str(i), attn)

    def forward(self, f_m=None, f_e=None, mask=None):
        """
        ResNet Generator Network
        :param z: latent vector
        :param f_m: feature of valid regions for conditional VAG-GAN
        :param f_e: previous encoder feature for short+long term attention layer
        :return results: different scale generation outputs
        """

        # the features come from mask regions and valid regions, we directly add them togethers
        out = self.generator(f_m)
        results = []
        attn_f = 0
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(self.up(out))
            if i > self.layers - self.output_scale - 1:
                model = getattr(self, 'out' + str(i))
                output = torch.tanh(model(F.leaky_relu(out, 2e-1)))
                results.append(output)
            if i == (self.layers - 3) and self.use_attn:
                # auto attention
                model = getattr(self, 'attn' + str(i))
                out, attn = model(out, f_e, mask)
                attn_f = out

        return results, attn_f


class ResDiscriminator(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """

    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm='none', activation='LeakyReLU', output_scale=1,
                 use_spect=True, use_attn=True, use_minbatch=False):
        super(ResDiscriminator, self).__init__()

        self.layers = layers
        self.use_attn = use_attn
        self.output_scale = output_scale
        self.use_minibatch = use_minbatch
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.nonlinearity = get_nonlinearity_layer(activation_type=activation)

        # encoder part

        mult = 1
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            # self-attention
            if i == 2 and use_attn:
                attn = Auto_Attn(ndf * mult_prev, norm, activation, use_spect)
                setattr(self, 'attn' + str(i), attn)
            block = ResBlock(ndf * mult_prev, ndf * mult, norm, activation, use_spect)
            if i < self.output_scale:
                rgb2f = spectral_norm_func(nn.Conv2d(input_nc, ndf * mult_prev, 1, 1, 0), use_spect)
                setattr(self, 'rgb2f' + str(i), rgb2f)
                if i > 0:
                    block = ResBlock(ndf * mult_prev * 2, ndf * mult, norm, activation, use_spect)
            setattr(self, 'encoder' + str(i), block)

        if use_minbatch:
            self.batch_discriminator = MinibatchStdDev()
            self.conv0 = ResBlock(ndf * mult + 1, ndf * mult, norm, activation, use_spect)
        else:
            self.conv0 = ResBlock(ndf * mult, ndf * mult, norm, activation, use_spect)
        self.conv1 = nn.Conv2d(ndf * mult, 1, 4)

    def forward(self, x):
        for i in range(self.layers):
            if i == 2 and self.use_attn:
                attn = getattr(self, 'attn' + str(i))
                out, attention = attn(out)
            if i < self.output_scale:
                model = getattr(self, 'rgb2f' + str(i))
                rgb2f = model(x[-i - 1])
                if i > 0:
                    out = torch.cat([rgb2f, out], dim=1)
                else:
                    out = rgb2f
            model = getattr(self, 'encoder' + str(i))
            out = model(self.down(out))
        if self.use_minibatch:
            out = self.batch_discriminator(out)
            out = self.conv0(out)
        else:
            out = self.conv0(out)
        out = self.conv1(self.nonlinearity(out))
        return out


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm='none', activation='LeakyReLU', output_scale=1,
                 use_spect=True, use_attn=True, use_minbatch=False):
        super().__init__()

        self.layers = layers
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, self.layers):
            nf_prev = ndf
            ndf = min(ndf * 2, img_f)
            stride = 1 if n == self.layers - 1 else 2
            sequence += [[nn.Conv2d(nf_prev, ndf, kernel_size=kw, stride=stride, padding=padw),
                          nn.InstanceNorm2d(ndf, affine=False),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(ndf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input[-1]]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        return results[-1]


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out