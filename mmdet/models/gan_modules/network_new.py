from .base_function_new import *
import torchvision


##############################################################################################################
# Network function
##############################################################################################################

def define_e(input_nc=3, ngf=64, img_f=512, layers=5, L=6, norm='none', activation='ReLU', use_spect=True):

    net = ResEncoder(input_nc, ngf, img_f, layers, L, norm, activation, use_spect)

    return net


def define_g(output_nc=3, ngf=64, img_f=512, layers=5, norm='instance', activation='ReLU', output_scale=1,
                 use_spect=True, use_attn=True):

    net = ResDecoder(output_nc, ngf, img_f, layers, norm, activation, output_scale, use_spect, use_attn)

    return net


def define_d(label_nc=0, input_nc=3, ndf=64, img_f=512, layers=4, norm='none', activation='LeakyReLu',
                 input_scale=1, use_spect=True, use_attn=False, use_minibatch=False, model='toy'):

    # if model.startswith('patch'):
    #     net = NLayerDiscriminator(label_nc, output_nc, ndf, layers, norm)
    # elif model.startswith('toy'):
    #     net = ToyDiscriminator(label_nc, output_nc, ndf, layers, norm)
    if model.startswith('res'):
        net = ResDiscriminator(label_nc, input_nc, ndf, img_f, layers, norm, activation, input_scale, use_spect,
                               use_attn, use_minibatch)
    else:
        raise NotImplementedError('Discriminator %s is not recognized' % model)

    return net


#############################################################################################################
# Network structure
#############################################################################################################

class ResEncoder(nn.Module):
    """
    ResNet Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param L: Number of dilation operation
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'RelU, SELU, LeakyReLU, PReLU'
    :param use_spect: spectral function
    """
    def __init__(self, input_nc=3, ngf=64, img_f=512, layers=5, L=6, norm='none', activation='ReLU', use_spect=True):
        super(ResEncoder, self).__init__()

        self.layers = layers
        self.L = L
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        # encoder part
        self.block0 = spectral_norm_func(nn.Conv2d(input_nc, ngf, kernel_size=7, stride=2, padding=3), use_spect)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, norm, activation, use_spect=use_spect)
            setattr(self, 'encoder'+str(i), block)

        # dilation part
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, norm, activation, dilation=2**(i+1), use_spect=use_spect, use_res=False)
            setattr(self, 'infer'+str(i), block)

    def forward(self, img, mask=None):
        """
        :param img: image with mask regions I_m
        :param mask:
        :return:
        """
        # encoder part
        out = self.block0(img)
        features = [out]
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(self.down(out))
            features.append(out)

        # dilation part
        for i in range(self.L):
            infer = getattr(self, 'infer' + str(i))
            out = infer(out)
        features.append(out)

        return features


class ResDecoder(nn.Module):
    """
    ResNet Decoder Network
    :param output_nc: number of channels of outputs
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param output_scale: different output sceles
    :param use_spect: spectral function
    :param use_attn: using short-long term patch-attention
    """
    def __init__(self, output_nc=3, ngf=64, img_f=512, layers=5, norm='instance', activation='LeakyReLu', output_scale=1,
                 use_spect=True, use_attn=True):
        super(ResDecoder, self).__init__()

        self.layers = layers
        self.output_scale = output_scale
        self.use_attn = use_attn
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # transform part
        mult = min(2 ** (layers), img_f // ngf)
        self.trans0 = ResBlock(ngf * mult, ngf * mult, norm, activation, bias=True, use_spect=use_spect, use_res=False)

        # decoder part
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers - i - 1), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, norm, activation, bias=True, use_spect=use_spect)
            setattr(self, 'decoder' + str(i), block)
            # output_part
            if i > layers - output_scale - 1:
                outconv = nn.Conv2d(ngf * mult, output_nc, 3, 1, 1, bias=True)
                setattr(self, 'out' + str(i), outconv)
            # short+long term patch attention part
            if i == (layers - 4) and use_attn:
                attn = Auto_Attn(ngf * mult, norm, activation, use_spect=use_spect)
                setattr(self, 'attn' + str(i), attn)

    def forward(self, f_m=None, f_e=None, mask=None):
        """
        :param f_m: feature of valid regions
        :param f_e: previous encoder feature for shot+long term patch attention layer
        :param mask: visible mask
        :return:
        """
        out = self.trans0(f_m)
        results = []
        attn_f = 0
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(self.up(out))
            if i > self.layers - self.output_scale - 1:
                model = getattr(self, 'out' + str(i))
                output = torch.tanh(model(F.leaky_relu(out, 2e-1)))
                results.append(output)
            if i == (self.layers - 4) and self.use_attn:
                model = getattr(self, 'attn' + str(i))
                out, attn = model(out, f_e, mask)
                attn_f = out

        return results, attn_f


class ResDiscriminator(nn.Module):
    """
    ResNet Discriminator Network
    :param label_nc: semantic channels for class specific discriminator
    :param input_nc: number of channles in input
    :param ndf: base fileter channel
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param input_scale: different input sceles
    :param use_spect: spectral function
    :param use_attn: using short-long term patch-attention
    :param use_minibatch: using the minibacth norm layer
    """
    def __init__(self, label_nc=0, input_nc=3, ndf=64, img_f=512, layers=5, norm='none', activation='LeakyReLu',
                 input_scale=1, use_spect=True, use_attn=True, use_minibatch=True):
        super(ResDiscriminator, self).__init__()

        input_nc = label_nc + input_nc
        self.layers = layers
        self.input_scale = input_scale
        self.use_atten = use_attn
        self.use_minibatch = use_minibatch
        self.nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        mult = 1
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            if i == 3 and use_attn:
                attn = Auto_Attn(ndf * mult_prev, norm, activation, use_spect=use_spect)
                setattr(self, 'attn' + str(i), attn)
            block = ResBlock(ndf * mult_prev, ndf * mult, norm, activation, use_spect=use_spect)
            if i < self.input_scale:
                rgb2f = spectral_norm_func(nn.Conv2d(input_nc, ndf * mult_prev, 1, 1, 0), use_spect)
                setattr(self, 'rgb2f' + str(i), rgb2f)
                if i > 0:
                    block = ResBlock(ndf * mult_prev * 2, ndf * mult, norm, activation, use_spect=use_spect)
            setattr(self, 'encoder' + str(i), block)

        if use_minibatch:
            self.batch_discriminator = MinibatchStdDev()
            self.conv0 = ResBlock(ndf * mult + 1, ndf * mult, norm, activation, use_spect=use_spect)
        else:
            self.conv0 = ResBlock(ndf * mult, ndf * mult, norm, activation, use_spect=use_spect)
        self.conv1 = nn.Conv2d(ndf * mult, 1, 4)

    def forward(self, x):
        for i in range(self.layers):
            if i == 3 and self.use_atten:
                model = getattr(self, 'attn' + str(i))
                out, attn = model(out)
            if i < self.input_scale:
                model = getattr(self, 'rgb2f' + str(i))
                rgb2f = model(x[-i-1])
                if i > 0:
                    out = torch.cat([rgb2f, out], dim=1)
                else:
                    out = rgb2f
            model = getattr(self, 'encoder' + str(i))
            out = model(self.down(out))
        if self.use_minibatch:
            out = self.batch_discriminator(out)
        out = self.conv0(out)
        out = self.conv1(self.nonlinearity(out))

        return out


# Define the Original GAN structure
class ToyDiscriminator(BaseNetwork):
    def __init__(self, label_nc=35, output_nc=3, ndf=64, layers=4, norm='none'):
        super(ToyDiscriminator, self).__init__()

        input_nc = label_nc + output_nc
        nf = ndf

        sequence = [
            nn.Conv2d(input_nc, nf, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, False)
        ]

        for n in range(1, 3):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence +=[
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2),
                nn.LeakyReLU(0.2, False)
            ]

        self.conv = nn.Sequential(*sequence)
        self.linear = nn.Linear(nf, 1)

    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(-1, out.size(1)*out.size(2)*out.size(3))
        out = self.linear(out)

        return out

# Defines the PatchGAN discriminator with the specified arguments
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, label_nc=35, output_nc=3, ndf=64, layers=4, norm='spectralinstance'):
        super(NLayerDiscriminator).__init__()

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        input_nc = label_nc + output_nc
        nf = ndf

        norm_layer = get_nonspade_norm_layer(norm)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == layers - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for model in self.children():
            output = model(results[-1])
            results.append(output)

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
        if X.size(2) > 224:
            X = torch.nn.functional.interpolate(X, [224, 224], mode='bilinear', align_corners=True)
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out