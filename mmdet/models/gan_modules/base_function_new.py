import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils import spectral_norm
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import functools

######################################################################################
# base function for network structure
######################################################################################

def init_weights(net, init_type='normal', gain=0.02):
    """Get different initial method for the network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_norm_layer(norm_type='batch'):
    """Get the normalization layer for the networks"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, momentum=0.1, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'pixel':
        norm_layer = functools.partial(PixelwiseNorm)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.2)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f M' % (num_params/1e6))


def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True


def spectral_norm_func(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return spectral_norm(module)
    else:
        return module

######################################################################################
# Pixelwise feature vector normalization.
# reference:
# https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
######################################################################################


class PixelwiseNorm(nn.Module):
    def __init__(self, input_nc):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super().__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y


class ResBlock(nn.Module):
    """Define a Residual Block"""
    def __init__(self, input_nc, output_nc, norm_type='none', activation='ReLU', dilation=1, bias=False,
                 use_spect=True, use_res=True):
        super(ResBlock, self).__init__()

        self.norm = get_norm_layer(norm_type)
        self.actvn = get_nonlinearity_layer(activation)
        self.pad = nn.ReflectionPad2d(dilation)
        self.res = use_res
        self.learned_short = (input_nc != output_nc)
        hidden_nc = min(input_nc, output_nc)

        # creat conv layers
        self.conv_0 = spectral_norm_func(nn.Conv2d(input_nc, hidden_nc, kernel_size=3,
                                                   dilation=dilation, bias=bias), use_spect)
        self.conv_1 = spectral_norm_func(nn.Conv2d(hidden_nc, output_nc, kernel_size=3,
                                                   dilation=dilation, bias=bias), use_spect)

        if self.res and self.learned_short:
            self.conv_s = spectral_norm_func(nn.Conv2d(input_nc, output_nc, kernel_size=1), use_spect)

        # get normalization layers
        if self.norm is not None:
            self.norm_0 = self.norm(input_nc)
            self.norm_1 = self.norm(hidden_nc)

    def forward(self, x):
        x_0 = self.conv_0(self.actvn(self.norm_0(self.pad(x)) if self.norm is not None else self.pad(x)))
        x_1 = self.conv_1(self.actvn(self.norm_1(self.pad(x_0)) if self.norm is not None else self.pad(x_0)))

        if self.res:
            x_s = self.short_cut(x)
            return x_1 + x_s
        else:
            return x_1

    def short_cut(self, x):
        if self.learned_short:
            return self.conv_s(x)
        else:
            return x


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        """print the network"""
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1e6 ))

    def init_weights(self, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class Auto_Attn(nn.Module):
    """ Short+Long attention Layer"""

    def __init__(self, input_nc, norm_type='none', activation='ReLU', use_spect=True):
        super(Auto_Attn, self).__init__()
        self.input_nc = input_nc

        self.query_conv = nn.Conv2d(input_nc, input_nc // 8, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

        self.model = ResBlock(input_nc * 2, input_nc, 'none', activation, use_spect)

    def forward(self, x, pre=None, mask=None):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        kernel = 3
        B, C, W, H = x.size()
        proj_query = self.query_conv(x)
        proj_query_value = proj_query.pow(2.0).sum(dim=1, keepdim=True).sqrt()
        proj_query = proj_query / torch.max(proj_query_value, 1e-5*torch.ones_like(proj_query_value)) # B X C X W X H
        proj_key = self.extract_patches(proj_query, kernel)  # B X C X W X H X K X K
        proj_key = proj_key.contiguous().view(B, -1, W*H, kernel, kernel).permute(0, 2, 1, 3, 4) # B X N(WH) X C X K X K

        energy = self.calu_attention(proj_query, proj_key)
        attention = F.softmax(10 * energy, dim=1)
        proj_value = x.view(B, -1, W * H)  # B X C X N

        out = torch.bmm(proj_value, attention)
        out = out.view(B, C, W, H)

        out = self.gamma * out + x

        if pre is not None:
            mask_n = mask.view(B, -1, W * H).permute(0, 2, 1).expand_as(energy)
            mask_n = torch.where(mask_n == 1, mask_n, torch.zeros_like(energy))
            energy = energy * mask_n
            energy_conf, _ = energy.max(dim=1, keepdim=True)
            attention = F.softmax(10 * energy, dim=1) * mask_n
            context_flow = torch.bmm(pre.view(B, -1, W * H), attention).view(B, -1, W, H)
            energy_conf = energy_conf.detach().view(B, -1, H, W)/(kernel*kernel)
            # threshold = self.alpha * energy_conf
            threshold = torch.where(energy_conf > 0.5, torch.ones_like(energy_conf), torch.zeros_like(energy_conf))
            out = self.model(torch.cat([(1 - threshold) * out, threshold * context_flow], dim=1))
            # out = self.model(torch.cat([out, threshold * context_flow], dim=1))

        return out, attention.permute(0, 2, 1)

    def calu_attention(self, proj_querys, proj_keys):

        energys = []
        pad = int(proj_keys.size(3)/2)
        for proj_query, proj_key in zip(proj_querys, proj_keys):
            proj_query = proj_query.unsqueeze(0) # 1 X C X W X H
            proj_query = F.pad(proj_query, [pad, pad, pad, pad], mode='reflect')
            energy = F.conv2d(proj_query, proj_key, stride=1, padding=0) # 1 X N X W X H
            energys.append(energy.view(energy.size(1), energy.size(1)))
        energys = torch.stack(energys, dim=0)
        return energys

    def extract_patches(self, x, kernel=3, stride=1):
        pad = int(kernel/2)
        x = F.pad(x, [pad, pad, pad, pad], mode='reflect')
        all_patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)

        return all_patches