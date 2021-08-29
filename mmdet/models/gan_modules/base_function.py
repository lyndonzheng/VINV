import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils import spectral_norm
from torch.optim import lr_scheduler
import torch.nn.functional as F
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


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler"""
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.iter_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


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


###################################################################
# multi scale and min depth pooling layer for the scene recomposition
###################################################################


def scale_img(img, size):
    h_ratio = img.size(-1) // size[-1]
    w_ratio = img.size(-2) // size[-2]
    scaled_img = F.avg_pool2d(img, kernel_size=(w_ratio, h_ratio), stride=(w_ratio, h_ratio))
    return scaled_img


def scale_pyramid(img, num_scales):
    scaled_imgs = [img]

    for i in range(1, num_scales):
        ratio = 2**i
        scaled_img = F.avg_pool2d(img, kernel_size=ratio, stride=ratio)
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse()
    return scaled_imgs


def min_depth_pooling(depths, rgbs):
    """scene re-composition based on the depth value"""
    B, N, H, W = rgbs.size()
    # depths = torch.where(depths == -1, torch.ones_like(depths), depths)
    min_depth, index = torch.min(depths, dim=0)
    r_rgb = torch.zeros_like(rgbs[0])
    for i, rgb in enumerate(rgbs):
        index_i = index == i
        r_rgb += index_i.type_as(rgb) * rgb
    # min_depth = torch.where(min_depth == 1, -1 * min_depth, min_depth)
    min_depth = min_depth.view(1, 1, H, W)
    r_rgb = r_rgb.view(1, N, H, W)
    return min_depth, r_rgb


def patch_min_depth_pooling(depths, rgbs, patch_size=3):
    """scene re-composition based on the patch depth value"""
    if patch_size > 1:
        filters = torch.ones((1, 1, patch_size, patch_size)).type_as(depths)
        padding = torch.nn.ReflectionPad2d(int(patch_size/2))
        pad_depths = padding(depths)
        mask = (pad_depths > -1).type_as(pad_depths)
        depth_sum = F.conv2d(pad_depths, filters)
        mask_sum = F.conv2d(mask, filters)
        mask_sum = torch.where(depths == -1, torch.ones_like(mask_sum), mask_sum)
        average = torch.clamp(depth_sum / mask_sum, -1, 1)
        depths = torch.where(depths == -1, -1 * torch.ones_like(average), average)

    return min_depth_pooling(depths, rgbs)


###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


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


######################################################################################
# Network basic function
######################################################################################
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


class PCResBlock(nn.Module):
    """Define a Residual Block"""
    def __init__(self, input_nc, output_nc, norm_type='none', activation='ReLU', use_spect=True):
        super(PCResBlock, self).__init__()

        self.actvn = get_nonlinearity_layer(activation)
        self.learned_short = (input_nc!=output_nc)
        hidden_nc = min(input_nc, output_nc)

        # creat conv layers
        self.conv_0 = spectral_norm_func(PartialConv2d(input_nc, hidden_nc, kernel_size=3, padding=1, return_mask=True), use_spect)
        self.conv_1 = spectral_norm_func(PartialConv2d(hidden_nc, output_nc, kernel_size=3, padding=1, return_mask=True), use_spect)
        if self.learned_short:
            self.conv_s = spectral_norm_func(PartialConv2d(input_nc, output_nc, kernel_size=1, bias=False, return_mask=True))

        # get normalization layers
        self.norm = get_norm_layer(norm_type)
        if type(self.norm) != type(None):
            self.norm_0 = self.norm(input_nc)
            self.norm_1 = self.norm(hidden_nc)

    def forward(self, x, mask_in=None):
        x_s, mask_s = self.short_cut(x, mask_in)

        x_0, mask = self.conv_0(self.actvn(self.norm_0(x) if type(self.norm) != type(None) else x), mask_in)
        x_1, mask = self.conv_1(self.actvn(self.norm_1(x_0) if type(self.norm) != type(None) else x_0), mask)

        out = x_1 + x_s
        mask = (mask + mask_s) * 0.5

        return out, mask

    def short_cut(self, x, mask_in=None):
        if self.learned_short:
            return self.conv_s(x, mask_in)
        else:
            return x, mask_in


class Block(nn.Module):
    """Define a Block without short cut"""
    def __init__(self, input_nc, output_nc, norm_type='none', activation='ReLU', bias=False, use_spect=True, dilation=1):
        super(Block, self).__init__()

        self.actvn = get_nonlinearity_layer(activation)
        hidden_nc = min(input_nc, output_nc)

        # creat conv layers
        self.conv_0 = spectral_norm_func(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, padding=dilation,
                                                   dilation=dilation, bias=bias), use_spect)
        self.conv_1 = spectral_norm_func(nn.Conv2d(hidden_nc, output_nc, kernel_size=3, padding=dilation,
                                                   dilation=dilation, bias=bias), use_spect)

        # get normalization layers
        self.norm = get_norm_layer(norm_type)
        if type(self.norm) != type(None):
            self.norm_0 = self.norm(input_nc)
            self.norm_1 = self.norm(hidden_nc)

    def forward(self, x):

        x_0 = self.conv_0(self.actvn(self.norm_0(x) if type(self.norm) != type(None) else x))
        x_1 = self.conv_1(self.actvn(self.norm_1(x_0) if type(self.norm) != type(None) else x_0))

        out = x_1

        return out


class ResBlock(nn.Module):
    """Define a Residual Block"""
    def __init__(self, input_nc, output_nc, norm_type='none', activation='ReLU', bias=False, use_spect=True, dilation=1):
        super(ResBlock, self).__init__()

        self.actvn = get_nonlinearity_layer(activation)
        self.learned_short = (input_nc!=output_nc)
        hidden_nc = min(input_nc, output_nc)

        # creat conv layers
        self.conv_0 = spectral_norm_func(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, padding=dilation,
                                                   dilation=dilation, bias=bias), use_spect)
        self.conv_1 = spectral_norm_func(nn.Conv2d(hidden_nc, output_nc, kernel_size=3, padding=dilation,
                                                   dilation=dilation, bias=bias), use_spect)
        if self.learned_short:
            self.conv_s = spectral_norm_func(nn.Conv2d(input_nc, output_nc, kernel_size=1, bias=False))

        # get normalization layers
        self.norm = get_norm_layer(norm_type)
        if type(self.norm) != type(None):
            self.norm_0 = self.norm(input_nc)
            self.norm_1 = self.norm(hidden_nc)

    def forward(self, x):
        x_s = self.short_cut(x)

        x_0 = self.conv_0(self.actvn(self.norm_0(x) if type(self.norm) != type(None) else x))
        x_1 = self.conv_1(self.actvn(self.norm_1(x_0) if type(self.norm) != type(None) else x_0))

        out = x_1 + x_s

        return out

    def short_cut(self, x):
        if self.learned_short:
            return self.conv_s(x)
        else:
            return x

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
            # attention = torch.where(energy == energy_conf, torch.ones_like(energy), torch.zeros_like(energy))
            # attention = attention / attention.sum(dim=1, keepdim=True)
            # pre = self.extract_patches(pre, kernel).contiguous().view(B, -1, W*H, kernel, kernel)
            # pad = int(kernel / 2)
            # context_flow = []
            # for atten, f in zip(attention, pre):
            #     atten = atten.view(1, W*H, W, H)
            #     atten = F.pad(atten, [pad, pad, pad, pad], mode='reflect')
            #     flow = F.conv2d(atten, f, stride=1, padding=0) / (kernel*kernel) # 1 X C X W X H
            #     context_flow.append(flow.view(-1, W, H))
            # context_flow = torch.stack(context_flow, dim=0)
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