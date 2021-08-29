import torch.nn.functional as F
import torch

###################################################################
# multi scale for image generation
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
    min_depth, index = torch.min(depths, dim=0)
    r_rgb = torch.zeros_like(rgbs[0])
    for i, rgb in enumerate(rgbs):
        index_i = index == i
        r_rgb += index_i.type_as(rgb).to(rgb.device) * rgb
    min_depth = min_depth.view(1, 1, H, W)
    r_rgb = r_rgb.view(1, N, H, W)
    return min_depth, r_rgb


def patch_min_depth_pooling(depths, rgbs, patch_size=3, void_depth=100):
    """scene re-composition based on the patch depth value"""
    if patch_size > 1:
        filters = torch.ones((1, 1, patch_size, patch_size)).type_as(depths)
        padding = torch.nn.ReflectionPad2d(int(patch_size/2))
        pad_depths = padding(depths)
        mask = (pad_depths != void_depth).type_as(pad_depths)
        depth_sum = F.conv2d(pad_depths, filters)
        mask_sum = F.conv2d(mask, filters)
        mask_sum = torch.where(depths == void_depth, torch.ones_like(mask_sum), mask_sum)
        average = depth_sum / mask_sum
        depths = torch.where(depths == void_depth, void_depth * torch.ones_like(average), average)

    return min_depth_pooling(depths, rgbs)

