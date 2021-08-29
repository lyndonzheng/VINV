import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmdet.core import auto_fp16, force_fp32, rgba_target
from mmdet.models.gan_modules.base_function_new import spectral_norm_func, BaseNetwork
from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module
class RGBAHead(BaseNetwork):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 out_channels=None,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_rgba=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
                 ):
        super(RGBAHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError('Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_rgba = build_loss(loss_rgba)

        for i in range(self.num_convs):
            in_channels = (self.in_channels if i == 0 else self.conv_out_channels)
            dilation = int(2**(i+1))
            block = nn.Sequential(
                nn.ReflectionPad2d(dilation),
                nn.InstanceNorm2d(in_channels, affine=True),
                nn.LeakyReLU(0.2),
                spectral_norm_func(nn.Conv2d(in_channels, self.conv_out_channels,
                                             kernel_size=3, dilation=dilation), use_spect=False)
            )
            setattr(self, 'Block'+str(i), block)

        # upsample layers
        self.layer = 0
        while upsample_ratio // 2 > 0:
            upsample_ratio = upsample_ratio // 2
            input_nc = self.conv_out_channels
            output_nc = int(self.conv_out_channels / 2)
            if self.upsample_method == 'deconv':
                up = nn.Sequential(
                    nn.InstanceNorm2d(input_nc, affine=True),
                    nn.LeakyReLU(0.2),
                    spectral_norm_func(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=self.conv_kernel_size,
                                                          stride=2, padding=1, output_padding=1), use_spect=False)
                )
            else:
                self.upsample = nn.Upsample(scale_factor=2, mode=self.upsample_method)
                up = nn.Sequential(
                    nn.ReflectionPad2d(int(self.conv_kernel_size / 2)),
                    nn.InstanceNorm2d(input_nc, affine=True),
                    nn.LeakyReLU(0.2),
                    spectral_norm_func(nn.Conv2d(input_nc, output_nc, kernel_size=self.conv_kernel_size,),
                                       use_spect=False)
                )
            setattr(self, 'up'+str(self.layer), up)
            self.layer += 1
            self.conv_out_channels = output_nc

        if out_channels is None:
            out_channels = 1 if self.class_agnostic else self.num_classes
        self.out = nn.Conv2d(self.conv_out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    @auto_fp16()
    def forward(self, x):
        # transform
        for i in range(self.num_convs):
            block = getattr(self, 'Block'+str(i))
            x = block(x)
        # upsample
        if self.layer > 0:
            for i in range(self.layer):
                if self.upsample_method in ['nearest', 'bilinear']:
                    x = self.upsample(x)
                upconv = getattr(self, 'up' + str(i))
                x = upconv(x)
        # output
        rgba_pred = self.out(nn.functional.leaky_relu(x, 2e-1))
        return rgba_pred

    def get_target(self, sampling_results, gt_masks, gt_rgbs, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        rgba_targets = rgba_target(pos_proposals, pos_assigned_gt_inds, gt_masks, gt_rgbs, rcnn_train_cfg)

        return rgba_targets

    def get_layered_target(self, gt_f_masks, gt_rgbs, l_orders):

        layered_target_wbg = []
        layered_target_wobg = []
        for (gt_f_mask, gt_rgb, l_order) in zip(gt_f_masks, gt_rgbs, l_orders):
            gt_rgb_bg = gt_rgb[-1].type(torch.float)
            gt_rgb_wobg = gt_rgb[:-1].type(torch.float)
            for i in range(l_order.max() + 1):
                inds = l_order == i
                if inds.any():
                    f_mask_ = gt_f_mask[inds]
                    rgb_ = gt_rgb_wobg[inds]
                    layered_f_mask = f_mask_.sum(dim=0).unsqueeze(0)
                    layered_rgb = rgb_.sum(dim=0)
                    layered_rgb_bg = torch.where(layered_f_mask > 0, layered_rgb, gt_rgb_bg)
                    layered_target_wobg.append(layered_rgb.unsqueeze(0))
                    layered_target_wbg.append(layered_rgb_bg.unsqueeze(0))
            layered_target_wobg.append(gt_rgb_bg.unsqueeze(0))

        layered_target_wbg = torch.cat(layered_target_wbg)
        layered_target_wobg = torch.cat(layered_target_wobg)

        return layered_target_wobg, layered_target_wbg

    @force_fp32(apply_to=('rgba_pred', ))
    def loss(self, rgba_pred, rgba_targets, rgba_weights, mean=0.0, std=1.0):
        losses = dict()
        rgba_targets = (rgba_targets - mean) / std
        b, c, w, h = rgba_pred.size()
        rgba_targets = torch.nn.functional.interpolate(rgba_targets, [w, h], mode='bilinear')
        losses['loss_rgba'] = self.loss_rgba(
            rgba_pred,
            rgba_targets,
            rgba_weights,
            avg_factor=None
        )

        return losses

    def get_seg_masks(self, rgba_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """
        Get segmentation masks from mask pred and bboxes
        :param rgba_pred: shape (n, 4, h, w)
        :param det_bboxes: shape (n, 4/5)
        :param det_labels: shape (n, )
        :param rcnn_test_cfg: rcnn testing config
        :param ori_shape: original image size
        :param scale_factor:
        :param rescale:
        :return: list: encoded masks
        """
        if isinstance(rgba_pred, torch.Tensor):
            rgba_pred = rgba_pred.tanh().cpu().numpy()
        assert  isinstance(rgba_pred, np.ndarray)
        # when enabling mixed precision training, rgba pred may be float16 numpy array
        rgba_pred = rgba_pred.astype(np.float32)
        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            if not isinstance(scale_factor, (float, np.ndarray)):
                scale_factor = scale_factor.cpu().numpy()
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if self.out_channels is None and not self.class_agnostic:
                mask_pred = rgba_pred[i, label, :, :]
            else:
                mask_pred = rgba_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox_mask = mmcv.imresize(mask_pred, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)

        return cls_segms