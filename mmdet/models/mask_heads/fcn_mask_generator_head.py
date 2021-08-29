import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmdet.core import auto_fp16, force_fp32, mask_target
from mmdet.models.gan_modules.base_function_new import ResBlock, spectral_norm_func, BaseNetwork
from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module
class GeneratorMaskHead(BaseNetwork):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(GeneratorMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)

        for i in range(self.num_convs):
            dilation = min(2**i, roi_feat_size // 2)
            in_channels = (self.in_channels if i ==0 else self.conv_out_channels)
            block = nn.Sequential(
                nn.ReflectionPad2d(dilation),
                #nn.InstanceNorm2d(in_channels),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels, self.conv_out_channels, kernel_size=self.conv_kernel_size,
                                             dilation=dilation)
            )
            setattr(self, 'Block'+str(i), block)

        # upsample layers
        self.layer = 0
        while upsample_ratio // 2 > 0:
            self.layer += 1
            upsample_ratio = upsample_ratio // 2

        if self.upsample_method == 'deconv':
            for i in range(self.layer):
                input_nc = self.conv_out_channels
                output_nc = int(self.conv_out_channels / 2)
                self.conv_out_channels = output_nc
                deconv = nn.Sequential(
                    #nn.InstanceNorm2d(input_nc),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(input_nc, output_nc, self.conv_kernel_size, stride=2,
                                                          padding=1, output_padding=1)
                )
                setattr(self, 'up'+str(i), deconv)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode=self.upsample_method)
            for i in range(self.layer):
                input_nc = self.conv_out_channels
                output_nc = int(self.conv_out_channels / 2)
                self.conv_out_channels = output_nc
                block = nn.Sequential(
                    nn.ReflectionPad2d(int(self.conv_kernel_size/2)),
                    #nn.InstanceNorm2d(input_nc),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(input_nc, output_nc, kernel_size=self.conv_kernel_size)
                )
                setattr(self, 'up'+str(i), block)

        out_channels = 1 if self.class_agnostic else self.num_classes
        self.out = nn.Conv2d(self.conv_out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    @auto_fp16()
    def forward(self, x):
        # transform
        for i in range(self.num_convs):
            block = getattr(self, 'Block'+str(i))
            x = block(x)
        # upsample
        for i in range(self.layer):
            if self.upsample_method in ['nearest', 'bilinear']:
                x = self.upsample(x)
            upconv = getattr(self, 'up' + str(i))
            x = upconv(x)
        # output
        mask_pred = self.out(nn.functional.leaky_relu(x, 2e-1))
        return mask_pred

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)
        # when enabling mixed precision training, mask_pred may be float16
        # numpy array
        mask_pred = mask_pred.astype(np.float32)

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

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)

        return cls_segms
