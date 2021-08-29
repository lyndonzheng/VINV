import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, bbox_target, force_fp32)
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS


@HEADS.register_module
class OccHead(nn.Module):
    """RoI head"""

    def __init__(self,
                 with_avg_pool=False,
                 with_occ=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=2,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 loss_occ=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0
                 )):
        super(OccHead, self).__init__()
        assert with_occ
        self.with_avg_pool = with_avg_pool
        self.with_occ = with_occ
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        self.fp16_enabled = False

        self.loss_occ = build_loss(loss_occ)
        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_occ:
            self.fc_occ = nn.Linear(in_channels, num_classes)
        self.debug_imgs = None

    def init_weights(self):
        if self.with_occ:
            nn.init.normal_(self.fc_occ.weight, 0, 0.01)
            nn.init.constant(self.fc_occ.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        occ_score = self.fc_occ(x) if self.with_occ else None
        return occ_score

    @force_fp32(apply_to=('occ_score'))
    def loss(self,
             occ_score,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if occ_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)
            losses['loss_occ'] = self.loss_occ(
                occ_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override
            )
            losses['acc_occ'] = accuracy(occ_score, labels)
        return losses

    def get_target(self, sampling_results, gt_occ, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        gt_occ = [res.gt_occ for res in sampling_results]
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            gt_occ=gt_occ,
            target_means=self.target_means,
            target_stds=self.target_stds
        )
        return cls_reg_targets