import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, bbox_target, force_fp32)
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS


@HEADS.register_module
class BinaryOccHead(nn.Module):
    """Binary absolution occlusion order"""

    def __init__(self,
                 with_avg_pool=False,
                 with_occ=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_fcs=2,
                 fc_out_channels=512,
                 num_classes=2,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 loss_occ=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0
                 )
                 ):
        super(BinaryOccHead, self).__init__()
        assert with_occ
        self.with_avg_pool = with_avg_pool
        self.with_occ = with_occ
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_fcs = num_fcs
        self.fc_out_channels = fc_out_channels
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

        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_fcs > 0:
            for i in range(num_fcs):
                fc_in_channels = (in_channels if i == 0 else self.fc_out_channels)
                branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            in_channels = self.fc_out_channels
        self.occ_fcs = branch_fcs

        self.relu = nn.ReLU(inplace=True)
        if self.with_occ:
            self.fc_occ = nn.Linear(in_channels, num_classes)
        self.debug_imgs = None

    def init_weights(self):
        for module_list in [self.occ_fcs, self.fc_occ]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        if self.num_fcs > 0:
            for fc in self.occ_fcs:
                x = self.relu(fc(x))
        occ_score = self.fc_occ(x) if self.with_occ else None
        return occ_score

    @force_fp32(apply_to=('occ_score'))
    def loss(self,
             occ_score,
             labels,
             label_weights,
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
        pos_gt_occ = [res.pos_gt_occ for res in sampling_results]
        labels = torch.cat(pos_gt_occ, 0)
        label_weights = torch.ones_like(labels)
        return labels, label_weights