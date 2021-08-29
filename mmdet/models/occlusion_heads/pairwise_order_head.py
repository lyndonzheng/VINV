import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

import pycocotools.mask as maskUtils
from mmdet.core import (auto_fp16, force_fp32)
from ..builder import build_loss
from ..losses import accuracy
from  ..registry import HEADS


@HEADS.register_module
class PairwiseOccHead(nn.Module):
    """Pairwise occlusion order"""

    def __init__(self,
                 with_avg_pool=False,
                 with_occ=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_fcs=2,
                 fc_out_channels=1024,
                 num_classes=2,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 loss_occ=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0
                 ),):
        super(PairwiseOccHead, self).__init__()
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
        self.fp16_enabled = False

        self.loss_occ = build_loss(loss_occ)
        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *=self.roi_feat_area

        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_fcs > 0:
            for i in range(num_fcs):
                fc_in_channels = (in_channels if i==0 else self.fc_out_channels)
                branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            in_channels = self.fc_out_channels
        self.occ_fcs = branch_fcs

        self.relu = nn.ReLU(inplace=True)
        # define the occlusion head
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
    def loss(self, occ_score, labels, label_weights, reduction_override=None):
        losses = dict()
        if occ_score is not None:
            avg_factor = max(torch.sum(label_weights>0).float().item(), 1.0)
            losses['loss_occ'] = self.loss_occ(
                occ_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override,
            )
            losses['acc_occ'] = accuracy(occ_score, labels)

        return losses

    def intersect(self, bboxes1, bboxes2, mode='over'):
        """ We resize both tensors to [A,B,2] without new malloc:
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                    bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap
        return ious

    def get_p_orders(self, rois, occ_preds, overlay_inds):
        p_orders = []
        obj_inds = []
        _, occ_labels = occ_preds.topk(1, dim=1)
        occ_labels = torch.where(occ_labels[:, 0] == 1, -1 * torch.ones_like(occ_labels[:, 0]), torch.zeros_like(occ_labels[:, 0]))
        for i in range(0, int(rois[:,0].max() + 1)):
            rois_ = rois[rois[:, 0] == i]
            overlay_inds_ = overlay_inds[overlay_inds[:, 0] == i]
            occ_labels_ = occ_labels[overlay_inds[:, 0] == i]
            obj_inds.append(torch.arange(0, rois_.size(0), dtype=torch.long, device=rois_.device))
            p_order = torch.zeros(rois_.size(0), rois_.size(0)).type_as(occ_labels_)
            for j in range(0, rois_.size(0)):
                occ_ind = overlay_inds_[overlay_inds_[:, 1] == j][:, 2].type(torch.long)
                p_order[j, occ_ind] = occ_labels_[overlay_inds_[:, 1] == j]
            p_orders.append(p_order)
        obj_inds = torch.cat(obj_inds)

        return p_orders, obj_inds

    def get_overlay_inds(self, rois):
        """get the overlay regions and indexes"""
        overlay_inds = []
        overlay_rois = []
        for i in range(0, int(rois[:, 0].max() + 1)):
            rois_ = rois[rois[:, 0] == i]
            inter_areas = self.intersect(rois_[:, 1:], rois_[:, 1:])
            for j, (roi, inter_area) in enumerate(zip(rois, inter_areas)):
                inter_area[j] = 0
                occ_roi_inds = inter_area > 49
                occ_rois = rois[occ_roi_inds]
                overlay_roi = torch.zeros_like(occ_rois)
                overlay_roi[:, 0] = occ_rois[:, 0]
                overlay_roi[:, 1:3] = torch.min(occ_rois[:, 1:3], torch.ones_like(occ_rois[:, 1:3]) * roi[1:3])
                overlay_roi[:, 3:] = torch.max(occ_rois[:, 3:], torch.ones_like(occ_rois[:, 3:]) * roi[3:])
                overlay_rois.append(overlay_roi)
                overlay_ind = torch.zeros(occ_rois.size(0), 3)
                overlay_ind[:, 0], overlay_ind[:, 1] = i, j
                overlay_ind[:, 2] = torch.where(torch.tensor(occ_roi_inds).to(rois.device))[0]
                overlay_inds.append(overlay_ind)

        return torch.cat(overlay_inds), torch.cat(overlay_rois)

    def get_target(self, rois, gt_inds, gt_f_masks, p_orders):
        occ_targets = []
        overlay_inds = []
        overlay_rois = []

        for i in range(0, int(rois[:, 0].max() + 1)):
            gt_f_masks_ = gt_f_masks[i]
            p_orders_ = p_orders[i]
            rois_ = rois[rois[:, 0] == i]
            gt_inds_ = gt_inds[rois[:, 0] == i]
            inter_areas = self.intersect(rois_[:, 1:], rois_[:, 1:])
            for j, (roi, inter_area, gt_ind) in enumerate(zip(rois_, inter_areas, gt_inds_)):
                gt_f_mask = gt_f_masks_[gt_ind]
                p_order = p_orders_[gt_ind]
                p_order = torch.where(p_order == -1, torch.ones_like(p_order), torch.zeros_like(p_order))
                occ_roi_inds = inter_area > 49
                occ_gt_inds = gt_inds_ != gt_ind
                occ_rois = rois_[occ_roi_inds & occ_gt_inds]
                occ_gts = gt_inds_[occ_roi_inds & occ_gt_inds]
                occ_targets.append(p_order[occ_gts])
                overlay_roi = torch.zeros_like(occ_rois)
                overlay_roi[:, 0] = occ_rois[:, 0]
                overlay_roi[:, 1:3] = torch.min(occ_rois[:, 1:3], torch.ones_like(occ_rois[:, 1:3]) * roi[1:3])
                overlay_roi[:, 3:] = torch.max(occ_rois[:, 3:], torch.ones_like(occ_rois[:, 3:]) * roi[3:])
                overlay_rois.append(overlay_roi)
                overlay_ind = torch.zeros(occ_rois.size(0), 3)
                overlay_ind[:, 0], overlay_ind[:, 1] = i, j
                overlay_ind[:, 2] = torch.where(occ_roi_inds & occ_gt_inds)[0]
                overlay_inds.append(overlay_ind)

        occ_targets = torch.cat(occ_targets)
        overlay_inds = torch.cat(overlay_inds)
        overlay_rois = torch.cat(overlay_rois)

        return occ_targets, overlay_inds, overlay_rois
