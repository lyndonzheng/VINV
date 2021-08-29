from ..registry import DETECTORS
from .two_stage import TwoStageDetector
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np

@DETECTORS.register_module
class FMaskRCNN(TwoStageDetector):
    """new mask rcnn struture for full mask"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FMaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        # define the writer for the tensorboard
        self.writer = SummaryWriter()

    def visualize_img(self, data, name, mean=None, std=None):
        """visualize the trianing image at tensorboard"""
        if self.iters % 1000 == 0:
            assert data.dim() == 4
            b, ch, h, w = data.size()
            if mean is None:
                mean = torch.tensor([0.5]).repeat(ch).view(1, ch, 1, 1)
                std = torch.tensor([0.5]).repeat(ch).view(1, ch, 1, 1)
            else:
                mean = torch.tensor(mean).view(1, ch, 1, 1)
                std = torch.tensor(std).view(1, ch, 1, 1)
            data = data * std.type_as(data) + mean.type_as(data)
            grid_data = make_grid(data)
            self.writer.add_image(name, grid_data, self.iters)

    def visualize_value(self, data, name):
        """visualize the training loss at tensorboard"""
        if self.iters % 1000 == 0:
            self.writer.add_scalar(name, data, self.iters)

    def _parse_understanding_data(self, img, img_meta, gt_a_labels, gt_v_bboxes=None, gt_bboxes_ignore=None,
                                  gt_v_masks=None, gt_f_bboxes=None, gt_f_masks=None, f_rgbs=None,):
        """parse data for training"""
        self.img_mean, self.img_std = torch.tensor(img_meta[0]['img_norm_cfg']['mean']).type_as(img).repeat(
            img.size(0)).view(img.size(0), img.size(1), 1, 1), \
                                      torch.tensor(img_meta[0]['img_norm_cfg']['std']).type_as(img).repeat(
                                          img.size(0)).view(img.size(0), img.size(1), 1, 1)
        self.visualize_img(img, 'original_scene', mean=img_meta[0]['img_norm_cfg']['mean'] / 256,
                           std=img_meta[0]['img_norm_cfg']['std'] / 256)

        # using only the visible or full bboxes and masks
        if gt_f_bboxes is not None:
            bboxes = gt_f_bboxes
        elif gt_v_bboxes is not None:
            bboxes = gt_v_bboxes
        if gt_f_masks is not None:
            masks = gt_f_masks
        elif gt_v_masks is not None:
            masks = gt_v_masks
        # ignore the BG for the object detection and mask prediction
        lables_wobg = [lable[:-1] for lable in gt_a_labels]
        bboxes_wobg = [bbox[:-1] for bbox in bboxes]
        masks_wobg = [mask[:-1] for mask in masks]
        # visualize the mask
        visual_mask = np.vstack(masks_wobg)
        b, h, w = visual_mask.shape
        if b > 0:
            self.visualize_img(torch.tensor(visual_mask).view(b, 1, h, w).type_as(img), 'gt_mask', mean=[0], std=[1])

        return img, lables_wobg, bboxes_wobg, masks_wobg

    def forward_train_alpha(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):

        losses = dict()

        x = self.extract_feat(img)

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)
            # visualize the loss
            for name, value in loss_bbox.items():
                self.visualize_value(value.data, 'Loss/train_' + str(name))

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(sampling_results,
                                                     gt_masks,
                                                     self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)

            # visualize the predict and ground truth sampling mask
            b, c, h, w = mask_pred.size()
            inds = torch.arange(0, b, dtype=torch.long, device=mask_pred.device)
            mask_pred_label = mask_pred[inds, pos_labels].squeeze(1)
            if b > 0:
                self.visualize_img(torch.nn.functional.sigmoid(mask_pred_label.view(b, 1, h, w)), 'mask_pred', mean=[0],
                                   std=[1])
                self.visualize_img(mask_targets.view(b, 1, h, w), 'mask_targets', mean=[0], std=[1])

            losses.update(loss_mask)
            # visualize the loss
            for name, value in loss_mask.items():
                self.visualize_value(value.data, 'Loss/train_' + str(name))

            return losses

    def forward_train(self,
                      img_o,
                      img_meta,
                      gt_labels,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_semantic_seg=None,
                      depth=None,
                      gt_f_bboxes=None,
                      gt_f_masks=None,
                      f_rgbs=None,
                      f_depths=None,
                      l_order=None,
                      p_order=None,
                      iters=None):

        self.iters = iters
        # get the training lable
        img, labels, bboxes, masks = self._parse_understanding_data(img_o, img_meta, gt_labels, gt_bboxes,
                                                                             gt_bboxes_ignore, gt_masks, gt_f_bboxes,
                                                                             gt_f_masks, f_rgbs)

        # training only for the mask(alpha) channel
        losses = self.forward_train_alpha(img, img_meta, bboxes, labels, gt_bboxes_ignore, masks)

        return losses