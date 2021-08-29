import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np

from .. import builder
from ..registry import DETECTORS
from .base import *
from .test_mixins import RPNTestMixin

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from mmdet.datasets.layer_data_prepare import *


@DETECTORS.register_module
class CompletedRGBHTC(BaseDetector, RPNTestMixin):

    def __init__(self,
                 mode=None,
                 num_stages=None,
                 backbone=None,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 occ_roi_extractor=None,
                 occ_head=None,
                 mask_refined_roi_extractor=None,
                 mask_refined_head=None,
                 amodal_roi_extractor=None,
                 amodal_head=None,
                 rgb_completion=None,
                 layered_feats_extractor=None,
                 completed_rgba_head=None,
                 mask_gan=None,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 mask_info_flow=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,):
        super(CompletedRGBHTC, self).__init__()
        assert not self.with_shared_head  # shared head not supported
        if mode == 'decomposition' or mode == 'end':
            assert backbone is not None
        elif mode == 'completion' or mode == 'end':
            assert rgb_completion is not None
        self.mode = mode
        # build structured scene decomposition network
        self.num_stages = num_stages
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        if shared_head is not None:
            self.share_head = builder.build_shared_head(shared_head)
        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)
        if bbox_head is not None:
            self.bbox_roi_extractor = nn.ModuleList()
            self.bbox_head = nn.ModuleList()
            if not isinstance(bbox_roi_extractor, list):
                bbox_roi_extractor = [
                    bbox_roi_extractor for _ in range(num_stages)
                ]
            if not isinstance(bbox_head, list):
                bbox_head = [bbox_head for _ in range(num_stages)]
            assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
            for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
                self.bbox_roi_extractor.append(
                    builder.build_roi_extractor(roi_extractor))
                self.bbox_head.append(builder.build_head(head))
        if mask_head is not None:
            self.mask_head = nn.ModuleList()
            if not isinstance(mask_head, list):
                mask_head = [mask_head for _ in range(num_stages)]
            assert len(mask_head) == self.num_stages
            for head in mask_head:
                self.mask_head.append(builder.build_head(head))
            if mask_roi_extractor is not None:
                self.share_roi_extractor = False
                self.mask_roi_extractor = nn.ModuleList()
                if not isinstance(mask_roi_extractor, list):
                    mask_roi_extractor = [
                        mask_roi_extractor for _ in range(num_stages)
                    ]
                assert len(mask_roi_extractor) == self.num_stages
                for roi_extractor in mask_roi_extractor:
                    self.mask_roi_extractor.append(
                        builder.build_roi_extractor(roi_extractor))
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
        if semantic_head is not None:
            self.semantic_roi_extractor = builder.build_roi_extractor(
                semantic_roi_extractor)
            self.semantic_head = builder.build_head(semantic_head)
        # add refined head
        if occ_head is not None:
            self.occ_roi_extractor = builder.build_roi_extractor(occ_roi_extractor)
            self.occ_head = builder.build_head(occ_head)
        if mask_refined_head is not None:
            self.mask_refined_roi_extractor = builder.build_roi_extractor(mask_refined_roi_extractor)
            self.mask_refined_head = builder.build_head(mask_refined_head)
        if amodal_head is not None:
            if amodal_roi_extractor is not None:
                self.amodal_roi_extractor = builder.build_roi_extractor(amodal_roi_extractor)
            else:
                self.amodal_roi_extractor = None
            self.amodal_head = builder.build_head(amodal_head)
        # directly use the use ResNet features for layered scene completion
        if completed_rgba_head is not None:
            self.layered_feats_extractor = builder.build_roi_extractor(layered_feats_extractor)
            self.completed_rgba_head = builder.build_head(completed_rgba_head)
        if mask_gan is not None:
            self.mask_gan = builder.build_mask_gan(mask_gan)
        # build the individual completion network
        if rgb_completion is not None:
            self.rgb_completion = builder.build_rgb(rgb_completion)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow
        # define the writer for the tensorboard
        self.writer = SummaryWriter()

        self.init_weights(pretrained=pretrained)

    @property
    def with_semantic(self):
        return hasattr(self, 'semantic_head') and self.semantic_head is not None

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_occ(self):
        return hasattr(self, 'occ_head') and self.occ_head is not None

    @property
    def with_completed_rgba(self):
        return hasattr(self, 'completed_rgba_head') and self.completed_rgba_head is not None

    @property
    def with_mask_gan(self):
        return hasattr(self, 'mask_gan') and self.mask_gan is not None

    @property
    def with_mask_refined(self):
        return hasattr(self, 'mask_refined_head') and self.mask_refined_head is not None

    @property
    def with_amodal(self):
        return hasattr(self, 'amodal_head') and self.amodal_head is not None

    @property
    def with_completion(self):
        return hasattr(self, 'rgb_completion') and self.rgb_completion is not None

    def init_weights(self, pretrained=None):
        super(CompletedRGBHTC, self).init_weights(pretrained)
        if self.backbone is not None:
            self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.with_neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.num_stages is not None:
            for i in range(self.num_stages):
                if self.with_bbox:
                    self.bbox_roi_extractor[i].init_weights()
                    self.bbox_head[i].init_weights()
                if self.with_mask:
                    if not self.share_roi_extractor:
                        self.mask_roi_extractor[i].init_weights()
                    self.mask_head[i].init_weights()
        if self.with_occ:
            self.occ_roi_extractor.init_weights()
            self.occ_head.init_weights()
        if self.with_semantic:
            self.semantic_roi_extractor.init_weights()
            self.semantic_head.init_weights()
        if self.with_mask_gan:
            self.mask_gan.init_weights()
        if self.with_mask_refined:
            self.mask_refined_roi_extractor.init_weights()
            self.mask_refined_head.init_weights()
        if self.with_amodal:
            self.amodal_head.init_weights()
            if self.amodal_roi_extractor:
                self.amodal_roi_extractor.init_weights()
        if self.with_completed_rgba:
            self.completed_rgba_head.init_weights()
        if self.with_completion:
            self.rgb_completion.init_weights()

    def visualize_img(self, data, name, mean=None, std=None):
        """visualize the trianing image at tensorboard"""
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
        if self.iters % 1000 == 0:
            self.writer.add_image(name, grid_data, self.iters)

    def visualize_value(self, data, name):
        """visualize the training loss at tensorboard"""
        if self.iters % 1000 == 0:
            self.writer.add_scalar(name, data, self.iters)

    def _update_l_orders(self, l_orders):
        """update l_orders to re-composite different images for each view"""
        if self.train_cfg.data_mode == 'layer':
            new_l_orders = [item - 1 for item in l_orders]
        elif self.train_cfg.data_mode == 'random':
            new_l_orders = []
            for item in l_orders:
                l_order = torch.ones_like(item)
                inds = torch.rand(l_order.size()) > 0.5
                l_order[inds] = -1
                new_l_orders.append(l_order)
        elif self.train_cfg.data_mode == 'original':
            new_l_orders = l_orders
        else:
            raise NotImplementedError('no data mode is selected')

        return new_l_orders

    def _parse_decomposition_data(self, img, img_meta, gt_lables, gt_v_bboxes=None, gt_bboxes_ignore=None, gt_v_masks=None,
                    gt_f_bboxes=None, gt_f_masks=None, f_rgbs=None, l_orders=None, p_orders=None, l_sce_labels=None):
        """parse data for decomposition network"""
        # get layer gt
        l_labels, l_v_bboxes, l_v_masks, l_f_bboxes, l_f_masks, l_l_orders, l_p_orders, l_occs = parse_batch_layer_anns(
            gt_lables, gt_v_bboxes, gt_v_masks, gt_f_bboxes, gt_f_masks, l_orders, p_orders, l_sce_labels
        )

        # visualize the mask
        visual_v_mask = np.vstack(l_v_masks)
        visual_f_mask = np.vstack(l_f_masks)
        b, h, w = visual_v_mask.shape
        if b > 0:
            self.visualize_img(torch.tensor(visual_v_mask).view(b, 1, h, w).type_as(img), 'gt_v_mask', mean=[0], std=[1])
            self.visualize_img(torch.tensor(visual_f_mask).view(b, 1, h, w).type_as(img), 'gt_f_mask', mean=[0], std=[1])

        # update the layer order for each training
        new_l_orders = self._update_l_orders(l_orders)

        return img, l_labels, l_v_bboxes, l_v_masks, l_f_bboxes, l_f_masks, l_l_orders, l_p_orders, new_l_orders

    def _parse_completion_data(self, rgbs, depths, masks, labels, l_orders):
        """parse data for completion network"""

        # get layer gt
        l_sce_rgbs, l_sce_depths, l_sce_labels, l_v_masks = parse_batch_layer_data(rgbs, depths, masks,
                                                                                   labels, l_orders)
        masked_l_sce_rgbs = l_sce_rgbs * l_v_masks

        # visualize the training data
        self.visualize_img(l_sce_rgbs, 'layer_rgb_truth')
        self.visualize_img(masked_l_sce_rgbs, 'maksed_layer_rgb')
        self.visualize_img(l_sce_labels, 'layer_semantic_map', mean=[0], std=[0.025])
        self.visualize_img(l_v_masks, 'layer_v_mask', mean=[0], std=[1])

        # update the layer order for each training
        new_l_orders = self._update_l_orders(l_orders)

        return masked_l_sce_rgbs, l_sce_rgbs, l_sce_depths, l_sce_labels, l_v_masks, new_l_orders

    def extract_feat(self, img):
        """extract the base feature from the Resnet head"""
        x = self.backbone(img)
        if self.with_neck:
            # extract multi-scale features
            x = self.neck(x)
        return x

    def forward_train_decomposition(self,
                      img,
                      img_meta,
                      gt_labels,
                      gt_v_bboxes=None,
                      gt_bboxes_ignore=None,
                      gt_v_masks=None,
                      gt_f_bboxes=None,
                      gt_f_masks=None,
                      p_orders=None,
                      l_orders=None,
                      gt_semantic_seg=None,
                      proposals=None,):

        losses = dict()

        x = self.extract_feat(img)

        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg
            self.visualize_value(loss_seg.data, 'Loss/train_loss_semantic')
        else:
            semantic_feat = None

        # RPN part, the same as normal two-stage detectors
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_f_bboxes, img_meta, self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
            bbox_sampler = build_sampler(rcnn_train_cfg.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_f_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_f_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            loss_bbox, rois, bbox_targets, bbox_pred = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_f_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_targets[0]

            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (
                    value * lw if 'loss' in name else value)
                self.visualize_value(value.data, 'Loss/train_de_s' + str(i)+ '_'+ str(name))

            # mask head forward and loss
            if self.with_mask:
                # interleaved execution: use regressed bboxes by the box branch
                # to train the mask branch
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []
                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(
                                proposal_list[j], gt_f_bboxes[j],
                                gt_bboxes_ignore[j], gt_labels[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_f_bboxes[j],
                                gt_labels[j],
                                feats=[lvl_feat[j][None] for lvl_feat in x])
                            sampling_results.append(sampling_result)
                loss_mask, pos_rois, pos_labels, mask_targets, mask_pred = \
                    self._mask_forward_train(i, x, sampling_results,
                                             gt_f_masks, rcnn_train_cfg, semantic_feat)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)
                    self.visualize_value(value.data, 'Loss/train_de_s' + str(i) + '_' + str(name))

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
        # visualize the predict and ground truth sampling mask
        b, c, h, w = mask_pred.size()
        inds = torch.arange(0, b, dtype=torch.long, device=mask_pred.device)
        mask_pred_label = mask_pred[inds, pos_labels].sigmoid()
        if b > 0:
            self.visualize_img(mask_pred_label.view(b, 1, h, w), 'mask_pred', mean=[0], std=[1])
            self.visualize_img(mask_targets.view(b, 1, h, w), 'mask_targets', mean=[0], std=[1])
        # occlusion head forward and loss
        if self.with_occ:
            gt_inds = torch.cat([res.pos_assigned_gt_inds for res in sampling_results])
            select_inds = []
            num = 0
            for i in range(0, len(gt_f_bboxes)):
                pos_rois_ = pos_rois[pos_rois[:, 0] == i]
                if pos_rois_.size(0) > 40:
                    select_ind = torch.arange(num, num+40, dtype=torch.long, device=pos_rois.device)
                else:
                    select_ind = torch.arange(num, num+pos_rois_.size(0), dtype=torch.long, device=pos_rois.device)
                select_inds.append(select_ind)
                num += pos_rois_.size(0)
            select_inds = torch.cat(select_inds)
            occ_pos_rois = pos_rois[select_inds]
            occ_gt_inds = gt_inds[select_inds]
            occ_targets, overlay_inds, overlay_rois = self.occ_head.get_target(occ_pos_rois,
                                                                               occ_gt_inds, gt_f_masks, p_orders)
            if occ_targets.size(0) > 0:
                occ_feats = self.occ_roi_extractor(x[:self.occ_roi_extractor.num_inputs], occ_pos_rois,
                                                   overlay_rois, overlay_inds)
                occ_pred = self.occ_head(occ_feats)
                loss_occ = self.occ_head.loss(occ_pred, occ_targets, torch.ones_like(occ_targets))
                losses.update(loss_occ)
                # visualize the loss
                for name, value in loss_occ.items():
                    self.visualize_value(value.data, 'Loss/train_' + str(name))
            # mask refined head forward and loss
            if self.with_mask_refined:
                mask_targets = self.mask_refined_head.get_target(sampling_results, gt_f_masks, self.train_cfg.rcnn)
                refined_pos_rois = pos_rois[select_inds]
                refined_gt_inds = gt_inds[select_inds]
                refined_mask_pred = mask_pred_label[select_inds]
                refined_mask_targets = mask_targets[select_inds]
                refined_pos_labels = pos_labels[select_inds]
                mask_refined_feats = self.mask_refined_roi_extractor(
                    x[:self.mask_refined_roi_extractor.num_inputs], refined_pos_rois, refined_mask_pred,
                    p_orders, refined_gt_inds, img_meta[0]['img_shape']
                )
                mask_refined = self.mask_refined_head(mask_refined_feats)
                loss_mask = self.mask_refined_head.loss(mask_refined, refined_mask_targets, refined_pos_labels)
                loss_mask['loss_refined_mask'] = loss_mask.pop('loss_mask')

                losses.update(loss_mask)
                # visualize the loss
                for name, value in loss_mask.items():
                    self.visualize_value(value.data, 'Loss/train_'+str(name))
                    # visualize the predicted mask
                    b, c, h, w = mask_refined.size()
                    inds = torch.arange(0, b, dtype=torch.long, device=mask_refined.device)
                    mask_refined_label = mask_refined[inds, refined_pos_labels].squeeze(1)
                    if b > 0:
                        self.visualize_img(mask_refined_label.sigmoid().view(b, 1, h, w), 'mask_refined', mean=[0], std=[1])
        # amodal mask forward and loss
        if self.with_amodal:
            amodal_targets = self.amodal_head.get_target(sampling_results, gt_f_masks, rcnn_train_cfg)
            if self.with_occ:
                amodal_feats = mask_refined_feats
                amodal_targets = amodal_targets[select_inds]
                pos_labels = refined_pos_labels
            else:
                amodal_feats = self.amodal_roi_extractor(x[:self.amodal_roi_extractor.num_inputs], pos_rois)
            amodal_pred = self.amodal_head(amodal_feats)
            loss_amodal = self.amodal_head.loss(amodal_pred, amodal_targets, pos_labels)
            loss_amodal['loss_amodal'] = loss_amodal.pop('loss_mask')

            losses.update(loss_amodal)
            # visualize the loss
            for name, value in loss_amodal.items():
                self.visualize_value(value.data, 'Loss/train_' + str(name))
            # visualize the predicted amodal
            b, c, h, w = amodal_pred.size()
            inds = torch.arange(0, b, dtype=torch.long, device=amodal_pred.device)
            amodal_pred_label = amodal_pred[inds, pos_labels].squeeze(1)
            if b > 0:
                self.visualize_img(amodal_pred_label.sigmoid().view(b, 1, h, w), 'amodal_mask', mean=[0], std=[1])
                self.visualize_img(amodal_targets.view(b, 1, h, w), 'amodal_targets', mean=[0], std=[1])

        return losses

    def forward_train_completion(self,
                            img,
                            masked_img,
                            v_mask,
                            semantic_map):
        """layered scene completion"""

        losses = dict()
        results = dict()

        # forward pass
        g_img = self.rgb_completion.forward(masked_img, mask=v_mask)
        self.visualize_img(g_img[-1], 'output_img')

        # get scale ground truth
        scale_gt = self.rgb_completion.get_scale_img(img, len(g_img))
        scale_mask = self.rgb_completion.get_scale_img(v_mask, len(g_img))

        # get the gan loss for the discriminator
        D_loss = self.rgb_completion.optimizer_d(scale_gt, g_img)
        for name, value in D_loss.items():
            self.visualize_value(value.data, 'Loss/train_co_'+str(name))

        # get the loss for the generation model
        G_loss = self.rgb_completion.G_loss(scale_gt, g_img, scale_mask)
        losses.update(G_loss)
        for name, value in G_loss.items():
            self.visualize_value(value.data, 'Loss/train_co_'+str(name))

        # fill in the completed pixel to the orignal image
        results['completed_img'] = (1 - v_mask) * g_img[-1] + v_mask * masked_img
        self.visualize_img(results['completed_img'], 'completed_img')

        return losses, results

    def forward_train(self,
                      img,
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
                      l_orders=None,
                      p_orders=None,
                      iters=None,
                      epoch=None):

        results = dict()
        self.iters = iters
        self.epoch = epoch

        # visualize the input image
        self.img_mean, self.img_std = torch.tensor(img_meta[0]['img_norm_cfg']['mean']).type_as(img).repeat(
            img.size(0)).view(img.size(0), img.size(1), 1, 1), \
            torch.tensor(img_meta[0]['img_norm_cfg']['std']).type_as(img).repeat(img.size(0)).view(img.size(0), img.size(1), 1, 1)

        self.visualize_img(img, 'original_scene', mean=img_meta[0]['img_norm_cfg']['mean'] / 256,
                           std=img_meta[0]['img_norm_cfg']['std'] / 256)

        if self.mode == 'decomposition':
            add_orders = [l_order + 1 for l_order in l_orders]
            _, l_sce_rgbs, _, l_sce_labels, _, _ = self._parse_completion_data(f_rgbs, f_depths, gt_f_masks, gt_labels, add_orders)
            img, l_labels, l_v_bboxes, l_v_masks, l_f_bboxes, l_f_masks, l_l_orders, l_p_orders, new_l_orders = \
                self._parse_decomposition_data(img, img_meta, gt_labels, gt_bboxes, gt_bboxes_ignore, gt_masks,
                                               gt_f_bboxes, gt_f_masks, f_rgbs, l_orders, p_orders, l_sce_labels)
            # normalize the data to original mask-rcnn range
            img = ((l_sce_rgbs + 1) * 128 - self.img_mean) / self.img_std
            l_sce_labels = F.interpolate(l_sce_labels, size=[gt_semantic_seg.size(2), gt_semantic_seg.size(3)], mode='nearest')
            losses = self.forward_train_decomposition(img, img_meta, l_labels, l_v_bboxes, gt_bboxes_ignore, l_v_masks,
                                                      l_f_bboxes, l_f_masks, l_p_orders, l_l_orders, l_sce_labels)
            results['l_orders'] = new_l_orders
            results['img'] = img
        elif self.mode == 'completion':
            masked_l_sce_rgbs, l_sce_rgbs, l_sce_depths, l_sce_labels, l_v_masks, new_l_orders = \
                self._parse_completion_data(f_rgbs, f_depths, gt_f_masks, gt_labels, l_orders)
            losses, results = self.forward_train_completion(l_sce_rgbs, masked_l_sce_rgbs, l_v_masks, l_sce_labels)
            results['l_orders'] = new_l_orders
            results['img'] = img
        else:
            raise NotImplementedError('no mode is selected')

        return losses, results

    def forward_test(self,
                     img,
                     img_meta,
                     gt_labels=None,
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
                     l_orders=None,
                     p_orders=None,
                     rescale=False):
        """rewrite the testing code with re-composition scene"""
        de_results = []
        co_results = []
        self.iters = 1
        if isinstance(img, list):
            img = img[0]

        num_augs = img.size(0)
        assert num_augs == 1
        if num_augs != len(img_meta):
            raise ValueError('num of augmentations ({}) != num of image meta ({})'.format(len(img), len(img_meta)))

        self.img_mean, self.img_std = torch.tensor(img_meta[0]['img_norm_cfg']['mean']).type_as(img).repeat(
            img.size(0)).view(img.size(0), img.size(1), 1, 1), \
                                      torch.tensor(img_meta[0]['img_norm_cfg']['std']).type_as(img).repeat(
                                          img.size(0)).view(img.size(0), img.size(1), 1, 1)

        if f_rgbs is not None:
            add_orders = [l_order + 1 for l_order in l_orders]
            _, img, _, l_sce_labels, _, _ = self._parse_completion_data(f_rgbs, f_depths, gt_f_masks, gt_labels,
                                                                               add_orders)
            img = ((img + 1) * 128 - self.img_mean) / self.img_std
        if self.mode == 'decomposition':
            de_results = self.forward_test_decomposition(img, img_meta, proposals, rescale, gt_f_bboxes, p_orders)
        elif self.mode == 'completion':
            img = (img * self.img_std + self.img_mean) / 128 - 1
            co_results = self.forward_test_completion(img, img_meta, gt_f_masks[0][:-1], p_orders[0][:-1][:,:-1])
        elif self.mode == 'end':
            de_results, co_results = self.forward_test_end(img, img_meta, proposals, rescale)
        else:
            raise NotImplementedError('no mode is selected')

        return de_results, co_results

    def forward_test_decomposition(self, img, img_meta, proposals=None, rescale=False, gt_f_bboxes=None, p_orders=None):
        """testing the decomposition network"""
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']

        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            cls_score, bbox_pred = self._bbox_forward_test(i, x, rois, semantic_feat)
            ms_scores.append(cls_score)
            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = self.bbox_head[i].regress_by_class(rois, bbox_label, bbox_pred, img_meta[0])
        cls_score = sum(ms_scores) / float(len(ms_scores))
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=rescale, cfg=rcnn_test_cfg
        )
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head[-1].num_classes)

        if self.with_mask:
            segm_results, _, _ = self.simple_test_occlusion_refined_mask(x, img_meta, det_bboxes,
                                                                         det_labels, semantic_feat, rescale)
            return bbox_results, segm_results
        else:
            return bbox_results

    def simple_test_occlusion_refined_mask(self, x, img_meta, det_bboxes, det_labels, semantic_feat = None, rescale=False):
        """refined the mask based on the occlusion order"""
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        sec_masks = None
        p_orders = None

        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head[-1].num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs
            _bboxes = (det_bboxes[:, :4]*scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            aug_masks = []
            mask_feats = self.mask_roi_extractor[-1](x[:len(self.mask_roi_extractor[-1].featmap_strides)], mask_rois)
            if self.with_semantic and 'mask' in self.semantic_fusion:
                mask_semantic_feat = self.semantic_roi_extractor([semantic_feat], mask_rois)
                mask_feats += mask_semantic_feat
            last_feat = None
            for i in range(self.num_stages):
                if self.mask_info_flow:
                    mask_pred, last_feat = self.mask_head[i](mask_feats, last_feat)
                else:
                    mask_pred = self.mask_head[i](mask_feats)
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, [img_meta] * self.num_stages, self.test_cfg.rcnn)
            mask_pred = torch.tensor(merged_masks).type_as(mask_pred)
            # select the high confidence objects to refine the mask
            if self.with_occ:
                conf_inds = det_bboxes[:, -1] > 0.5
                ref_bboxes, ref_lables, ref_mask_rois = det_bboxes[conf_inds], det_labels[conf_inds], mask_rois[conf_inds]
                overlay_inds, overlay_rois = self.occ_head.get_overlay_inds(ref_mask_rois)
                # get occlusion labels
                occ_feats = self.occ_roi_extractor(x[:self.occ_roi_extractor.num_inputs], ref_mask_rois,
                                                   overlay_rois, overlay_inds)
                if occ_feats.size(0) > 0:
                    occ_pred = self.occ_head(occ_feats)
                    p_orders, obj_inds = self.occ_head.get_p_orders(ref_mask_rois, occ_pred, overlay_inds)
                    # get refined features and mask
                    mask_pred_label = mask_pred[conf_inds, ref_lables + 1]
                    mask_refined_feats = self.mask_refined_roi_extractor(x[:len(self.mask_refined_roi_extractor.featmap_strides)],
                                                                         ref_mask_rois, mask_pred_label, p_orders, obj_inds, ori_shape)
                    # mask_feats[conf_inds] = mask_refined_feats
                    # mask_pred = self.mask_refined_head(mask_feats)
                    # mask_pred = self.amodal_head(mask_feats)
                    mask_pred[conf_inds] = self.mask_refined_head(mask_refined_feats).sigmoid()
                    sec_masks = self.mask_head[-1].trans_pred_masks(mask_pred[conf_inds], ref_bboxes, ref_lables,
                                                                self.test_cfg.rcnn, ori_shape, scale_factor, rescale)

            mask_pred = mask_pred.cpu().numpy()
            segm_result = self.mask_head[-1].get_seg_masks(mask_pred, _bboxes, det_labels, self.test_cfg.rcnn,
                                                       ori_shape, scale_factor, rescale)

        return segm_result, sec_masks, p_orders

    def forward_test_completion(self, img, img_meta, f_masks, p_orders):
        """testing the completion network"""
        results = []
        nums = p_orders.size(0)
        del_inds = []
        l_results = [img]
        f_masks = torch.tensor(f_masks).type_as(img).unsqueeze(dim=1)
        p_orders = torch.tensor(p_orders).type_as(img)
        while(nums != len(del_inds)):
            del_objs = []
            p_orders_sum = (p_orders==-1).sum(dim=1)
            if len(del_inds) > 0:
                p_orders_sum[torch.tensor(del_inds)] = 100  # label the deleted objects has largest
            unocc_inds = (p_orders_sum == p_orders_sum.min()).nonzero()
            l_v_mask = torch.ones_like(f_masks[[0]])
            for unocc_ind in unocc_inds:
                l_v_mask = torch.where(f_masks[unocc_ind] > 0, torch.zeros_like(l_v_mask), l_v_mask)
                del_obj = l_results[-1] * f_masks[unocc_ind]
                del_obj = torch.cat([del_obj, f_masks[unocc_ind]], dim=1)
                del_objs.append((del_obj, unocc_ind[0]))
                del_inds.append(unocc_ind[0])
            p_orders[unocc_inds] = 0
            p_orders[:, unocc_inds] = 0
            maskd_img = l_results[-1] * l_v_mask
            g_img = self.rgb_completion.forward(maskd_img, mask=l_v_mask)
            co_img = (1 - l_v_mask) * g_img[-1] + l_v_mask * maskd_img
            l_results.append(co_img)
            results.append((g_img[-1], del_objs))

        return results

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            cls_score, bbox_pred = self._bbox_forward_test(
                i, x, rois, semantic_feat=semantic_feat)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])

        cls_score = sum(ms_scores) / float(len(ms_scores))
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                mask_classes = self.mask_head[-1].num_classes - 1
                segm_result = [[] for _ in range(mask_classes)]
            else:
                _bboxes = (
                    det_bboxes[:, :4] *
                    scale_factor if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None
                for i in range(self.num_stages):
                    mask_head = self.mask_head[i]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks,
                                               [img_meta] * self.num_stages,
                                               self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result

        if self.with_mask:
            results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'])
        else:
            results = ms_bbox_result['ensemble']

        return results

    def _bbox_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            rcnn_train_cfg,
                            semantic_feat=None):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat

        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes,
                                            gt_labels, rcnn_train_cfg)
        loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
        return loss_bbox, rois, bbox_targets, bbox_pred

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            semantic_feat=None):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        pos_rois)

        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             pos_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat

        # mask information flow
        # forward all previous mask heads to obtain last_feat, and fuse it
        # with the normal mask feature
        if self.mask_info_flow:
            last_feat = None
            for i in range(stage):
                last_feat = self.mask_head[i](
                    mask_feats, last_feat, return_logits=False)
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
        else:
            mask_pred = mask_head(mask_feats)

        mask_targets = mask_head.get_target(sampling_results, gt_masks,
                                            rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
        return loss_mask, pos_rois, pos_labels, mask_targets, mask_pred

    def _bbox_forward_test(self, stage, x, rois, semantic_feat=None):
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(
            x[:len(bbox_roi_extractor.featmap_strides)], rois)
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat
        cls_score, bbox_pred = bbox_head(bbox_feats)
        return cls_score, bbox_pred

    def _mask_forward_test(self, stage, x, bboxes, semantic_feat=None):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_rois = bbox2roi([bboxes])
        mask_feats = mask_roi_extractor(
            x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             mask_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat
        if self.mask_info_flow:
            last_feat = None
            last_pred = None
            for i in range(stage):
                mask_pred, last_feat = self.mask_head[i](mask_feats, last_feat)
                if last_pred is not None:
                    mask_pred = mask_pred + last_pred
                last_pred = mask_pred
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
            if last_pred is not None:
                mask_pred = mask_pred + last_pred
        else:
            mask_pred = mask_head(mask_feats)
        return mask_pred

    def forward_dummy(self, img):
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # semantic head
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None
        # bbox heads
        rois = bbox2roi([proposals])
        for i in range(self.num_stages):
            cls_score, bbox_pred = self._bbox_forward_test(
                i, x, rois, semantic_feat=semantic_feat)
            outs = outs + (cls_score, bbox_pred)
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            mask_roi_extractor = self.mask_roi_extractor[-1]
            mask_feats = mask_roi_extractor(
                x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_semantic and 'mask' in self.semantic_fusion:
                mask_semantic_feat = self.semantic_roi_extractor(
                    [semantic_feat], mask_rois)
                mask_feats += mask_semantic_feat
            last_feat = None
            for i in range(self.num_stages):
                mask_head = self.mask_head[i]
                if self.mask_info_flow:
                    mask_pred, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_pred = mask_head(mask_feats)
                outs = outs + (mask_pred, )
        return outs

    def aug_test(self, imgs, img_metas, proposals=None, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        if self.with_semantic:
            semantic_feats = [
                self.semantic_head(feat)[1]
                for feat in self.extract_feats(imgs)
            ]
        else:
            semantic_feats = [None] * len(img_metas)

        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)

        rcnn_test_cfg = self.test_cfg.rcnn
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, semantic in zip(
                self.extract_feats(imgs), img_metas, semantic_feats):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                cls_score, bbox_pred = self._bbox_forward_test(
                    i, x, rois, semantic_feat=semantic)
                ms_scores.append(cls_score)

                if i < self.num_stages - 1:
                    bbox_label = cls_score.argmax(dim=1)
                    rois = bbox_head.regress_by_class(rois, bbox_label,
                                                      bbox_pred, img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes -
                                              1)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta, semantic in zip(
                        self.extract_feats(imgs), img_metas, semantic_feats):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip)
                    mask_rois = bbox2roi([_bboxes])
                    mask_feats = self.mask_roi_extractor[-1](
                        x[:len(self.mask_roi_extractor[-1].featmap_strides)],
                        mask_rois)
                    if self.with_semantic:
                        semantic_feat = semantic
                        mask_semantic_feat = self.semantic_roi_extractor(
                            [semantic_feat], mask_rois)
                        if mask_semantic_feat.shape[-2:] != mask_feats.shape[
                                -2:]:
                            mask_semantic_feat = F.adaptive_avg_pool2d(
                                mask_semantic_feat, mask_feats.shape[-2:])
                        mask_feats += mask_semantic_feat
                    last_feat = None
                    for i in range(self.num_stages):
                        mask_head = self.mask_head[i]
                        if self.mask_info_flow:
                            mask_pred, last_feat = mask_head(
                                mask_feats, last_feat)
                        else:
                            mask_pred = mask_head(mask_feats)
                        aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg.rcnn)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return bbox_result, segm_result
        else:
            return bbox_result
