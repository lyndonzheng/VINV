import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np

from .. import builder
from ..registry import DETECTORS
from .base import *
from .test_mixins import BBoxTestMixin,MaskTestMixin, RPNTestMixin

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.datasets.layer_data_prepare import *

@DETECTORS.register_module
class CompletedRGBARCNN(BaseDetector, RPNTestMixin, BBoxTestMixin, MaskTestMixin):
    """Completed rgba rcnn structure with pairwise occlusion relationship"""
    def __init__(self,
                 mode=None,
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
                 layered_feats_extractor=None,
                 completed_rgba_head=None,
                 rgb_completion=None,
                 mask_gan=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):
        super(CompletedRGBARCNN, self).__init__()
        self.mode = mode
        if self.mode == 'decomposition' or self.mode == 'end':
            assert backbone is not None
        if self.mode == 'completion' or self.mode == 'end':
            assert rgb_completion is not None
        # build structured scene decomposition network
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)
        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)
        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)
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
        if completed_rgba_head is not None:
            if layered_feats_extractor is not None:
                self.layered_feats_extractor = builder.build_roi_extractor(layered_feats_extractor)
            self.completed_rgba_head = builder.build_head(completed_rgba_head)

        # build the mask gan network
        if mask_gan is not None:
            self.mask_gan = builder.build_mask_gan(mask_gan)
        # build the completion network
        if rgb_completion is not None:
            self.rgb_completion = builder.build_rgb(rgb_completion)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # define the writer for the tensorboard
        self.writer = SummaryWriter()

        self.init_weights(pretrained)

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
        super(CompletedRGBARCNN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()
        if self.with_occ:
            self.occ_head.init_weights()
            self.occ_roi_extractor.init_weights()
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

    def extract_feat(self, imgs):
        """extract the base feature from the Resnet head"""
        x = self.backbone(imgs)
        if self.with_neck:
            # extract multi-scale features
            x = self.neck(x)
        return x

    def forward_train_decomposition(self,
                            img,
                            img_meta,
                            gt_lables,
                            gt_v_bboxes,
                            gt_bboxes_ignore=None,
                            gt_v_masks=None,
                            gt_f_bboxes=None,
                            gt_f_masks=None,
                            p_orders=None,
                            l_orders=None,
                            gt_f_rgbs=None,
                            proposals=None):
        """one time mask and occlusion relationship prediction"""

        losses = dict()

        x = self.extract_feat(img)

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_f_bboxes, img_meta, self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_f_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_lables[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_f_bboxes[i],
                    gt_lables[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x]
                )
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results, gt_f_bboxes, gt_lables, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            losses.update(loss_bbox)
            # visualize the loss
            for name, value in loss_bbox.items():
                self.visualize_value(value.data, 'Loss/train_' + str(name))

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(res.pos_bboxes.shape[0], device=device, dtype=torch.uint8)
                    )
                    pos_inds.append(
                        torch.zeros(res.neg_bboxes.shape[0], device=device, dtype=torch.uint8)
                    )
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(sampling_results, gt_v_masks, self.train_cfg.rcnn)

            pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets, pos_labels)

            # visualize the predict and ground truth sampling mask
            b, c, h, w = mask_pred.size()
            inds = torch.arange(0, b, dtype=torch.long, device=mask_pred.device)
            mask_pred_label = mask_pred[inds, pos_labels].sigmoid()
            if b > 0:
                self.visualize_img(mask_pred_label.view(b, 1, h, w), 'mask_pred', mean=[0], std=[1])
                self.visualize_img(mask_targets.view(b, 1, h, w), 'mask_targets', mean=[0], std=[1])

            losses.update(loss_mask)
            # visualize the loss
            for name, value in loss_mask.items():
                self.visualize_value(value.data, 'Loss/train_'+str(name))

        # occlusion head or refined head
        if self.with_occ or self.with_mask_refined:
            mask_targets = self.mask_head.get_target(sampling_results, gt_f_masks, self.train_cfg.rcnn)
            gt_inds = torch.cat([res.pos_assigned_gt_inds for res in sampling_results])
            conf_inds = []
            sample_num = 0
            for i in range(0, len(gt_f_bboxes)):
                ious_bbox = self.occ_head.intersect(pos_rois[pos_rois[:, 0]==i, 1:], gt_f_bboxes[i], mode='iou')
                value, ind = torch.max(ious_bbox, dim=1)
                conf_ind = value > 0.5
                gt_ind = gt_inds[sample_num:sample_num+ious_bbox.size(0)] == ind
                conf_inds.append(gt_ind & conf_ind)
                sample_num +=ious_bbox.size(0)
            conf_inds = torch.cat(conf_inds)
            refined_pos_rois = pos_rois[conf_inds]
            refined_mask_pred = mask_pred_label[conf_inds]
            refined_gt_inds = gt_inds[conf_inds]
            refined_mask_targets = mask_targets[conf_inds]
            refined_pos_labels = pos_labels[conf_inds]
            refined_feats = mask_feats[conf_inds]

            # occlusion head forward and loss
            if self.with_occ and refined_pos_rois.size(0) > 0:
                select_inds = []
                num = 0
                for i in range(0, len(gt_f_bboxes)):
                    refined_pos_rois_ = refined_pos_rois[refined_pos_rois[:, 0] == i]
                    if refined_pos_rois_.size(0) > 40:
                        select_ind = torch.arange(num, num+40, dtype=torch.long, device=refined_pos_rois.device)
                    else:
                        select_ind = torch.arange(num, num+refined_pos_rois_.size(0),
                                                  dtype=torch.long, device=refined_pos_rois.device)
                    select_inds.append(select_ind)
                    num = num+refined_pos_rois_.size(0)
                select_inds = torch.cat(select_inds)
                occ_pos_rois = refined_pos_rois[select_inds]
                occ_gt_inds = refined_gt_inds[select_inds]
                occ_targets, overlay_inds, overlay_rois = self.occ_head.get_target(occ_pos_rois,
                                                            occ_gt_inds, gt_f_masks, p_orders)
                occ_feats = self.occ_roi_extractor(x[:self.occ_roi_extractor.num_inputs], occ_pos_rois,
                                                   overlay_rois, overlay_inds)
                if occ_feats.size(0) > 0:
                    occ_pred = self.occ_head(occ_feats)
                    loss_occ = self.occ_head.loss(occ_pred, occ_targets, torch.ones_like(occ_targets))
                    losses.update(loss_occ)
                    # visualize the loss
                    for name, value in loss_occ.items():
                        self.visualize_value(value.data, 'Loss/train_' + str(name))

            # mask refined head forward and loss
            if self.with_mask_refined and refined_pos_rois.size(0) > 0:
                # mask_refined_feats = torch.zeros_like(mask_feats)

                mask_refined_feats = self.mask_refined_roi_extractor(
                    x[:self.mask_refined_roi_extractor.num_inputs], refined_pos_rois, refined_mask_pred, #mask_targets, #
                    p_orders, refined_gt_inds, img_meta[0]['img_shape']
                )
                combine_feats = torch.cat([refined_feats, mask_refined_feats], dim=1)

                mask_refined = self.mask_refined_head(combine_feats)
                loss_refined_mask = self.mask_refined_head.loss(mask_refined, refined_mask_targets, refined_pos_labels)
                loss_refined_mask['loss_refined_mask'] = loss_refined_mask.pop('loss_mask')

                losses.update(loss_refined_mask)
                # visualize the loss
                for name, value in loss_refined_mask.items():
                    self.visualize_value(value.data, 'Loss/train_'+str(name))

                # visualize the predicted mask
                b, c, h, w = mask_refined.size()
                inds = torch.arange(0, b, dtype=torch.long, device=mask_refined.device)
                mask_refined_label = mask_refined[inds, refined_pos_labels].squeeze(1)
                if b > 0:
                    self.visualize_img(mask_refined_label.sigmoid().view(b, 1, h, w), 'mask_refined', mean=[0], std=[1])

        # amodal mask forward and loss
        if self.with_amodal:
            amodal_targets = self.amodal_head.get_target(sampling_results, gt_f_masks, self.train_cfg.rcnn)
            if self.with_occ:
                amodal_feats = mask_refined_feats
                amodal_targets = amodal_targets[conf_inds]
                pos_labels = refined_pos_labels
            else:
                amodal_feats = mask_feats

            amodal_pred = self.amodal_head(amodal_feats)
            loss_amodal = self.amodal_head.loss(amodal_pred, amodal_targets, pos_labels)
            loss_amodal['loss_amodal'] = loss_amodal.pop('loss_mask')

            losses.update(loss_amodal)

            # visualize the loss
            for name, value in loss_amodal.items():
                self.visualize_value(value.data, 'Loss/train_'+str(name))

            # visualize the predicted amodal
            b, c, h, w = amodal_pred.size()
            inds = torch.arange(0, b, dtype=torch.long, device=amodal_pred.device)
            amodal_pred_label = amodal_pred[inds, pos_labels].squeeze(1)
            if b > 0:
                self.visualize_img(amodal_pred_label.sigmoid().view(b, 1, h, w), 'amodal_mask', mean=[0], std=[1])
                self.visualize_img(amodal_targets.view(b, 1, h, w), 'amodal_targets', mean=[0], std=[1])

        # completed rgba forward and loss
        if self.with_completed_rgba:
            """first try to use the ground truth label for training"""
            # transfer the mask to tensor
            gt_v_masks = [torch.tensor(gt_v_mask, dtype=torch.float, device=x[0].device) for gt_v_mask in gt_v_masks]
            gt_f_masks = [torch.tensor(gt_f_mask, dtype=torch.float, device=x[0].device) for gt_f_mask in gt_f_masks]
            layered_feats = self.layered_feats_extractor(x[0], gt_v_masks, gt_f_masks, l_orders)
            completed_rgba = self.completed_rgba_head(layered_feats).tanh()

            completed_rgba_targets, completed_rgba_real = self.completed_rgba_head.get_layered_target(gt_f_masks, gt_f_rgbs, l_orders)
            loss_completed_rgba = self.completed_rgba_head.loss(completed_rgba, completed_rgba_targets,
                                                        torch.ones_like(completed_rgba), mean=128.0, std=128.0)

            losses.update(loss_completed_rgba)
            # visualize the loss
            for name, value in loss_completed_rgba.items():
                self.visualize_value(value.data, 'Loss/train_'+str(name))
            # visualize the generated completed layered rgba
            b, c, h, w = completed_rgba.size()
            if b > 0:
                self.visualize_img(completed_rgba, 'layered_completed_rgba', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                self.visualize_img(completed_rgba_targets, 'layered_completed_targets', mean=[0, 0, 0], std=[1/256, 1/256, 1/256])

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

    # def forward_train_end(self):

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
                                      torch.tensor(img_meta[0]['img_norm_cfg']['std']).type_as(img).repeat(
                                          img.size(0)).view(img.size(0), img.size(1), 1, 1)
        self.visualize_img(img, 'original_scene', mean=img_meta[0]['img_norm_cfg']['mean'] / 256,
                           std=img_meta[0]['img_norm_cfg']['std'] / 256)

        if self.mode == 'decomposition':
            add_orders = [l_order + 1 for l_order in l_orders]
            _, l_sce_rgbs, _, l_sce_labels, _, _ = self._parse_completion_data(f_rgbs, f_depths, gt_f_masks, gt_labels, add_orders)
            img, l_labels, l_v_bboxes, l_v_masks, l_f_bboxes, l_f_masks, l_l_orders, l_p_orders, new_l_orders = \
                self._parse_decomposition_data(img, img_meta, gt_labels, gt_bboxes, gt_bboxes_ignore, gt_masks,
                                                    gt_f_bboxes, gt_f_masks, f_rgbs, l_orders, p_orders, l_sce_labels)
            # normalize the data to original mask-rcnn range
            img = ((l_sce_rgbs + 1)*128 - self.img_mean)/self.img_std
            losses = self.forward_train_decomposition(img, img_meta, l_labels, l_v_bboxes, gt_bboxes_ignore, l_v_masks,
                                                      l_f_bboxes, l_f_masks, l_p_orders, l_l_orders)
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

        assert self.with_bbox, 'Bbox head must be implemented'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels = self.simple_test_bboxes(x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)

        if self.with_mask:
            # segm_results = self.simple_test_refined_mask(x, img_meta, det_bboxes, det_labels,
            #                                              gt_f_bboxes=[gt_f_bboxes[0][:-1]], p_order=p_orders)
            segm_results, _, _ = self.simple_test_occlusion_refined_mask(x, img_meta, det_bboxes, det_labels, rescale)
            return bbox_results, segm_results

        else:
            return bbox_results

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

    def forward_test_end(self, img, img_meta, proposals=None, rescale=False):
        """testing the completed scene decomposition"""

        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)

        segm_results, sec_masks, p_orders = self.simple_test_occlusion_refined_mask(x, img_meta, det_bboxes, det_labels, rescale)
        de_results = (bbox_results, segm_results)

        # layered scene completion
        img = (img * self.img_std + self.img_mean) / 128 - 1
        co_results = self.forward_test_completion(img, img_meta, sec_masks, p_orders[0])

        return de_results, co_results

    def simple_test_occlusion_refined_mask(self, x, img_meta, det_bboxes, det_labels, rescale=False):
        """refined the mask based on the occlusion order"""
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        sec_masks = None
        p_orders = None

        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs
            _bboxes = (det_bboxes[:, :4]*scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            if self.with_occ:
                # select the high confidence objects to refine the mask
                conf_inds = det_bboxes[:, -1] > 0.5
                ref_bboxes, ref_labels, ref_mask_rois = det_bboxes[conf_inds], det_labels[conf_inds], mask_rois[conf_inds]
                overlay_inds, overlay_rois = self.occ_head.get_overlay_inds(ref_mask_rois)
                # get occlusion labels
                occ_feats = self.occ_roi_extractor(x[:self.occ_roi_extractor.num_inputs], ref_mask_rois,
                                                   overlay_rois, overlay_inds)
                if occ_feats.size(0) > 0:
                    occ_pred = self.occ_head(occ_feats)
                    p_orders, obj_inds = self.occ_head.get_p_orders(ref_mask_rois, occ_pred, overlay_inds)
                    # get refined features and mask
                    mask_pred_label = mask_pred[conf_inds, ref_labels + 1].squeeze(1).sigmoid()
                    mask_refined_feats = self.mask_refined_roi_extractor(x[:len(self.mask_refined_roi_extractor.featmap_strides)],
                                                                         ref_mask_rois, mask_pred_label, p_orders, obj_inds, ori_shape)
                    # mask_feats[conf_inds] = mask_refined_feats
                    # mask_pred = self.mask_refined_head(mask_feats)
                    # mask_pred[conf_inds] = self.amodal_head(mask_refined_feats)
                    refined_feats = mask_feats[conf_inds]
                    combine_feats = torch.cat([refined_feats, mask_refined_feats], dim=1)
                    mask_pred[conf_inds] = self.mask_refined_head(combine_feats)
                    sec_masks = self.mask_head.trans_pred_masks(mask_pred[conf_inds], ref_bboxes, ref_labels,
                                                                self.test_cfg.rcnn, ori_shape, scale_factor, rescale)

            segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes, det_labels, self.test_cfg.rcnn,
                                                       ori_shape, scale_factor, rescale)

        return segm_result, sec_masks, p_orders

    def match_result_to_gt(self, gt_f_bboxes, pred_bboxes):
        """get the gt inds for the occlusion order"""

        ious_bbox = self.occ_head.intersect(pred_bboxes, gt_f_bboxes[0])
        value, ind = torch.max(ious_bbox, dim=1)
        conf_ind = value > 0.7
        gt_ind = ind[conf_ind]

        return gt_ind, conf_ind

    # def simple_test_refined_mask(self,
    #                      x,
    #                      img_meta,
    #                      det_bboxes,
    #                      det_labels,
    #                      rescale=False,
    #                      gt_f_bboxes=None,
    #                      p_order=None,):
    #     # image shape pf the first image int the batch (only one)
    #     ori_shape = img_meta[0]['ori_shape']
    #     scale_factor = img_meta[0]['scale_factor']
    #
    #     if det_bboxes.shape[0] == 0:
    #         segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
    #     else:
    #         # if det_bboxes is rescaled to the original image size, we need to
    #         # rescale it back to the testing scale to obtain RoIs.
    #         _bboxes = (det_bboxes[:, :4]*scale_factor if rescale else det_bboxes)
    #         mask_rois = bbox2roi([_bboxes])
    #         mask_feats = self.mask_roi_extractor(
    #             x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
    #         if self.with_shared_head:
    #             mask_feats = self.shared_head(mask_feats)
    #         mask_pred = self.mask_head(mask_feats)
    #         # b, c, h, w = mask_pred.size()
    #         # inds = torch.arange(0, b, dtype=torch.long, device=mask_pred.device)
    #         # mask_pred_label = mask_pred[inds, det_labels].squeeze(1)
    #         gt_inds, conf_inds = self.match_result_to_gt(gt_f_bboxes, _bboxes[:, :4])
    #         if gt_inds.size(0) > 0:
    #             refined_bboxes, refined_det_labels, refined_mask_rois = _bboxes[conf_inds], det_labels[conf_inds], mask_rois[conf_inds]
    #             mask_pred_label = mask_pred[conf_inds, refined_det_labels+1].squeeze(1).sigmoid()
    #             mask_refined_feats = self.mask_refined_roi_extractor(
    #                     x[:len(self.mask_refined_roi_extractor.featmap_strides)], refined_mask_rois, mask_pred_label,# mask_targets, #
    #                     p_order, gt_inds, img_meta[0]['img_shape']
    #                 )
    #             mask_refined = self.mask_refined_head(mask_refined_feats)
    #             mask_pred[conf_inds] = mask_refined
    #             # mask_feats[conf_inds] = mask_refined_feats
    #
    #         # mask_pred = self.amodal_head(mask_feats)
    #         # mask_pred = self.mask_refined_head(mask_feats)
    #
    #         segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes,
    #                                                    det_labels, self.test_cfg.rcnn,
    #                                                    ori_shape, scale_factor,
    #                                                    rescale)
    #
    #     return segm_result

    async def async_simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation"""
        assert self.with_bbox, 'Bbox head must be implemented'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.simple_test_rpn(x, img_meta, self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels = await self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale
        )
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)

        if self.with_mask:
            segm_results =await self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale
            )
            return bbox_results, segm_results
        else:
            return bbox_results

    # def simple_test(self, img, img_meta, proposals=None, rescale=False, gt_f_bboxes=None, p_order=None,):
    #     """Test without augmentation"""
    #     assert self.with_bbox, 'Bbox head must be implemented'
    #
    #     x = self.extract_feat(img)
    #
    #     if proposals is None:
    #         proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn)
    #     else:
    #         proposal_list = proposals
    #
    #     det_bboxes, det_labels = self.simple_test_bboxes(
    #         x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale
    #     )
    #     bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #
    #     if self.with_mask:
    #         if self.with_mask_refined:
    #             segm_results = self.simple_test_refined_mask(x, img_meta, det_bboxes, det_labels,
    #                                                          gt_f_bboxes=[gt_f_bboxes[0][:-1]], p_order=p_order)
    #         else:
    #             segm_results = self.simple_test_mask(x, img_meta, det_bboxes, det_labels, rescale=rescale)
    #         return bbox_results, segm_results
    #     else:
    #         return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentation"""
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(self.extract_feats(imgs), img_metas, self.test_cfg.rpn)

        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,  self.test_cfg.rcnn
        )

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels, self.bbox_head.num_classes)

        # det bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels
            )
            return bbox_results, segm_results
        else:
            return bbox_results

    def show_result(self, data, result, dataset=None, score_thr=0.3):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            class_ind6 = labels == 6
            class_ind20 = labels == 20
            class_ind19 = labels == 6
            class_inds = class_ind6 | class_ind20 | class_ind19
            score_inds = bboxes[:, -1] > score_thr
            inds = np.where(class_inds & score_inds)[0]
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                # inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = (labels[i] * np.array([2 ** 11 - 1, 2 ** 21 - 1, 2 ** 31 - 1]) % 255).astype(np.uint8)#np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            bboxes = bboxes[inds]
            labels = labels[inds]
            mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr)