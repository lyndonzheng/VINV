from __future__ import division

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .. import builder
from ..registry import DETECTORS
from .base import *
from .test_mixins import RPNTestMixin
from mmdet.datasets.layer_data_prepare import *

from mmdet.core import (bbox2result, bbox2roi, build_assigner, build_sampler,
                        merge_aug_masks, occ2result, imshow_det_bboxes, mkdirs)


@DETECTORS.register_module
class LBLCompletedRGBHTC(BaseDetector, RPNTestMixin):

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
                 rgb_completion=None,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 semantic_fusion=('bbox', 'mask', 'occ'),
                 interleaved=True,
                 mask_info_flow=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):
        super(LBLCompletedRGBHTC, self).__init__()
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
        # add the binary occlusion extractor and occlusion file
        if occ_head is not None:
            self.occ_head = nn.ModuleList()
            if not isinstance(occ_head, list):
                occ_head = [occ_head for _ in range(num_stages)]
            assert len(occ_head) == self.num_stages
            for head in occ_head:
                self.occ_head.append(builder.build_head(head))
            if occ_roi_extractor is not None:
                self.occ_roi_extractor = nn.ModuleList()
                if not isinstance(occ_roi_extractor, list):
                    occ_roi_extractor = [occ_roi_extractor for _ in range(num_stages)]
                assert len(occ_roi_extractor) == self.num_stages
                for roi_extractor in occ_roi_extractor:
                    self.occ_roi_extractor.append(builder.build_roi_extractor(roi_extractor))
            else:
                self.occ_roi_extractor = self.bbox_roi_extractor
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
    def with_completion(self):
        return hasattr(self, 'rgb_completion') and self.rgb_completion is not None

    @property
    def with_completion_wogt(self):
        return False

    def init_weights(self, pretrained=None):
        super(LBLCompletedRGBHTC, self).init_weights(pretrained)
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
                    if not self.share_roi_extractor:
                        self.occ_roi_extractor[i].init_weights()
                    self.occ_head[i].init_weights()
        if self.with_semantic:
            self.semantic_roi_extractor.init_weights()
            self.semantic_head.init_weights()
        if self.with_completion:
            self.rgb_completion.init_weights()

    def visualize_img(self, data, name, mean=None, std=None):
        """visualize the trianing image at tensorboard"""
        assert data.dim() == 4
        if self.iters % 1000 == 0:
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

    def _update_l_orders(self, l_orders):
        """update l_orders to re-composite different images for each view"""
        if self.train_cfg is None:
            return l_orders
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

    def _parse_decomposition_data(self, img, img_meta, gt_lables, gt_v_bboxes=None, gt_v_masks=None, gt_f_bboxes=None,
                                  gt_f_masks=None, l_orders=None, p_orders=None):
        """parse data for decomposition network"""
        # get layer gt
        l_labels, l_v_bboxes, l_v_masks, l_f_bboxes, l_f_masks, l_l_orders, l_p_orders, l_occs = parse_batch_layer_anns(
            img_meta, gt_lables, gt_v_bboxes, gt_v_masks, gt_f_bboxes, gt_f_masks, l_orders, p_orders)

        # visualize the mask, as cocoa may load different size masks, only show the mask for first image
        visual_v_mask = np.vstack([l_v_masks[0]])
        visual_f_mask = np.vstack([l_f_masks[0]])
        b, h, w = visual_v_mask.shape
        if b > 0:
            self.visualize_img(torch.tensor(visual_v_mask).view(b, 1, h, w).type_as(img), 'gt_v_mask', mean=[0], std=[1])
            self.visualize_img(torch.tensor(visual_f_mask).view(b, 1, h, w).type_as(img), 'gt_f_mask', mean=[0], std=[1])

        # update the layer order for each training
        new_l_orders = self._update_l_orders(l_orders)

        return img, l_labels, l_v_bboxes, l_v_masks, l_f_bboxes, l_f_masks, l_l_orders, l_p_orders, l_occs, new_l_orders

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
                                    gt_bboxes=None,
                                    gt_bboxes_ignore=None,
                                    gt_masks=None,
                                    gt_occ=None,
                                    gt_semantic_seg=None,
                                    proposals=None):

        losses = dict()

        x = self.extract_feat(img)

        # semantic segmentation part
        # outputs: segmentation prediction and embedded features
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
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta, self.train_cfg.rpn)
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
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j],
                                                     gt_occ=gt_occ[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    gt_occ=gt_occ[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            loss_bbox, rois, bbox_targets, bbox_pred = \
                    self._bbox_forward_train(
                        i, x, sampling_results, gt_bboxes, gt_labels,
                        rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_targets[0]
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (
                    value * lw if 'loss' in name else value)
                self.visualize_value(value.data, 'Loss/train_de_s' + str(i) + '_' + str(name))

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
                                proposal_list[j], gt_bboxes[j],
                                gt_bboxes_ignore[j], gt_labels[j], gt_occ[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j],
                                gt_labels[j],
                                gt_occ[j],
                                feats=[lvl_feat[j][None] for lvl_feat in x])
                            sampling_results.append(sampling_result)
                loss_mask = self._mask_forward_train(i, x, sampling_results,
                                             gt_masks, rcnn_train_cfg, semantic_feat)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)
                    self.visualize_value(value.data, 'Loss/train_de_s' + str(i) + '_' + str(name))

            # occlusion head forward and loss
            if self.with_occ:
                loss_occ = self._occ_forward_train(i, x, sampling_results, gt_occ,
                                         rcnn_train_cfg, semantic_feat)
                for name, value in loss_occ.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)
                    self.visualize_value(value.data, 'Loss/train_de_' + str(i) + '_' + str(name))

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
        return losses

    def forward_train_completion(self,
                            imgs,
                            img_metas,
                            masked_imgs,
                            v_masks):
        """layered scene completion"""

        losses = dict()
        results = dict()

        # ignore the pad part
        if img_metas[0]['img_shape'] != img_metas[0]['ori_shape']:
            imgs_nopad, masked_imgs_nopad, v_masks_nopad = [], [], []
            for img, img_meta, masked_img, v_mask in zip(imgs, img_metas, masked_imgs, v_masks):
                img_shape = img_meta['img_shape']
                imgs_nopad.append(F.interpolate(img.unsqueeze(0)[:,:,:img_shape[0], :img_shape[1]], (512, 512)))
                masked_imgs_nopad.append(F.interpolate(masked_img.unsqueeze(0)[:, :, :img_shape[0], :img_shape[1]], (512, 512)))
                v_masks_nopad.append(F.interpolate(v_mask.unsqueeze(0)[:, :, :img_shape[0], :img_shape[1]], (512, 512)))
            in_imgs = torch.cat(imgs_nopad)
            in_masked_imgs = torch.cat(masked_imgs_nopad)
            in_v_masks = torch.cat(v_masks_nopad)
        else:
            in_imgs = imgs
            in_masked_imgs = masked_imgs
            in_v_masks = v_masks

        # forward pass
        g_imgs = self.rgb_completion.forward(in_masked_imgs, mask=in_v_masks)
        self.visualize_img(g_imgs[-1], 'output_img')

        # get scale ground truth
        scale_gt = self.rgb_completion.get_scale_img(in_imgs, len(g_imgs))
        scale_mask = self.rgb_completion.get_scale_img(in_v_masks, len(g_imgs))

        # get the gan loss for the discriminator
        D_loss = self.rgb_completion.optimizer_d(scale_gt, g_imgs)
        for name, value in D_loss.items():
            self.visualize_value(value.data, 'Loss/train_co_'+str(name))

        # get the loss for the generation model
        G_loss = self.rgb_completion.G_loss(scale_gt, g_imgs, scale_mask)
        losses.update(G_loss)
        for name, value in G_loss.items():
            self.visualize_value(value.data, 'Loss/train_co_'+str(name))

        # fill in the completed pixel to the orignal image
        if img_metas[0]['img_shape'] != img_metas[0]['ori_shape']:
            out_imgs = torch.zeros_like(imgs)
            for i, (img_meta, g_img) in enumerate(zip(img_metas, g_imgs[-1])):
                img_shape = img_meta['img_shape']
                out_imgs[i, :, :img_shape[0], :img_shape[1]] = F.interpolate(g_img.unsqueeze(0), (img_shape[0], img_shape[1]))
        else:
            out_imgs = g_imgs[-1]
        results['completed_img'] = (1 - v_masks) * out_imgs + v_masks * masked_imgs
        self.visualize_img(results['completed_img'], 'completed_img')

        return losses, results

    def forward_train_end(self,
                          imgs,
                          img_metas,
                          ori_imgs,
                          gt_labels,
                          gt_v_bboxes=None,
                          gt_bboxes=None,
                          gt_bboxes_ignore=None,
                          gt_v_masks=None,
                          gt_masks=None,
                          gt_semantic_seg=None,
                          proposals=None,
                          f_rgbs=None,
                          f_depths=None,
                          l_orders=None,
                          p_orders=None):
        """End-to-end training, that mask-rcnn will contribute for the completion network"""
        losses = dict()
        results = dict()
        dec_flag = True

        # detect the non-occluded objects
        update_orders = []
        l_pre_masks = []
        # proposal_list = []
        for l_order, p_order, img, img_meta, gt_label, gt_bbox, gt_mask in zip(l_orders,
                        p_orders, imgs, img_metas, gt_labels, gt_bboxes, gt_masks):
            with torch.no_grad():
                img_meta['flip'] = False
                result = self.forward_test_decomposition(img.unsqueeze(0), [img_meta], proposals=proposals)
            # proposal_list.append(torch.tensor(np.vstack(result[0])))
            result = select_result_by_order(result, score_thr=0.5)
            del_inds, masks = match_result_to_gt(gt_label[:-1], gt_mask[:-1], gt_bbox[:-1], l_order, result)
            l_pre_masks.append(torch.cat(masks).unsqueeze(1).sum(dim=0, keepdim=True)==0)
            update_order = l_order_update(l_order, del_inds, p_order)
            update_orders.append(update_order)
        l_pre_masks = torch.cat(l_pre_masks).type_as(imgs)
        obj_orders = [item + 1 for item in update_orders]
        # get batch training data for completion network
        masked_l_sce_rgbs, l_sce_rgbs, l_sce_depths, l_sce_labels, l_v_masks, _ = \
                self._parse_completion_data(f_rgbs, f_depths, gt_masks, gt_labels, obj_orders)
        if self.epoch > 5:
            l_v_masks[:, :, :l_pre_masks.size(2), :l_pre_masks.size(3)] = l_pre_masks
        # normalize the trained image for completion network
        imgs = (imgs * self.img_std + self.img_mean) / 128 - 1
        masked_l_sce_rgbs = imgs * l_v_masks
        self.visualize_img(masked_l_sce_rgbs, 'masked_rgb')
        losses_co, results_co = self.forward_train_completion(l_sce_rgbs, img_metas, masked_l_sce_rgbs, l_v_masks)
        # collect the completion loss
        for name, value in losses_co.items():
            losses[name] = value

        # update the order
        for i, (update_order, gt_label) in enumerate(zip(update_orders, gt_labels)):
            if (update_order[:-1] > -1).sum() == 0 or (gt_label[update_order > 0] > 0).sum() == 0:
                update_orders[i][-1] = -1
                dec_flag=False
        # only when we have objects in the scene, we do the scene decomposition
        # normalize the trained image for decomposition network
        results['img'] = ((results_co['completed_img'] + 1) * 128 - self.img_mean) / self.img_std

        if dec_flag:
            # get batch training data for decomposition network
            img, l_labels, _, _, l_f_bboxes, l_f_masks, l_l_orders, l_p_orders, l_occs, new_l_orders = \
                self._parse_decomposition_data(imgs, img_metas, gt_labels, gt_v_bboxes, gt_v_masks, gt_bboxes, gt_masks,
                                               update_orders, p_orders)
            l_sce_labels = F.interpolate(l_sce_labels, size=[gt_semantic_seg.size(2), gt_semantic_seg.size(3)], mode='nearest')
            losses_de = self.forward_train_decomposition(results['img'], img_metas, l_labels, l_f_bboxes,
                                                         gt_bboxes_ignore, l_f_masks, l_occs, l_sce_labels, proposals)
            # collect the decomposition loss
            for name, value in losses_de.items():
                losses[name] = value
        # collect the results
        results['l_orders'] = update_orders
        results['img'] = results['img'].detach() # completed image only has one gradient back propagate
        # store the proposal list in first step
        # if self.steps == 0:
        #     results['proposals'] = proposal_list
        # else:
        #     results['proposals'] = proposals

        return losses, results

    def forward_train(self,
                      img,
                      img_meta,
                      ori_img,
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
                      epoch=None,
                      steps=None):

        results = dict()
        self.iters = iters
        self.epoch = epoch
        self.steps = steps
        self.img_mean, self.img_std = torch.tensor(img_meta[0]['img_norm_cfg']['mean']).type_as(img).repeat(
            img.size(0)).view(img.size(0), img.size(1), 1, 1), \
                                      torch.tensor(img_meta[0]['img_norm_cfg']['std']).type_as(img).repeat(
                                          img.size(0)).view(img.size(0), img.size(1), 1, 1)

        self.visualize_img(img, 'original_scene', mean=img_meta[0]['img_norm_cfg']['mean'] / 256,
                           std=img_meta[0]['img_norm_cfg']['std'] / 256)
        l_sce_labels = gt_semantic_seg

        if self.mode == 'decomposition':
            img, l_labels, l_v_bboxes, l_v_masks, l_f_bboxes, l_f_masks, l_l_orders, l_p_orders, l_occs, new_l_orders = \
                self._parse_decomposition_data(img, img_meta, gt_labels, gt_bboxes, gt_masks, gt_f_bboxes, gt_f_masks, l_orders,
                                               p_orders)
            if f_rgbs is not None:  # recompose the different scene with instance gt for training
                add_orders = [l_order + 1 for l_order in l_orders]
                _, l_sce_rgbs, _, l_sce_labels, _, _ = self._parse_completion_data(f_rgbs, f_depths, gt_f_masks,
                                                                                   gt_labels, add_orders)
                # normalize the data to original mask-rcnn range
                img = ((l_sce_rgbs + 1) * 128 - self.img_mean) / self.img_std
            if gt_semantic_seg is not None:
                    l_sce_labels = F.interpolate(l_sce_labels, size=[gt_semantic_seg.size(2), gt_semantic_seg.size(3)], mode='nearest')
            losses = self.forward_train_decomposition(img, img_meta, l_labels, l_f_bboxes, gt_bboxes_ignore, l_f_masks,
                                                          l_occs, l_sce_labels, proposals)
            results['l_orders'] = new_l_orders
            results['img'] = img
        elif self.mode == 'completion':
            masked_l_sce_rgbs, l_sce_rgbs, l_sce_depths, l_sce_labels, l_v_masks, new_l_orders = \
                    self._parse_completion_data(f_rgbs, f_depths, gt_f_masks, gt_labels, l_orders)
            if self.epoch > 10 and img.max() < 1:
                masked_l_sce_rgbs = l_v_masks * img
            else:
                masked_l_sce_rgbs = l_v_masks * l_sce_rgbs
            losses, results = self.forward_train_completion(l_sce_rgbs, img_meta, masked_l_sce_rgbs, l_v_masks)
            results['l_orders'] = new_l_orders
            results['img'] = results['completed_img'].detach()
        elif self.mode == 'end':
            losses, results = self.forward_train_end(img, img_meta, ori_img, gt_labels, gt_bboxes, gt_f_bboxes, gt_bboxes_ignore,
                                    gt_masks, gt_f_masks, gt_semantic_seg, proposals, f_rgbs, f_depths, l_orders, p_orders)
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
        """rewrite the testing code the re-composition scene"""
        de_results = []
        co_results = []
        self.iters = 1

        if isinstance(img, list):
            img = img[0]
        if isinstance(img_meta[0], list):
            img_meta = img_meta[0]

        num_augs = img.size(0)
        assert num_augs == 1
        if num_augs != len(img_meta):
            raise ValueError('num of augmentations ({}) != num of image meta ({})'.format(len(img), len(img_meta)))

        self.img_mean, self.img_std = torch.tensor(img_meta[0]['img_norm_cfg']['mean']).type_as(img).repeat(
            img.size(0)).view(img.size(0), img.size(1), 1, 1), \
                                      torch.tensor(img_meta[0]['img_norm_cfg']['std']).type_as(img).repeat(
                                          img.size(0)).view(img.size(0), img.size(1), 1, 1)
        if self.mode == 'decomposition':
            if f_rgbs is not None:
                add_orders = [l_order + 1 for l_order in l_orders]
                _, img, _, l_sce_labels, _, _ = self._parse_completion_data(f_rgbs, f_depths, gt_f_masks, gt_labels,
                                                                            add_orders)
                img = ((img + 1) * 128 - self.img_mean) / self.img_std
            # if gt_f_bboxes is not None and gt_labels is not None:  # testing the model with grount truth bbox
            #     proposals = []
            #     for gt_f_bbox, gt_label in zip(gt_f_bboxes, gt_labels):
            #         if gt_f_bbox.size(0) == 0:
            #             proposals = None
            #             break
            #         else:
            #             proposals.append(
            #                 torch.cat([gt_f_bbox, (gt_label - 1).unsqueeze(-1).type_as(gt_f_bbox)], dim=-1))
            if f_rgbs:
                de_results = self.forward_test_decomposition_gt_completion(img, img_meta, gt_labels, gt_f_bboxes,
                                                            gt_f_masks, f_rgbs, f_depths, l_orders, proposals, rescale)
            else:
                de_results = self.forward_test_decomposition(img, img_meta, proposals, rescale)
        elif self.mode == 'completion':
            co_results = self.forward_test_completion_gt_decomposition(img, img_meta, gt_labels, gt_f_bboxes,
                                                                       gt_f_masks, f_rgbs, f_depths, l_orders)
        elif self.mode == 'end':
            de_results, co_results = self.forward_test_end(img, img_meta, proposals, rescale, iters=1,
                                                           occ_thr=0.5, score_thr=0.3)

        return de_results, co_results

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
            mask_pred = mask_head(mask_feats, return_feat=False)

        mask_targets = mask_head.get_target(sampling_results, gt_masks,
                                            rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
        return loss_mask

    def _occ_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_occ,
                            rcnn_train_cfg,
                            semantic_feat=None
                            ):
        occ_roi_extractor = self.occ_roi_extractor[stage]
        occ_head = self.occ_head[stage]
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        occ_feats = occ_roi_extractor(x[:occ_roi_extractor.num_inputs], pos_rois)
        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'occ' in self.semantic_fusion:
            occ_semantic_feat = self.semantic_roi_extractor([semantic_feat], pos_rois)
            if occ_semantic_feat.shape[-2:] != occ_feats.shape[-2:]:
                occ_semantic_feat = F.adaptive_avg_pool2d(occ_semantic_feat, occ_feats.shape[-2:])
            occ_feats += occ_semantic_feat

        occ_score = occ_head(occ_feats)
        occ_targets = occ_head.get_target(sampling_results, gt_occ, rcnn_train_cfg)
        loss_occ = occ_head.loss(occ_score, *occ_targets)
        return loss_occ

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

    def _occ_forward_test(self, stage, x, bboxes, semantic_feat=None):
        occ_roi_extractor = self.occ_roi_extractor[stage]
        occ_head = self.occ_head[stage]
        occ_rois = bbox2roi([bboxes])
        occ_feats = occ_roi_extractor(x[:len(occ_roi_extractor.featmap_strides)], occ_rois)
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            occ_semantic_feat = self.semantic_roi_extractor([semantic_feat], occ_rois)
            if occ_semantic_feat.shape[-2:] != occ_feats.shape[-2:]:
                occ_semantic_feat = F.adaptive_avg_pool2d(occ_semantic_feat, occ_feats.shape[-2:])
            occ_feats += occ_semantic_feat
        occ_score = occ_head(occ_feats)
        return occ_score

    def forward_test_decomposition(self, img, img_meta, proposals=None, rescale=False):
        """testing the decomposition network"""
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

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
                segm_result = [[] for _ in range(self.mask_head[-1].num_classes - 1)]
            else:
                _bboxes = (det_bboxes[:, :4] * torch.tensor(scale_factor).type_as(det_bboxes) if rescale else det_bboxes)
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
                    elif 'HTCMaskHead' in self.mask_head[i]._get_name():
                        mask_pred, last_feat = self.mask_head[i](mask_feats)
                    else:
                        mask_pred = self.mask_head[i](mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks, [img_meta] * self.num_stages, self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                                                            ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result

        if self.with_occ:
            ms_occ_result = {}
            ms_occ_scores = []
            if det_bboxes.shape[0] == 0:
                occ_result = [np.zeros((0, 2), dtype=np.float32) for _ in range(self.bbox_head[-1].num_classes - 1)]
            else:
                _bboxes = (det_bboxes[:, :4] * torch.tensor(scale_factor).type_as(det_bboxes) if rescale else det_bboxes)
                for i in range(self.num_stages):
                    occ_score = self._occ_forward_test(i, x, _bboxes, semantic_feat=semantic_feat)
                    ms_occ_scores.append(occ_score)
                occ_score = sum(ms_occ_scores) / float(len(ms_occ_scores))
                occ_score = F.softmax(occ_score, dim=1)
                occ_result = occ2result(occ_score, det_labels, self.bbox_head[-1].num_classes)
            ms_occ_result['ensemble'] = occ_result

        if self.with_mask and self.with_occ:
            results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'], ms_occ_result['ensemble'])
        elif self.with_mask:
            results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'])
        else:
            results = ms_bbox_result['ensemble']

        return results

    def forward_test_decomposition_gt_completion(self,
                                                img,
                                                img_meta,
                                                gt_labels=None,
                                                gt_bboxes=None,
                                                gt_masks=None,
                                                f_rgbs=None,
                                                f_depths=None,
                                                l_orders=None,
                                                proposals=None,
                                                rescale=False):
        """top line for the layer-by-layer completed scene decomposition"""
        de_results = []
        layer_num = 0
        iters = 10
        while((l_orders[0][:-1]>-1).sum() > 0 and layer_num < iters):
            add_orders = [l_order + 1 for l_order in l_orders]
            _, img, _, l_sce_labels, _, _ = self._parse_completion_data(f_rgbs, f_depths, gt_masks, gt_labels,
                                                                               add_orders)
            img = ((img + 1) * 128 - self.img_mean) / self.img_std
            result = self.forward_test_decomposition(img, img_meta, proposals=proposals, rescale=rescale)
            select_result = select_result_by_order(result, score_thr=0.5)
            del_inds, _ = match_result_to_gt(gt_labels[0][:-1], gt_masks[0][:-1], gt_bboxes[0][:-1], l_orders[0], select_result)
            l_orders = [torch.where(del_inds, -1*torch.ones_like(l_orders[0]), l_orders[0])]
            if layer_num == iters - 1:
                result = add_pre_layer(result, pre_layer=layer_num)
            else:
                result = add_pre_layer(select_result, pre_layer=layer_num)
            de_results.append(result)
            layer_num += 1
        de_results = collect_layer_results(de_results)

        return de_results

    def forward_test_completion_gt_decomposition(self,
                                                 img,
                                                 img_meta,
                                                 gt_labels=None,
                                                 gt_bboxes=None,
                                                 gt_masks=None,
                                                 f_rgbs=None,
                                                 f_depths=None,
                                                 l_orders=None,
                                                 ):
        """top line for the layer-by-layer completed scene decomposition"""
        co_results = []
        g_img = None
        img = (img * self.img_std + self.img_mean) / 128 - 1
        while((l_orders[0]>0).sum() > 0):
            del_inds = torch.nonzero(l_orders[0] == 0, as_tuple=False)
            l_del_masks = torch.tensor(gt_masks[0]).type_as(img)[del_inds]
            l_del_mask_sum = l_del_masks.sum(dim=0, keepdim=True)
            l_v_mask = (l_del_mask_sum == 0).type_as(img)
            del_objs = []
            if g_img is None:
                g_img = img
            for del_mask, del_ind in zip(l_del_masks, del_inds):
                del_mask = del_mask.type_as(img).view(1, -1, img.size(2), img.size(3))
                del_mask = torch.where(del_mask > 0, del_mask, -1 * torch.ones_like(del_mask))
                del_obj = torch.cat([g_img, del_mask], dim=1)
                del_objs.append((del_obj, del_ind[0]))
            # img, _, _, _, _, _ = self._parse_completion_data(f_rgbs, f_depths, gt_masks, gt_labels, l_orders)

            m_img = img*l_v_mask
            g_img = self.rgb_completion.forward(m_img, l_v_mask)
            co_results.append((g_img[-1], del_objs))
            l_orders[0] -=1
            g_img = torch.where(l_v_mask > 0, img, g_img[-1])
            img = g_img

        return co_results

    def forward_test_end_gt(self,
                         img,
                         img_meta,
                         proposals=None,
                         rescale=False,
                         gt_labels=None,
                         gt_bboxes=None,
                         gt_masks=None,
                         l_orders=None,
                         p_orders=None,
                         f_rgbs=None,
                         pre_layer=0,
                         iters=10,
                         occ_thr=0.5,
                         score_thr=0.5):
        """end-to-end testing without any gt"""
        flag = True
        de_results = []
        co_results = []
        while(flag):
            # scene decomposition
            de_result = self.forward_test_decomposition(img, img_meta, proposals=proposals, rescale=rescale)
            norm_img = (img * self.img_std + self.img_mean) / 128 - 1  # normalize to [-1, 1]
            img_shape = img_meta[0]['img_shape']
            ori_shape = img_meta[0]['ori_shape']
            ori_img = F.interpolate(norm_img[:,:,:img_shape[0], :img_shape[1]], (ori_shape[0], ori_shape[1]))
            flag_check = iter_flag(de_result, score_thr=score_thr)
            select_de_result = select_result_by_order(de_result, order_thr=occ_thr, score_thr=score_thr)

            bbox_results, segm_results, occ_results = select_de_result
            bbox = np.vstack(bbox_results)
            del_objs = []

            if f_rgbs is not None:
                del_inds = match_result_to_gt(gt_labels[:-1], gt_masks[:-1], gt_bboxes[:-1], l_orders, select_de_result)
            else:
                del_inds = match_result_to_gt(gt_labels, gt_masks, gt_bboxes, l_orders, select_de_result, mask_thr=0.4, bbox_thr=0.5)

            if del_inds.sum() == 0 or pre_layer == (iters - 1):
                flag = False
            else:
                # decoder the mask to mask the object out
                masks = torch.tensor(gt_masks[del_inds.cpu().numpy()])
                l_v_mask = (masks.sum(dim=0) == 0).view(1, -1, ori_shape[0], ori_shape[1]).type_as(img)
                if len(masks) == 0:
                    flag = False
                else:
                    de_result_order = add_pre_layer(select_de_result, pre_layer=pre_layer)
                    for mask in masks:
                        mask = mask.view(1, -1, ori_shape[0], ori_shape[1]).type_as(norm_img)
                        mask = torch.where(mask > 0, mask, -1 * torch.ones_like(mask))
                        del_obj = torch.cat([ori_img, mask], dim=1)
                        del_objs.append(del_obj)
            flag = flag_check if flag else flag
            if not flag:
                de_result_order = add_pre_layer(de_result, pre_layer=pre_layer)
                l_v_mask = torch.ones_like(ori_img).chunk(3, dim=1)[0]
            m_img = l_v_mask * ori_img
            # scene completion
            g_img = self.rgb_completion.forward(m_img, l_v_mask)
            # g_img = [torch.zeros_like(ori_img)]
            # update results
            norm_img[:, :, :img_shape[0], :img_shape[1]] = F.interpolate(
                (1 - l_v_mask) * g_img[-1] + l_v_mask * ori_img, (img_shape[0], img_shape[1]))
            img = ((norm_img + 1) * 128 - self.img_mean) / self.img_std
            de_results.append(de_result_order)
            co_results.append((g_img[-1], del_objs))
            pre_layer +=1

        de_results = collect_layer_results(de_results)

        return de_results, co_results

    def forward_test_end(self,
                         img,
                         img_meta,
                         proposals=None,
                         rescale=False,
                         pre_layer=0,
                         iters=10,
                         occ_thr=0.5,
                         score_thr=0.5):
        """end-to-end testing without any gt"""
        flag = True
        de_results = []
        co_results = []
        proposal_list = []
        while(flag):
            # scene decomposition
            de_result = self.forward_test_decomposition(img, img_meta, proposals=proposals, rescale=rescale)
            bbox = np.vstack(de_result[0])
            if pre_layer == 0 and bbox.shape[0] > 0:
                bbox[:, :-1] = bbox[:, :-1] * img_meta[0]['scale_factor']
                proposal_list.append(torch.tensor(bbox).to(img.device))
                proposals = proposal_list
            norm_img = (img * self.img_std + self.img_mean) / 128 - 1  # normalize to [-1, 1]
            img_shape = img_meta[0]['img_shape']
            ori_shape = img_meta[0]['ori_shape']
            ori_img = F.interpolate(norm_img[:,:,:img_shape[0], :img_shape[1]], (ori_shape[0], ori_shape[1]))
            flag_check = iter_flag(de_result, score_thr=score_thr)
            select_de_result = select_result_by_order(de_result, order_thr=occ_thr, score_thr=score_thr)

            bbox_results, segm_results, occ_results = select_de_result
            bbox = np.vstack(bbox_results)
            del_objs = []
            if bbox.shape[0] == 0 or pre_layer == (iters - 1):
                flag = False
                l_v_mask = torch.ones_like(ori_img).chunk(3, dim=1)[0]
            else:
                # decoder the mask to mask the object out
                l_v_mask, masks = rle2mask(select_de_result, score_thr=score_thr)
                l_v_mask = l_v_mask.view(1, -1, ori_shape[0], ori_shape[1]).type_as(img)
                if len(masks) == 0:
                    flag = False
                else:
                    de_result_order = add_pre_layer(select_de_result, pre_layer=pre_layer)
                    for mask in masks:
                        mask = mask.view(1, -1, ori_shape[0], ori_shape[1]).type_as(norm_img)
                        mask = torch.where(mask > 0, mask, -1 * torch.ones_like(mask))
                        del_obj = torch.cat([ori_img, mask], dim=1)
                        del_objs.append(del_obj)
            flag = flag_check if flag else flag
            if not flag:
                de_result_order = add_pre_layer(de_result, pre_layer=pre_layer)
            m_img = l_v_mask * ori_img
            # scene completion
            g_img = self.rgb_completion.forward(m_img, l_v_mask)
            # g_img = [torch.zeros_like(ori_img)]
            # update results
            norm_img[:, :, :img_shape[0], :img_shape[1]] = F.interpolate(
                (1 - l_v_mask) * g_img[-1] + l_v_mask * ori_img, (img_shape[0], img_shape[1]))
            img = ((norm_img + 1) * 128 - self.img_mean) / self.img_std
            de_results.append(de_result_order)
            # co_results.append((g_img[-1], del_objs))
            co_results.append((F.interpolate(norm_img[:, :, :img_shape[0], :img_shape[1]], (ori_shape[0], ori_shape[1])), del_objs))
            pre_layer +=1

        de_results = collect_layer_results(de_results)

        return de_results, co_results

    def show_result(self, data, result, dataset=None, score_thr=0.3, pre_order=True, show=False, out_file=None):
        segm_result, occ_result = None, None
        if isinstance(result, tuple):
            if len(result) == 3:
                bbox_result, segm_result, occ_result = result
            else:
                bbox_result, segm_result = result
        else:
            bbox_result= result

        if isinstance(data['img_meta'], list):
            img_tensor = data['img'][0]
        else:
            img_tensor = data['img'].data[0]
        if isinstance(data['img_meta'], list):
            img_metas = data['img_meta'][0].data[0]
        else:
            img_metas = data['img_meta'].data[0]
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
            bboxes[:, :-1] = bboxes[:, :4] * img_meta['scale_factor']
            if occ_result is not None:
                occs = np.vstack(occ_result)
                if occs.shape[1] == 3:
                    occ_labels = np.argmax(occs[:,:-1], axis=1)
                else:
                    occ_labels = np.argmax(occs, axis=1)
                if pre_order:
                    occ_labels = occs[:,-1].astype(np.int64)
            else:
                occ_labels = None
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i])
                    mask = mmcv.imresize(mask, (w, h)).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
                    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE,
                                                           cv2.CHAIN_APPROX_SIMPLE)
                    img_show = cv2.drawContours(img_show, contours, -1, (int(color_mask[0][0]), int(color_mask[0][1]), int(color_mask[0][2])), 3)
            # build save files
            if out_file is not None:
                file_name = img_metas[0]['filename']
                content = file_name.split('/')
                mkdirs(out_file)
                out_file = out_file + '/' + content[-3] + '_' + content[-1]
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                occ_labels=occ_labels,
                class_names=class_names,
                score_thr=score_thr,
                show=show,
                out_file=out_file)