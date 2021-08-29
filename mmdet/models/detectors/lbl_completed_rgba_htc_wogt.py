from __future__ import division

import copy
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .. import builder
from ..registry import DETECTORS
from .base import *
from .lbl_completed_rgba_htc import LBLCompletedRGBHTC
from mmdet.datasets.layer_data_prepare import *

from mmdet.core import (bbox2result, bbox2roi, build_assigner, build_sampler,
                        merge_aug_masks, occ2result, imshow_det_bboxes, mkdirs)


@DETECTORS.register_module
class LBLCompletedRGBHTCWoGT(LBLCompletedRGBHTC):

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
        super(LBLCompletedRGBHTCWoGT, self).__init__(mode, num_stages, backbone, neck, shared_head, rpn_head,
                 bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, occ_roi_extractor, occ_head, rgb_completion,
                 semantic_roi_extractor, semantic_head, semantic_fusion, interleaved, mask_info_flow, train_cfg,
                 test_cfg,pretrained,)

        if rgb_completion is not None:
            self.rgb_completion_synthesis = copy.deepcopy(self.rgb_completion)

    @property
    def with_completion_wogt(self):
        return True

    def _parse_completion_data_wogt(self, ori_img, masks, l_orders):
        """parse data for completion network without gt"""
        l_v_masks = []
        for (mask, l_order) in zip(masks, l_orders):
            mask = torch.tensor(mask).unsqueeze(1)
            del_inds = torch.nonzero(l_order == 0, as_tuple=True)
            l_del_mask = mask[del_inds]
            l_del_mask_sum = l_del_mask.sum(dim=0, keepdim=True)
            l_v_mask = (l_del_mask_sum == 0).view(l_del_mask_sum.size(2), l_del_mask_sum.size(3))
            l_v_mask = l_v_mask.detach().cpu().numpy().astype(np.uint8)*255
            dila = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            dilation = cv2.dilate(255 - l_v_mask, dila, iterations=1)
            l_v_mask = torch.tensor(dilation == 0).view(1, 1, mask.size(2), mask.size(3)).type_as(ori_img)
            l_v_masks.append(l_v_mask)
        l_v_masks = torch.cat(l_v_masks)
        l_sce_rgbs = (ori_img * self.img_std + self.img_mean) / 128 - 1

        # visualize the training data
        self.visualize_img(l_sce_rgbs, 'layer_rgb_truth')
        self.visualize_img(l_v_masks, 'layer_v_mask', mean=[0], std=[1])

        return l_sce_rgbs, l_v_masks

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
        # pseudo ground truth
        with torch.no_grad():
            pse_g_imgs = self.rgb_completion_synthesis.forward(in_masked_imgs, mask=in_v_masks)
            self.visualize_img(pse_g_imgs[-1], 'pseudo_output_img')

        # get scale ground truth
        scale_gt = self.rgb_completion.get_scale_img(in_imgs, len(g_imgs))
        scale_mask = self.rgb_completion.get_scale_img(in_v_masks, len(g_imgs))

        # get the gan loss for the discriminator
        D_loss = self.rgb_completion.optimizer_d(scale_gt, g_imgs)
        for name, value in D_loss.items():
            self.visualize_value(value.data, 'Loss/train_co_'+str(name))

        # get the loss for the generation model
        G_loss = self.rgb_completion.G_loss(pse_g_imgs, g_imgs, scale_mask)
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
            del_inds, masks = match_result_to_gt(gt_label, gt_mask, gt_bbox, l_order, result, mask_thr=0.5, bbox_thr=0.6)
            l_pre_masks.append(torch.cat(masks).unsqueeze(1).sum(dim=0, keepdim=True)==0)
            update_order = l_order_update(l_order, del_inds, p_order)
            update_orders.append(update_order)
        l_pre_masks = torch.cat(l_pre_masks).type_as(imgs)
        obj_orders = [item + 1 for item in update_orders]
        # get batch training data for completion network without ground truth
        l_sce_rgbs, l_v_masks = self._parse_completion_data_wogt(ori_imgs, gt_masks, obj_orders)
        l_sce_labels = gt_semantic_seg
        if self.epoch > 10:
            l_v_masks[:,:,:l_pre_masks.size(2),:l_pre_masks.size(3)] = l_pre_masks
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