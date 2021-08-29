import torch

import pycocotools.mask as maskUtils
from mmdet.core import force_fp32
from ..registry import ROI_EXTRACTORS
from .single_level import SingleRoIExtractor


@ROI_EXTRACTORS.register_module
class RefinedRoIExtractor(SingleRoIExtractor):
    """Refine the RoI features based on the occlusion order

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.

    """
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        super(RefinedRoIExtractor, self).__init__(
            roi_layer=roi_layer,
            out_channels=out_channels,
            featmap_strides=featmap_strides,
            finest_scale=finest_scale)

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, mask_preds, p_orders, obj_inds, ori_shape, roi_scale_factor=None):

        img_h, img_w = ori_shape[:2]
        out_size = self.roi_layers[0].out_size
        _, m_w, m_h = mask_preds.size()
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = feats[0].new_zeros(rois.size(0), self.out_channels, *out_size)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i, roi in enumerate(rois):
            obj_ind = obj_inds[i]
            p_order = p_orders[int(roi[0])][obj_ind]
            occ_inds = (p_order == -1).nonzero()
            # get mask hot map
            b, _, f_w, f_h = feats[target_lvls[i]].size()
            mask_map = torch.zeros((b, 1, f_w, f_h), dtype=torch.float, device=mask_preds.device)
            x0, y0, x1, y1 = roi[1]*f_w/img_w, roi[2]*f_h/img_h, roi[3]*f_w/img_w, roi[4]*f_h/img_h
            w = int(max((x1 - x0 + 1), 1))
            h = int(max((y1 - y0 + 1), 1))
            bbox_mask = torch.nn.functional.interpolate(mask_preds[i].view(1, 1, m_w, m_h), [h, w], mode='bilinear')
            mask_map[int(roi[0]),:,int(y0):int(y0)+h, int(x0):int(x0)+w] = bbox_mask.view(h, w)
            if occ_inds.size(0) != 0:
                for occ_ind in occ_inds:
                    batch_ind = rois[:, 0] == roi[0]
                    occ_mask_inds = obj_inds[batch_ind] == occ_ind
                    occ_rois = rois[batch_ind,:][occ_mask_inds, :]
                    occ_masks = mask_preds[batch_ind,:][occ_mask_inds, :]
                    for occ_roi, occ_mask in zip(occ_rois, occ_masks):
                        x0, y0, x1, y1 = occ_roi[1] * f_w / img_w, occ_roi[2] * f_h / img_h, \
                                             occ_roi[3] * f_w / img_w, occ_roi[4] * f_h / img_h
                        w = int(max((x1 - x0 + 1), 1))
                        h = int(max((y1 - y0 + 1), 1))
                        bbox_mask = torch.nn.functional.interpolate(occ_mask.view(1, 1, m_w, m_h), [h, w], mode='bilinear')
                        mask_map[int(occ_roi[0]), :, int(y0):int(y0) + h, int(x0):int(x0) + w] = 1-bbox_mask.view(h, w)
            roi_feats_refined = self.roi_layers[target_lvls[i]](feats[target_lvls[i]]*mask_map, roi.view(1, -1))
            roi_feats[i] = roi_feats_refined
        return roi_feats