import torch
import torch.nn as nn

import pycocotools.mask as maskUtils
from mmdet.core import force_fp32
from ..registry import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module
class LayeredFeatsExtractor(nn.Module):
    def __init__(self):
        super(LayeredFeatsExtractor, self).__init__()

        self.fp16_enabled = False

    def forward(self, feats, v_masks, f_masks, l_orders):
        """extract layered features"""

        layered_feats = []

        for i, (v_mask, f_mask, l_order) in enumerate(zip(v_masks, f_masks, l_orders)):
            feat = feats[i]
            _, f_w, f_h = feat.size()
            v_mask = torch.nn.functional.interpolate(v_mask.unsqueeze(0), [f_w, f_h], mode='bilinear')
            f_mask = torch.nn.functional.interpolate(f_mask.unsqueeze(0), [f_w, f_h], mode='bilinear')
            for j in range(l_order.max() + 1):
                inds = l_order == j
                if inds.any():
                    v_mask_ = v_mask[:,inds]
                    f_mask_ = f_mask[:,inds]
                    layered_v_mask = v_mask_.sum(dim=1)
                    layered_f_mask = f_mask_.sum(dim=1)
                    layered_feat = torch.where(layered_v_mask > 0, feat, torch.zeros_like(feat))
                    layered_feat = torch.where(layered_f_mask > 0, layered_feat, torch.ones_like(feat))
                    layered_feats.append(layered_feat.unsqueeze(0))
            layered_v_mask = v_mask.sum(dim=1)
            layered_feat = torch.where(layered_v_mask > 0, feat, torch.zeros_like(feat))
            layered_feats.append(layered_feat.unsqueeze(0))

        layered_feats = torch.cat(layered_feats)

        return layered_feats


