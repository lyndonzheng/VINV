import torch


from mmdet.core import force_fp32
from ..registry import ROI_EXTRACTORS
from .single_level import SingleRoIExtractor


@ROI_EXTRACTORS.register_module
class OccRoIExtractor(SingleRoIExtractor):
    """ Extract the Occlusion features from overlay ROI regions
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56
                 ):
        super(OccRoIExtractor, self).__init__(
            roi_layer=roi_layer,
            out_channels=out_channels,
            featmap_strides=featmap_strides,
            finest_scale=finest_scale)

    @force_fp32(apply_to=('feats',), out_fp16=True)
    def forward(self, feats, rois, overlay_rois, overlay_inds, roi_scale_factor=None):

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        rois_expand = []
        for i in range(0, int(rois[:, 0].max()+1)):
            inds = overlay_rois[:, 0] == i
            overlay_ind = overlay_inds[inds, 1].type(torch.long)
            rois_expand.append(rois[rois[:, 0] == i][overlay_ind])
        rois_expand = torch.cat(rois_expand)
        target_lvls = self.map_roi_levels(rois_expand, num_levels)
        roi_feats = feats[0].new_zeros(rois_expand.size(0), self.out_channels*2, *out_size)
        if roi_scale_factor is not None:
            rois_expand = self.roi_rescale(rois_expand, roi_scale_factor)
            overlay_rois = self.roi_rescale(overlay_rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois_expand[inds]
                overlay_rois_ = overlay_rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                overlay_roi_feats_t = self.roi_layers[i](feats[i], overlay_rois_)
                roi_feats[inds] = torch.cat([roi_feats_t, overlay_roi_feats_t], dim=1)

        return roi_feats