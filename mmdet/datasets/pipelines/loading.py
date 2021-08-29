import os.path as osp
import warnings

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmcv.image.transforms.colorspace import bgr2rgb

from ..registry import PIPELINES

@PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        filename = filename.replace('rgb', 're_rgb')
        img = mmcv.imread(filename)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=False,
                 with_depth=False,
                 with_f_bbox=False,
                 with_f_mask=False,
                 with_f_rgb=False,
                 with_f_depth=False,
                 with_l_order=False,
                 with_p_order=False,
                 skip_img_without_anno=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.with_depth = with_depth
        self.with_f_bbox = with_f_bbox
        self.with_f_mask = with_f_mask
        self.with_f_rgb = with_f_rgb
        self.with_f_depth = with_f_depth
        self.with_l_order = with_l_order
        self.with_p_order = with_p_order
        self.skip_img_without_anno = skip_img_without_anno

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']
        if len(results['gt_bboxes']) == 0 and self.skip_img_without_anno:
            if results['img_prefix'] is not None:
                file_path = osp.join(results['img_prefix'],
                                     results['img_info']['filename'])
            else:
                file_path = results['img_info']['filename']
            warnings.warn(
                'Skip the image "{}" that has no valid gt bbox'.format(
                    file_path))
            return None

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_f_bboxes(self, results):
        results['gt_f_bboxes'] = results['ann_info']['f_bboxes']
        results['bbox_fields'].append('gt_f_bboxes')
        return results

    def _load_depth(self, results):
        filename = osp.join(results['img_prefix'], results['img_info']['depth_name'])
        depth = mmcv.imread(filename, flag='unchanged').astype(np.float32) / 1000.0
        results['depth'] = depth
        return results

    def _load_f_depths(self, results):
        f_depths = results['ann_info']['f_depths']
        depths = []
        for f_depth in f_depths:
            filename = osp.join(results['img_prefix'], f_depth)
            depth = mmcv.imread(filename, flag='unchanged').astype(np.float32) / 1000.0
            depths.append(depth)
        results['f_depths'] = depths
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _load_l_orders(self, results):
        results['l_orders'] = results['ann_info']['l_orders']
        return results

    def _load_p_orders(self, results):
        results['p_orders'] = results['ann_info']['p_orders']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, np.ndarray):
            mask = mask_ann
        else:
            if isinstance(mask_ann, list):
                # polygon -- a single object might consist of multiple parts
                # we merge all parts into one mask rle code
                rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
                rle = maskUtils.merge(rles)
            elif isinstance(mask_ann['counts'], list):
                # uncompressed RLE
                rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            else:
                # rle
                rle = mask_ann
            mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        else:
            masks = []
            for gt_mask in gt_masks:
                gt_mask = osp.join(results['img_prefix'], gt_mask)
                mask = mmcv.imread(gt_mask)[:, :, -1]
                masks.append((mask / 255).astype(np.uint8))
            gt_masks = masks
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_f_rgb_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        if 'f_masks' in results['ann_info']:
            f_masks = results['ann_info']['f_masks']
            f_masks = [self._poly2mask(mask, h, w) for mask in f_masks]
        else:
            f_rgb_masks = results['ann_info']['f_rgbs']
            f_rgbs = []
            f_masks = []
            for f_rgb_mask in f_rgb_masks:
                filename = osp.join(results['img_prefix'], f_rgb_mask)
                f_rgb_mask_np = mmcv.imread(filename, flag='unchanged')
                f_masks.append((f_rgb_mask_np[:,:,3]>0).astype(np.uint8))
                f_rgbs.append(bgr2rgb(f_rgb_mask_np[:,:,:-1]))
            results['f_rgbs'] = f_rgbs
        results['gt_f_masks'] = f_masks
        results['mask_fields'].append('gt_f_masks')
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        if self.with_depth:
            results = self._load_depth(results)
        if self.with_f_depth:
            results = self._load_f_depths(results)
        if self.with_f_bbox:
            results = self._load_f_bboxes(results)
        if self.with_f_mask or self.with_f_rgb:
            results = self._load_f_rgb_masks(results)
        if self.with_l_order:
            results = self._load_l_orders(results)
        if self.with_p_order:
            results = self._load_p_orders(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str


@PIPELINES.register_module
class LoadProposals(object):

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(num_max_proposals={})'.format(
            self.num_max_proposals)
