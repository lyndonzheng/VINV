import numpy as np
from mmdet.datasets.visualize import KINS
from mmdet.core.utils import pairwise_ranking

from .custom import CustomDataset
from .registry import DATASETS
from pycocotools import mask as maskUtils

@DATASETS.register_module
class KinsDataset(CustomDataset):

    CLASSES = ('cyclist', 'pedestrian', 'car',
               'tram', 'truck', 'van', 'misc')

    def load_annotations(self, ann_file):
        self.kins = KINS(ann_file)
        self.cat_ids = [1, 2, 4, 5, 6, 7, 8]
        self.cat2label = {
            cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.kins.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.kins.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.kins.getAnnIds(imgIds=[img_id])
        ann_info = self.kins.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.kins.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        # shared annotations for visible and full objects
        gt_labels = []
        gt_layer_orders = []
        gt_pair_orders = []
        gt_bboxes_ignore = []
        # visible annotations
        gt_v_bboxes = []
        gt_v_masks = []
        # full annotations
        gt_f_bboxes = []
        gt_f_masks = []
        w, h = img_info['width'], img_info['height']

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            if 'inmodal_bbox' in ann.keys(): # processing the old kins dataset
                ann['a_bbox'] = ann['bbox']
                ann['i_bbox'] = ann['inmodal_bbox']
            x1, y1, w1, h1 = ann['a_bbox']
            x2, y2, w2, h2 = ann['i_bbox']
            if w1 < 1 or h1 < 1 or w2 < 1 or h2 < 1:
                continue
            f_bbox = [x1, y1, x1 + w1 - 1, y1 + h1 - 1]
            gt_f_bboxes.append(f_bbox)
            v_bbox = [x2, y2, x2 + w2 - 1, y2 + h2 - 1]
            gt_v_bboxes.append(v_bbox)

            gt_labels.append(self.cat2label[ann['category_id']])
            if 'layer_order' in ann.keys():
                ann['ico_id'] = ann['layer_order']
            gt_layer_orders.append(ann['ico_id'])
            if 'inmodal_seg' in ann.keys():
                ann['i_segm'] = ann['inmodal_seg']
            gt_v_masks.append(ann['i_segm'])
            if 'segmentation' in ann.keys():
                gt_f_masks.append(ann['segmentation'])
            else:
                # decide the amodal mask first to get pairwise order
                rles = maskUtils.frPyObjects(ann['a_segm'], h, w)
                rle = maskUtils.merge(rles)
                f_mask = maskUtils.decode(rle).squeeze()
                gt_f_masks.append(f_mask)
            if 'pair_order' in ann.keys():
                gt_pair_orders.append(list(ann['pair_order'].values()))

        if 'pair_order' not in ann.keys():
            gt_pair_orders = pairwise_ranking(gt_f_masks, gt_layer_orders)

        if gt_f_bboxes:
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_layer_orders = np.array(gt_layer_orders, dtype=np.int64)
            gt_pair_orders = np.array(gt_pair_orders, dtype=np.int64)
            gt_v_bboxes = np.array(gt_v_bboxes, dtype=np.float32)
            gt_f_bboxes = np.array(gt_f_bboxes, dtype=np.float32)
        else:
            gt_labels = np.array([], dtype=np.int64)
            gt_layer_orders = np.array([], dtype=np.int64)
            gt_pair_orders = np.array([], dtype=np.int64)
            gt_v_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_f_bboxes = np.zeros((0, 4), dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_v_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_v_masks,
            f_bboxes=gt_f_bboxes,
            f_masks=gt_f_masks,
            l_orders=gt_layer_orders,
            p_orders=gt_pair_orders,
            cat2label=self.cat2label
        )

        return ann