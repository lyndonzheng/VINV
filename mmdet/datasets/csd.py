import numpy as np
from mmdet.datasets.visualize.csd_tools import CSD

from .custom import CustomDataset
from .registry import DATASETS

@DATASETS.register_module
class CsdDataset(CustomDataset):

    CLASSES = ('BG', 'void', 'cabinet', 'counter', 'refridgerator',
               'desk', 'chair', 'table', 'television', 'door', 'lamp', 'window',
               'night_stand', 'dresser', 'otherprop', 'sink', 'shower_curtain', 'mirror',
               'toilet', 'floor_mat', 'sofa', 'shelves', 'bed', 'bookshelf', 'picture',
               'otherstructure', 'curtain', 'blinds', 'books', 'person', 'bathtub', 'whiteboard',
               'pillow', 'clothes')

    def load_annotations(self, ann_file):
        self.csd = CSD(ann_file)
        self.cat_ids = self.csd.getCatIds()
        self.cat2label = {
            cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.csd.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.csd.loadImgs([i])[0]
            info['filename'] = info['img_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.csd.getAnnIds(imgIds=[img_id])
        ann_info = self.csd.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['id'] for _ in self.csd.anns.values())
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
        gt_f_rgbs = []
        gt_f_depths = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w1, h1 = ann['f_bbox']
            x2, y2, w2, h2 = ann['v_bbox']
            if w1 < 4 or h1 < 4 or w2 < 2 or h2 < 2:
                continue
            f_bbox = [x1, y1, x1 + w1 - 1, y1 + h1 - 1]
            gt_f_bboxes.append(f_bbox)
            v_bbox = [x2, y2, x2 + w2 - 1, y2 + h2 - 1]
            gt_v_bboxes.append(v_bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
            gt_layer_orders.append(ann['layer_order'])
            gt_pair_orders.append(list(ann['pair_order'].values()))
            gt_v_masks.append(ann['v_mask_name'])
            gt_f_rgbs.append(ann['f_img_name'])
            gt_f_depths.append(ann['f_depth_name'])

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

        seg_map = img_info['seg_name']

        ann = dict(
            bboxes=gt_v_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_v_masks,
            seg_map=seg_map,
            f_bboxes=gt_f_bboxes,
            f_rgbs=gt_f_rgbs,
            f_depths=gt_f_depths,
            l_orders=gt_layer_orders,
            p_orders=gt_pair_orders,
            cat2label=self.cat2label
        )

        return ann