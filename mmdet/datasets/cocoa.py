import numpy as np
from mmdet.datasets.visualize import COCOA

from .custom import CustomDataset
from .registry import DATASETS
from pycocotools import mask as maskUtils
from mmdet.core.utils import layer_ranking

@DATASETS.register_module
class CocoaDataset(CustomDataset):

    CLASSES = ('thing', 'stuff')

    # CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #            'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
    #            'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
    #            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #            'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
    #            'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
    #            'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #            'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #            'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
    #            'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
    #            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #            'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def load_annotations(self, ann_file):
        self.cocoa = COCOA(ann_file)
        self.cat_ids = [i for i in range(len(self.CLASSES))]
        self.cat2label = {
            cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.cocoa.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.cocoa.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.cocoa.getAmodalAnnIds(imgIds=[img_id])
        ann_info = self.cocoa.loadAnns(ann_ids)[0]
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths"""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.cocoa.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def mask_to_bbox(self, mask):
        """
        get bbox from polygon segmentation mask
        :param mask: polygon segmentation mask
        :return: bbox with xywh size
        """
        mask = (mask == 1)
        if np.all(~mask):
            return 0, 0, 0, 0
        assert len(mask.shape) == 2
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return cmin.item(), rmin.item(), cmax.item() + 1 - cmin.item(), rmax.item() + 1 - rmin.item()  # xywh

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
        object_ind = []
        w, h = img_info['width'], img_info['height']

        for i, ann in enumerate(ann_info['regions']):
            if ann['isStuff']:
                object_ind.append(False)
                continue
                # gt_labels.append(2)
                # gt_bboxes_ignore.append(f_bbox)
            else:
                object_ind.append(True)
                gt_labels.append(1)
            # parse the bbox and mask
            rles = maskUtils.frPyObjects([ann['segmentation']], h, w)
            rle = maskUtils.merge(rles)
            f_mask = maskUtils.decode(rle).squeeze()
            if 'visible_mask' in ann.keys():
                rle = [ann['visible_mask']]
                v_mask = maskUtils.decode(rle).squeeze()
            else:
                v_mask = f_mask

            x1, y1, w1, h1 = self.mask_to_bbox(f_mask)
            x2, y2, w2, h2 = self.mask_to_bbox(v_mask)
            if w1 == 0 or h1 == 0: # cocoa consider the full occluded object
                f_bbox = [x1, y1, x1 , y1]
            else:
                f_bbox = [x1, y1, x1 + w1 - 1, y1 + h1 - 1]
            if w2 == 0 or h2 == 0:
                v_bbox = [x2, y2, x2, y2 ]
            else:
                v_bbox = [x2, y2, x2 + w2 - 1, y2 + w2 - 1]
            gt_v_bboxes.append(v_bbox)
            gt_f_bboxes.append(f_bbox)
            gt_v_masks.append(v_mask)
            gt_f_masks.append(f_mask)

        # parse the orders
        num = ann_info['size']
        pair_order = np.zeros((num, num), dtype=np.int64)
        order_str = ann_info['depth_constraint']
        if len(order_str) > 0:
            order_str = order_str.split(',')
            for o in order_str:
                idx1, idx2 = o.split('-')
                idx1, idx2 = int(idx1) - 1, int(idx2) - 1
                pair_order[idx1, idx2] = 1
                pair_order[idx2, idx1] = -1
        # pair_order = pair_order[object_ind][:, object_ind]
        layer_order = layer_ranking(pair_order)
        layer_order = layer_order[object_ind]
        pair_order = pair_order[object_ind][:, object_ind]

        if gt_f_bboxes:
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_layer_orders = np.array(layer_order, dtype=np.int64)
            gt_pair_orders = np.array(pair_order, dtype=np.int64)
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