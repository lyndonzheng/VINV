import numpy as np
from mmdet.core.utils import pairwise_ranking, pairwise_order_area, pairwise_order_yaxis, infer_gt_order, layer_ranking
from collections import defaultdict
from .csd_eval import CSDeval, ParamsCsd
from pycocotools import mask as maskUtils


class COCOAeval(CSDeval):
    # Interface for evaluating detection on the COCOA dataset and ignore the class category
    def __init__(self, cocoaGt=None, cocoaDt=None, iouType='segm', order_method='depth', fullMask=True, fullBbox=True):
        """
        Initialize CocoaEval using cocoa APIs for gt and dt
        :param cocoaGt: cocoa object with ground truth annotations
        :param cocoaDt: cocoa object with predicted results
        :param iouType: set iouType to 'segm', 'bbox' or 'keypoints'
        :return: None
        """
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoaGt = cocoaGt               # ground truth COCO API
        self.cocoaDt = cocoaDt               # detections COCO API
        self.params = {}                    # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}                      # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = ParamsCsd(iouType=iouType)  # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        self.fullMask = fullMask            # mask flag
        self.fullBbox = fullBbox            # bbox flag
        self.order_method = order_method    # depth flag
        if not cocoaGt is None:
            self.params.imgIds = sorted(cocoaGt.getImgIds())
            self.params.catIds = [i for i in range(2)]

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """
        def _toMask(anns, cocoa):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = cocoa.annToRLE(ann)
                ann['segmentation'] = rle

        def _gtMaskBBox(anns, cocoa):
            # convert ground truth polygon to mask
            id = 1
            for ann in anns:
                if 'regions' in ann:
                    layer_order = self.gt_l_orders[ann['image_id']]
                    for region in ann['regions']:
                        if layer_order[region['order']-1] > -1:
                            gt = region
                            gt['image_id'] = ann['image_id']
                            gt['category_id'] = 1 if gt['isStuff'] else 0
                            if isinstance(gt['segmentation'], list):
                                gt['segmentation'] = [gt['segmentation']]
                                rle = cocoa.annToRLE(gt)
                                gt['segmentation'] = rle
                            if not self.fullMask and 'visible_mask' in gt.keys():
                                    gt['segmentation'] = gt['visible_mask']
                            gt['bbox'] = maskUtils.toBbox(gt['segmentation'])
                            gt['ignore'] = 1 if gt['isStuff'] else 0
                            gt['iscrowd'] = 0
                            gt['id'] = id
                            id = id + 1
                            self._gts[gt['image_id'], gt['category_id']].append(gt)
            print(id)

        def _dtMaskBBox(anns, cocoa):
            # convert ground truth polygon to mask
            for ann in anns:
                if 'regions' in ann:
                    for region in ann['regions']:
                        gt = region
                        gt['area'] = maskUtils.area(gt['segmentation'])
                        gt['image_id'] = ann['image_id']
                        gt['category_id'] = 1 if gt['isStuff'] else 0
                        gt['ignore'] = 1 if gt['isStuff'] or gt['score']<0.99 else 0
                        gt['iscrowd'] = 0
                        self._dts[gt['image_id'], gt['category_id']].append(gt)

        def _gtOrder(imgIds):
            # get pairwise order from anotations
            for imgId in imgIds:
                anns = self.cocoaGt.loadAnns(self.cocoaGt.getAnnIds(imgIds=[imgId]))[0]
                num = anns['size']
                pair_order = np.zeros((num, num), dtype=np.int64)
                order_str = anns['depth_constraint']
                if len(order_str) > 0:
                    order_str = order_str.split(',')
                    for o in order_str:
                        idx1, idx2 = o.split('-')
                        idx1, idx2 = int(idx1) - 1, int(idx2) - 1
                        pair_order[idx1, idx2] = 1
                        pair_order[idx2, idx1] = -1
                self.gt_p_orders[imgId] = pair_order
                self.gt_l_orders[imgId] = layer_ranking(pair_order)

        def _dtOrder(imgIds):
            for imgId in imgIds:
                anns = self.cocoaDt.loadAnns(self.cocoaDt.getAnnIds(imgIds=[imgId]))
                f_masks = []
                f_rles = []
                layer_orders = []
                for ann in anns:
                    if 'layer' in ann.keys():
                        if len(ann['layer']) == 3:
                            layer_orders.append(ann['layer'][-1])
                        else:
                            layer_orders.append(np.argmax(ann['layer']))
                    f_rles.append(ann['segmentation'])
                    f_masks.append(maskUtils.decode(ann['segmentation']))
                if self.order_method == 'depth':
                    self.dt_p_orders[imgId] = pairwise_ranking(f_masks, layer_orders)
                elif self.order_method == 'iou':
                    anns = self.cocoaGt.loadAnns(self.cocoaGt.getAnnIds(imgIds=[imgId]))[0]
                    v_masks = []
                    v_rles = []
                    for region in anns['regions']:
                        rle = region['visible_mask'] if 'visible_mask' in region.keys() else region['segmentation']
                        v_rles.append(rle)
                        v_mask = maskUtils.decode(rle)
                        v_masks.append(v_mask)
                    if len(f_rles) == 0:
                        self.dt_p_orders[imgId] = np.zeros((0, 0), dtype=np.int64)
                    else:
                        iou = maskUtils.iou(v_rles, f_rles, [0 for i in v_rles])
                        inds = np.argmax(iou, axis=0)
                        v_masks_new = []
                        for i in inds:
                            v_masks_new.append(v_masks[i])
                        self.dt_p_orders[imgId] = infer_gt_order(v_masks_new, f_masks)
                elif self.order_method == 'area':
                    self.dt_p_orders[imgId] = pairwise_order_area(f_masks, above='smaller')
                elif self.order_method == 'yaxis':
                    self.dt_p_orders[imgId] = pairwise_order_yaxis(f_masks)
                else:
                    raise Exception('No such order method: {}'.format(self.order_method))

        p = self.params
        gts = self.cocoaGt.loadAnns(self.cocoaGt.getAnnIds(imgIds=p.imgIds))
        dts = self.cocoaDt.loadAnns(self.cocoaDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm' or p.iouType == 'depth':
            _toMask(dts, self.cocoaDt)
        if p.iouType == 'depth':
            self.dt_p_orders = defaultdict()  # dt pairwise order for evaluation
            _dtOrder(p.imgIds)
        self.gt_p_orders = defaultdict()  # gt pairwise order for evaluation
        self.gt_l_orders = defaultdict()
        _gtOrder(p.imgIds)
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        _gtMaskBBox(gts, self.cocoaGt)
        _dtMaskBBox(dts, self.cocoaDt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}