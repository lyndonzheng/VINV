import numpy as np
from mmdet.core.utils import pairwise_ranking, pairwise_order_area, pairwise_order_yaxis, infer_gt_order
from collections import defaultdict
from .csd_eval import CSDeval, ParamsCsd
from pycocotools import mask as maskUtils


class KINSeval(CSDeval):
    # Interface for evaluating detection on the KINS dataset
    def __init__(self, kinsGt=None, kinsDt=None, iouType='segm', order_method='depth', fullMask=True, fullBbox=True):
        """
        Initialize CocoaEval using cocoa APIs for gt and dt
        :param kinsGt: kins object with ground truth annotations
        :param kinsDt: kins object with predicted results
        :param iouType: set iouType to 'segm', 'bbox' or 'keypoints'
        :return: None
        """
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.kinsGt = kinsGt               # ground truth COCO API
        self.kinsDt = kinsDt               # detections COCO API
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
        self.order_method = order_method  # depth flag
        if not kinsGt is None:
            self.params.imgIds = sorted(kinsGt.getImgIds())
            self.params.catIds = sorted(kinsGt.getCatIds())

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''

        def _toMask(anns, kins):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = kins.annToRLE(ann)
                ann['segmentation'] = rle

        def _gtMask(anns, kins):
            # convert ground truth polygon to mask
            for ann in anns:
                if self.fullMask:
                    ann['segmentation'] = ann['a_segm'] if 'a_segm' in ann.keys() else ann['segmentation']
                else:
                    ann['segmentation'] = ann['i_segm'] if 'i_segm' in ann.keys() else ann['inmodal_seg']
                if isinstance(ann['segmentation'], list):
                    rle = kins.annToRLE(ann)
                    ann['segmentation'] = rle

        def _getBbox(anns):
            # get the ground truth visible or full bbox
            for ann in anns:
                if self.fullBbox:
                    ann['bbox'] = ann['a_bbox'] if 'a_bbox' in ann.keys() else ann['bbox']
                else:
                    ann['bbox'] = ann['i_bbox'] if 'i_bbox' in ann.keys() else ann['inmodal_bbox']

        def _addAnn(anns):
            # add ignore and crowd to the ann for coco evaluation
            for ann in anns:
                ann['ignore'] = 0
                ann['area'] = ann['i_area'] if 'i_area' in ann.keys() else ann['area']

        def _gtOrder(imgIds):
            # get pairwise order from anotations
            for imgId in imgIds:
                anns = self.kinsGt.loadAnns(self.kinsGt.getAnnIds(imgIds=[imgId]))
                f_masks = []
                layer_orders = []
                for ann in anns:
                    if 'layer_order' in ann.keys():
                        ann['ico_id'] = ann['layer_order']
                    layer_orders.append(ann['ico_id'])
                    f_masks.append(maskUtils.decode(ann['segmentation']))
                self.gt_p_orders[imgId] = pairwise_ranking(f_masks, layer_orders)

        def _dtOrder(imgIds):
            for imgId in imgIds:
                anns = self.kinsDt.loadAnns(self.kinsDt.getAnnIds(imgIds=[imgId]))
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
                    anns = self.kinsGt.loadAnns(self.kinsGt.getAnnIds(imgIds=[imgId]))
                    v_masks = []
                    v_rles = []
                    for ann in anns:
                        rle = ann['i_segm'] if 'i_segm' in ann.keys() else ann['inmodal_seg']
                        v_rles.append(rle)
                        v_mask = maskUtils.decode(rle)
                        v_masks.append(v_mask)
                    iou = maskUtils.iou(v_rles, f_rles, [0 for i in v_rles])
                    inds = np.argmax(iou, axis=0)
                    v_masks_new = []
                    for i in inds:
                        v_masks_new.append(v_masks[i])
                    self.dt_p_orders[imgId] = infer_gt_order(v_masks_new, f_masks)
                elif self.order_method == 'area':
                    self.dt_p_orders[imgId] = pairwise_order_area(f_masks, above='larger')
                elif self.order_method == 'yaxis':
                    self.dt_p_orders[imgId] = pairwise_order_yaxis(f_masks)
                else:
                    raise Exception('No such order method: {}'.format(self.args.order_method))

        p = self.params
        if p.useCats:
            gts = self.kinsGt.loadAnns(self.kinsGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.kinsDt.loadAnns(self.kinsDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.kinsGt.loadAnns(self.kinsGt.getAnnIds(imgIds=p.imgIds))
            dts = self.kinsDt.loadAnns(self.kinsDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm' or p.iouType == 'depth':
            _gtMask(gts, self.kinsGt)
            _toMask(dts, self.kinsDt)
        _getBbox(gts)
        _addAnn(gts)
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        if p.iouType == 'depth':
            self.gt_p_orders = defaultdict()  # gt pairwise order for evaluation
            self.dt_p_orders = defaultdict()  # dt pairwise order for evaluation
            _gtOrder(p.imgIds)
            _dtOrder(p.imgIds)
        for gt in gts:
            # if gt['category_id'] == 0:
            # continue
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}