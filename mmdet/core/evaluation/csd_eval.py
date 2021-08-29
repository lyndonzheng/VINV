import numpy as np
import os
import mmcv
import datetime
import time
import copy
from mmdet.core.utils import pairwise_ranking, eval_order, pairwise_order_area, pairwise_order_yaxis,infer_gt_order
from collections import defaultdict
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval, Params


class CSDeval(COCOeval):
    def __init__(self, csdGt=None, csdDt=None, iouType='segm', order_method='depth',
                 fullMask=True, fullBbox=True, imgPrefix='csd_new/'):
        """
        Initialize CSDeval using csd APIs for gt and dt
        :param dataroot: data root the visible and full mask
        :param csdGt: csd object with ground truth annotations
        :param csdDt: csd object with detection results
        :param iouType:
        :param fullMask: evaluate with full mask or visible mask
        :param fullBbox: evaluate with full bbox or visible bbox
        """
        if not iouType:
            print("iouType not specified, use default iouType segm")
        self.csdGt = csdGt                # ground truth CSD API
        self.csdDt = csdDt                # detections CSD API
        self.params = {}                    # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KXAXI] elements
        self.eval = {}                      # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = ParamsCsd(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result sumarization
        self.ious = {}                      # ious between all gts and dts
        self.imgPrefix=imgPrefix            # img root for the binary mask
        self.fullMask = fullMask            # mask flag
        self.fullBbox = fullBbox            # bbox flag
        self.order_method = order_method    # depth flag
        if not csdGt is None:
            self.params.imgIds = sorted(csdGt.getImgIds())
            self.params.catIds = sorted(csdDt.getCatIds())

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """
        def _toMask(anns, csd):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = csd.annToRLE(ann)
                ann['segmentation'] = rle

        def _imgToMask(anns):
            # convert ground truth image to mask
            for ann in anns:
                if self.fullMask:
                    # full mask
                    filename = os.path.join(self.imgPrefix, ann['f_img_name'])
                    f_rgba = mmcv.imread(filename, flag='unchanged')
                    f_mask = (f_rgba[:, :, 3] > 0).astype(np.uint8)
                    rle = maskUtils.encode(np.array(f_mask[:, :, np.newaxis], order='F'))[0]
                else:
                    # visible mask
                    filename = os.path.join(self.imgPrefix, ann['v_mask_name'])
                    v_mask = (mmcv.imread(filename)[:, :, -1] / 255).astype(np.uint8)
                    rle = maskUtils.encode(np.array(v_mask[:, :, np.newaxis], order='F'))[0]
                ann['segmentation'] = rle
                ann['area'] = maskUtils.area(ann['segmentation'])

        def _getBbox(anns):
            # get the ground truth visible or full bbox
            for ann in anns:
                if self.fullBbox:
                    ann['bbox'] = ann['f_bbox']
                else:
                    ann['bbox'] = ann['v_bbox']
                ann['area'] = ann['bbox'][2] * ann['bbox'][3]

        def _addAnn(anns):
            # add ignore and crowd to the ann for coco evaluation
            for ann in anns:
                ann['ignore'] = 0
                ann['iscrowd'] = 0

        def _gtOrder(imgIds):
            # get pairwise order from annotations
            for imgId in imgIds:
                anns = self.csdGt.loadAnns(self.csdGt.getAnnIds(imgIds=[imgId]))
                gt_pair_orders = []
                for ann in anns:
                    gt_pair_orders.append(list(ann['pair_order'].values()))
                gt_pair_orders = np.array(gt_pair_orders, dtype=np.int8)
                self.gt_p_orders[imgId] = gt_pair_orders

        def _dtOrder(imgIds):
            for imgId in imgIds:
                anns = self.csdDt.loadAnns(self.csdDt.getAnnIds(imgIds=[imgId]))
                f_masks = []
                f_rles =[]
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
                    anns = self.csdGt.loadAnns(self.csdGt.getAnnIds(imgIds=[imgId]))
                    v_masks = []
                    v_rles = []
                    for ann in anns:
                        filename = os.path.join(self.imgPrefix, ann['v_mask_name'])
                        v_mask = (mmcv.imread(filename)[:, :, -1] / 255).astype(np.uint8)
                        v_masks.append(v_mask)
                        rle = maskUtils.encode(np.array(v_mask[:, :, np.newaxis], order='F'))[0]
                        v_rles.append(rle)
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
                    raise Exception('No such order method: {}'.format(self.args.order_method))

        p = self.params
        new_Ids = []
        for i in p.imgIds:
            gt = self.csdGt.loadAnns(self.csdGt.getAnnIds(imgIds=[i]))
            if len(gt) > 19:
                new_Ids.append(i)
        p.imgIds = new_Ids
        print(len(p.imgIds))
        if p.useCats:
            gts = self.csdGt.loadAnns(self.csdGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.csdDt.loadAnns(self.csdDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.csdGt.loadAnns(self.csdGt.getAnnIds(imgIds=p.imgIds))
            dts = self.csdDt.loadAnns(self.csdDt.getAnnIds(imgIds=p.imgIds))
        # convert the full or visible bbox to 'bbox'
        print(len(gts))
        _getBbox(gts)
        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm' or p.iouType == 'depth':
            _imgToMask(gts)
            _toMask(dts, self.csdDt)
        _addAnn(gts)
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        if p.iouType == 'depth':
            self.gt_p_orders = defaultdict()  # gt pairwise order for evaluation
            self.dt_p_orders = defaultdict()  # dt pairwise order for evaluation
            _gtOrder(p.imgIds)
            _dtOrder(p.imgIds)
        for gt in gts:
            if gt['category_id'] == 0:
                continue
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p
        # prepare the gt and predication based on the category
        self._prepare()

        if p.useCats == 0 or p.iouType == 'depth':
            # ignore the category for depth order evaluation
            p.useCats = 0
        else:
            p.useCats = 1
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        self.params = p
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox' or p.iouType == 'depth':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm' or p.iouType == 'depth': # recompute the ious of segmentation for ignoring category
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def accumulateDepth(self, p = None):
        '''
                Accumulate per image evaluation results and store the result in self.eval
                :param p: input params for evaluation
                :return: None
                '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
            p.maxDets = [100]
            p.recThrs = [1.00]
            p.occpair = True
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        nd = len(tp)
                        rc = tp / npig

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                    for t in range(T):
                        allpair_trues, allpairs, occpair_trues, occpairs = [], [], [], []
                        for e in E:
                            imgId = e['image_id']
                            gt_p_order = self.gt_p_orders[imgId]
                            dt_p_order = self.dt_p_orders[imgId]
                            gtIds = np.array(e['gtIds'])
                            gtIgs = np.array(e['gtIgnore'])
                            dtIds = e['gtMatches'][t]
                            gtPosIds = np.logical_and(dtIds, np.logical_not(gtIgs))
                            matchNums = np.count_nonzero(gtPosIds)
                            if matchNums > 1:
                                gtIds_t = (gtIds[gtPosIds]).astype(np.uint32) - np.min(e['gtIds'])
                                dtIds_t = (dtIds[gtPosIds]).astype(np.uint32) - np.min(e['dtIds'])
                                try:
                                    gt_order_matrix = gt_p_order[gtIds_t][:, gtIds_t]
                                    dt_order_matrix = dt_p_order[dtIds_t][:, dtIds_t] #np.zeros_like(gt_order_matrix) #
                                    allpair_true, allpair, occpair_true, occpair, _ = eval_order(
                                        dt_order_matrix, gt_order_matrix)
                                    allpair_trues.append(allpair_true)
                                    allpairs.append(allpair)
                                    occpair_trues.append(occpair_true)
                                    occpairs.append(occpair)
                                except:
                                    print(gtIds_t)
                        if np.sum(allpairs) > 0 and np.sum(occpairs) > 0:
                            if p.occpair:
                                precision[t,:,k,a,m] = np.sum(occpair_trues) / float(np.sum(occpairs))
                            else:
                                precision[t,:,k,a,m] = np.sum(allpair_trues) / float(np.sum(allpairs))

        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeDepths():
            stats = np.zeros((10,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[-1])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[-1])
            stats[3] = _summarize(1, iouThr=.85, maxDets=self.params.maxDets[-1])
            stats[4] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[-1])
            stats[5] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[-1])
            stats[6] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[-1])
            stats[7] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[-1])
            stats[8] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[-1])
            stats[9] = _summarize(0, iouThr=.85, maxDets=self.params.maxDets[-1])
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'depth':
            summarize = _summarizeDepths
        self.stats = summarize()

    def __str__(self):
        self.summarize()

    def PRCurve(self):
        """
        Computer and draw the PR curve for evaluarion results
        """
        p = self.params
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            return mean_s

        T = len(p.iouThrs)
        precision = -np.ones(T)  # -1 for the precision of absent categories
        recall = -np.ones(T)
        for i, iouThr in enumerate(p.iouThrs):
            precision[i] = _summarize(1, iouThr=iouThr, maxDets=self.params.maxDets[-1])
            recall[i] = _summarize(0, iouThr=iouThr, maxDets=self.params.maxDets[-1])
        if (precision == -1).sum() > 0 or (recall == -1).sum() > 0:
            print('precision or recall has invalid values, please check the results')

        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(precision, recall)
        plt.show()
        # plt.savefig('p-r.png')


class ParamsCsd(Params):
    '''
        Params for coco evaluation api
        '''

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox' or iouType == 'depth':
            self.setDetParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None