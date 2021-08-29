import itertools

import mmcv
import numpy as np
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from .recall import eval_recalls
from mmdet.core.evaluation.cocoa_eval import COCOAeval
from mmdet.core.evaluation.kins_eval import KINSeval
from mmdet.core.evaluation.csd_eval import CSDeval


def eval(result_files, result_types, dataset, cfg, max_dets=(100, 300, 1000),
         classwise=False, order_method='depth', full_mask=True, full_bbox=True, pr_curve=False):
    for res_type in result_types:
        assert res_type in [
            'proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints', 'depth'
        ]
    dataname = cfg.test.type[:-7].upper()
    if dataname == 'CSD':
        data_class = dataset.csd
    elif dataname == 'KINS':
        data_class = dataset.kins
    elif dataname == 'COCOA':
        data_class = dataset.cocoa
    elif dataname == 'COCO':
        data_class = dataset.coco
    else:
        print('{} model and dataset if provided'.format(dataname))

    for res_type in result_types:
        if isinstance(result_types, str):
            result_file = result_files
        elif isinstance(result_files, dict):
            result_file = result_files[res_type]
        else:
            assert TypeError('result_files must be a str or dict')
        assert result_file.endswith('.json')

        dets = data_class.loadRes(result_file)
        # dets = data_class.loadRes('/media/lyndon/2e91762c-97d9-40c9-9af1-6f318aca4771/results/VIV_TPAMI/amodal_results/modal_cocoa.json')
        img_ids = list(np.unique(data_class.getImgIds()))
        iou_type = 'bbox' if res_type == 'proposal' else res_type

        if dataname == 'CSD':
            Eval = CSDeval(data_class, dets, iou_type, order_method, fullMask=full_mask, fullBbox=full_bbox, imgPrefix=cfg.test.img_prefix)
            Eval.params.catIds = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                      21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36]
        elif dataname == 'KINS':
            Eval = KINSeval(data_class, dets, iou_type, order_method, fullMask=full_mask, fullBbox=full_bbox)
            Eval.params.catIds = [1, 2, 4, 5, 6, 7, 8]
        elif dataname == 'COCOA':
            Eval = COCOAeval(data_class, dets, iou_type, order_method, fullMask=full_mask, fullBbox=full_bbox)
            Eval.params.useCats = 0 # cocoa do not provide special category for each instance,
        else:
            Eval = COCOeval(data_class, dets, iou_type)
        Eval.params.imgIds = img_ids
        if pr_curve:
            Eval.params.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .01) + 1, endpoint=True)
        Eval.evaluate()
        if res_type == 'depth':
            Eval.accumulateDepth()
        else:
            Eval.accumulate()
        if pr_curve:
            Eval.PRCurve()
        else:
            Eval.summarize()

        if classwise and dataname != 'COCOA' and res_type != 'depth':
            # compute per-category AP
            precisions = Eval.eval['precision']
            catIds = Eval.params.catIds
            # precision has dims (iou, recall, cls, area range, max dets)
            assert len(catIds) == precisions.shape[2]

            results_per_category = []
            for idx, catId in enumerate(catIds):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = data_class.loadCats(int(catId))[0]
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float('nan')
                results_per_category.append(
                    ('{}'.format(nm['supercategory']),
                     '{:0.3f}'.format(float(ap * 100))))

            N_COLS = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (N_COLS // 2)
            results_2d = itertools.zip_longest(
                *[results_flatten[i::N_COLS] for i in range(N_COLS)])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print(table.table)


def fast_eval_recall(results,
                     coco,
                     max_dets,
                     iou_thrs=np.arange(0.5, 0.96, 0.05)):
    if mmcv.is_str(results):
        assert results.endswith('.pkl')
        results = mmcv.load(results)
    elif not isinstance(results, list):
        raise TypeError(
            'results must be a list of numpy arrays or a filename, not {}'.
            format(type(results)))

    gt_bboxes = []
    img_ids = coco.getImgIds()
    for i in range(len(img_ids)):
        ann_ids = coco.getAnnIds(imgIds=img_ids[i])
        ann_info = coco.loadAnns(ann_ids)
        if len(ann_info) == 0:
            gt_bboxes.append(np.zeros((0, 4)))
            continue
        bboxes = []
        for ann in ann_info:
            if ann.get('ignore', False) or ann['iscrowd']:
                continue
            x1, y1, w, h = ann['bbox']
            bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.shape[0] == 0:
            bboxes = np.zeros((0, 4))
        gt_bboxes.append(bboxes)

    recalls = eval_recalls(
        gt_bboxes, results, max_dets, iou_thrs, print_summary=False)
    ar = recalls.mean(axis=1)
    return ar


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def proposal2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                json_results.append(data)
    return json_results


def segm2json(dataset, results, with_occ=False):
    bbox_json_results = []
    segm_json_results = []
    layer_json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        if with_occ:
            det, seg, occ = results[idx]
        else:
            det, seg = results[idx]
        for label in range(len(det)):
            # bbox results
            bboxes = det[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                bbox_json_results.append(data)

            # segm results
            # some detectors use different score for det and segm
            if isinstance(seg, tuple):
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(mask_score[i])
                data['category_id'] = dataset.cat_ids[label]
                if isinstance(segms[i]['counts'], bytes):
                    segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)

            # layer order results
            if with_occ:
                occes = occ[label]
                for i in range(occes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = dataset.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    data['layer'] = list(occes[i].astype(np.float64))
                    if data['score'] > 0.3:
                        layer_json_results.append(data)
    return bbox_json_results, segm_json_results, layer_json_results


def results2json(dataset, results, out_file, with_occ=False):
    result_files = dict()
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        mmcv.dump(json_results, result_files['bbox'])
    elif isinstance(results[0], tuple):
        json_results = segm2json(dataset, results, with_occ)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
        result_files['depth'] = '{}.{}.json'.format(out_file, 'depth')
        mmcv.dump(json_results[0], result_files['bbox'])
        mmcv.dump(json_results[1], result_files['segm'])
        if with_occ:
            mmcv.dump(json_results[2], result_files['depth'])
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
        mmcv.dump(json_results, result_files['proposal'])
    else:
        raise TypeError('invalid type of results')
    return result_files