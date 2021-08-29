import json
import os
import glob
import torch
import numpy as np
from mmdet.datasets.visualize.cocoa_tools import COCOA
from pycocotools import mask as maskUtils
from mmdet.ops.nms import nms_cpu


def createAmodalAnn(image_id, ann_id):
    ann = {}
    ann['id'] = ann_id
    ann['category_id'] = 1 # fake label
    ann['image_id'] = image_id
    ann['regions']  =[]
    return ann


def createAmodalRegion(ann, id):
    region = {}
    region['id'] = id #used for gt/dt matching
    region['segmentation'] = ann['segmentation']
    bbox = maskUtils.toBbox(ann['segmentation'])
    region['bbox'] = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
    region['score'] = ann['score']
    region['isStuff'] = 0  # default things
    if 'foreground_ness' in ann:
        region['foreground_ness'] = ann['foreground_ness']
    if 'invisibleMask' in ann:
        region['invisible_mask'] = ann['invisibleMask']
    if 'amodalMask' in ann:
        region['amodal_mask'] = ann['amodalMask']
    return region


def nms(dets, iou_thr):
    if dets.shape[0] == 0:
        inds = dets.new_zeros(0, dtype=np.int8)
    else:
        dets_thr = torch.from_numpy(dets).to('cpu')
        inds = nms_cpu.nms(dets_thr, iou_thr)

    inds = inds.cpu().numpy()

    return dets[inds, :], inds


# filter coco dt, according to amodal gt imgId
# coco dt is formated as instance. This function transform data into image_id view
def filterDtFile(resFiles, amodalGtImgIds):
    amodalDt = {}
    id = 0
    ann_id = 0
    for i, file in enumerate(resFiles):
        print ("processing json %d in total %d" %(i+1, len(resFiles)))
        anns = json.load(open(file))
        for ann in anns:
            image_id = ann['image_id']
            if image_id in amodalGtImgIds:
                id = id + 1
                if image_id not in amodalDt:
                    amodalDt[image_id] = createAmodalAnn(image_id, ann_id)
                    ann_id = ann_id + 1
                region = createAmodalRegion(ann, id)
                amodalDt[image_id]['regions'].append(region)
    res = []
    # for image_id, ann in amodalDt.iteritems():
    for key in amodalDt.keys():
        anns = amodalDt[key]
        dets = []
        regions = []
        for ann in anns['regions']:
            det = np.zeros((5), np.float32)
            det[:-1] = ann['bbox']
            det[-1] = ann['score']
            dets.append(det)
        dets = np.array(dets)
        dets, inds = nms(dets, 0.7)
        for ind in inds:
            regions.append(anns['regions'][ind])
        anns['regions'] = regions
        res.append(anns)
    return res

dataDir='/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/coco/val2014/' # 'train2014','test2014'
gtFile = '/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/coco/annotations/COCO_amodal_val2014.json'
dtFile = '/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/coco/annotations/'
amodalDtFile = '/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/coco/annotations/AmodalMask_amodalDt_nums.json'

dtFiles = []

for filename in glob.glob(dtFile + 'amodal-props-*.json'):
    dtFiles.append(filename)

amodal=COCOA(gtFile)
imgIds = sorted(amodal.getImgIds())
amodalDt = filterDtFile(dtFiles, imgIds)

with open(amodalDtFile, 'w') as output:
    json.dump(amodalDt, output, indent=4)