import json
import numpy as np
from mmdet.datasets.visualize.kins_tools import KINS
from pycocotools.coco import maskUtils
from mmdet.core.utils.misc import infer_gt_order, layer_ranking

print("show KINS examples")

dataDir = "/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/KINS/testing/image_2"
gtFile = "/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/KINS/instances_val.json"

kins = KINS(gtFile)

# random select one image
imgIds = kins.getImgIds(imgIds=[], catIds=[])
data_kins = {}

data_kins['categories'] = kins.dataset['categories']
annotations = []
images = []

for i in range(0, len(imgIds)):
    img = kins.loadImgs(imgIds[i])[0]

    images.append(img)
    # load the annotations
    annIds = kins.getAnnIds(imgIds=img['id'])
    print(img['id'])
    anns = kins.loadAnns(annIds)
    inmodal = []
    amodal = []
    for ann in anns:
        inmodal.append(maskUtils.decode(ann['inmodal_seg']))
        rles = maskUtils.frPyObjects(ann['segmentation'], img['height'], img['width'])
        rle = maskUtils.merge(rles)
        amodal.append(maskUtils.decode((rle)))
    gt_order_matrix = infer_gt_order(np.array(inmodal), np.array(amodal))
    layer_order = layer_ranking(gt_order_matrix)

    for i, ann in enumerate(anns):
        ann['layer_order'] = int(layer_order[i])
        order = {}
        for j, item in enumerate(gt_order_matrix[i]):
            order[int(j)] = int(item)
        ann['pair_order'] = order
        annotations.append(ann)
data_kins['annotations'] = annotations
data_kins['images'] = images

json.dump(data_kins, open("/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/KINS/instances_val_depth.json", 'w'), indent=4)