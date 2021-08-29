from PIL import Image
import numpy as np
from mmdet.datasets.csd import CSD

dataDir='/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/coco/val2014/' # 'train2014','test2014'
gtFile = '/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/coco/annotations/COCO_amodal_val2014.json'
dtFile = '/media/lyndon/2e91762c-97d9-40c9-9af1-6f318aca4771/results/VIV_TPAMI/amodal_results/amodalcomp_val_deocclusion_gtv_cocoa.json'

dataType = 'test'
annFile = '{}/sosc_new_{}_order.json'.format('/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/suncg_data', dataType)

csd = CSD(annFile)
amodal = csd.loadRes(dtFile)

imgId = np.random.choice(amodal.dataset['images'])['id']
imgId = 487450 # cached image demo for coco 473821
annIds = amodal.getAnnIds(imgIds=[imgId])
anns = amodal.loadAnns(annIds)