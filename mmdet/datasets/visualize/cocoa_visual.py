import os
import random
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from mmdet.datasets.visualize.cocoa_tools import COCOA
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

print("show COCOA examples")

dataDir='/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/coco/train2014/' # 'train2014','test2014'
gtFile = '/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/coco/annotations/COCO_amodal_train2014.json'
dtFile = '/media/lyndon/2e91762c-97d9-40c9-9af1-6f318aca4771/results/VIV_TPAMI/amodal_results/amodalcomp_val_deocclusion_gtv_cocoa.json'

amodal=COCOA(gtFile)

# random select one image
imgId = random.choice(amodal.dataset['images'])['id']
# imgId = 538574 # cached image demo for coco 473821
imgIds = amodal.get_img_ids(img_ids=[])
for imgId in imgIds:
    annIds = amodal.getAmodalAnnIds(imgIds=imgId)
    anns = amodal.loadAnns(annIds)
    ann = random.choice(anns)
    img = amodal.loadImgs(imgId)[0]
    I = io.imread(os.path.join(dataDir, img['file_name']))
    plt.figure()
    plt.imshow(I)
    plt.show()
    print(img['file_name'])

# display the full image annoation: draw all annotated instances with depth ordering effect
plt.figure()
plt.imshow(I)
amodal.showAmodalAnns(ann)
plt.show()

# display the edge map
plt.figure()
plt.imshow(np.zeros(I.shape))
amodal.showEdgeMap(ann)
plt.show()

# display each annotated amodal instance

for ins in range(ann['size']):
    plt.figure()
    plt.imshow(I)
    #amodal.showModalInstance(ann, ins+1) # show k-th object, with only visible mask
    amodal.showAmodalInstance(ann, ins+1) # show k-th object, with invisible_mask highlighted
    ax = plt.gca()
    nameStr = "region name: " + ann['regions'][ins]['name']
    depthStr = "depth order: " + str(ann['regions'][ins]['order'])
    stuffStr = "isStuff: " + str(ann['regions'][ins]['isStuff'])
    rateStr = "occlude rate: " + '%0.3f' % ann['regions'][ins]['occlude_rate']

    # show properties of the instance
    ax.annotate(nameStr, xy=(1, 1), xytext=(0, -5), fontsize=15)
    ax.annotate(depthStr, xy=(1, 1), xytext=(160, -5), fontsize=15)
    ax.annotate(stuffStr, xy=(1, 1), xytext=(280, -5), fontsize=15)
    ax.annotate(rateStr, xy=(1, 1), xytext=(350, -5), fontsize=15)
    plt.show()
