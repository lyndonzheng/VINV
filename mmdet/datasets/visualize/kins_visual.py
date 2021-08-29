import os
import random
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from mmdet.datasets.visualize.kins_tools import KINS

print("show KINS examples")

dataDir = "/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/KINS/testing/image_2"
gtFile = "/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/KINS/instances_val_depth.json"

kins = KINS(gtFile)

# random select one image
imgIds = kins.getImgIds(imgIds=[], catIds=[])
imgIds = kins.getImgIds(imgIds=[1053])

for i in range(0, len(imgIds)):
    img = kins.loadImgs(imgIds[i])[0]

    # load and display image
    I = io.imread(os.path.join(dataDir, img['file_name']))
    plt.figure()
    plt.imshow(I)
    plt.show()
    print(os.path.join(dataDir, img['file_name']))

    # load the annotations
    annIds = kins.getAnnIds(imgIds=img['id'])
    anns = kins.loadAnns(annIds)

    # display the full image annoation: draw all annotated instances with depth ordering effect
    plt.figure()
    plt.imshow(I)
    kins.showAmodalAnns(anns)
    plt.show()
    #
    # for ann in anns:
    #     plt.figure()
    #     plt.imshow(I)
    #     # amodal.showModalInstance(ann, ins+1) # show k-th object, with only visible mask
    #     kins.showAmodalInstance(ann)
    #     ax = plt.gca()
    #     nameStr = "region name: " + kins.loadCats(ids=[ann['category_id']])[0]['name']
    #     if 'layer_order' in ann.keys():
    #         ann['ico_id'] = ann['layer_order']
    #     depthStr = "depth order: " + str(ann['ico_id'])
    #     if 'area' in ann.keys():
    #         ann['a_area'] = ann['area']
    #         ann['i_area'] = ann['inmodal_bbox'][-1]*ann['inmodal_bbox'][-2]
    #     rateStr = "occlude rate: " + '%0.3f' % (float(ann['a_area'] - ann['i_area'])/float(ann['a_area']))
    #
    #     # show properties of the instance
    #     ax.annotate(nameStr, xy=(1, 1), xytext=(0, -5), fontsize=15)
    #     ax.annotate(depthStr, xy=(1, 1), xytext=(400, -5), fontsize=15)
    #     ax.annotate(rateStr, xy=(1, 1), xytext=(800, -5), fontsize=15)
    #     plt.show()