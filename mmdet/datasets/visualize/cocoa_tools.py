import json
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
from pycocotools import mask as maskUtils
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools.coco import COCO


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class COCOA(COCO):
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations
        :param annotation_file: location of annotation file
        """

        COCO.__init__(self, annotation_file)

    def createIndex(self):
        # create index
        print('creating index..')
        anns, imgs = {}, {}
        regions = []
        imgToAnns= defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann
                if 'region' in ann:
                    for region in ann['regions']:
                        region['image_id'] = ann['image_id']
                        regions.append(region)

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img
        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.imgs = imgs
        self.regions = regions

    def getAmodalAnnIds(self, imgIds = []):
        """
        Get amodal ann id that satisfy given fiter conditions.
        :param imgIds (int array): get anns for given imgs
        :return: ids (int array) : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]

        if len(imgIds) == 0:
            anns = self.dataset['annotations']
        else:
            lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
            anns = list(itertools.chain.from_iterable(lists))
        ids = [ann['id'] for ann in anns]

        return ids

    def labels2Colors(self, label, palette):
        """Simple function that add fixed colors depending on the class"""
        colors = label * palette
        colors = (colors % 255).astype(np.uint8)

        return colors

    def showAmodalAnns(self, anns):
        """
        Display a set of amodal Ann object.
        :param anns: a dict object
        return: None
        """
        if type(anns) == list:
            print("anns cannot be a list! Should be a dict.")
            return 0
        ax = plt.gca()
        polygons = []
        lines = []
        color = []
        for ann in reversed(anns['regions']):
            c = np.random.random((1, 3)).tolist()[0]
            line = np.ones((4))
            if type(ann['segmentation']) == list and ann['isStuff'] == 0:
                # polygon
                seg = ann['segmentation']
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                polygons.append(Polygon(poly, True, alpha=0.4))
                color.append(c)
                line[:-1] = c
                lines.append(line)
            else:
                print("todo")
                raise NotImplementedError

        p = PatchCollection(polygons, facecolors=color, edgecolors=lines, linewidths=4, alpha=0.5)
        ax.add_collection(p)

    def showmodalAnns(self, anns):
        """
        Display a set of amodal Ann object.
        :param anns: a dict object
        return: None
        """
        if type(anns) == list:
            print("anns cannot be a list! Should be a dict.")
            return 0
        ax = plt.gca()
        polygons = []
        lines = []
        color = []
        for ann in reversed(anns['regions']):
            c = np.random.random((1, 3)).tolist()[0]
            line = np.ones((4))
            if ann['isStuff'] == 0 :
                if 'visible_mask' in ann:
                    mm = maskUtils.decode([ann['visible_mask']])
                    img = np.ones((mm.shape[0], mm.shape[1], 3))
                    color_mask = c
                    for i in range(3):
                        img[:, :, i] = color_mask[i]
                    ax.imshow(np.dstack((img, mm * 0.5)))
                elif type(ann['segmentation']) == list:
                    # polygon
                    seg = ann['segmentation']
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    polygons.append(Polygon(poly, True, alpha=0.2))
                    color.append(c)
                    line[:-1] = c
                    lines.append(line)
            # else:
            #     print("todo")
            #     raise NotImplementedError

        p = PatchCollection(polygons, facecolors=color, edgecolors=lines, linewidths=3, alpha=0.5)
        ax.add_collection(p)

    def showEdgeMap(self, anns):
        """
        Show edge map for an annontation
        :param anns: a dict object
        return: None
        """
        if type(anns) == list:
            print("anns cannot be a list! Should be a dict")
            return 0
        ax = plt.gca()
        polygons = []
        lines = []
        color = []
        for ann in reversed(anns['regions']):
            c = np.zeros([1, 3]).tolist()[0]
            if type(ann['segmentation']) == list:
                # polygon
                seg = ann['segmentation']
                poly = np.array(seg).reshape(int((len(seg) / 2)), 2)
                polygons.append(Polygon(poly, True, alpha=0.2))
                color.append(c)
            else:
                print("todo")
                raise NotImplementedError

        p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 1, 1), linewidths=1, alpha=1)
        ax.add_collection(p)

    def showMask(self, M, ax, c=[0, 1, 0]):
        m = maskUtils.decode([M])
        img = np.ones((m.shape[0], m.shape[1], 3))

        # get boundary quickly
        B = np.zeros((m.shape[0], m.shape[1]))
        for aa in range(m.shape[0] - 1):
            for bb in range(m.shape[1] - 1):
                # kk = aa*m.shape[1]+bb
                if m[aa, bb] != m[aa, bb + 1]:
                    B[aa, bb], B[aa, bb + 1] = 1, 1
                if m[aa, bb] != m[aa + 1, bb]:
                    B[aa, bb], B[aa + 1, bb] = 1, 1
                if m[aa, bb] != m[aa + 1, bb + 1]:
                    B[aa, bb], B[aa + 1, bb + 1] = 1, 1

        for i in range(3):
            img[:, :, i] = c[i]
            ax.imshow(np.dstack((img, B * 1)))
            ax.imshow(np.dstack((img, m * 0.3)))

    def showAmodalInstance(self, anns, k=-1):
        """
        Display k-th instance only: print segmentation first, then print invisible_mask
        anns: a single annotation
        k: the depth order of anns, 1-index. If k = -1, just visulize input
        """
        ax = plt.gca()
        c = np.random.random((1, 3)).tolist()[0]
        c = [0.0, 1.0, 0.0]  # green

        if k < 0:
            self.showMask(anns['segmentation'], ax)
            return

        if type(anns) == list:
            print("ann cannot be a list! Should be a dict")
            return 0
        ann = anns['regions'][k - 1]
        polygons = []
        color = []
        # draw whole mask
        if type(ann['segmentation']) == list:
            # polygon
            seg = ann['segmentation']
            poly = np.array(seg).reshape(int((len(seg) / 2)), 2)
            polygons.append(Polygon(poly, True, alpha=0.2))
            color.append(c)
            p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 1, 1), linewidths=3, alpha=0.2)
            ax.add_collection(p)
        else:
            self.showMask(ann['segmentation'], ax)

        # draw invisible_mask
        if 'invisible_mask' in ann:
            self.showMask(ann['invisible_mask'], ax, [1, 0, 0])

    def showModalInstance(self, anns, k):
        """
        Display k-th instance: print its visible mask
        anns: a single annotation
        k: the depth order of anns, 1-index
        """
        if type(anns) == list:
            print("ann cannot be a list! Should be a dict")
            return 0
        ax = plt.gca()
        c = np.random.random((1, 3)).tolist()[0]
        c = [0.0, 1.0, 0.0]  # green
        ann = anns['regions'][k - 1]
        polygons = []
        color = []
        # draw whole mask
        if 'visible_mask' in ann:
            mm = maskUtils.decode([ann['visible_mask']])
            img = np.ones((mm.shape[0], mm.shape[1], 3))
            color_mask = c
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, mm * 0.6)))
        else:
            if type(ann['segmentation']) == list:
                # polygon
                seg = ann['segmentation']
                poly = np.array(seg).reshape((len(seg) / 2, 2))
                polygons.append(Polygon(poly, True, alpha=0.2))
                color.append(c)
            else:
                # mask
                mm = maskUtils.decode([ann['segmentation']])
                img = np.ones((mm.shape[0], mm.shape[1], 3))
                color_mask = c
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, mm * 0.6)))

            p = PatchCollection(polygons, facecolors=color, edgecolors=(0, 0, 0, 1), linewidths=3, alpha=0.4)
            ax.add_collection(p)

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCOA()
        res.dataset['images'] = [img for img in self.dataset['images']]
        print ('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
            'Results do not correspond to current coco set'
        if 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id+1
                ann['iscrowd'] = 0
        print ('DONE (t=%0.2fs)' % (time.time() - tic))
        res.dataset['annotations'] = anns
        res.createIndex()
        return res