import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools.coco import COCO, maskUtils
from skimage import measure
import copy

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class KINS(COCO):
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations
        :param annotation_file: location of annotation file
        """

        COCO.__init__(self, annotation_file)

    def showAmodalAnns(self, anns):
        """
        Display a set of amodal Ann object.
        :param anns: a dict object
        return: None
        """
        ax = plt.gca()
        polygons = []
        lines = []
        color = []
        for ann in anns:
            c = np.random.random((1, 3)).tolist()[0]
            line = np.ones((4))
            if 'segmentation' in ann.keys():
                ann['a_segm'] = ann['segmentation']
            if type(ann['a_segm']) == list:
                # ploygon
                seg = ann['a_segm'][0]
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                polygons.append(Polygon(poly, True, alpha=0.2))
                color.append(c)
                line[:-1] = c
                lines.append(line)
            else:
                print("todo")
                raise NotImplementedError

        p = PatchCollection(polygons, facecolors=color, edgecolors=lines, linewidths=3, alpha=0.5)
        ax.add_collection(p)

    def showAmodalInstance(self, ann):
        """Display one instance only: print amodal first, then print invisible mask"""
        ax = plt.gca()
        c = [0.0, 1.0, 0.0]

        polygons = []
        color = []
        # draw amodal mask
        if 'segmentation' in ann.keys():
            ann['a_segm'] = ann['segmentation']
        seg = ann['a_segm'][0]
        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
        polygons.append(Polygon(poly, True, alpha=0.2))
        color.append(c)
        p = PatchCollection(polygons, facecolors=color, edgecolors='green', linewidths=3, alpha=0.2)
        ax.add_collection(p)
        # draw modal mask
        if 'inmodal_seg' in ann.keys():
            ground_truth_binary_mask = maskUtils.decode(ann['inmodal_seg'])
            contours = measure.find_contours(ground_truth_binary_mask, 0.5)
            ann['i_segm'] = []
            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                ann['i_segm'].append(segmentation)
        seg = ann['i_segm'][0]
        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
        polygons.append(Polygon(poly, True, alpha=0.2))
        color.append([1.0, 0.0, 0.0])
        p = PatchCollection(polygons, facecolors=color, edgecolors='red', linewidths=3, alpha=0.5)
        ax.add_collection(p)

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = KINS()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
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
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res