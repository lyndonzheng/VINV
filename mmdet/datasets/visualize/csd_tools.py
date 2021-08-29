import json
import time
import numpy as np
import itertools
from collections import defaultdict
from PIL import Image, ImageDraw
from pycocotools import mask as maskUtils
import copy


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class CSD:
    def __init__(self, annotation_file=None):
        """SOSC helper class for reading and visualizing annotation"""
        # load dataset
        self.dataset, self.anns, self.cats, self.images = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImages = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        """create index for image, annotations, categories"""
        print('creating index...')
        anns, cats, images = {}, {}, {}
        imgToAnns, catToImages = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for image in self.dataset['images']:
                images[image['id']] = image

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImages[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImages = catToImages
        self.images = images
        self.cats = cats

    def getAnnIds(self, imgIds=[], catIds=[]):
        """
        Get ann ids that satisfy given filter conditions
        :param imgds: get anns for given images
        :param catIds: get anns for given cats
        :return: ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
        ids = [ann['id'] for ann in anns]

        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        Get category ids that satisfy given filter conditions
        :param catNms: get cats for given cat names
        :param supNms: get cats for given supercategory names
        :param catIds: get cats for given cat ids
        :return: integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id'] in catIds]
        ids = [cat['id'] for cat in cats]
        ids = list(set(ids))
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param sceIds (int array) : get images for given ids
        :param catIds (int array) : get images with all given cats
        :return: ids (int array)  : integer array of images ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.images.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImages[catId])
                else:
                    ids &= set(self.catToImages[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.images[id] for id in ids]
        elif type(ids) == int:
            return [self.images[ids]]

    def labels2Colors(self, label, palette):
        """Simple function that add fixed colors depending on the class"""
        colors = label * palette
        colors = (colors % 255).astype(np.uint8)

        return colors

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        t = self.images[ann['image_id']]
        h, w = t['height'], t['width']
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def showAnns(self, image, anns, f_bbox=True, v_bbox=False, f_mask=False, v_mask=True,
                 name=True, layer=True, object_id=False, pair_order=False, dataDir='.'):
        """
        Adds the predicted boxes on the top of the image
        :param image: an image as returned by opencv
        :param anns: annotations contains of full bbox, visible bbox
        :param show_f: flag to show the full bbox or not
        :param show_v: flag to show the full bbox or not
        :return: overlay image
        """
        if len(anns) == 0:
            return image
        # used to make colors for each class
        palette = np.array([2 ** 11 - 1, 2 ** 21 - 1, 2 ** 31 - 1])
        # image size
        for ann in anns:
            label = ann['category_id']
            color = tuple(self.labels2Colors(label, palette))
            if f_mask:
                image = np.array(image)
                rgba = Image.open('%s/%s' % (dataDir, ann['f_img_name']))
                r, g, b, a = rgba.split()
                for c in range(3):
                    image[:, :, c] = np.where(np.array(a) > 0, image[:, :, c] * 0.5 + color[c] * 0.5, image[:, :, c])
                image = Image.fromarray(image)
            if v_mask:
                image = np.array(image)
                v = Image.open('%s/%s' % (dataDir, ann['v_mask_name']))
                for c in range(3):
                    image[:, :, c] = np.where(np.array(v) > 0,
                                              image[:, :, c] * 0.5 + (color[c] + 50).astype(np.uint8) * 0.5,
                                              image[:, :, c])
                image = Image.fromarray(image)
            d = ImageDraw.Draw(image)
            if f_bbox:
                f_box = ann['f_bbox']
                d.rectangle((f_box[0], f_box[1], f_box[0] + f_box[2], f_box[1] + f_box[3]), outline=(0, 255, 0))
            if v_bbox:
                v_box = ann['v_bbox']
                d.rectangle((v_box[0], v_box[1], v_box[0] + v_box[2], v_box[1] + v_box[3]), outline=(0, 255, 0))
            if name:
                v_box = ann['v_bbox']
                class_name = self.cats[ann['category_id']]['supercategory']
                d.text((v_box[0], v_box[1]), class_name, fill=(255, 255, 255))
            if layer:
                v_box = ann['v_bbox']
                layer_name = str(ann['layer_order'])
                d.text((v_box[0] + v_box[2] - 10, v_box[1] + v_box[3] - 10), layer_name, fill=(255, 255, 255))
            if object_id:
                v_box = ann['v_bbox']
                class_name = self.cats[ann['category_id']]['supercategory']
                object_name = str(ann['id'])
                d.text((v_box[0] + len(class_name) * 6 + 2, v_box[1]), object_name, fill=(255, 255, 255))
            if pair_order:
                print(ann['id'], ann['pair_order'])

        return image

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = CSD()
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