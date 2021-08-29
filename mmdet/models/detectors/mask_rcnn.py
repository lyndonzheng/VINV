from .two_stage import TwoStageDetector
from ..registry import DETECTORS
from .base import *
from mmdet.datasets.layer_data_prepare import *

from mmdet.core import (imshow_det_bboxes, mkdirs)

@DETECTORS.register_module
class MaskRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(MaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def show_result(self, data, result, dataset=None, score_thr=0.3, pre_order=True, show=False, out_file=None):
        segm_result, occ_result = None, None
        if isinstance(result, tuple):
            if len(result) == 3:
                bbox_result, segm_result, occ_result = result
            else:
                bbox_result, segm_result = result
        else:
            bbox_result= result

        if isinstance(data['img_meta'], list):
            img_tensor = data['img'][0]
        else:
            img_tensor = data['img'].data[0]
        if isinstance(data['img_meta'], list):
            img_metas = data['img_meta'][0].data[0]
        else:
            img_metas = data['img_meta'].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            bboxes = np.vstack(bbox_result)
            bboxes[:, :-1] = bboxes[:, :4] * img_meta['scale_factor']
            if occ_result is not None:
                occs = np.vstack(occ_result)
                if occs.shape[1] == 3:
                    occ_labels = np.argmax(occs[:,:-1], axis=1)
                else:
                    occ_labels = np.argmax(occs, axis=1)
                if pre_order:
                    occ_labels = occs[:,-1].astype(np.int64)
            else:
                occ_labels = None
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i])
                    mask = mmcv.imresize(mask, (w, h)).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
                    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE,
                                                           cv2.CHAIN_APPROX_SIMPLE)
                    img_show = cv2.drawContours(img_show, contours, -1, (int(color_mask[0][0]), int(color_mask[0][1]), int(color_mask[0][2])), 2)
            # build save files
            if out_file is not None:
                file_name = img_metas[0]['filename']
                content = file_name.split('/')
                mkdirs(out_file)
                out_file = out_file + '/' + content[-3] + '_' + content[-1]
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                occ_labels=occ_labels,
                class_names=class_names,
                score_thr=score_thr,
                show=show,
                out_file=out_file)
