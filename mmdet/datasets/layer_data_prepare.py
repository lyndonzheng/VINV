import torch
import numpy as np
import cv2


import mmcv
from mmdet.models.gan_modules.task import patch_min_depth_pooling
from pycocotools import mask as maskUtils


######################################################################################
# prepare the re-composition data for training
######################################################################################

def data_normalization(f_rgbs, f_depths, f_masks, void_depth=100):
    """normalize the ground truth image, depth, mask"""
    f_rgbs = f_rgbs.type_as(f_depths) / 256 * 2 - 1
    f_depths = torch.where(f_depths > 0, f_depths, torch.ones_like(f_depths) * void_depth)
    f_masks = torch.tensor(f_masks).type_as(f_depths).unsqueeze(1)

    return f_rgbs, f_depths, f_masks


def parse_layer_data(f_rgbs, f_depths, f_masks, gt_labels, l_orders):
    """select the visible object based on the layer order, 0 will be masked out, > 0 been selected"""
    # select the visible object for training
    v_inds = torch.nonzero(l_orders > 0, as_tuple=True)
    l_f_rgbs = f_rgbs[v_inds]
    l_f_depths = f_depths[v_inds]
    l_f_masks = f_masks[v_inds]
    l_f_labels = gt_labels[v_inds].view(-1, 1, 1, 1).type_as(l_f_masks) * l_f_masks
    # min depth pooling to get the visible image and depth and labels
    sce_depth_re, sce_rgb_re = patch_min_depth_pooling(l_f_depths, l_f_rgbs, void_depth=100)
    sce_depth_re, sce_labels_re = patch_min_depth_pooling(l_f_depths, l_f_labels, void_depth=100)
    # get visible mask
    del_inds = torch.nonzero(l_orders == 0, as_tuple=True)
    l_del_mask = f_masks[del_inds]
    l_del_mask_sum = l_del_mask.sum(dim=0, keepdim=True)
    l_v_mask = (l_del_mask_sum == 0).type_as(l_f_masks)

    return sce_rgb_re, sce_depth_re, sce_labels_re, l_v_mask


def parse_batch_layer_data(f_rgbs, f_depths, f_masks, gt_labels, l_orders):
    """transfer the data from Dataloader (list for image)"""
    l_sce_rgbs, l_sce_depths, l_sce_labels, l_v_masks = [], [], [], []
    for (f_rgb, f_depth, f_mask, gt_label, l_order) in zip(f_rgbs, f_depths, f_masks, gt_labels, l_orders):
        if 'ByteTensor' in f_rgb.type():
            f_rgb, f_depth, f_mask = data_normalization(f_rgb, f_depth, f_mask)
        sce_rgb_re, sce_depth_re, sce_labels_re, l_v_mask = parse_layer_data(f_rgb, f_depth, f_mask, gt_label, l_order)
        l_sce_rgbs.append(sce_rgb_re)
        l_sce_depths.append(sce_depth_re)
        l_sce_labels.append(sce_labels_re)
        l_v_masks.append(l_v_mask)
    # get batch gt for layer-by-layer data
    l_sce_rgbs = torch.cat(l_sce_rgbs)
    l_sce_depths = torch.cat(l_sce_depths)
    l_sce_labels = torch.cat(l_sce_labels)
    l_v_masks = torch.cat(l_v_masks)

    return l_sce_rgbs, l_sce_depths, l_sce_labels, l_v_masks


def parse_batch_layer_anns(img_metas, gt_labels, gt_v_bboxes=None, gt_v_masks=None, gt_f_bboxes=None, gt_f_masks=None,
                           l_orders=None, p_orders=None):
    """select the visible annotations based on the layer order, 0 is non-occluded object, 1 is been occluded object"""
    l_labels, l_v_bboxes, l_v_masks, l_f_bboxes, l_f_masks, l_l_orders, l_p_orders, l_occs = [], [], [], [], [], [], [], []
    for i, (l_order, img_meta) in enumerate(zip(l_orders, img_metas)):
        if "suncg" in img_meta['filename']:
            l_order = l_order[:-1] # ignore the BG for the object detection and mask prediction for synthesis data
        obj_inds = torch.nonzero(l_order > -1, as_tuple=True)
        l_labels.append(gt_labels[i][obj_inds])
        l_v_bboxes.append(gt_v_bboxes[i][obj_inds])
        l_f_bboxes.append(gt_f_bboxes[i][obj_inds])
        l_f_mask = gt_f_masks[i][obj_inds[0].cpu()]
        if l_f_mask.ndim == 2:
            l_f_mask = np.expand_dims(l_f_mask, axis=0)
        l_f_masks.append(l_f_mask)
        l_order = l_order[obj_inds]
        l_l_orders.append(l_order)
        l_occs.append((l_order != 0).type_as(gt_labels[0]))
        l_v_mask = np.zeros_like(l_f_mask)
        l_p_orders.append(p_orders[i][obj_inds[0]][:, obj_inds[0]])
        for j, l_p_order in enumerate(l_p_orders[-1]):
            occ_inds = (l_p_order == -1).nonzero()
            l_v_mask[j] = l_f_mask[j]
            for occ_ind in occ_inds:
                l_v_mask[j] = np.where(l_f_mask[occ_ind] > 0, np.zeros_like(l_v_mask[j]), l_v_mask[j])
        l_v_masks.append(l_v_mask)

    return l_labels, l_v_bboxes, l_v_masks, l_f_bboxes, l_f_masks, l_l_orders, l_p_orders, l_occs


######################################################################################
# matching ground truth to delete to reduce the error
######################################################################################
def select_result_by_order(results, order_thr=0.5, score_thr=0.3):
    """select the predicted results based on the predicted order and score"""
    if len(results) == 3:
        bbox_results, segm_results, occ_results = results
        select_bbox, select_segm, select_occ = [], [], []
    else:
        raise NotImplementedError('selected results must consist of occ result')

    flag = False
    max_i_ind = 0
    max_obj_ind = 0
    max_conf = 0

    for i, (bbox_result, segm_result, occ_result) in enumerate(zip(bbox_results, segm_results, occ_results)):
        inds = np.where(bbox_result[:, -1] > score_thr)[0]
        bbox_result, occ_result = bbox_result[inds], occ_result[inds]
        segm_result = [segm_result[ind] for ind in inds]
        if occ_result.shape[0] > 0:
            occ_labels = np.argmax(occ_result, axis=1)
            occ_conf = np.max(occ_result, axis=1)
            # select the fully visible object
            obj_ind = np.nonzero((occ_labels == 0) & (occ_conf > order_thr))
            if len(obj_ind[0]) == 0:
                select_occ.append(np.zeros((0, occ_result.shape[1]), dtype=np.float32))
                select_bbox.append(np.zeros((0, 5), dtype=np.float32))
                select_segm.append([])
                # if no fully visible object, select the object based on the largest occlusion score
                ind = np.argmax(occ_result, axis=0)
                if occ_result[ind[0]][0] > max_conf:
                    max_i_ind, max_obj_ind, max_conf = i, inds[ind[0]], occ_result[ind[0]][0]
            else:
                select_occ.append(occ_result[obj_ind])
                select_bbox.append(bbox_result[obj_ind])
                select_segm.append([segm_result[index] for index in obj_ind[0]])
                flag = True
        else:
            select_occ.append(np.zeros((0, occ_result.shape[1]), dtype=np.float32))
            select_bbox.append(np.zeros((0, 5), dtype=np.float32))
            select_segm.append([])

    if not flag:
        bbox = np.vstack(bbox_results)
        bbox = bbox[np.where(bbox[:, -1] > score_thr)[0]]
        if bbox.shape[0] != 0:
            select_occ[max_i_ind] = results[2][max_i_ind][max_obj_ind].reshape(1, -1)
            select_bbox[max_i_ind] = results[0][max_i_ind][max_obj_ind].reshape(1, -1)
            select_segm[max_i_ind] = [results[1][max_i_ind][max_obj_ind]]

    select_results = (select_bbox, select_segm, select_occ)

    return select_results


def match_result_to_gt(gt_labels, gt_f_masks, gt_f_bboxes, l_order, results, mask_thr=0.6, bbox_thr=0.8):
    """match the predicted mask to gt for training"""
    if len(results) == 3:
        bbox_results, segm_results, occ_results = results
    else:
        raise NotImplementedError('selected results must consist of occ result')

    del_inds = l_order < 0
    max_i_ind = 0
    max_ious = 0
    max_bbox = 0
    flag = False
    masks = []
    for i, (gt_label, gt_f_mask, gt_f_bbox) in enumerate(zip(gt_labels, gt_f_masks, gt_f_bboxes)):
        if l_order[i] < 0:
            continue
        pre_bbox = bbox_results[gt_label - 1]
        if len(pre_bbox) == 0 or pre_bbox[:, -1].max() < 0.3:
            continue
        pre_rle = segm_results[gt_label - 1]
        h, w = pre_rle[0]['size']
        gt_rle = maskUtils.encode(np.array(gt_f_mask[:h, :w, None], order='F'))
        # caculate the iou for mask and bbox
        ious_mask_all = maskUtils.iou(pre_rle, gt_rle, [0])
        ious_mask = ious_mask_all.max()
        ious_bbox = maskUtils.iou(pre_bbox[:, :-1], [gt_f_bbox.data.cpu().numpy()], [0]).max()
        if ious_mask > mask_thr and ious_bbox > bbox_thr:
            del_inds[i] = True
            flag = True
            masks.append(torch.tensor(maskUtils.decode(pre_rle[np.argmax(ious_mask_all)])).unsqueeze(0))
        elif ious_mask > max_ious or ious_bbox > max_bbox:
            max_i_ind = i
            max_ious = ious_mask
            max_bbox = ious_bbox

    # at least masked one object out each time
    if not flag:
        del_inds[max_i_ind] = True
        masks.append(torch.tensor(gt_f_masks[max_i_ind]).unsqueeze(0))

    return del_inds, masks


def l_order_update(l_order, del_inds, p_order):
    """update the occlusion order for end-to-end training"""
    updated_order = torch.where(l_order < 0, l_order - 1, l_order)
    for i, item in enumerate(del_inds):
        if item and updated_order[i] > -1:
            updated_order[i] = -1
            p_order[i] = 0
            p_order[:, i] = 0
    for i, item in enumerate(updated_order):
        if item > 0:
            value = p_order[i] == -1
            # if all front objects have been selected out, change the value to 0
            if value.sum() == 0:
                updated_order[i] = 0
    return updated_order


######################################################################################
# collect the layered results
######################################################################################
def add_pre_layer(results, pre_layer=0):
    """add the current layer number to the binary predicted layer order"""
    if len(results) == 3:
        bbox_results, segm_results, occ_results = results
    else:
        raise NotImplementedError('selected results must consist of occ result')

    for i, occ_result in enumerate(occ_results):
        if occ_result.shape[0] > 0:
            pre_layer_numpy = np.array(pre_layer).repeat(occ_result.shape[0]).reshape(-1, 1)
            occ_results[i] = np.append(occ_result, pre_layer_numpy, axis=1)
        else:
            occ_results[i] = np.zeros((0, 3), dtype=np.float32)

    update_results = (bbox_results, segm_results, occ_results)

    return update_results


def collect_layer_results(results):
    """collect the decomposition results from different layer"""
    if len(results) == 1:
        return results[0]
    else:
        bbox_results, segm_results, occ_results = results[0]
        for i in range(1, len(results)):
            result = results[i]
            bbox_results_n, segm_results_n, occ_results_n = result
            for j, (bbox_results_j, segm_results_j, occ_results_j) in enumerate(zip(bbox_results_n, segm_results_n, occ_results_n)):
                bbox_results[j] = np.vstack((bbox_results[j], bbox_results_j))
                occ_results[j] = np.vstack((occ_results[j], occ_results_j))
                segm_results[j].extend(segm_results_j)

        results = (bbox_results, segm_results, occ_results)

    return results


def iter_flag(results, score_thr=0.3):
    """get iteration flag for testing, if no objects are detected, stop the testing"""
    if len(results) == 3:
        bbox_results, segm_results, occ_results = results
    else:
        raise NotImplementedError('selected results must consist of occ result')

    bboxes = np.vstack(bbox_results)
    occs = np.vstack(occ_results)

    ind = np.where(bboxes[:, -1] > score_thr)[0]
    occs = occs[ind]
    occ_labels = np.argmax(occs, axis=1)

    if (occ_labels == 1).sum() == 0 or bboxes.shape[0] == 1:
        return False
    else:
        return True


def rle2mask(results, score_thr=0.3):
    """get segmented masks for the completion network"""
    if len(results) == 3:
        bbox_results, segm_results, occ_results = results
    else:
        raise NotImplementedError('selected results must consist of occ result')

    bboxes = np.vstack(bbox_results)
    masks = []

    if segm_results is not None:
        segms = mmcv.concat_list(segm_results)
        v_mask = np.ones_like(maskUtils.decode(segms[0]).astype(np.bool))
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            binary = mask.astype(np.uint8) * 255
            dila = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilation = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, dila, iterations=1)
            masks.append(torch.tensor(dilation > 0).view(1, -1,))
            dila = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            dilation = cv2.dilate(binary, dila, iterations=1)
            v_mask = np.where(dilation > 0, np.zeros_like(v_mask), v_mask)#np.where(mask > 0, np.zeros_like(v_mask), v_mask)#
    else:
        v_mask = None

    v_mask = torch.tensor(v_mask).view(1, -1,)

    return v_mask, masks