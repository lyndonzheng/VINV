from functools import partial

import cv2
import mmcv
import numpy as np
from six.moves import map, zip


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret


def layer_ranking(occ_order):
    """transfer the directed occlusion graph to layer ordering"""
    layer_num = 0
    order, new_order = occ_order.copy(), occ_order.copy()
    num = occ_order.shape[0]
    layer_order = np.zeros(num, dtype=int)
    idx = [i for i in range(num)]
    new_idx = idx.copy()
    while(len(idx)):
        for i in idx:
            value_n1 = (order[i] == -1).astype(np.uint8)
            if value_n1.sum() == 0:
                layer_order[i] = layer_num
                del new_idx[new_idx.index(i)]
                new_order[i] = 0
                new_order[:, i] = 0
        if len(idx) == len(new_idx):
            print('cycle direct map. select the least occluded value')
            del new_idx[new_idx.index(idx[0])]
            new_order[idx[0]] = 0
            new_order[:, idx[0]] = 0
        idx = new_idx.copy()
        order = new_order.copy()
        layer_num += 1

    return layer_order


def pairwise_ranking(f_masks, layer_orders):
    """transfer the layer order to directed occlusion graph"""
    num = len(f_masks)
    pairwise_order_matrix = np.zeros((num, num), dtype=np.int64)
    if num == 0:
        return pairwise_order_matrix
    for i in range(num):
        f_mask_i = f_masks[i]
        for j in range(i + 1, num):
            f_mask_j = f_masks[j]
            overlay = ((f_mask_i > 0) & (f_mask_j > 0)).astype(np.uint8)
            if overlay.sum() > 5 or bordering(f_masks[i], f_masks[j]):
                if layer_orders[i] < layer_orders[j]:
                    pairwise_order_matrix[i, j] = 1
                    pairwise_order_matrix[j, i] = -1
                elif layer_orders[i] > layer_orders[j]:
                    pairwise_order_matrix[i, j] = -1
                    pairwise_order_matrix[j, i] = 1
                else:
                    pairwise_order_matrix[i, j] = -1
                    pairwise_order_matrix[j, i] = -1
    return pairwise_order_matrix


def bordering(a, b):
    dilate_kernel = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.uint8)
    a_dilate = cv2.dilate(a.astype(np.uint8), dilate_kernel, iterations=1)
    return np.any((a_dilate == 1) & b)


def pairwise_order_area(f_masks, above='smaller'):
    """infer the occlusion order using mask area"""
    num = len(f_masks)
    pairwise_order_matrix = np.zeros((num, num), dtype=np.int64)
    if num == 0:
        return pairwise_order_matrix
    for i in range(num):
        for j in range(i + 1, num):
            overlay = ((f_masks[i] > 0) & (f_masks[j]> 0)).astype(np.uint8)
            if overlay.sum() > 5 or bordering(f_masks[i], f_masks[j]):
                area_i = f_masks[i].sum()
                area_j = f_masks[j].sum()
                if (area_i < area_j and above == 'larger') or \
                   (area_i >= area_j and above == 'smaller'):
                    pairwise_order_matrix[i, j] = -1
                    pairwise_order_matrix[j, i] = 1
                else:
                    pairwise_order_matrix[i, j] = 1
                    pairwise_order_matrix[j, i] = -1
    return pairwise_order_matrix


def pairwise_order_yaxis(f_masks):
    """infer the occlusion order using y axis"""
    num = len(f_masks)
    pairwise_order_matrix = np.zeros((num, num), dtype=np.int64)
    if num == 0:
        return pairwise_order_matrix
    for i in range(num):
        for j in range(i + 1, num):
            overlay = ((f_masks[i] > 0) & (f_masks[j] > 0)).astype(np.uint8)
            if overlay.sum() > 5 or bordering(f_masks[i], f_masks[j]):
                center_i = [coord.mean() for coord in np.where(f_masks[i] == 1)]
                center_j = [coord.mean() for coord in np.where(f_masks[j] == 1)]
                if center_i[0] < center_j[0]:
                    pairwise_order_matrix[i, j] = -1
                    pairwise_order_matrix[j, i] = 1
                else:
                    pairwise_order_matrix[i, j] = 1
                    pairwise_order_matrix[j, i] = -1
    return pairwise_order_matrix


def infer_gt_order(inmodal, amodal):
    """infer the pairwise order from visible and amodal mask"""
    if isinstance(inmodal, list):
        num = len(inmodal)
    else:
        num = inmodal.shape[0]
    gt_order_matrix = np.zeros((num, num), dtype=np.int)
    for i in range(num):
        for j in range(i + 1, num):
            overlay = ((amodal[i] > 0) & (amodal[j] > 0)).astype(np.uint8)
            if overlay.sum() > 5 or bordering(inmodal[i], inmodal[j]):
                occ_ij = ((inmodal[i] == 1) & (amodal[j] == 1)).sum()
                occ_ji = ((inmodal[j] == 1) & (amodal[i] == 1)).sum()
                if occ_ij == 0 and occ_ji == 0: # bordering but not occluded
                    continue
                gt_order_matrix[i, j] = 1 if occ_ij >= occ_ji else -1
                gt_order_matrix[j, i] = -gt_order_matrix[i, j]
    return gt_order_matrix


def eval_order(order_matrix, gt_order_matrix):
    """evaluate the pairwise order for the given order matric"""
    inst_num = order_matrix.shape[0]
    allpair_true = ((order_matrix == gt_order_matrix).sum() - inst_num) / 2
    allpair = (inst_num * inst_num - inst_num) / 2

    occpair_true = ((order_matrix == gt_order_matrix) & (gt_order_matrix != 0)).sum() / 2
    occpair = (gt_order_matrix != 0).sum() / 2

    err = np.where(order_matrix != gt_order_matrix)
    gt_err = gt_order_matrix[err]
    pred_err = order_matrix[err]
    show_err = np.concatenate([np.array(err).T + 1, gt_err[:, np.newaxis], pred_err[:, np.newaxis]], axis=1)
    return allpair_true, allpair, occpair_true, occpair, show_err
