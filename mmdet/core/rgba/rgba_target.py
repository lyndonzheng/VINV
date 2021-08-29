import mmcv
import numpy as np
import torch
from torch.nn.modules.utils import _pair


def rgba_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, gt_rgb, cfg):
    """
    :param pos_proposals_list: positive objects bounding box list
    :param pos_assigned_gt_inds_list: positive objects index list
    :param gt_masks_list: ground truth mask list
    :param gt_rgb: ground truth rgb list, (full image or visible image)
    :param cfg:
    :return: rgba target list
    """
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    rgba_targets = map(rgba_target_single, pos_proposals_list, pos_assigned_gt_inds_list,
                       gt_masks_list, gt_rgb, cfg_list)
    rgba_targets = torch.cat(list(rgba_targets))

    return rgba_targets


def rgba_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, gt_rgbs, cfg):
    mask_size = _pair(cfg.mask_size)
    keep_ratio = cfg.keep_ratio
    num_pos = pos_proposals.size(0)
    rgba_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        gt_rgbs = gt_rgbs.cpu().numpy()
        _, maxh, maxw = gt_masks.shape
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw - 1)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh - 1)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            if gt_rgbs.ndim == 4:
                gt_rgb = gt_rgbs[pos_assigned_gt_inds[i]]
            else:
                gt_rgb = gt_rgbs
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            if keep_ratio:
                # if keep ratio, using the cube to crop the target
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                l = w if w > h else h
                x1 = (center_x - l / 2).astype(np.int32)
                y1 = (center_y - l / 2).astype(np.int32)
                w = l
                h = l
            a_target = mmcv.imresize(gt_mask[y1: y1 + h, x1: x1 + w], mask_size[::-1]).reshape(-1, mask_size[0], mask_size[1])
            rgb_target = mmcv.imresize(gt_rgb[:, y1: y1 + h, x1: x1 + w].transpose(1,2,0), mask_size[::-1]).transpose(2,0,1)
            rgb_target = np.where(a_target > 0, rgb_target, -1*np.ones_like(rgb_target))
            rgba_target = np.concatenate((a_target.astype(np.float) * 2 -1, rgb_target), axis=0)
            rgba_targets.append(rgba_target)
        rgba_targets = torch.from_numpy(np.stack(rgba_targets)).float().to(pos_proposals.device)

    else:
        rgba_targets = pos_proposals.new_zeros((0, ) + mask_size)
    return rgba_targets