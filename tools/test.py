import argparse
import os
import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.core import eval, results2json, wrap_fp16_model, tensor2im, save_image, mkdirs
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def single_gpu_test(model, data_loader, show=False, path=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
            with torch.no_grad():
                de_result, co_result = model(return_loss=False, rescale=True, **data)
            results.append(de_result)

            if len(co_result) > 0:
                save_results(co_result, data['img_meta'], path=path+'completion')

            if show:
                model.module.show_result(data, de_result, show=False, pre_order=True, out_file=path+'decomposition', score_thr=0.5)

            if isinstance(data['img'], list):
                batch_size = data['img'][0].size(0)
            else:
                batch_size = data['img'].data[0].size(0)
            for _ in range(batch_size):
                prog_bar.update()
    return results


def save_results(outputs, img_meta, path=None):
    """save the completion results to disk"""
    if isinstance(img_meta, list):
        img_meta = img_meta[0]
    filename = img_meta.data[0][0]['filename']
    content = filename.split('/')
    scene_path = path+'/scene/'+content[-3]
    object_path = path+'/object/'+content[-3]
    mkdirs(scene_path)
    mkdirs(object_path)
    layer = 1
    nums = 0
    for data in outputs:
        co_scene, obj_dets = data
        sce_numpy = tensor2im(co_scene)
        sce_name = content[-1].split('.')[0]
        name = sce_name + '_' + format(layer, '02d') + '.png'
        img_path = os.path.join(scene_path, name)
        save_image(sce_numpy, img_path)
        if obj_dets is not None:
            for obj_det in obj_dets:
                if len(obj_det) == 2:
                    obj_det, ind = obj_det
                else:
                    ind = nums
                obj_numpy = tensor2im(obj_det)
                name = sce_name + '_' + format(ind+1, '02d') + '.png'
                img_path = os.path.join(object_path, name)
                save_image(obj_numpy, img_path)
                nums +=1
        layer += 1


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--out_path', default='.', help='output_path to store the completed image')
    parser.add_argument('--with_occ', action='store_true', help='decompose the result with occlusion label')
    parser.add_argument('--order_method', default='depth', help='how to infer the occlusion based on the mask')
    parser.add_argument('--json_out', help='output result file name without extension', type=str)
    parser.add_argument('--eval', type=str, nargs='+', choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints', 'depth'],help='eval types')
    parser.add_argument('--show', action='store_true', help='show and save results')
    parser.add_argument('--gpu_collect', action='store_true', help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    for pipeline in cfg.data.test['pipeline']:
        if pipeline['type'] == 'LoadAnnotations':
            # some ablation study need to load the ground truth annotation
            cfg.data.test.test_mode = False

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # if json file is existed, directly testing
    if os.path.exists(args.out):
        outputs = mmcv.load(args.out)
        result_files = results2json(dataset, outputs, args.out, with_occ=args.with_occ)
        eval(result_files, args.eval, dataset, cfg.data, classwise=True, order_method=args.order_method,
             full_mask=True, full_bbox=True, pr_curve=False)
    else:
        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.out_path)

        rank, _ = get_dist_info()
        if args.out and rank == 0:
            print('\nwriting results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)
            eval_types = args.eval
            if eval_types:
                print('Starting evaluate {}'.format(' and '.join(eval_types)))
                result_files = results2json(dataset, outputs, args.out, with_occ=args.with_occ)
                eval(result_files, args.eval, dataset, cfg.data, classwise=True, order_method=args.order_method,
                     full_mask=True, full_bbox=True, pr_curve=False)


if __name__ == '__main__':
    main()
