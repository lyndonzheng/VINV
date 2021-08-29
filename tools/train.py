from __future__ import division
import argparse
import os

import torch
from mmcv import Config
from mmcv.runner import init_dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet import __version__
from mmdet.apis import get_root_logger, set_random_seed, train_detector, Runner
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')
    parser.add_argument('--validate', action='store_true', help='whether to evaluate the checkpoint during training')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--launcher',choices=['none', 'pytorch', 'slurm', 'mpi'],default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale_lr', action='store_true', help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('MMDetection Version: {}'.format(__version__))
    logger.info('Config: {}'.format(cfg.text))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # distribute the data and model to GPU
    if distributed:
        data_loaders = [build_dataloader(ds, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, dist=True) for ds in datasets]
        model = MMDistributedDataParallel(model.cuda())
    else:
        data_loaders = [build_dataloader(ds, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, cfg.gpus, dist=False) for ds in datasets]
        model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build the runner
    runner = Runner(model, cfg)
    # register the checkpoint and log hook/ remove the lr and optimizer hook for we use different oprimizers
    # and different learning rate for different tasks
    runner.register_training_hooks(cfg.lr_config, cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from, resume_optimizer=False)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


if __name__ == '__main__':
    main()
