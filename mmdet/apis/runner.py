import logging
import os.path as osp
import time
from collections import OrderedDict
from torch.nn.utils import clip_grad

import torch

import mmcv
from mmcv.runner import hooks, get_dist_info
from mmcv.runner.checkpoint import load_checkpoint, save_checkpoint
from mmcv.runner.hooks import (CheckpointHook, Hook, IterTimerHook, LrUpdaterHook, lr_updater)
from mmcv.runner.log_buffer import LogBuffer
from mmcv.runner.priority import get_priority
from mmcv.runner.utils import get_host_info, get_time_str, obj_from_dict


class Runner(object):
    """training forward and backward, rewrite the runner in mmcv"""
    def __init__(self, model, cfg=None):

        self.cfg = cfg
        self.model = model
        self.optimizers = []
        self.optimizer = None
        self.optimizer_sce_g = None
        optimizer_cfg = cfg.optimizer.deepcopy()
        # define the optimizer for scene understanding
        if hasattr(model, 'module'):
            model = model.module
        param_list = []
        if model.backbone:
            param_list.append(dict(params=model.backbone.parameters(), lr=optimizer_cfg['lr']))
            param_list.append(dict(params=model.neck.parameters(), lr=optimizer_cfg['branch_lr']))
            if model.with_rpn:
                param_list.append(dict(params=model.rpn_head.parameters(), lr=optimizer_cfg['branch_lr']))
            if model.with_shared_head:
                param_list.append(dict(params=model.shared_head.parameters(), lr=optimizer_cfg['branch_lr']))
            if model.with_bbox:
                param_list.append(dict(params=model.bbox_head.parameters(), lr=optimizer_cfg['branch_lr']))
            if model.with_mask:
                param_list.append(dict(params=model.mask_head.parameters(), lr=optimizer_cfg['branch_lr']))
            if model.with_occ:
                param_list.append(dict(params=model.occ_head.parameters(), lr=optimizer_cfg['branch_lr']))
            if model.with_semantic:
                param_list.append(dict(params=model.semantic_head.parameters(), lr=optimizer_cfg['branch_lr']))
            optimizer_cfg.pop('branch_lr')
            self.optimizer = obj_from_dict(optimizer_cfg, torch.optim, dict(params=param_list))
            self.optimizers.append(self.optimizer)
        # define the optimizer for refinement network
        if model.with_completion:
            self.optimizer_sce_g = torch.optim.Adam(
                [{'params': list(filter(lambda p: p.requires_grad, model.rgb_completion.net_G.parameters())), 'lr': 0.0001},
                 {'params': list(filter(lambda p: p.requires_grad, model.rgb_completion.net_E.parameters()))}], lr=0.0001,
                betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_sce_g)

        # create work_dir
        if mmcv.is_str(cfg.work_dir):
            self.work_dir = osp.abspath(cfg.work_dir)
            mmcv.mkdir_or_exist(self.work_dir)
        elif cfg.work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        if cfg.log_level is not None:
            self.logger = self.init_logger(cfg.work_dir, cfg.log_level)
            self.log_buffer = LogBuffer()

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._l_max_iters = cfg.l_max_iters
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def _add_file_handler(self, logger, filename=None, mode='w', level=logging.INFO):
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    def _forward_flag(self, l_orders):
        # keep training for one view
        flag = True
        if not isinstance(l_orders, list):
            l_orders = l_orders.data[0]
        for l_order in l_orders:
            if (l_order[:-1] > 0).sum() == 0:
                flag = False
        return flag

    def _update_data(self, data_batch, results):
        # update data for layer-by-layer training based on the results
        data_batch['l_orders'] = results['l_orders']
        data_batch['img'] = results['img']
        if 'proposals' in results.keys():
            data_batch['proposals'] = results['proposals']

        return data_batch

    def init_logger(self, log_dir=None, level=logging.INFO):
        """Init the logger.

        Args:
            log_dir(str, optional): Log file directory. If not specified, no
            log file will be used.
            level (int or str): See the built-in python logging module.

        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        logging.basicConfig( format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        logger = logging.getLogger(__name__)
        if log_dir and self.rank == 0:
            filename = '{}.log'.format(self.timestamp)
            log_file = osp.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        return logger

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        current_lr = []
        if len(self.optimizers)==0:
            raise RuntimeError('lr is not applicable because optimizer does not exist.')
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                current_lr.append(group['lr'])
        return current_lr

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError('"args" must be either a Hook object'
                            ' or dict, not {}'.format(type(args)))

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        # self._epoch = checkpoint['meta']['epoch']
        # self._iter = checkpoint['meta']['iter']
        # load the synthesis completion weight to real completion network
        if self.model.module.with_completion and self.model.module.with_completion_wogt:
            for name, param in self.model.module.rgb_completion.state_dict().items():
                self.model.module.rgb_completion_synthesis.state_dict()[name].copy_(param)
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def parse_losses(self, losses):
        """parse different loss"""
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError('{} is not a tensor or list of tensors'.format(loss_name))

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for name in log_vars:
            log_vars[name] = log_vars[name].item()

        return loss, log_vars

    def forward(self, data):
        """forward to get the loss and output"""
        losses, results = self.model(**data)
        loss, log_vars = self.parse_losses(losses)

        self.outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return self.outputs, results

    def optimize_parameters(self):
        """backward the loss and update the network"""
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        self.outputs['loss'].backward()
        if self.cfg.optimizer_config is not None and self.optimizer is not None:
            clip_grad.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), **self.cfg.optimizer_config['grad_clip'])
        # optimization
        for optimizer in self.optimizers:
            optimizer.step()

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch')
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            steps = 0
            data_batch['ori_img'] = data_batch['img']
            while(self._forward_flag(data_batch['l_orders']) and (steps<self._l_max_iters)):
                self.call_hook('before_train_iter')
                data_batch['iters'] = self._iter
                data_batch['epoch'] = self._epoch
                data_batch['steps'] = steps
                outputs, results = self.forward(data_batch)
                self.optimize_parameters()
                data_batch = self._update_data(data_batch, results)
                if 'log_vars' in outputs:
                    self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
                self.call_hook('after_train_iter')
                self._iter += 1
                steps +=1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=False, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.
                            format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def register_lr_hooks(self, lr_config):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            # from .hooks import lr_updater
            hook_name = lr_config['policy'].title() + 'LrUpdaterHook'
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError('"lr_config" must be either a LrUpdaterHook object'
                            ' or dict, not {}'.format(type(lr_config)))

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = obj_from_dict(
                info, hooks, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='VERY_LOW')

    def register_training_hooks(self,
                                lr_config,
                                checkpoint_config=None,
                                log_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        if checkpoint_config is None:
            checkpoint_config = {}
        if self.optimizer is not None:
            self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config)