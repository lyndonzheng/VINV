# model settings
model = dict(
    type='LBLCompletedRGBHTC',
    mode='end',
    num_stages=3,
    pretrained='torchvision://resnet50',
    interleaved=True,
    mask_info_flow=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=[
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=35,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=True,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=35,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.05, 0.05, 0.1, 0.1],
            reg_class_agnostic=True,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=35,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.033, 0.033, 0.067, 0.067],
            reg_class_agnostic=True,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    ],
    occ_head=dict(
            type='OccHead',
            with_avg_pool=False,
            with_occ=True,
            in_channels=256,
            num_classes=2,
            loss_occ=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0)),
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    mask_head=dict(
        type='HTCMaskHead',
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=35,
        loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
    semantic_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[8]),
    semantic_head=dict(
        type='FusedSemanticHead',
        num_ins=5,
        fusion_level=1,
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=35,
        ignore_label=0,
        loss_weight=0.2),
    rgb_completion=dict(
        type='RGB_COMPLETION',
        input_nc=3,
        output_nc=3,
        output_scale=4,
        base_nc=64,
        down_sampling=4,
        dilation=4,
        init_type='xavier',
        loss_d=dict(type='GANloss', gan_mode='lsgan', loss_weight=1.0),
        loss_g=dict(
                loss_g=dict(type='GANloss', gan_model='lsgan', loss_weight=1.0),
                loss_rec=dict(type='l1loss', loss_weight=10.0),
                loss_vgg=dict(type='l1loss', loss_weight=1.0)
            )
        ),
    )
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weights=[1, 0.5, 0.25])
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.3,#0.001,#
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
        mask_thr_binary=0.5),
    keep_all_stages=False)
# dataset settings
dataset_type = 'CsdDataset'
data_root = '/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/suncg_data/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(
    mean=[128, 128, 128], std=[128, 128, 128], to_rgb=True)
semantic_ignore_list = []#[0, 2, 3, 4, 37, 38, 39, 40]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_depth=True, with_label=True, with_f_bbox=True, with_bbox=False, with_f_mask=True,
         with_mask=False, with_seg=True, with_f_rgb=True, with_f_depth=True, with_l_order=True, with_p_order=True,
         poly2mask=False),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegResizeFlipPadRescale', scale_factor=1 / 8, ignore_list=semantic_ignore_list, seg_num=41),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'depth', 'gt_f_bboxes', 'gt_labels', 'gt_f_masks', 'l_order', 'p_order', 'gt_semantic_seg', 'f_rgbs', 'f_depths']
    ),
]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_depth=True, with_label=True, with_f_bbox=True, with_bbox=False, with_f_mask=True,
#                 with_mask=False, with_seg=True, with_f_rgb=True, with_f_depth=True, with_l_order=True, with_p_order=True,
#                 poly2mask=False),
#     dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.0),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='SegResizeFlipPadRescale', scale_factor=1 / 8, ignore_list=semantic_ignore_list, seg_num=41),
#     dict(type='DefaultFormatBundle'),
#     dict(
#         type='Collect',
#         keys=['img', 'depth', 'gt_f_bboxes', 'gt_labels', 'gt_f_masks', 'l_order', 'p_order', 'gt_semantic_seg', 'f_rgbs', 'f_depths']
#     ),
# ]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# data = dict(
#    imgs_per_gpu=1,
#    workers_per_gpu=1,
#    train=dict(
#        type=dataset_type,
#        data_root=data_root,
#        ann_file=data_root + 'sosc_new_train_order.json',
#        img_prefix=data_root + 'sosc_new/',
#        seg_prefix=data_root + 'sosc_new/',
#        pipeline=train_pipeline),
#    val=dict(
#        type=dataset_type,
#        data_root=data_root,
#        ann_file=data_root + 'sosc_new_test_order.json',
#        img_prefix=data_root + 'sosc_new/',
#        seg_prefix=data_root + 'sosc_new/',
#        pipeline=test_pipeline),
#    test=dict(
#        type=dataset_type,
#        data_root=data_root,
#        ann_file=data_root + 'sosc_new_test_order.json',
#        img_prefix=data_root + 'sosc_new/',
#        seg_prefix=data_root + 'sosc_new/',
#        pipeline=test_pipeline))
# data = dict(
#     imgs_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'sosc_new_train_order.json',
#         img_prefix=data_root + 'sosc_new_copy/',
#         seg_prefix=data_root + 'sosc_new_copy/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'sosc_new_test_order.json',
#         img_prefix=data_root + 'sosc_new_copy/',
#         seg_prefix=data_root + 'sosc_new_copy/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'sosc_new_test_order_short.json',
#         img_prefix=data_root + 'sosc_new_copy/',
#         seg_prefix=data_root + 'sosc_new_copy/',
#         pipeline=test_pipeline))
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sosc_new_train_order.json',
        img_prefix=data_root + 'sosc_new_copy1/',
        seg_prefix=data_root + 'sosc_new_copy1/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sosc_new_test_order.json',
        img_prefix=data_root + 'sosc_new_copy1/',
        seg_prefix=data_root + 'sosc_new_copy1/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sosc_new_train_example.json',
        img_prefix=data_root + 'sosc_new_copy1/',
        seg_prefix=data_root + 'sosc_new_copy1/',
        pipeline=test_pipeline))
# optimizer
# optimizer = dict(type='SGD', lr=0.000, momentum=0.9, weight_decay=0.0001, branch_lr=0.0025)
optimizer = dict(type='SGD', lr=0.0000, momentum=0.9, weight_decay=0.0001, branch_lr=0.00025)
# optimizer = dict(type='SGD', lr=0.000, momentum=0.9, weight_decay=0.0001, branch_lr=0.0000)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/sosc/htc_r50_fpn_1x_end'
load_from = None
resume_from = ''
workflow = [('train', 1)]
