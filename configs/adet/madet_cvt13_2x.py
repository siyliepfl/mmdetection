_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='MAdet',
    backbone=dict(
        type='MultiCvTAdet',
        in_chans=3,
        init= 'trunc_norm',
        num_stages=3,
        spec = dict( PATCH_SIZE=[7, 3, 3],
        PATCH_STRIDE= [4, 2, 2],
        PATCH_PADDING= [2, 1, 1],
        DIM_EMBED= [64, 192, 384],
        NUM_HEADS= [1, 3, 6],
        DEPTH= [1, 2, 10],
        MLP_RATIO= [4.0, 4.0, 4.0],
        ATTN_DROP_RATE= [0.0, 0.0, 0.0],
        DROP_RATE= [0.0, 0.0, 0.0],
        DROP_PATH_RATE= [0.0, 0.0, 0.1],
        QKV_BIAS= [True, True, True],
        CLS_TOKEN= [False, False, False],
        POS_EMBED= [False, False, False],
        QKV_PROJ_METHOD= ['dw_bn', 'dw_bn', 'dw_bn'],
        KERNEL_QKV= [3, 3, 3],
        PADDING_KV= [1, 1, 1],
        STRIDE_KV= [2, 2, 2],
        PADDING_Q= [1, 1, 1],
        STRIDE_Q= [1, 1, 1],
        FREEZE_BN= True),
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='saved_models/cvt_weights/CvT-13-384x384-IN-22k.pth')
            ),
    neck=dict(
        type='FPN',
        in_channels=[64, 192, 384],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=3,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='MAdetHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32],
        regress_ranges = ((-1, 64), (64, 128), (128, 256)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            img_db_path='data/lvis/train_imgs.hdf5',
            backend='hdf5',
            type='lvis')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            img_db_path='data/lvis/val2017.h5',
            backend='hdf5',
            type='lvis')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=2,
#     train=dict(pipeline=train_pipeline),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/lvis/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# # optimizer
# optimizer = dict(
#     lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='constant',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[16, 22])
# runner = dict(type='EpochBasedRunner', max_epochs=24)

find_unused_parameters = True
load_from='saved_models/cvt_weights/CvT-13-384x384-IN-22k-backbone.pth'