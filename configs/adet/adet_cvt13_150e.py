_base_ = [
    '../_base_/datasets/coco_one_shot_detection.py',
    # '../_base_/schedules/schedule_2x.py',
    # '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='Adet',
    backbone=dict(
        type='CvTAdet',
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
        #     checkpoint='saved_models/cvt_weights/CvT-21-384x384-IN-22k.pth')
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 192, 384],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
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
        max_per_img=100),

    nms_cfg=dict(
        class_agnostic=True,
        batch_nms_cfg = dict(
            iou_thr=0.5,
        ),
        max_per_img=100
    )
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[100])
runner = dict(type='EpochBasedRunner', max_epochs=150)
checkpoint_config = dict(interval=5)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(jsonfile_prefix='results/adet_cvt13_150e/',
classwise=True, interval=5)
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
find_unused_parameters = True
load_from='saved_models/cvt_weights/CvT-13-384x384-IN-22k-backbone.pth'
