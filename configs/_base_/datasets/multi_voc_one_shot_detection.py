# dataset settings
dataset_type = 'VocMultiOneShotDataset'
data_root = 'data/voc/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            img_db_path='data/voc/voc_0712_imgs.h5',
            backend='hdf5',
            type='lvis')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'query_img_list',
                               'query_labels_list', 'query_targets_list']),
]

query_train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            img_db_path='data/voc/voc_0712_imgs.h5',
            backend='hdf5',
            type='lvis')),
dict(type='SampleTarget', output_sz=128, search_area_factor=1),
dict(type='RandomFlip', flip_ratio=0.5),
dict(type='Normalize', **img_norm_cfg),
dict(type='DefaultFormatBundle'),
dict(type='Collect', keys=['img']),
]

query_test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            img_db_path='data/voc/voc_0712_imgs.h5',
            backend='hdf5',
            type='lvis')),
    dict(type='SampleTarget', output_sz=128, search_area_factor=1),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            img_db_path='data/voc/voc_0712_imgs.h5',
            backend='hdf5',
            type='lvis')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', pad_to_square=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

split = 0
average_num = 2
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/voc0712_train.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
        query_pipeline=query_train_pipeline,
        average_num = average_num,
        split=split,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/voc07_test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        query_pipeline=query_test_pipeline,
    average_num = average_num,
    split=split,
    query_json = data_root + '/voc0712_train.json',
    ),
    test=dict(
        type=dataset_type,
        # classes = data_root + 'oneshot/test_split_0.txt',
        ann_file=data_root + 'voc07_test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        query_pipeline=query_test_pipeline,
    average_num = average_num,
    split=split,
    query_json = data_root + 'voc0712_train.json',
    test_type = 'all'
    ))
evaluation = dict(interval=1, metric='mAP')
