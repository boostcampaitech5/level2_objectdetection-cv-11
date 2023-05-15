_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[0.48488008,0.46039949,0.4316564], std=[0.21189889,0.20929213,0.214843], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='RandomResizedCrop',height=512,width=512,scale=[0.2,0.4,0.6,0.8,1.0], ratio=[0.7,1.5],interpolation=1,p=0.5),
    # dict(type='Cutout',num_holes=8, max_h_size=48, max_w_size=48, p=0.5),
    # dict(type='HorizontalFlip',p=0.5),
    # dict(type='VerticalFlip',p=0.5),
    # dict(type='RandomRotate90',p=0.5),
    # dict(type='HueSaturationValue',p=0.5),
    # dict(type='CLAHE',p=0.5),
    # dict(type='RandomBrightnessContrast',brightness_limit=[-0.2, 0.4],contrast_limit=[-0.5, 0.5],p=0.5),
    # dict(type='CenterCrop',height=864,width=864,p=0.5),
    # dict(type='RGBShift',r_shift_limit=10,g_shift_limit=10,b_shift_limit=10,p=0.5),
    # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.5),
    dict(type='ChannelShuffle', p=0.5),
    # dict(type='MedianBlur', blur_limit=3, p=0.5),
    # dict(type='Mosaic', img_scale=(1024, 1024), pad_val=img_norm_cfg["mean"][::-1], prob=0.3),
    # dict(type="Resize",img_scale=[(512 + 64 * i, 512 + 64 * i) for i in range(9)],multiscale_mode="value",keep_ratio=True),
    # dict(type='MixUp', pad_val=img_norm_cfg["mean"][::-1]),
    # dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
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