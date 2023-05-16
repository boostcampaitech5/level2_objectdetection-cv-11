_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/albu_coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


dataset_type = 'CocoDataset'
data_root = '/opt/ml/dataset/'
img_norm_cfg = dict(
    mean=[123.6444204, 117.40186995, 110.072382], std=[54.03421695, 53.36949315, 54.784965], to_rgb=True)
albu_train_transforms = [
    #dict(type='RandomResizedCrop',height=1024,width=1024,scale=[0.2,1.0], ratio=[0.7,1.5],interpolation=1,p=0.5),
    #dict(type='Cutout',num_holes=8, max_h_size=48, max_w_size=48, p=0.5),
    #dict(type='HorizontalFlip',p=0.5),
    #dict(type='VerticalFlip',p=0.5),
    #dict(type='RandomRotate90',p=0.5),
    #dict(type='HueSaturationValue',p=0.5),
    #dict(type='CLAHE',p=0.5),
    dict(type='RandomBrightnessContrast',brightness_limit=[-0.2, 0.4],contrast_limit=[-0.5, 0.5],p=0.5),
    #dict(type='CenterCrop',height=864,width=864,p=0.5),
    #dict(type='RGBShift',r_shift_limit=10,g_shift_limit=10,b_shift_limit=10,p=0.5),
    #dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.5),
    #dict(type='ChannelShuffle', p=0.5),
    #dict(type='MedianBlur', blur_limit=3, p=0.5),
]
train_pipeline = [
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    #dict(type='Mosaic', img_scale=(1024, 1024), pad_val=img_norm_cfg["mean"][::-1], prob=0.3),
    #dict(type='MixUp', pad_val=img_norm_cfg["mean"][::-1]),
    #dict(type="Resize",img_scale=[(512 + 64 * i, 512 + 64 * i) for i in range(9)],multiscale_mode="value",keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',keys=['img', 'gt_bboxes', 'gt_labels'],
        # meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
        #            'pad_shape', 'scale_factor')
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=[(512 + 64 * i, 512 + 64 * i) for i in range(9)],
        img_scale = (1024,1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

data = dict(
    samples_per_gpu=14,
    workers_per_gpu=2,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + '2___train_MultiStfKFold.json',
            img_prefix=data_root,
            classes=classes,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ]
            ),
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '2___val_MultiStfKFold.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(save_best='bbox_mAP_50', metric='bbox')