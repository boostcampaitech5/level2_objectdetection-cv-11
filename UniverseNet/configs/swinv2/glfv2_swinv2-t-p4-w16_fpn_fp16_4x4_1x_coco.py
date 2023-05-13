_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window16_256.pth'  # noqa
model = dict(
    type='GFL', # 변경
    backbone=dict(
        type='SwinTransformerV2',
        pretrain_img_size=256,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=16,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(1, 2, 3),
        use_checkpoint=False,
        pretrained_window_sizes=[0, 0, 0, 0],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead', # 변경
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        # bbox_coder=dict(
        #     type='DeltaXYWHBBoxCoder',
        #     target_means=[.0, .0, .0, .0],
        #     target_stds=[0.1, 0.1, 0.2, 0.2]),
        # loss_cls=dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0), # glfv2일때 사용
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        use_dgqp=True,
        reg_topk=4,
        reg_channels=64,
        add_mean=True,
        one_more_cls_out_channels=False,  # True to load official weights
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
        # loss_centerness=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)), # gflv2에서는 사용하지 않음
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))


# data = dict(samples_per_gpu=4)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'cpb_mlp': dict(decay_mult=0.),
            'logit_scale': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000)

fp16 = dict(loss_scale=dict(init_scale=512))
