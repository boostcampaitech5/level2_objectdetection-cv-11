_base_ = ['./tood_r101_fpn_mstrain_2x_coco.py'
    '../_base_/datasets/trash_augmentation.py',
    '../_base_/schedules/schedule_adamw.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    bbox_head=dict(num_dcn=2))
