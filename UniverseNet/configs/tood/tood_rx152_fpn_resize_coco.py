_base_ = './tood_rx101_fpn_resize_coco.py'

model = dict(
    backbone=dict(
        depth=152,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet152')))
