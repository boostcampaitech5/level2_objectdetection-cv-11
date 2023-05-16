# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00003,
    betas=(0.9, 0.999),
    weight_decay=0.03,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'cpb_mlp': dict(decay_mult=0.),
            'logit_scale': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        })
)

lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=7,
    step_ratio_up=0.3,
    gamma=0.9,
    by_epoch=False
)
runner = dict(type='EpochBasedRunner', max_epochs=28)

optimizer_config = dict(grad_clip=None)
