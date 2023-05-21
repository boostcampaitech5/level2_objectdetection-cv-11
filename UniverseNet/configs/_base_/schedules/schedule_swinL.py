optimizer_config = dict(grad_clip=None)
optimizer = dict(
    type='AdamW',
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=1e-8,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# 학습률 스케줄러 설정
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-3
)
runner = dict(type='EpochBasedRunner', max_epochs=30)

