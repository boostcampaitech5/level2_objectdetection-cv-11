optimizer = dict(
    type="AdamW",
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        )
    ),
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 15],
)


# fp16 = dict(loss_scale="dynamic")
runner = dict(
    type="EpochBasedRunner",  # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=10,
)  # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`
