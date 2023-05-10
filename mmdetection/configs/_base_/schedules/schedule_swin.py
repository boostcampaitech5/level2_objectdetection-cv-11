# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))

# learning policy chapGPT 추천 config
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-6,
)


runner = dict(type='EpochBasedRunner', max_epochs=20)

# 설정 내용
# optimizer를 AdamW로 변경, learning rate는 0.0001, weight decay는 0.01로 설정
# optimizer grad_clip 설정 추가
# lr_config를 CosineAnnealingWarmRestarts로 변경
# learning rate의 최솟값인 eta_min을 1e-6으로 설정
# 학습률을 첫번째 주기(periods)동안은 1, 두번째 주기동안은 0.5로 가중치를 두어 주기마다 학습률이 감소하도록 설정
# 총 학습 epochs는 12로 설정