## Multilabel_Kfold.py 돌리는 방법

1. python3 level2_objectdetection-cv-11/tools/multilabel_Kfold.py --K {kfold 분할 개수(default:5)}
2. multilabel_Kfold.py 폴더 경로에 들어가서 python3 multilabel_Kfold.py

```bash
python3 level2_objectdetection-cv-11/tools/multilabel_Kfold.py --K 5
```

<br></br>

## main.py 돌리는 방법
```bash
python3 level2_objectdetection-cv-11/tools/main.py
```

main.py 폴더 경로에 들어가서 python3 main.py

### parser

- --max_epoch: default=20, 학습 돌릴 epoch 횟수

- --model: config 파일에서 실행할 .py 이름

- --folder: config 파일에서 실행할 .py의 폴더 제목

- --augmentation: default=False, augmentation 여부

- --trainset: default='2___train_MultiStfKFold.json', input 데이터셋

- --validset, default='2___val_MultiStfKFold.json', input validset

- --resize, default=1024

- --inference_epoch: default="latest", inference할 때 어떤 epoch 사용할 것인지 (숫자 or latest, best)

<br></br>
## train.sh 파일 돌리는 방법
```bash
nohup sh level2_objectdetection-cv-11/tools/train.sh
```
- train.sh 파일에 python3 main.py 명령어 작성
```bash
python3 main.py --model cascade_rcnn_r50_fpn_1x_coco --folder cascade_rcnn --resize 512 --max_epoch 20 --inference_epoch best
```

## UniverseNet 오류 해결방법
```bash
pip install -v -e .
```
모델 backbone SyncBN에서 BN으로 수정함. (SyncBN은 특정 상황에서만 작동)
