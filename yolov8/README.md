## 1. ultralytics 설치
```bash
pip install ultralytics
```
## 2. 폴더 구성하기

![image](https://github.com/boostcampaitech5/level2_objectdetection-cv-11/assets/69185594/6449d48e-b1c4-44d1-8a97-3d6645bc9406)

## 3. coco2yolo.py 명령어
- valid 없이 학습할때
```bash
python3 coco2yolo.py -j '/opt/ml/dataset/train.json' -o '/opt/ml/dataset/labels'
```
- kfold로 train, valid 나누어서 학습하고 싶을 때
```bash
python3 coco2yolo.py -j '/opt/ml/dataset/images/2___train_MultiStfKFold.json' -o '/opt/ml/dataset/labels'
```

## 4. train.py 명령어
```bash
python3 train.py
```
## 5. inference.py 명령어
```bash
python3 inference.py --model {best.pt 경로}
```

## 주요 파일
ultralytics/yolo/cfg/default.yaml
- train 파라미터 바꾸는 부분

ultralytics/datasets/recycle.yaml
- 학습데이터셋 파일

