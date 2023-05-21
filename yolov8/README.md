
# Folder Structure
```bash
├─.github
│  ├─ISSUE_TEMPLATE
│  └─workflows
├─ultralytics
│    ├─datasets
│   ├─hub
│   ├─models
│   │  ├─v3
│   │  ├─v5
│   │  └─v8
│   ├─nn
│   ├─tracker
│   │  ├─cfg
│   │  ├─trackers
│   │  └─utils
│   └─yolo
│       ├─cfg
│       ├─data
│       │  ├─dataloaders
│       │  └─scripts
│       ├─engine
│       ├─utils
│       │  └─callbacks
│       └─v8
│           ├─classify
│           ├─detect
│           └─segment
├─train.py
├─inference.py
├─coco2yolo.py
├─setup.py
└─setup.cfg


```
<br></br>

# Cloning
```bash
git clone https://github.com/Kim-jy0819/yolov8.git
```
<br></br>

# Installation
```bash
cd yolov8
pip install ultralytics
```

<br></br>

# Configure a data folder
```bash
dataset
└── images
      ├── train (*.jpg)
      ├── valid
      └── test
└── labels
      ├── train (*.txt)
      └── valid
├── train.json
└── test.json
```

<br></br>

# Convert coco json to yolo format
```bash
python3 coco2yolo.py -j 'dataset/train.json' -o 'dataset/labels'
```
<br></br>

# Usage
```bash
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="recycle.yaml", epochs=100)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# predict on an image
success = model.export(format="onnx")  # export the model to ONNX format
```
## train.py
```bash
python3 train.py
```
```bash
yolo mode=train model=yolov8x.pt data=/home/jun/project/level2_objectdetection-cv-11/yolov8/ultralytics/datasets/recycle.yaml imgsz=1024 epochs=70 batch=3 optimizer=AdamW max_det=300 lr0=0.0001 mosaic=0.5 close_mosaic = 35
```

## inference.py
```bash
python3 inference.py --model {best.pt path}
```
```bash
yolo mode=predict model=yolov8x.pt data=/home/jun/project/level2_objectdetection-cv-11/yolov8/ultralytics/datasets/recycle.yaml imgsz=1024 epochs=70 batch=3 optimizer=AdamW max_det=300 lr0=0.0001 mosaic=0.5 close_mosaic = 35
```
<br></br>

# Key Files
## ultralytics/yolo/cfg/default.yaml
- part that changes the train parameters

## ultralytics/datasets/recycle.yaml
- Train Dataset Files

<br></br>

# Models

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models) download automatically from the latest
Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

<details open><summary>Detection</summary>

See [Detection Docs](https://docs.ultralytics.com/tasks/detect/) for usage examples with these models.

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](http://cocodataset.org) dataset.
  <br>Reproduce by `yolo val detect data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)
  instance.
  <br>Reproduce by `yolo val detect data=coco128.yaml batch=1 device=0|cpu`

</details>
