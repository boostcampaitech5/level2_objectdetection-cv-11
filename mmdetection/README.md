# Folder Structure
```bash
├─configs
│  ├─cascade_rcnn
│  ├─pvt
│  ├─retinanet
│  ├─swin
│  ├─tood
│  └─_base_
│      ├─datasets
│      ├─models
│      └─schedules
├─mmdet
│  ├─.mim
│  ├─apis
│  ├─core
│  ├─datasets
│  ├─models
│  └─utils
├─tools
│  ├─main.py
│  └─train.sh
```
<br></br>

# Installation
```bash
pip install -r requirements.txt
```
<br></br>

# main.py 
Code that executes train and reference continuously
You can enter the main folder path and enter the command or enter the multilabel_kfold directly into the command
```bash
python3 level2_objectdetection-cv-11/mmdetection/tools/main.py
```

## parser
```bash
--max_epoch: default=20, Number of epochs to rotate learning

--model: py file name of the model to run in the config

--folder: Name of the py folder to run in the config file

--augmentation: default=False, augmented or not

--trainset: default='2___train_MultiStfKFold.json', input train dataset

--validset: default='2___val_MultiStfKFold.json', input valid dataset

--resize: default=1024

--inference_epoch: default="best", Which epoch to use for the inference (epoch number or latest, best)
```
