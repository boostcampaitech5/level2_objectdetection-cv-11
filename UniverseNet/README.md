# UniverseNet

This is the official repository of "[USB: Universal-Scale Object Detection Benchmark](https://arxiv.org/abs/2103.14027)" (BMVC 2022).

We established a new benchmark *USB* with fair protocols and designed state-of-the-art detectors *UniverseNets* for universal-scale object detection.
This repository extends MMDetection with more features and allows for more comprehensive benchmarking and development.

## Introduction

![universal-scale object detection](https://user-images.githubusercontent.com/42844407/113513063-b5aa2780-95a2-11eb-8413-2fb470256a1a.png)

Benchmarks, such as COCO, play a crucial role in object detection. However, existing benchmarks are insufficient in scale variation, and their protocols are inadequate for fair comparison. In this paper, we introduce the Universal-Scale object detection Benchmark (USB). USB has variations in object scales and image domains by incorporating COCO with the recently proposed Waymo Open Dataset and Manga109-s dataset. To enable fair comparison and inclusive research, we propose training and evaluation protocols. They have multiple divisions for training epochs and evaluation image resolutions, like weight classes in sports, and compatibility across training protocols, like the backward compatibility of the Universal Serial Bus. Specifically, we request participants to report results with not only higher protocols (longer training) but also lower protocols (shorter training). Using the proposed benchmark and protocols, we conducted extensive experiments using 15 methods and found weaknesses of existing COCO-biased methods.
<br></br>

# Folder Structure
```bash
├─configs
│  ├─dyhead
│  ├─gflv2
│  ├─pvtv2_original
│  ├─swinv2
│  │  └─models
│  ├─tood
│  ├─universenet
│  │  ├─ablation
│  │  └─models
│  └─_base_
│      ├─datasets
│      ├─models
│      └─schedules
├─mmcv_custom
│  └─runner
├─mmdet
│  ├─apis
│  ├─core
│  ├─datasets
│  ├─models
│  └─utils
├─tools
│  ├─main.py
│  └─main_dyhead.py

```
<br></br>
# Cloning
```bash
git clone https://github.com/shinya7y/UniverseNet.git
```
<br></br>

# Installation
```bash
cd UniverseNet
pip install -v -e .
```
Modified model backbone SyncBN to BN. (SyncBN only works in certain situations)
<br></br>

# main.py & main_dyhead.py
dyhead model can be run as main_dyhead.py, and the rest of the model can be run with main.py
```bash
python3 level2_objectdetection-cv-11/UniverseNet/tools/main.py
or
python3 level2_objectdetection-cv-11/UniverseNet/tools/main_dyhead.py
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



