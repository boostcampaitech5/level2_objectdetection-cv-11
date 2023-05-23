# Gender and age classification for wearing mask image
This project is the Naver Boost Camp CV11 team's submission code for the trash object detection competition.
Given an image containing a garbage object, it is a matter of specifying the location of the garbage and classifying the class.

![Untitled](https://user-images.githubusercontent.com/77565951/206111215-d4dc677e-1ba5-4e37-99ee-50a1c5b40f58.png)

### Team Members

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/hykhhijk">
            <img src="https://avatars.githubusercontent.com/u/58303938?v=4" alt="김용희 프로필" width=120 height=120 />
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/HipJaengYiCat">
          <img src="https://avatars.githubusercontent.com/u/78784633?v=4" alt="박승희 프로필" width=120 height=120 />
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/imsmile2000">
          <img src="https://avatars.githubusercontent.com/u/69185594?v=4" alt="이윤표 프로필" width=120 height=120 />
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/junha-lee">
          <img src="https://avatars.githubusercontent.com/u/44857783?v=4" alt="이준하 프로필" width=120 height=120 />
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/JaiyoungJoo">
          <img src="https://avatars.githubusercontent.com/u/103994779?v=4" alt="주재영 프로필" width=120 height=120 />
        </a>
      </td>
    </tr>
    <tr>
      <td align="center">
        <a href="https://github.com/hykhhijk">
          김용희
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/HipJaengYiCat">
          박승희
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/imsmile2000">
          이윤표
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/junha-lee">
          이준하
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/JaiyoungJoo">
          주재영
        </a>
      </td>
    </tr>
  </table>
</div>

<br/>
<div id="5"></div>
 
# Environment
- OS : Linux Ubuntu 18.04.5
- GPU : Tesla V100 (32GB)


# Folder Structure
```bash
├─eda
├─ensemble
├─mmdetection
├─mmdetection3
├─UniverseNet
├─yolov8
├─multilabel_kfold.py
└─streamlit
```
<br></br>

# Usage

## Install Requirements

- `pip install -r requirements.txt`


## Multilabel_Kfold.py

You can enter the multilabel_kfold.py folder path and enter the command or enter the multilabel_kfold directly into the command
```bash
python3 level2_objectdetection-cv-11/multilabel_Kfold.py --K {kfold split count}
```
<br></br>

## train.sh
1. Move the path to the tools folder where the train.sh file is located

2. Write python3 main.py command in train.sh file
    ```bash
    python3 main.py --model {.py name} --folder {folder_name} --resize {size} --max_epoch {epoch} --inference_epoch {best/latest}
    ```

3. Run
    ```bash
    nohup sh train.sh
    ```


# Result

## Augmentation experiment result
**Metric** : mAP score

| augmentation | mAP50 | General trash(0) | Plastic(5) | total |
| --- | --- | --- | --- | --- |
| Normalize | 0.346174 | 0.173843 | 0.280453 | 0.80047 |
| HorizontalFlip | 0.317365 | 0.17596 | 0.254375 | 0.7477 |
| Mosaic | 0.328192 | 0.16313 | 0.252239 | 0.743561 |
| RGBShift | 0.324352 | 0.152785 | 0.248293 | 0.72543 |
| MedianBlur | 0.317173 | 0.145847 | 0.25099 | 0.71401 |
| HueSaturation | 0.310931 | 0.163762 | 0.238093 | 0.712786 |
| Cutout | 0.311202 | 0.157962 | 0.2429 | 0.712064 |
| CLAHE | 0.308945 | 0.158813 | 0.240687 | 0.708445 |
| JpegCompression | 0.32155 | 0.16028 | 0.22024 | 0.70207 |
| RandomBrightnessContrast | 0.312545 | 0.144382 | 0.238167 | 0.695094 |
| Multiresize | 0.29278 | 0.15752 | 0.235295 | 0.685595 |

## Final Model
**Metric** : mAP score

| Library | TYPE | Method | Backbone | Neck | Datasets | Scheduler | Runtime | Optimizer | mAP(public) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mmdetection2 | 2stage | Cascade RCNN | Swin transformer base | FPN | albu_coco_detection | schedule | default_runtime | AdamW | 0.6671 |
| UniverseNet | 1stage | UniverseNet | Res2Net_101 | FPN / SEPC | albu_coco_detection | schedule_20e | default_runtime | AdamW | 0.61 |
| UniverseNet | 1stage | ATSS | Swin transformer large | FPN / Dyhead | albu_coco_detection | schedule_20e | default_runtime | AdamW | 0.6237 |
| UniverseNet | 2stage | GFLv2 | PVT_v2 | FPN | albu_coco_detection | schedule_2x | default_runtime | SGD | 0.5693 |
| UniverseNet | 1stage | TOOD | Swin transformer v2 tiny | FPN | albu_coco_detection | schedule_1x | default_runtime | AdamW | 0.54 |
| YOLOv8 | 1stage | YOLOv8_l | Darknet | - | recycle.yaml | - | - | - | 0.3822 |

## Ensemble

![image](https://github.com/boostcampaitech5/level2_objectdetection-cv-11/assets/69185594/fde682dc-11bc-4165-ace7-42e9cb2c4b33)

- Final submission : Public : 0.6878(5th) / Private : 0.6720(5th)
