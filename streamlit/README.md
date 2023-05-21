# Environment
- OS : Linux Ubuntu 18.04.5
- GPU : Tesla V100 (32GB)
- python 3.10.6

## Install Requirements

- `pip install -r requirements.txt`

```
pickle5==0.0.11
streamlit==1.22.0
scikit-learn==1.2.2
numpy==1.24.3
pandas==1.5.1
altair<5
```

- `pip install ultralytics streamlit pafy`


## 디렉토리 구조

![Untitled](streamlit%20a7c73382094d41a1a02b07edd401e773/Untitled.png)

## 학습한 가중치 공유링크 :

[best_bbox_mAP_50_epoch_20.pth](streamlit%20a7c73382094d41a1a02b07edd401e773/best_bbox_mAP_50_epoch_20.pth)

[](https://yonsei-my.sharepoint.com/:u:/g/personal/junha4304_o365_yonsei_ac_kr/EZF6JWj4kcpJlJqA6lk7eKkBMK87y7iAysS8BZlja1ujcw?e=UgiIdJ)

weights 폴더 생성 > weights 폴더에 가중치 넣기

## 수정할 코드 - [settings.py](http://settings.py) :

![Untitled](https://github.com/boostcampaitech5/level2_objectdetection-cv-11/blob/main/streamlit/assets/Untitled%201.png)

가상환경 activate → streamlit run [app.py](http://app.py/)

# Result

## Changes with adjustment of confidence score 

### Confidence 40 :

![Untitled](https://github.com/boostcampaitech5/level2_objectdetection-cv-11/blob/main/streamlit/assets/Untitled%202.png)

### Confidence 95 :

![Untitled](https://github.com/boostcampaitech5/level2_objectdetection-cv-11/blob/main/streamlit/assets/Untitled%203.png)

### Trash detection for video :

![Untitled](https://github.com/boostcampaitech5/level2_objectdetection-cv-11/blob/main/streamlit/assets/Untitled%204.png)

### Realtime trash detection :

![Untitled](https://github.com/boostcampaitech5/level2_objectdetection-cv-11/blob/main/streamlit/assets/Untitled%205.png)
