# 환경 구성

[GitHub - CodingMantras/yolov8-streamlit-detection-tracking: YOLOv8 object detection algorithm and Streamlit framework for Real-Time Object Detection and tracking in video streams.](https://github.com/CodingMantras/yolov8-streamlit-detection-tracking)

python 3.10.6

requirements.txt

```
pickle5==0.0.11
streamlit==1.22.0
scikit-learn==1.2.2
numpy==1.24.3
pandas==1.5.1
altair<5
```

`pip install ultralytics streamlit pafy`

- [https://torbjorn.tistory.com/679](https://torbjorn.tistory.com/679)

## 디렉토리 구조

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4f042c7b-1386-4a1b-bfc0-6ff21cf2a1a0/Untitled.png)

## 학습한 가중치 공유링크 :

[best_bbox_mAP_50_epoch_20.pth](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fc067244-6bad-4afb-9295-798cc56a2c8a/best_bbox_mAP_50_epoch_20.pth)

[](https://yonsei-my.sharepoint.com/:u:/g/personal/junha4304_o365_yonsei_ac_kr/EZF6JWj4kcpJlJqA6lk7eKkBMK87y7iAysS8BZlja1ujcw?e=UgiIdJ)

weights 폴더 생성 > weights 폴더에 가중치 넣기

## 수정할 코드 - [settings.py](http://settings.py) :

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/28febe85-1e4f-44a4-96ea-47c2c20d9212/Untitled.png)

가상환경 activate → streamlit run [app.py](http://app.py/)

# 결과

## confidence score 조절에 따른 박스 변화

### confidence 40 :

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/111cedf5-87ae-4846-ac73-a12ec456a603/Untitled.png)

### confidence 95 :

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f660b748-11e9-4a33-87c2-c71ee2267772/Untitled.png)

### video에 대한 trash detection :

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e2719644-9116-4fb7-aec9-6e33dc2399cb/Untitled.png)

### webcam을 사용한 실시간 vidio detection :

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bd70dd7c-42d5-4dcd-a315-552eec133081/Untitled.png)

# 정리

가장 박스가 적은 yolov8l 모델을 사용했음에도 confidence 40에서는 박스를 많이 잡고, 95까지 올려도 잡아내는 모습이다. mAP 50 성능이 높았던 대회에 최종 제출한 모델을 사용했다면, 모든 물체에 박스가 생성 되었을 것이다.

동영상에 대한 trash detection 또한 잘 작동하는 것을 알 수 있었고, 다만, 교육용 데이터셋 처럼 위에서 찍은 사진이 아닌 경우 쓰레기의 옆 모습만 보고 해당 쓰레기를 검출하는 능력은 많이 떨어졌다.

특히 webcam을 이용한 trash detection에서는 대부분의 물체를 일반 쓰레기로 분류하였으며, 심지어 내 얼굴또한 일반 쓰레기로 분류되었다. 대회에서는 좋은 성적을 거두었지만, 사용화하려면 보다 많은 것들을 고려해야 한다는 것을 느꼈고, 특히 대회를 위해 주어진 대부분의 이미지는 쓰레기만을 특정 각도에서 촬영한 것이 많았는데, 실제 object detection이 필요한 상황은 쓰레기가 아닌 물체들이 많은 상황에서 특정 쓰레기를 감지할 수 있어야 하며, 촬영 각도 또한 다각도를 고려하여야한다.
