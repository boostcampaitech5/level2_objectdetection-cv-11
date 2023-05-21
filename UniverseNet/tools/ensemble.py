# 같은 폴더에 있는 csv 파일을 모두 앙상블
# 인자 폴더 경로, iou_thr, weights, skip_box_thr
# 폴더 명으로 CSV 파일 생성
import pandas as pd
from ensemble_boxes import *
import numpy as np
from pycocotools.coco import COCO
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--path', default='/opt/ml/sample_submission/', help='input your csv path')
parser.add_argument('--iou_thr', default=0.97, help='input your iou_thr')
parser.add_argument('--weights', default="1,0.9,9.7", help='input your weights')
parser.add_argument('--skip_box_thr', default=0.4, help='input your skip_box_thr')
# parser.add_argument('--conf_type',default="absent_model_aware_avg")
# parser.add_argument('--thresh',default=0.0013)
args = parser.parse_args()

# ensemble 시 설정할 iou threshold 이 부분을 바꿔가며 대회 metric에 알맞게 적용해봐요!
iou_thr = float(args.iou_thr)
weights = [float(i) for i in args.weights.split(',')]
skip_box_thr = float(args.skip_box_thr)
path = args.path
print(path)

submission_files = os.listdir(path)

submission_df = [pd.read_csv(path +'/'+ file, encoding='utf-8') for file in submission_files]

image_ids = submission_df[0]['image_id'].tolist()

annotation = '/opt/ml/dataset/images/test.json'
coco = COCO(annotation)

prediction_strings = []
file_names = []

# 각 image id 별로 submission file에서 box좌표 추출
for i, image_id in tqdm(enumerate(image_ids)):
    prediction_string = ''
    boxes_list = []
    scores_list = []
    labels_list = []
    image_info = coco.loadImgs(i)[0]
#     각 submission file 별로 prediction box좌표 불러오기
    for df in submission_df:
        predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
        predict_list = str(predict_string).split()
        
        if len(predict_list)==0 or len(predict_list)==1:
            continue
            
        predict_list = np.reshape(predict_list, (-1, 6))
        box_list = []
        
        for box in predict_list[:, 2:6].tolist():
            box[0] = float(box[0]) / image_info['width']
            box[1] = float(box[1]) / image_info['height']
            box[2] = float(box[2]) / image_info['width']
            box[3] = float(box[3]) / image_info['height']
            box_list.append(box)
            
        boxes_list.append(box_list)
        scores_list.append(list(map(float, predict_list[:, 1].tolist())))
        labels_list.append(list(map(int, predict_list[:, 0].tolist())))
    
#     예측 box가 있다면 이를 ensemble 수행
    if len(boxes_list):
        # boxes, scores, labels = nms(boxes_list, scores_list, labels_list,iou_thr=iou_thr,weights=weights)
        # boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, method=3, iou_thr=iou_thr, sigma=0.145, thresh=args.thresh, weights=weights)
        boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        # boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
#         sorted_indices = np.argsort(scores)[::-1]
#         sorted_boxes = boxes[sorted_indices]
#         sorted_scores = scores[sorted_indices]
#         sorted_labels = labels[sorted_indices]

#         # Keep only the top N boxes
#         top_n = 300
#         filtered_boxes = sorted_boxes[:top_n]
#         filtered_scores = sorted_scores[:top_n]
#         filtered_labels = sorted_labels[:top_n]
        for box, score, label in zip(boxes,scores,labels):
            prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '
    
    prediction_strings.append(prediction_string)
    file_names.append(image_id)

submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
model_name = path.split('/')[-1]
submission.to_csv(f'{path}/{model_name}+iou_{iou_thr}+top300.csv')

submission.head()