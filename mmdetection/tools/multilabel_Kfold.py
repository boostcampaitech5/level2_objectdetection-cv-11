import json
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedGroupKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from collections import Counter

parser = argparse.ArgumentParser(description='K')
parser.add_argument('--K', default=5, help='input your fold num')
args = parser.parse_args()
K = int(args.K)
print(f'start MultilabelStratified_{K}_fold')

# load json 

dataset_path = '/opt/ml/dataset/'
annotation = dataset_path + 'train.json'

with open(annotation) as f:
    data = json.load(f)

var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]

X = np.ones((len(data['annotations']),1))
y = np.array([v[1] for v in var])  # class
groups = np.array([v[0] for v in var])  # image(group)


train_df = pd.DataFrame(data['annotations'])
train_df = train_df[['image_id', 'category_id', 'area','id']]

bbox_num = train_df.groupby('image_id')['id'].count().to_list()

# 이미지 별 bbox 개수
bbox_num = train_df.groupby('image_id')['id'].count().to_list()

# 작은 box부터 큰 box까지 0,1,2,3, 이미지 별 bbox의 평균 크기 분류
area_mean_df = train_df.groupby('image_id')['area'].mean()
area_class = []
num_area = len(area_mean_df)
for i in range(num_area):
    if area_mean_df[i] <= 9997:
        area_class.append(0)
    elif area_mean_df[i] <= 38938:
        area_class.append(1)
    elif area_mean_df[i] <= 119122:
        area_class.append(2)
    else:
        area_class.append(3)

# 최빈 class 
category_group = train_df.groupby('image_id')['category_id'].value_counts()
most_class_list = []
for i in range(num_area):
    most_class = category_group[i].index.to_list()[0] # 동률일 경우 가장 빠른 index 선택
    most_class_list.append(most_class)

multi_label_df = pd.DataFrame({'image_id':[i for i in range(num_area)], 'bbox_num':bbox_num, 'area_class':area_class, 'most_class_list':most_class_list})

# multilabelstratifiedkfold
cv = MultilabelStratifiedKFold(n_splits=K, shuffle=True, random_state=42)
X = np.ones((len(bbox_num),1))
    
all_fold_train_list = []
all_fold_val_list = []
for fold_ind, (train_idx, val_idx) in enumerate(cv.split(multi_label_df, multi_label_df[['bbox_num', 'area_class']])):
    
    all_fold_train_list.append(list(train_idx))
    all_fold_val_list.append(list(val_idx))

# make kfold.json
for i in range(K):
    train_idx_list = all_fold_train_list[i]
    val_idx_list = all_fold_val_list[i]

    json_file = data.copy()
    new_data_images_train = [json_file['images'][j] for j in train_idx_list]
    new_data_images_val = [json_file['images'][j] for j in val_idx_list]

    new_ann_train = []
    new_ann_val = []
    for ann_id in range(len(data['annotations'])):
        ann_img_id = json_file['annotations'][ann_id]['image_id']
        if ann_img_id in train_idx_list:
            new_ann_train.append(json_file['annotations'][ann_id])
        if ann_img_id in val_idx_list:
            new_ann_val.append(json_file['annotations'][ann_id])
            
    json_train = json_file.copy()
    json_val = json_file.copy()

    json_train['images'] = new_data_images_train
    json_train['annotations'] = new_ann_train

    json_val['images'] = new_data_images_val
    json_val['annotations'] = new_ann_val

    with open(f'{dataset_path}{i+1}___train_MultiStfKFold.json', 'w') as t:
        json.dump(json_train, t)

    with open(f'{dataset_path}{i+1}___val_MultiStfKFold.json', 'w') as v:
        json.dump(json_val, v)
    
    print(f'{i+1}_fold is done')
        
print('All you needs are done')