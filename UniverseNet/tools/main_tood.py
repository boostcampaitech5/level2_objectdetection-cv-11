import mmcv
import os
import numpy as np
import pandas as pd
import argparse
import copy
import os.path as osp
import time
import warnings
import wandb
import multiprocessing
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset)
                            # replace_ImageToTensor)
from mmdet.utils import get_device
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.apis import set_random_seed
from pandas import DataFrame
from pycocotools.coco import COCO


parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--max_epoch', default=30, help='input your max_epoch')
parser.add_argument('--model', default="cascade_rcnn_x101_64x4d_fpn_1x_coco", help='input your model_name')
parser.add_argument('--folder', default="cascade_rcnn", help='input your folder_name')
parser.add_argument('--augmentation', default=True, help='input your augmentation')
parser.add_argument('--trainset', default='2___train_MultiStfKFold.json', help='input your trainset')
parser.add_argument('--validset', default='2___val_MultiStfKFold.json', help='input your validset')
parser.add_argument('--resize', default=1024, help='input your resize')
parser.add_argument('--inference_epoch', default="best", help='input your inference epoch')

args = parser.parse_args()

# ------------------ 변경할 부분-------------------
model_name = args.model
folder_name= args.folder
augmentation = True
# ------------------ 변경할 부분-------------------
cfg = Config.fromfile(f'../configs/{folder_name}/{model_name}.py')
cfg.runner.max_epochs = int(args.max_epoch) # 에포크 횟수 조정
resize = int(args.resize)
root='/opt/ml/dataset/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")


def data_config(cfg: Config) -> None:
    # cfg.data.train.classes = classes
    # cfg.data.train.img_prefix = root
    # cfg.data.train.ann_file = root + args.trainset # train json 정보
    # if "dataset" in cfg.data.train.keys():
    #     cfg.data.train.dataset.pipeline[2]['img_scale'] = (resize,resize)
    # else:
    #     cfg.data.train.pipeline[2]['img_scale'] = (resize,resize) # Resize

    # cfg.data.val.classes = classes
    # cfg.data.val.img_prefix = root
    # cfg.data.val.ann_file = root + args.validset # valid json 정보
    # cfg.data.val.pipeline[1]['img_scale'] = (resize,resize) # Resize

    # cfg.data.test.classes = classes
    # cfg.data.test.img_prefix = root
    # cfg.data.test.ann_file = root + 'test.json' # test json 정보
    # cfg.data.test.pipeline[1]['img_scale'] = (resize,resize) # Resize
    cfg.data.test.test_mode = True
    cfg.data.samples_per_gpu = 5
    cfg.data.workers_per_gpu = multiprocessing.cpu_count() // 2 # num_workers
    

def model_config(cfg: Config) -> None:
    if "roi_head" in cfg.model.keys():
        if type(cfg.model.roi_head.bbox_head) == dict:
            cfg.model.roi_head.bbox_head.num_classes = 10

        #In case of cascade RCNN : List[Dict]
        elif type(cfg.model.roi_head.bbox_head) == list:
            for each_head in cfg.model.roi_head.bbox_head:
                if hasattr(each_head, "num_classes"):
                    each_head.num_classes = 10 
                else: 
                    raise Exception("Num_classes가 없습니다")
    else:
        cfg.model.bbox_head.num_classes = 10


def train_config(cfg:Config) -> None:
    cfg.seed = 2022
    cfg.deterministic = True
    set_random_seed(2022, True)
    cfg.gpu_ids = [0]
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.device = get_device()
    # 모델 weight 저장 경로
    cfg.work_dir = f'../work_dirs/{model_name}_trash'
    cfg.evaluation = dict(save_best='bbox_mAP_50',metric='bbox')
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    # wandb 프로젝트 이름
    cfg.log_config.hooks[1].init_kwargs.name=f"{model_name}+aug={augmentation}"

def train(cfg,kfold=False):
    data_config(cfg)
    model_config(cfg)
    train_config(cfg)
    # build_dataset
    datasets = [build_dataset(cfg.data.train)]
    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()
    
    meta = dict()
    meta['seed'] = cfg.seed
    meta['exp_name'] = os.path.basename(f'../configs/{folder_name}/{model_name}.py')
    
    # train_detector(model, datasets, cfg, distributed=False, validate=True,meta=meta)


def inference(cfg):
    epoch = args.inference_epoch
    cfg.model.train_cfg = None
    # build dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
    
    # checkpoint path
    # 만약 'best' 일 경우 best가 들어간 pth를 찾아서 load
    checkpoint_path = ''
    # if args.inference_epoch == 'best':
    #     checkpoint_name = [i for i in os.listdir(cfg.work_dir) if 'best' in i][0]
    #     checkpoint_path = os.path.join(cfg.work_dir, checkpoint_name)
    # # 만약 'latest' 일 경우 latest.pth를 찾아서 load
    # elif args.inference_epoch == 'latest':
    #     checkpoint_path = os.path.join(cfg.work_dir, 'latest.pth')
    # # 그 외 숫자를 넣을 경우 해당 숫자에 해당하는 f'{epoch}.pth'를 찾아서 load
    # else:
    #     checkpoint_path = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')
    print('===================================')
    print("checkpoint_path:", checkpoint_path)
    print('===================================')

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    
    output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산
    
    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '
            
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])


    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=None)
    submission.head()
    

if __name__ == '__main__':
    train(cfg)
    inference(cfg)