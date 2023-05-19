python3 main.py --model glfv2_swinv2-t-p4-w16_fpn_weight_sgd --folder swinv2 --resize 768
python3 main_dyhead.py --model gflv2_pvt_v2_b2_fpn_sepc --folder gflv2 --resize 768
python3 main.py --model gflv2_pvt_v2_b2_fpn_sepc --folder gflv2 --resize 768
python3 main.py --model gflv2_pvt_v2_b2_fpn_dyhead --folder gflv2 --resize 768
python3 main.py --model gflv2_pvt_v2_b2_fpn_albu --folder gflv2

python3 main_valid.py --model gflv2_pvt_v2_b2_fpn_albu --folder gflv2 --validset 'train.json'

python3 main_uvs.py --model universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco --folder universenet


python3 main_tood.py --model tood_swinv2-t-p4-w16_fpn_resize_coco_adamw --folder tood

python3 main_uvs.py --model universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco --folder universenet