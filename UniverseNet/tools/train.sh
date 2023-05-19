python3 main.py --model sparse_rcnn_pvt_v2_b2_fpn_300_proposals_detraug_3x_coco --folder pvtv2_original --resize 768
python3 main.py --model atss_pvt_v2_b2_fpn_fp16_detraug_3x_coco --folder pvtv2_original
python3 main.py --model atss_swinv2-t-p4-w16_fpn_4x4_1x_coco --folder swinv2

python3 main.py --model atss_swin-l-p4-w16_fpn_fp16_4x4_1x_coco --folder swinv2
python3 main.py --model atss_swin-b-p4-w16_fpn_fp16_4x4_1x_coco --folder swinv2

python3 main_dyhead.py --model original_atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco --folder dyhead
python3 main_dyhead.py --model atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco --folder dyhead
python3 main_dyhead_aug.py --model albu_atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco --folder dyhead

python3 main.py --model universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco --folder universenet


python3 val_csv.py --model original_atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco --folder dyhead

        