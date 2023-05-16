python3 main.py --model glfv2_swinv2-t-p4-w16_fpn_weight_sgd --folder swinv2 --resize 768
python3 main_dyhead.py --model gflv2_pvt_v2_b2_fpn_sepc --folder gflv2 --resize 768
python3 main.py --model gflv2_pvt_v2_b2_fpn_sepc --folder gflv2 --resize 768
python3 main.py --model gflv2_pvt_v2_b2_fpn_dyhead --folder gflv2 --resize 768
python3 main.py --model gflv2_pvt_v2_b2_fpn_albu --folder gflv2