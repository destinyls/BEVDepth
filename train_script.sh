# CUDA_VISIBLE_DEVICES=2,3,4,5,6 python exps/bev_depth_lss_r50_256x704_128x128_24e.py --amp_backend native -b 6 --gpus 5
python exps/bev_depth_lss_r50_256x704_128x128_24e.py --amp_backend native -b 6 --gpus 7
# python exps/bev_depth_lss_r50_256x704_128x128_24e.py --ckpt outputs/bev_depth_lss_r50_256x704_128x128_24e/checkpoints/ -e -b 1 --gpus 7