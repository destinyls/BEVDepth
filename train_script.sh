export TORCH_HOME=./$TORCH_HOME
CUDA_VISIBLE_DEVICES=1,3,4,5,6 python exps/bev_depth_lss_r50_256x704_128x128_24e.py --amp_backend native -b 2 --gpus 5
CUDA_VISIBLE_DEVICES=1,3,4,5,6 python exps/bev_depth_lss_r50_256x704_128x128_24e.py --ckpt outputs/bev_depth_lss_r50_256x704_128x128_24e/checkpoints/ -e -b 2 --gpus 5