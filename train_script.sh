export TORCH_HOME=./$TORCH_HOME
python exps/bev_depth_lss_r50_256x704_128x128_24e.py --amp_backend native -b 4 --gpus 8
# python exps/bev_depth_lss_r50_256x704_128x128_24e.py --ckpt outputs/bev_depth_lss_r50_256x704_128x128_24e/checkpoints/ -e -b 6 --gpus 8
