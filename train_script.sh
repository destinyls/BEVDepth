export TORCH_HOME=./$TORCH_HOME
python exps/bev_height_lss_r50_256x704_128x128_24e.py --amp_backend native -b 2 --gpus 8
python exps/bev_height_lss_r50_256x704_128x128_24e.py --ckpt outputs/bev_height_lss_r50_256x704_128x128_24e/checkpoints/ -e -b 2 --gpus 1
