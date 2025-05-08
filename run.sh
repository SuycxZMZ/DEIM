# 跑测试

python train.py -c configs/deim_dfine/deim_hgnetv2_n_neudet.yml --use-amp --seed=0 -t pretrain_checkpoints/deim_dfine_hgnetv2_n_coco_160e.pth

# 模块测试
python train.py -c configs/deim_dfine/deim_hgnetv2_n_neudet_test.yml --use-amp --seed=0 -t pretrain_checkpoints/deim_dfine_hgnetv2_n_coco_160e.pth