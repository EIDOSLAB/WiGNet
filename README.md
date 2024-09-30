Checkpoint Path for WiGNet-Ti:
```
checkpoints/wignet_ti_256_gelu_shift_adapt_knn/model.pth.tar
```

Conda environment dumped in:
```
env_wignet.yaml
```

# ImageNet Classification


## Evaluation
```
python train.py --model wignn_ti_256_gelu \
--img-size 256 \
--knn 9 \
--use-shift 1 \
--adapt-knn 1 \
--data /path/to/imagenet \
-b 128 \
--resume  /path/to/checkpoint.pth.tar \
--evaluate 
```

## Training WiGNet-Ti on 8 GPUs
```
python -m torch.distributed.launch \
--nproc_per_node=8 train.py \
--model wignn_ti_256_gelu \
--img-size 256 \
--knn 9 \
--use-shift 1 \
--adapt-knn 1 \
--use-reduce-ratios 0 \
--data /path/to/imagenet \  
--sched cosine \
--epochs 300 \
--opt adamw -j 8 \
--warmup-lr 1e-6 \
--mixup .8 \
--cutmix 1.0 \
--model-ema \
--model-ema-decay 0.99996 \
--aa rand-m9-mstd0.5-inc1 \
--color-jitter 0.4 \
--warmup-epochs 20 \
--opt-eps 1e-8 \
--remode pixel \
--reprob 0.25 \
--amp \
--lr 2e-3 \
--weight-decay .05 \
--drop 0 \
--drop-path .1 \
-b 128 \
--output /path/to/save 
```



# Complexity Evaluation

## Memory & MACs
- WiGNet
```
python -m model.wignn
```

- ViG
```
python -m model.pyramid_vig
```

- GreedyViG
```
python -m model.greedyvig
```

- MobileViG
```
python -m model.mobilevig
```




# Transfer Learning
- WiGNet
```
python train_trasnfer_learning.py 
--model-type wignn_ti_256_gelu \
--use-shift 1 \
--adapt-knn 1 \
--batch-size 64  \
--checkpoint /path/to/checkpoint.pth.tar \
--crop-size 512 \
--dataset CelebA \
--epochs 30 \
--freeze 1 \
--loss cross_entropy \
--lr 0.001 \
--lr-scheduler constant  \
--opt adam \
--root /path/to/save/dataset \
--save-dir /path/to/save/outputs_tl_high_res/ \
--seed 1 
```

For ViG include `--num-gpu 8`