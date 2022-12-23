# FFNeRV: Flow-Guided Frame-Wise Neural Representations for Videos
Joo Chan Lee, Daniel Rho, Jong Hwan Ko†, and Eunbyung Park†

### [Project Page](https://maincold2.github.io/ffnerv/), ### [Paper(arxiv)](https://)

Our code is based on [NeRV](https://github.com/haochen-rye/NeRV)

## 0. Requirements
```
pip install -r requirements.txt 
```

## 1. Training (Encoding)
```bash
python main.py -e 600 --lower-width 24 --num-blocks 1 --dataset [data_dir] --outf [out_dir] --fc_hw_dim 9_16_48 --expansion 8 --loss Fusion6 --strides 5 3 2 2 2  --conv_type compact -b 1  --lr 0.0005 --agg_ind -2 -1 1 2 --lw 0.1 --wbit 8 --t_dim 300 600 --resol 1920 1080
```
- "--conv_type": the type of convolution blocks
- "--agg_ind": relative video indices for aggregation 
- "--wbit": for n-bit quantization (QAT)
- "--t_dim": multi-channels of temporal grids

More details can be found in "main.py"


## 2. Evaluation (+ Entropy coding)
```bash
python main.py -e 600 --lower-width 24 --num-blocks 1 --dataset [data_dir] --outf [out_dir] --fc_hw_dim 9_16_48 --expansion 8 --loss Fusion6 --strides 5 3 2 2 2  --conv_type compact -b 1  --lr 0.0005 --agg_ind -2 -1 1 2 --lw 0.1 --wbit 8 --t_dim 300 600 --resol 1920 1080 --eval_only
```
