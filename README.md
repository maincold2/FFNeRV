# FFNeRV: Flow-Guided Frame-Wise Neural Representations for Videos
#### Joo Chan Lee, Daniel Rho, Jong Hwan Ko†, and Eunbyung Park†

### [[Project Page](https://maincold2.github.io/ffnerv/)] [[Paper(arxiv)](https://arxiv.org/abs/2212.12294)]

Our code is based on [NeRV](https://github.com/haochen-rye/NeRV)

## Method overview
<img src="https://maincold2.github.io/ffnerv/img/fig_arch.png"  />

## 0. Requirements
```
pip install -r requirements.txt 
```
Dataset link: [UVG](https://ultravideo.fi/#testsequences)

## 1. Training (Encoding)
### For representation
```bash
python main.py -e 300 --lower-width 96 --num-blocks 1 --dataset [data_dir] --outf [out_dir] --fc-hw-dim 9_16_156 --expansion 1 --loss Fusion6 --strides 5 2 2 2 2  --conv-type conv -b 1  --lr 0.0005 --agg-ind -2 -1 1 2 --lw 0.1 --t-dim 64 128 256 512
```
### For compression
```bash
python main.py -e 600 --lower-width 24 --num-blocks 1 --dataset [data_dir] --outf [out_dir] --fc-hw-dim 9_16_48 --expansion 8 --loss Fusion6 --strides 5 3 2 2 2  --conv-type compact -b 1  --lr 0.0005 --agg-ind -2 -1 1 2 --lw 0.1 --wbit 8 --t-dim 300 600 --resol 1920 1080
```
- "--conv-type": the type of convolution blocks
- "--agg-ind": relative video indices for aggregation 
- "--wbit": for n-bit quantization (QAT)
- "--t-dim": multi-channels of temporal grids

More details can be found in "main.py"


## 2. Evaluation (+ Entropy coding)
```bash
python main.py --lower-width 24 --num-blocks 1 --dataset [data_dir] --outf [out_dir] --fc-hw-dim 9_16_48 --expansion 8 --strides 5 3 2 2 2  --conv-type compact -b 1 --agg-ind -2 -1 1 2 --wbit 8 --t-dim 300 600 --resol 1920 1080 --eval-only
```

## 3. Decoding
```bash
python main.py --lower-width 24 --num-blocks 1 --dataset [data_dir] --outf [out_dir] --fc-hw-dim 9_16_48 --expansion 8 --strides 5 3 2 2 2  --conv-type compact -b 1 --agg-ind -2 -1 1 2 --wbit 8 --t-dim 300 600 --resol 1920 1080 --eval-only --weight [weight_path] --dump-images
```
