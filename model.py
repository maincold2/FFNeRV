import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, vid_list=[None], resol=[1280, 720], frame_gap=1,  visualize=False, offset=0):
        self.main_dir = main_dir
        self.transform = transform
        frame_idx, self.frame_path = [], []
        accum_img_num = []
        all_imgs = os.listdir(main_dir)
        all_imgs.sort()

        num_frame = 0 
        for img_id in all_imgs:
            self.frame_path.append(img_id)
            frame_idx.append(num_frame)
            num_frame += 1  

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        #accum_img_num.append(num_frame)
        self.frame_idx = [float(x) / len(frame_idx) for x in frame_idx]
        if None not in vid_list:
            self.frame_idx = [self.frame_idx[i] for i in vid_list]
        self.frame_gap = frame_gap
        self.offset = offset
        self.h, self.w = resol

    def __len__(self):
        return len(self.frame_idx) // self.frame_gap

    def __getitem__(self, idx):
        valid_idx = idx * self.frame_gap + self.offset
        img_id = self.frame_path[valid_idx]
        img_name = os.path.join(self.main_dir, img_id)
        image = Image.open(img_name).convert("RGB")
        image = image.resize((self.h, self.w))
        tensor_image = self.transform(image)
        if tensor_image.size(1) > tensor_image.size(2):
            tensor_image = tensor_image.permute(0,2,1)
        frame_idx = torch.tensor(self.frame_idx[valid_idx])

        return tensor_image, frame_idx

class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)


def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = torch.sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):    
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer

class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**(k-1) - 1)
        out = torch.floor(torch.abs(input) * n) / n
        out = out*torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class weight_quantize_fn(nn.Module):
    def __init__(self, bit):
        super(weight_quantize_fn, self).__init__()
        self.wbit = bit
        assert self.wbit <= 8 or self.wbit == 32

    def forward(self, x):
        if self.wbit == 32:
            weight_q = x
        else:
            weight = torch.tanh(x)
            weight_q = qfn.apply(weight, self.wbit)
        return weight_q

class Conv2d_Q(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Q, self).__init__(*kargs, **kwargs)


def conv2d_quantize_fn(bit):
    class Conv2d_Q_(Conv2d_Q):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                     bias=True):
            super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                            bias)
            self.w_bit = bit
            self.quantize_fn = weight_quantize_fn(self.w_bit)

        def forward(self, input, order=None):
            weight_q = self.quantize_fn(self.weight)
            bias_q = self.quantize_fn(self.bias)
            return F.conv2d(input, weight_q, bias_q, self.stride, self.padding, self.dilation, self.groups)

    return Conv2d_Q_

class CustomConv(nn.Module):
    def __init__(self, **kargs):
        super(CustomConv, self).__init__()
        Conv2d = conv2d_quantize_fn(kargs['wbit'])
        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.conv_type = kargs['conv_type']
        if self.conv_type == 'conv':
            self.conv1 = Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'])
            self.conv2 = nn.Identity()
            self.up_scale = nn.PixelShuffle(stride)
        # compact convolution
        elif self.conv_type == 'compact':
            if stride == 5:
                self.conv1 = Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'], groups=1)
                self.conv2 = nn.Identity()
                self.up_scale = nn.PixelShuffle(stride)
            else:
                self.conv1 = Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'], groups=8)
                self.conv2 = Conv2d(new_ngf * stride * stride, new_ngf * stride * stride, 1, bias=kargs['bias'])
                self.up_scale = nn.PixelShuffle(stride)
        elif self.conv_type == 'deconv':
            self.conv = nn.ConvTranspose2d(ngf, new_ngf, stride, stride)
            self.up_scale = nn.Identity()
        elif self.conv_type == 'bilinear':
            self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.up_scale = Conv2d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.up_scale(out)


class ConvBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.conv = CustomConv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], stride=kargs['stride'], bias=kargs['bias'], 
            conv_type=kargs['conv_type'], wbit=kargs['wbit'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        Conv2d = conv2d_quantize_fn(kargs['wbit'])
        self.quantize_fn = weight_quantize_fn(kargs['wbit'])
        self.strides = kargs['stride_list']
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fc_hw_dim'].split('_')]
        # multi-resolution temporal grids
        self.video_grid = nn.ParameterList()
        for t in kargs['t_dim']:
            self.video_grid.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty(t,self.fc_dim//len(kargs['t_dim']),self.fc_h,self.fc_w))))
                     
        # BUILD CONV LAYERS
        self.layers, self.head_layers = [nn.ModuleList() for _ in range(2)]
        ngf = self.fc_dim
        for i, stride in enumerate(kargs['stride_list']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * kargs['expansion'])
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else kargs['reduction']), kargs['lower_width'])

            for j in range(kargs['num_blocks']):
                self.layers.append(ConvBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                    bias=kargs['bias'], norm=kargs['norm'], act=kargs['act'], conv_type=kargs['conv_type'], wbit=kargs['wbit']))
                ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            if kargs['sin_res']:
                if i == len(kargs['stride_list']) - 1:
                    head_layer = Conv2d(ngf, 3, 1, 1, bias=kargs['bias'])
                else:
                    head_layer = None
            else:
                if i == len(kargs['stride_list']) - 3:
                    head_layer = Conv2d(ngf, kargs['aggs']*3, 1, 1, 0, bias=kargs['bias']) 
                elif i == len(kargs['stride_list']) - 1:
                    head_layer = Conv2d(ngf, 5, 1, 1, 0, bias=kargs['bias']) 
                else:
                    head_layer = None
            self.head_layers.append(head_layer)
        self.sigmoid =kargs['sigmoid']

    
    def forward(self, input):
        out_list = []
        for param in self.video_grid: # multi-resolution grids
            vg = self.quantize_fn(param)
            # interpolate grid features
            inp = input*(param.size(0))
            left = torch.floor(inp+1e-6).long()
            right = torch.min(left+1, torch.tensor(param.size(0)-1))
            d_left = (inp - left).view(-1, 1, 1, 1)
            d_right = (right - inp).view(-1, 1, 1, 1)
            out_list.append(d_right*vg[left] + d_left*vg[right] - ((right-left-1).view(-1,1,1,1))*vg[left])
        output = out_list[0]
        # concat latent features from multi-resolution grids
        for i in range(len(out_list)-1):
            output = torch.cat([output, out_list[i+1]],dim=1)
        
        out_list = []
        for layer, head_layer in zip(self.layers, self.head_layers):
            output = layer(output) 
            if head_layer is not None:
                img_out = head_layer(output)
                out_list.append(img_out)
        # normalize the independent frame to [0,1]
        out_list[1][:,0:3,:,:] = torch.sigmoid(out_list[1][:,0:3,:,:]) if self.sigmoid else (torch.tanh(out_list[1][:,0:3,:,:]) + 1) * 0.5
        return  out_list
