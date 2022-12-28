
from __future__ import print_function

import argparse
import os
import random
import shutil
from datetime import datetime
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import CustomDataSet, Generator
from utils import *

def main():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--vid',  default=[None], type=int,  nargs='+', help='video id list for training')
    parser.add_argument('--scale', type=int, default=1, help='scale-up facotr for data transformation,  added to suffix!!!!')
    parser.add_argument('--frame-gap', type=int, default=1, help='frame selection gap')
    parser.add_argument('--augment', type=int, default=0, help='augment frames between frames,  added to suffix!!!!')
    parser.add_argument('--dataset', type=str, default='UVG', help='dataset',)
    parser.add_argument('--test-gap', default=1, type=int, help='evaluation gap')
    parser.add_argument('--resol', nargs='+', default=[1280, 720], type=int, help='frame resolution')
    parser.add_argument('--offset', default=0, type=int, help='dataset offset')

    # model parameters
    parser.add_argument('--t-dim', nargs='+', default=[256, 512], type=int, help='temporal resolution of grids')
    parser.add_argument('--fc-hw-dim', type=str, default='9_16_128', help='out size (h,w) for mlp')
    parser.add_argument('--expansion', type=float, default=8, help='channel expansion from fc to conv')
    parser.add_argument('--reduction', type=int, default=2)
    parser.add_argument('--strides', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='strides list')
    parser.add_argument('--num-blocks', type=int, default=1)

    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use', choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
    parser.add_argument('--lower-width', type=int, default=32, help='lowest channel width for output feature maps')
    parser.add_argument("--single-res", action='store_true', help='single resolution,  added to suffix!!!!')
    parser.add_argument("--conv-type", default='conv', type=str,  help='upscale methods, can add bilinear and deconvolution methods', choices=['conv', 'compact' ,'deconv', 'bilinear'])

    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--not-resume-epoch', action='store_true', help='resuming start_epoch from checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--cycles', type=int, default=1, help='epoch cycles for training')
    parser.add_argument('--warmup', type=float, default=0.2, help='warmup epoch ratio compared to the epochs, default=0.2,  added to suffix!!!!')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--lr-type', type=str, default='cosine', help='learning rate type, default=cosine')
    parser.add_argument('--lr-steps', default=[], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10,  added to suffix!!!!')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for adam. default=0.5,  added to suffix!!!!')
    parser.add_argument('--loss-type', type=str, default='L2', help='loss type, default=L2')
    parser.add_argument('--lw', type=float, default=1.0, help='loss weight,  added to suffix!!!!')
    parser.add_argument('--sigmoid', action='store_true', help='using sigmoid for output prediction')

    # evaluation parameters
    parser.add_argument('--eval-only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval-freq', type=int, default=50, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--dump-images', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--eval-fps', action='store_true', default=False, help='fwd multiple times to test the fps ')

    # pruning paramaters
    parser.add_argument('--prune-steps', type=float, nargs='+', default=[0.,], help='prune steps')
    parser.add_argument('--prune-ratio', type=float, default=1.0, help='pruning ratio')

    # distribute learning parameters
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--init-method', default='tcp://127.0.0.1:9888', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')

    # logging, output directory, 
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')  
    parser.add_argument('-p', '--print-freq', default=50, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")
    
    parser.add_argument('--agg-ind', nargs='+', default=[-2,-1,0,1,2], type=int, help='relative indices of neighboring frames to reference')
    parser.add_argument('--wbit', default=32, type=int, help='QAT weight bit width')

    args = parser.parse_args()
    
    
    global agg_ind
    agg_ind = args.agg_ind
        
    args.warmup = int(args.warmup * args.epochs)

    print(args)
    torch.set_printoptions(precision=4) 

    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    if args.prune_ratio < 1 and not args.eval_only: 
        prune_str = '_Prune{}_{}'.format(args.prune_ratio, ','.join([str(x) for x in args.prune_steps]))
    else:
        prune_str = ''
    extra_str = f'eval' if args.eval_only else 'train'

    exp_id = f'{args.dataset}/{extra_str}{prune_str}'
    args.exp_id = exp_id

    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
    	print('Will overwrite the existing output dir!')
    	shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2) 
    args.ngpus_per_node = torch.cuda.device_count()
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(None, args)

# used for warping function       
def get_grid(flow):
    m, n = flow.shape[-2:]
    shifts_x = torch.arange(0, n, 1, dtype=torch.float32, device=flow.device)
    shifts_y = torch.arange(0, m, 1, dtype=torch.float32, device=flow.device)
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)

    grid_dst = torch.stack((shifts_x, shifts_y)).unsqueeze(0)
    workspace = torch.tensor([(n - 1) / 2, (m - 1) / 2]).view(1, 2, 1, 1).to(flow.device)

    flow_grid = ((flow + grid_dst) / workspace - 1).permute(0, 2, 3, 1)

    return flow_grid

# warping function
def resample(feats, flow):
    scale_factor = float(feats.shape[-1]) / flow.shape[-1]
    flow = torch.nn.functional.interpolate(
        flow, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    flow = flow * scale_factor
    flow_grid = get_grid(flow)
    warped_feats = F.grid_sample(feats, flow_grid, mode="bilinear", padding_mode="border")
    return warped_feats

def train(local_rank, args):
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [torch.tensor(0) for _ in range(4)]
    is_train_best, is_val_best = False, False

    model = Generator(fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
        num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
        stride_list=args.strides,  sin_res=args.single_res,  lower_width=args.lower_width, sigmoid=args.sigmoid, aggs=len(args.agg_ind), wbit=args.wbit, t_dim=args.t_dim)

    ##### prune model params and flops #####
    prune_net = args.prune_ratio < 1
    # import pdb; pdb.set_trace; from IPython import embed; embed()
    if prune_net:
        param_list = []
        param_to_prune = []
        for k,v in model.named_parameters():
            print(k,)
            if 'video_field' in k:
                param_list.append(model.video_field[int(k.split('.')[-1])])
                #param_to_prune.append([model.video_field, k.split('.')[-1]])
            if 'weight' in k:
                if 'layers' in k[:6] and 'conv1' in k:
                    layer_ind = int(k.split('.')[1])
                    param_list.append(model.layers[layer_ind].conv.conv1)
                    param_to_prune.append([model.layers[layer_ind].conv.conv1,'weight'])
                elif 'layers' in k[:6] and 'conv2' in k:
                    if not isinstance(model.layers[layer_ind].conv.conv2, nn.Identity):
                        param_to_prune.append([model.layers[layer_ind].conv.conv2,'weight'])
                        param_list.append(model.layers[layer_ind].conv.conv2)
                
        #print(param_list)
        #param_to_prune = [(ele, 'weight') for ele in param_list]
        prune_base_ratio = args.prune_ratio ** (1. / len(args.prune_steps))
        args.prune_steps = [int(x * args.epochs) for x in args.prune_steps]
        prune_num = 0
        if args.eval_only:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )

    ##### get model params and flops #####
    total_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
    if local_rank in [0, None]:
        params = sum([p.data.nelement() for p in model.parameters()]) / 1e6

        print(f'{args}\n {model}\n Model Params: {params}M')
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(str(model) + '\n' + f'Params: {params}M\n')
        writer = SummaryWriter(os.path.join(args.outf, f'param_{total_params}M', 'tensorboard'))
    else:
        writer = None

    # distrite model to gpu or parallel
    print("Use GPU: {} for training".format(local_rank))
    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()        
        args.batchSize = int(args.batchSize / args.ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], \
                                                          output_device=local_rank, find_unused_parameters=False)
    elif args.ngpus_per_node > 1:
        model.cuda() #model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), betas=(args.beta, 0.999))

    # resume from args.weight
    checkpoint = None
    loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        orig_ckt = checkpoint['state_dict']
        new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()} 
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
            model.load_state_dict(new_ckt)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(new_ckt, strict=False)
        else:
            model.load_state_dict(new_ckt)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))        

    # resume from model_latest
    checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if prune_net:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
            
            sparsity_num = 0.
            for param in param_list:
                for param in param_list:
                    if isinstance(param, nn.Parameter):
                        sparsity_num += (param==0).sum()
                    else:
                        sparsity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{args.start_epoch}: {sparsity_num / 1e6 / total_params}')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    args.start_epoch = 0
    if checkpoint is not None:
        args.start_epoch = checkpoint['epoch'] 
        train_best_psnr = checkpoint['train_best_psnr'].to(torch.device(loc))
        train_best_msssim = checkpoint['train_best_msssim'].to(torch.device(loc))
        val_best_psnr = checkpoint['val_best_psnr'].to(torch.device(loc))
        val_best_msssim = checkpoint['val_best_msssim'].to(torch.device(loc))
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.not_resume_epoch:
        args.start_epoch = 0

    # setup dataloader
    img_transforms = transforms.ToTensor()
    DataSet = CustomDataSet
    train_data_dir = f'./datasets/{args.dataset.lower()}'
    val_data_dir = f'./datasets/{args.dataset.lower()}'

    train_dataset = DataSet(train_data_dir, img_transforms,vid_list=args.vid, resol=args.resol, frame_gap=args.frame_gap, )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True,
         num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

    val_dataset = DataSet(val_data_dir, img_transforms, vid_list=args.vid, resol=args.resol, frame_gap=args.test_gap, offset=args.offset )
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize,  shuffle=False,
         num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False, worker_init_fn=worker_init_fn)
    
    #used for both train and eval
    global data_size
    data_size = len(train_dataset)
    global len_agg
    len_agg = len(agg_ind)+1
    # frame buffer
    global key_frames
    key_frames = torch.cuda.FloatTensor(data_size+1, 3, args.resol[1], args.resol[0]).fill_(0).half()
    
    
    if args.eval_only:
        print('Evaluation ...')
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        print_str = f'{time_str}\t Results for checkpoint: {args.weight}\n'
        if prune_net:
            for param in param_to_prune:
                prune.remove(param[0], param[1])
            sparsity_num = 0.
            for param in param_list:
                sparsity_num += (param.weight == 0).sum()
            print_str += f'Model sparsity at Epoch{args.start_epoch}: {sparsity_num / 1e6 / total_params}\n'

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        val_psnr, val_msssim = evaluate(model, train_dataloader, val_dataloader, local_rank, args)
        print_str += f'PSNR/ms_ssim on validate set for bit {args.wbit}: {round(val_psnr.item(),2)}/{round(val_msssim.item(),4)}'
        print(print_str)
        with open('{}/eval.txt'.format(args.outf), 'a') as f:
            f.write(print_str + '\n\n')        
        return
    
    
    start = datetime.now()
    total_epochs = args.epochs * args.cycles
    
    updated = False
    for epoch in range(args.start_epoch, total_epochs):
        model.train()
        ##### prune the network if needed #####
        if prune_net and epoch in args.prune_steps:
            prune_num += 1 
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
            
            sparsity_num = 0.
            for param in param_list:
                sparsity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{epoch}: {sparsity_num / 1e6 / total_params}')
        
        epoch_start_time = datetime.now()
        psnr_list = []
        msssim_list = []
        
        # if frame buffer is empty, fill the buffer before training
        if updated == False:
            for i, (data,  norm_idx) in enumerate(train_dataloader):
                t_input = norm_idx
                if local_rank is not None:
                    data = data.cuda(local_rank, non_blocking=True)
                    t_input = t_input.cuda(local_rank, non_blocking=True)
                else:
                    data,  t_input = data.cuda(non_blocking=True),   t_input.cuda(non_blocking=True)
                # forward and backward
                output_list = model(t_input)
                key_frames[torch.round(norm_idx*data_size).long()] = output_list[1][:,0:3,:,:].detach().clone().half()
                
        soft = nn.Softmax(dim=1)
        # iterate over dataloader
        for i, (data,  norm_idx) in enumerate(train_dataloader):
            if i > 10 and args.debug:
                break
            t_input = norm_idx
            if local_rank is not None:
                data = data.cuda(local_rank, non_blocking=True)
                t_input = t_input.cuda(local_rank, non_blocking=True)
            else:
                data,  t_input = data.cuda(non_blocking=True),  t_input.cuda(non_blocking=True)
            # forward and backward
            output_list = model(t_input)
            
            idx = torch.round(norm_idx*data_size).long()
            # update frame buffer
            key_frames[idx] = output_list[1][:,0:3,:,:].detach().clone().half()
            # flow-guided frame aggregation
            # indexing for batched warping
            agg_idx = idx.unsqueeze(1).repeat(1, len_agg-1) + torch.tensor(agg_ind)
            agg_idx = torch.where(agg_idx >= 0, agg_idx, data_size)
            agg_idx = torch.where(agg_idx < data_size, agg_idx, data_size)
            agg_frames = key_frames[agg_idx.long()].to(torch.float32)
            agg_frames = agg_frames.reshape(-1, 3, agg_frames.shape[-2], agg_frames.shape[-1])
            flows = output_list[0][:,:2*(len_agg-1),:,:]
            flows = flows.reshape(-1, 2, flows.shape[-2], flows.shape[-1])
            # warping
            agg_frames = resample(agg_frames, flows).unsqueeze(0)
            agg_frames = agg_frames.reshape(-1, len_agg-1, 3, agg_frames.shape[-2], agg_frames.shape[-1])
            # first aggregation
            wei1 = output_list[0][:,(len_agg-1)*2:,:,:]
            wei1 = torch.nn.functional.interpolate(wei1, scale_factor=4, mode='nearest')
            wei1 = soft(wei1)
            # the aggregated frame
            agg_frame = torch.sum(agg_frames * wei1.unsqueeze(2), dim=1, keepdim=True)
            # second aggregation
            agg_frames = torch.cat([output_list[1][:,0:3,:,:].unsqueeze(1), agg_frame],dim=1)  
            wei2 = soft(output_list[1][:,3:5,:,:]).unsqueeze(2)
            # aggregated frame, independent frame, final frame
            output_list = [agg_frame.squeeze(1), output_list[1][:,0:3,:,:], torch.sum(agg_frames * wei2, dim=1)]
            
            target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
            loss_list = [loss_fn(output, target, args) for output, target in zip(output_list, target_list)]
            # weighted loss function
            loss_list = [loss_list[i] * (args.lw if i < len(loss_list) - 1 else 1) for i in range(len(loss_list))]
            loss_sum = sum(loss_list)
            lr = adjust_lr(optimizer, epoch % args.epochs, i, data_size, args)
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            # compute psnr and msssim
            psnr_list.append(psnr_fn(output_list, target_list))
            msssim_list.append(msssim_fn(output_list, target_list))
            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                train_psnr = torch.cat(psnr_list, dim=0) #(batchsize, num_stage)
                train_psnr = torch.mean(train_psnr, dim=0) #(num_stage)
                train_msssim = torch.cat(msssim_list, dim=0) #(batchsize, num_stage)
                train_msssim = torch.mean(train_msssim.float(), dim=0) #(num_stage)
                time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}'.format(
                    time_now_string, local_rank, epoch+1, args.epochs, i+1, len(train_dataloader), lr, 
                    RoundTensor(train_psnr, 2, False), RoundTensor(train_msssim, 4, False))
                print(print_str, flush=True)
                if local_rank in [0, None]:
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')
        
        updated = True
        
        # collect numbers from other gpus
        if args.distributed and args.ngpus_per_node > 1:
            train_psnr = all_reduce([train_psnr.to(local_rank)])
            train_msssim = all_reduce([train_msssim.to(local_rank)])

        # ADD train_PSNR TO TENSORBOARD
        if local_rank in [0, None]:
            h, w = output_list[-1].shape[-2:]
            is_train_best = train_psnr[-1] > train_best_psnr
            train_best_psnr = train_psnr[-1] if train_psnr[-1] > train_best_psnr else train_best_psnr
            train_best_msssim = train_msssim[-1] if train_msssim[-1] > train_best_msssim else train_best_msssim
            writer.add_scalar(f'Train/PSNR_{h}X{w}_gap{args.frame_gap}', train_psnr[-1].item(), epoch+1)
            writer.add_scalar(f'Train/MSSSIM_{h}X{w}_gap{args.frame_gap}', train_msssim[-1].item(), epoch+1)
            writer.add_scalar(f'Train/best_PSNR_{h}X{w}_gap{args.frame_gap}', train_best_psnr.item(), epoch+1)
            writer.add_scalar(f'Train/best_MSSSIM_{h}X{w}_gap{args.frame_gap}', train_best_msssim, epoch+1)
            print_str = '\t{}p: current: {:.2f}\t best: {:.2f}\t msssim_best: {:.4f}\t'.format(h, train_psnr[-1].item(), train_best_psnr.item(), train_best_msssim.item())
            print(print_str, flush=True)
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')
            writer.add_scalar('Train/lr', lr, epoch+1)
            epoch_end_time = datetime.now()
            print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                    (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch) ))

        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch+1,
            'state_dict': state_dict,
            'train_best_psnr': train_best_psnr,
            'train_best_msssim': train_best_msssim,
            'val_best_psnr': val_best_psnr,
            'val_best_msssim': val_best_msssim,
            'optimizer': optimizer.state_dict(),   
        }    
        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or epoch > total_epochs - 10:
            val_start_time = datetime.now()
            val_psnr, val_msssim = evaluate(model, train_dataloader, val_dataloader, local_rank, args)
            val_end_time = datetime.now()
            if args.distributed and args.ngpus_per_node > 1:
                val_psnr = all_reduce([val_psnr.to(local_rank)])
                val_msssim = all_reduce([val_msssim.to(local_rank)])            
            if local_rank in [0, None]:
                # ADD val_PSNR TO TENSORBOARD
                h, w = output_list[-1].shape[-2:]
                print_str = f'Eval best_PSNR at epoch{epoch+1}:'
                is_val_best = val_psnr[-1] > val_best_psnr
                val_best_psnr = val_psnr[-1] if is_val_best else val_best_psnr
                val_best_msssim = val_msssim[-1] if val_msssim[-1] > val_best_msssim else val_best_msssim
                writer.add_scalar(f'Val/PSNR_{h}X{w}_gap{args.test_gap}', val_psnr[-1], epoch+1)
                writer.add_scalar(f'Val/MSSSIM_{h}X{w}_gap{args.test_gap}', val_msssim[-1], epoch+1)
                writer.add_scalar(f'Val/best_PSNR_{h}X{w}_gap{args.test_gap}', val_best_psnr, epoch+1)
                writer.add_scalar(f'Val/best_MSSSIM_{h}X{w}_gap{args.test_gap}', val_best_msssim, epoch+1)
                print_str += '\t{}p: current: {:.2f}\tbest: {:.2f} \tbest_msssim: {:.4f}\t Time/epoch: {:.2f}'.format(h, val_psnr[-1].item(),
                     val_best_psnr.item(), val_best_msssim.item(), (val_end_time - val_start_time).total_seconds())
                print(print_str)
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
                if is_val_best:
                    torch.save(save_checkpoint, '{}/model_val_best.pth'.format(args.outf))

        if local_rank in [0, None]:
            # state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if is_train_best:
                torch.save(save_checkpoint, '{}/model_train_best.pth'.format(args.outf))

    print("Training complete in: " + str(datetime.now() - start))

# quantization function used for evaluation
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
    
@torch.no_grad()
def evaluate(model, train_dataloader, val_dataloader, local_rank, args):
    # Saved weights are not applied quantization -> applying qfn
    if args.wbit == 8 and args.eval_only:
        cur_ckt = model.state_dict()
        from dahuffman import HuffmanCodec
        quant_weight_list = []
        for k,v in cur_ckt.items():
            #applying qfn
            weight = torch.tanh(v)
            quant_v = qfn.apply(weight, args.wbit)
            valid_quant_v = quant_v
            quant_weight_list.append(valid_quant_v.flatten())
            cur_ckt[k] = quant_v
        cat_param = torch.cat(quant_weight_list)
        input_code_list = cat_param.tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))
        print(num_freq)
        
        # generating HuffmanCoding table
        codec = HuffmanCodec.from_data(input_code_list)

        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        avg_bits = total_bits / len(input_code_list)    
        # import pdb; pdb.set_trace; from IPython import embed; embed()       
        encoding_efficiency = avg_bits / args.wbit
        print_str = f'Entropy encoding efficiency for bit {args.wbit}: {encoding_efficiency}'
        print(print_str)
        if local_rank in [0, None]:
            with open('{}/eval.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')       
        model.load_state_dict(cur_ckt)

        # import pdb; pdb.set_trace; from IPython import embed; embed()

    psnr_list = []
    msssim_list = []
    if args.dump_images:
        from torchvision.utils import save_image
        visual_dir = f'{args.outf}/visualize'
        print(f'Saving predictions to {visual_dir}')
        if not os.path.isdir(visual_dir):
            os.makedirs(visual_dir)
    
    time_list = []
    model.eval()
    soft = nn.Softmax(dim=1)
    global key_frames
    for i, (data,  norm_idx) in enumerate(train_dataloader):
        if i > 10 and args.debug:
            break
        t_input = norm_idx
        if local_rank is not None:
            data = data.cuda(local_rank, non_blocking=True)
            t_input = t_input.cuda(local_rank, non_blocking=True)
        else:
            data,  t_input = data.cuda(non_blocking=True), t_input.cuda(non_blocking=True)

        # compute psnr and msssim
     
        output_list = model(t_input)
        key_frames[torch.round((norm_idx-1e-6)*data_size).long()] = output_list[1][:,0:3,:,:].detach().clone().half()
   
    for i, (data,  norm_idx) in enumerate(val_dataloader):
        t_input = norm_idx
        if local_rank is not None:
            data = data.cuda(local_rank, non_blocking=True)
            t_input = t_input.cuda(local_rank, non_blocking=True)
        else:
            data,  t_input = data.cuda(non_blocking=True), t_input.cuda(non_blocking=True)

        # compute psnr and msssim
        fwd_num = 10 if args.eval_fps else 1
        for _ in range(fwd_num):
            start_time = datetime.now()
            output_list = model(t_input)

            idx = torch.round((norm_idx-1e-6)*data_size).long()
            # indexing for batched warping
            agg_idx = idx.unsqueeze(1).repeat(1, len_agg-1) + torch.tensor(agg_ind)
            agg_idx = torch.where(agg_idx >= 0, agg_idx, data_size)
            agg_idx = torch.where(agg_idx < data_size, agg_idx, data_size)
            agg_frames = key_frames[agg_idx.long()].to(torch.float32)
            agg_frames = agg_frames.reshape(-1, 3, agg_frames.shape[-2], agg_frames.shape[-1])
            flows = output_list[0][:,:2*(len_agg-1),:,:]
            flows = flows.reshape(-1, 2, flows.shape[-2], flows.shape[-1])
            # warping
            agg_frames = resample(agg_frames, flows).unsqueeze(0)
            agg_frames = agg_frames.reshape(-1, len_agg-1, 3, agg_frames.shape[-2], agg_frames.shape[-1])
            
            # first aggregation
            wei1 = output_list[0][:,(len_agg-1)*2:,:,:]
            wei1 = torch.nn.functional.interpolate(wei1, scale_factor=4, mode='nearest')
            wei1 = soft(wei1)
            # the aggregate frame
            agg_frame = torch.sum(agg_frames * wei1.unsqueeze(2), dim=1, keepdim=True)
            # second aggregation
            agg_frames = torch.cat([output_list[1][:,0:3,:,:].unsqueeze(1), agg_frame],dim=1)  
            wei2 = soft(output_list[1][:,3:5,:,:]).unsqueeze(2)
            output_list = [torch.sum(agg_frames * wei2, dim=1)]

            torch.cuda.synchronize()
            # torch.cuda.current_stream().synchronize()
            time_list.append((datetime.now() - start_time).total_seconds())

        # dump predictions
        if args.dump_images:
            for batch_ind in range(args.batchSize):
                full_ind = i * args.batchSize + batch_ind
                save_image(output_list[-1][batch_ind], f'{visual_dir}/pred_{full_ind}.png')
                save_image(data[batch_ind], f'{visual_dir}/gt_{full_ind}.png')

        # compute psnr and ms-ssim
        target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
        psnr_list.append(psnr_fn(output_list, target_list))
        msssim_list.append(msssim_fn(output_list, target_list))
        val_psnr = torch.cat(psnr_list, dim=0)              #(batchsize, num_stage)
        val_psnr = torch.mean(val_psnr, dim=0)              #(num_stage)
        val_msssim = torch.cat(msssim_list, dim=0)          #(batchsize, num_stage)
        val_msssim = torch.mean(val_msssim.float(), dim=0)  #(num_stage)        
        if i % args.print_freq == 0:
            fps = fwd_num * (i+1) * args.batchSize / sum(time_list)
            print_str = 'Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {} FPS: {}'.format(
                local_rank, i+1, len(val_dataloader),
                RoundTensor(val_psnr, 2, False), RoundTensor(val_msssim, 4, False), round(fps, 2))
            print(print_str)
            if local_rank in [0, None]:
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
    model.train()

    return val_psnr, val_msssim


if __name__ == '__main__':
    main()
    

