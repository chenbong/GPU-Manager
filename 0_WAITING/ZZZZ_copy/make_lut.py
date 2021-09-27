import os
import time
import pickle
import torch
import torch.nn as nn
import argparse
from collections import namedtuple

from models.comp_mobilenet_v1 import comp_conv3x3_dw_pw as mobilenet_v1_block
from models.comp_mobilenet_v1 import CompModel as mobilenet_v1
from models.comp_mobilenet_v2 import CompInvertedResidual as mobilenet_v2_block
from models.comp_mobilenet_v2 import CompModel as mobilenet_v2
from utils.comm import make_divisible, adapt_channels

# block_type: Conv2d, MBV1Block, MBV2Block
BlockCfg = namedtuple('BlockCfg', 'block_type in_size cin cout cmid kernel_size stride padding', defaults=[None, None, None, None])

def get_model(model_name, width_mults=None):
    if model_name == 'mobilenet_v1':
        return mobilenet_v1(width_mults=width_mults)
    elif model_name == 'mobilenet_v2':
        return mobilenet_v2(width_mults=width_mults)
    else:
        raise NotImplementedError


def get_block(block_cfg: BlockCfg):
    if block_cfg.block_type == 'Conv2d':
        return nn.Sequential(
            nn.Conv2d(block_cfg.cin, block_cfg.cout, kernel_size=block_cfg.kernel_size, stride=block_cfg.stride, padding=block_cfg.padding, bias=False),
            nn.BatchNorm2d(block_cfg.cout),
            nn.ReLU6(inplace=True),
        )
    elif block_cfg.block_type == 'Linear':
        return nn.Sequential(nn.Linear(block_cfg.cin, block_cfg.cout))

    elif block_cfg.block_type == 'MBV1Block':
        return mobilenet_v1_block(cin = block_cfg.cin, cout = block_cfg.cout, stride = block_cfg.stride)

    elif block_cfg.block_type == 'MBV2Block':
        return mobilenet_v2_block(cin = block_cfg.cin, cout = block_cfg.cout, cmid = block_cfg.cmid, stride = block_cfg.stride)

    else:
        raise NotImplementedError


def add_lut_block_item(lut, block_type, in_size, min_width_mult, max_width_mult, c_step, cin, cout, cmid=None, cin_loop=True, cout_loop=True, kernel_size=None, stride=None, padding=None):
    cin_range = range(make_divisible(cin * min_width_mult), make_divisible(cin * max_width_mult)+1, c_step)
    cout_range = range(make_divisible(cout * min_width_mult), make_divisible(cout * max_width_mult)+1, c_step)

    if block_type == 'Conv2d':
        for cin_i in cin_range if cin_loop else [cin]:
            for cout_i in cout_range if cout_loop else [cout]:
                block_cfg = BlockCfg(block_type='Conv2d', in_size=in_size, cin=cin_i, cout=cout_i, kernel_size=kernel_size, stride=stride, padding=padding)
                lut[block_cfg] = None

    elif block_type == 'Linear':
        for cin_i in cin_range if cin_loop else [cin]:
            for cout_i in cout_range if cout_loop else [cout]:
                block_cfg = BlockCfg(block_type='Linear', in_size=in_size, cin=cin_i, cout=cout_i, kernel_size=None, stride=None, padding=None)
                lut[block_cfg] = None

    elif block_type == 'MBV1Block':
        for cin_i in cin_range:
            for cout_i in cout_range:
                block_cfg = BlockCfg(block_type='MBV1Block', in_size=in_size, cin=cin_i, cout=cout_i, cmid=None, kernel_size=None, stride=stride, padding=None)
                lut[block_cfg] = None

    elif block_type == 'MBV2Block':
        cmid_range = range(make_divisible(cmid * min_width_mult), make_divisible(cmid * max_width_mult)+1, c_step)
        for cin_i in cin_range:
            for cout_i in cout_range:
                for cmid_i in cmid_range:
                    block_cfg = BlockCfg(block_type='MBV2Block', in_size=in_size, cin=cin_i, cout=cout_i, cmid=cmid_i, kernel_size=None, stride=stride, padding=None)
                    lut[block_cfg] = None

    else:
        raise NotImplementedError
    return lut

def create_lut_model_items(model_name, resolution_range, resolution_step, min_width_mult, max_width_mult, c_step):
    lut = {}
    _kernel_size, _stride, _padding = 3, 2, 1
    if model_name == 'mobilenet_v1':
        for in_size in range(resolution_range[0], resolution_range[1]+1, resolution_step):
            # head
            add_lut_block_item(lut, 'Conv2d', in_size, min_width_mult, max_width_mult, c_step, cin=3, cout=32, cin_loop=False, kernel_size=3, stride=2, padding=1); in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1

            # blocks
            add_lut_block_item(lut, 'MBV1Block', in_size, min_width_mult, max_width_mult, c_step, cin=32, cout=64, stride=1)

            add_lut_block_item(lut, 'MBV1Block', in_size, min_width_mult, max_width_mult, c_step, cin=64, cout=128, stride=2); in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
            add_lut_block_item(lut, 'MBV1Block', in_size, min_width_mult, max_width_mult, c_step, cin=128, cout=128, stride=1)

            add_lut_block_item(lut, 'MBV1Block', in_size, min_width_mult, max_width_mult, c_step, cin=128, cout=256, stride=2); in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
            add_lut_block_item(lut, 'MBV1Block', in_size, min_width_mult, max_width_mult, c_step, cin=256, cout=256, stride=1)

            add_lut_block_item(lut, 'MBV1Block', in_size, min_width_mult, max_width_mult, c_step, cin=256, cout=512, stride=2); in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
            add_lut_block_item(lut, 'MBV1Block', in_size, min_width_mult, max_width_mult, c_step, cin=512, cout=512, stride=1)
            add_lut_block_item(lut, 'MBV1Block', in_size, min_width_mult, max_width_mult, c_step, cin=512, cout=512, stride=1)
            add_lut_block_item(lut, 'MBV1Block', in_size, min_width_mult, max_width_mult, c_step, cin=512, cout=512, stride=1)
            add_lut_block_item(lut, 'MBV1Block', in_size, min_width_mult, max_width_mult, c_step, cin=512, cout=512, stride=1)
            add_lut_block_item(lut, 'MBV1Block', in_size, min_width_mult, max_width_mult, c_step, cin=512, cout=512, stride=1)

            add_lut_block_item(lut, 'MBV1Block', in_size, min_width_mult, max_width_mult, c_step, cin=512, cout=1024, stride=2); in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
            add_lut_block_item(lut, 'MBV1Block', in_size, min_width_mult, max_width_mult, c_step, cin=1024, cout=1024, stride=1)

            # classifier
            in_size = 1
            add_lut_block_item(lut, 'Linear', in_size, min_width_mult, max_width_mult, c_step, cin=1024, cout=1000, cout_loop=False)

    elif model_name == 'mobilenet_v2':
        for in_size in range(resolution_range[0], resolution_range[1]+1, resolution_step):
            # head
            add_lut_block_item(lut, 'Conv2d', in_size, min_width_mult, max_width_mult, c_step, cin=3, cout=32, cin_loop=False, kernel_size=3, stride=2, padding=1); in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1

            # features
            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=32, cout=16, cmid=32*1, stride=1)

            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=16, cout=24, cmid=16*6, stride=2); in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=24, cout=24, cmid=24*6, stride=1)

            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=24, cout=32, cmid=24*6, stride=2); in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=32, cout=32, cmid=32*6, stride=1)
            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=32, cout=32, cmid=32*6, stride=1)

            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=32, cout=64, cmid=32*6, stride=2); in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=32, cout=64, cmid=32*6, stride=1)
            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=64, cout=64, cmid=64*6, stride=1)
            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=64, cout=64, cmid=64*6, stride=1)

            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=64, cout=96, cmid=64*6, stride=1)
            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=96, cout=96, cmid=96*6, stride=1)
            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=96, cout=96, cmid=96*6, stride=1)

            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=96, cout=160, cmid=96*6, stride=2); in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=160, cout=160, cmid=160*6, stride=1)
            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=160, cout=160, cmid=160*6, stride=1)

            add_lut_block_item(lut, 'MBV2Block', in_size, min_width_mult, max_width_mult, c_step, cin=160, cout=320, cmid=160*6, stride=1)

            add_lut_block_item(lut, 'Conv2d', in_size, min_width_mult, max_width_mult, c_step, cin=320, cout=1280, kernel_size=1, stride=1, padding=0)

            # classifier
            in_size = 1
            add_lut_block_item(lut, 'Linear', in_size, min_width_mult, max_width_mult, c_step, cin=1280, cout=1000, cout_loop=False)
    else:
        raise NotImplementedError

    return lut


def compute_model_lut(lut, device, warmup, iter, job_dir, save_freq=1000, verbose=False):
    if device == 'cpu':
        dev = torch.device('cpu')
        bs = 1
    elif device == 'gpu':
        dev = torch.device('cuda')
        bs = 128
    else:
        raise NotImplementedError

    sstart = time.time()
    for i, block_cfg in enumerate(lut):
        if block_cfg.block_type == 'Linear':
            x = torch.randn(bs, block_cfg.cin).to(dev)
        else:
            x = torch.randn(bs, block_cfg.cin, block_cfg.in_size, block_cfg.in_size).to(dev)
        block = get_block(block_cfg).to(dev).eval()
        if device == 'cpu':
            with torch.no_grad():
                for _ in range(warmup):
                    block(x)
                start = time.time()
                for _ in range(iter):
                    block(x)
                cost = time.time() - start

        elif device == 'gpu':
            with torch.no_grad():
                for _ in range(warmup):
                    block(x)
                    torch.cuda.current_stream().synchronize()
                start = time.time()
                for _ in range(iter):
                    block(x)
                    torch.cuda.current_stream().synchronize()
                cost = time.time() - start
        else:
            raise NotImplementedError
        
        ms = 1000.0 * cost / iter / bs
        lut[block_cfg] = ms
        if verbose:
            print(f'{i}, {block_cfg}: {lut[block_cfg]:.4f}ms, cost:{time.time() - sstart:.2f}s')
            sstart = time.time()
        
        if (i+1) % save_freq == 0:
            with open(os.path.join(job_dir, 'lut_ckpt', f'{i+1:05d}.pkl'), 'wb') as f:
                pickle.dump(lut, f)
    with open(os.path.join(job_dir, 'lut_ckpt', 'final.pkl'), 'wb') as f:
        pickle.dump(lut, f)


def create_model_block_list(model_cfg):
    block_list = []
    _kernel_size, _stride, _padding = 3, 2, 1
    if model_cfg['model_name'] == 'mobilenet_v1':
        base_stage_couts = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512,   512, 512, 1024, 1024]
        stage_cout_mults = model_cfg['stage_cout_mults']
        stage_couts = [make_divisible(base_stage_couts[i]*cout_mult) for i, cout_mult in enumerate(stage_cout_mults)]

        in_size = model_cfg['resolution']
        block_list += [BlockCfg('Conv2d', in_size,                3, stage_couts[0], kernel_size=3, stride=2, padding=1),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1

        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[0], stage_couts[1], cmid=None, kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[1], stage_couts[2], cmid=None, kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[2], stage_couts[3], cmid=None, kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[3], stage_couts[4], cmid=None, kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[4], stage_couts[5], cmid=None, kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[5], stage_couts[6], cmid=None, kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[6], stage_couts[7], cmid=None, kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[7], stage_couts[8], cmid=None, kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[8], stage_couts[9], cmid=None, kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[9], stage_couts[10], cmid=None, kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[10], stage_couts[11], cmid=None, kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[11], stage_couts[12], cmid=None, kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[12], stage_couts[13], cmid=None, kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('Linear',          1, stage_couts[13],          1000, cmid=None, kernel_size=None, stride=None, padding=None),]

    elif model_cfg['model_name'] == 'mobilenet_v2':
        base_stage_couts = [32, 16, 24, 32, 64, 96, 160, 320, 1280]                             # len=9
        base_block_cmids = [16*6]*1 + [24*6]*2 + [32*6]*3 + [64*6]*4 + [96*6]*3 + [160*6]*3     # len=16

        stage_cout_mults = model_cfg['stage_cout_mults']
        block_cmid_mults = model_cfg['block_cmid_mults']
        stage_couts = [make_divisible(base_stage_couts[i]*cout_mult) for i, cout_mult in enumerate(stage_cout_mults)]
        block_cmids = [make_divisible(base_block_cmids[i]*cout_mult) for i, cout_mult in enumerate(block_cmid_mults)]

        in_size = model_cfg['resolution']

        # head
        block_list += [BlockCfg('Conv2d', in_size,                3, stage_couts[0], kernel_size=3, stride=2, padding=1),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        
        # blocks
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[0], stage_couts[1], cmid=stage_couts[0], kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[1], stage_couts[2], cmid=block_cmids[0], kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[2], stage_couts[2], cmid=block_cmids[1], kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[2], stage_couts[3], cmid=block_cmids[2], kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[3], stage_couts[3], cmid=block_cmids[3], kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[3], stage_couts[3], cmid=block_cmids[4], kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[3], stage_couts[4], cmid=block_cmids[5], kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[4], stage_couts[4], cmid=block_cmids[6], kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[4], stage_couts[4], cmid=block_cmids[7], kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[4], stage_couts[4], cmid=block_cmids[8], kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[4], stage_couts[5], cmid=block_cmids[9], kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[5], stage_couts[5], cmid=block_cmids[10], kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[5], stage_couts[5], cmid=block_cmids[11], kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[5], stage_couts[6], cmid=block_cmids[12], kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[6], stage_couts[6], cmid=block_cmids[13], kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[6], stage_couts[6], cmid=block_cmids[14], kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[6], stage_couts[7], cmid=block_cmids[15], kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('Conv2d',    in_size, stage_couts[7], stage_couts[8], kernel_size=1, stride=1, padding=0),]

        # classifier
        block_list += [BlockCfg('Linear',          1, stage_couts[8],          1000, cmid=None, kernel_size=None, stride=None, padding=None),]


    else:
        raise NotImplementedError

    return block_list

class LatencyPredictor():
    def __init__(self, lut_dir) -> None:
        with open(lut_dir, 'rb') as f:
            self.lut = pickle.load(f)
    
    def predict_subnet_latency(self, model_cfg):
        subnet_block_list = create_model_block_list(model_cfg)
        total_ms = 0.
        for block_cfg in subnet_block_list:
            if block_cfg in self.lut:
                total_ms += self.lut[block_cfg]
            else:
                raise IndexError
        total_us = total_ms * 1000.
        return total_us

def build_comp_model(model_cfg):
    
    pass

def main():
    parser = argparse.ArgumentParser(description='make_lut')
    parser.add_argument('--model_name', type=str, default='mobilenet_v1', choices=['mobilenet_v1', 'mobilenet_v2'])
    parser.add_argument('--width_mults_range', type=int, default=[0.75, 1.0], nargs='+')
    parser.add_argument('--c_step', type=int, default=8)
    parser.add_argument('--resolution_range', type=int, default=[96, 224], nargs='+')
    parser.add_argument('--resolution_step', type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "gpu", "trt"], default="gpu", help="Use GPU, CPU or TensorRT latency")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Profiling iterations")
    parser.add_argument("--job_dir", default='./')
    parser.add_argument('--gpu_id', type=str, default='0')

    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    os.makedirs(os.path.join(args.job_dir, 'lut_ckpt'))
    lut = create_lut_model_items(
        args.model_name, 
        [args.resolution_range[0], args.resolution_range[1]], args.resolution_step, 
        args.width_mults_range[0], args.width_mults_range[1], args.c_step
    )
    print(len(lut))

    compute_model_lut(lut, args.device, args.warmup, args.iters, args.job_dir, save_freq=1000, verbose=True)


if __name__ == '__main__':
    main()

