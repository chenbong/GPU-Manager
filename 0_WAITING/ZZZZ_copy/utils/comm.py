import os
import torch
import torch.nn
import numpy as np

def make_divisible(v, divisor=8, min_value=8):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

def round_metric(flops, divisor, offset):
    return int(round((flops - offset) / divisor) * divisor + offset)


def linear(Vs, Ve, e, E):
    return Vs + (Ve-Vs) * (e/E) 

def sin(Vs, Ve, e, E):
    if Vs <= Ve:
        return Vs + (Ve-Vs)*np.sin(e*np.pi/2/E)
    else:
        return Ve + (Vs-Ve)*np.sin(np.pi/2 + e*np.pi/2/E)

def exp(Vs, Ve, e, E):
    return Vs * (Ve/Vs)**(e/E)


def query_gpu(qargs=[]):
    def parse(line, qargs):
        numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']#可计数的参数
        power_manage_enable=lambda v:(not 'Not Support' in v)#lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
        to_numberic=lambda v:float(v.upper().strip().replace('MIB','').replace('W',''))#带单位字符串去掉单位
        process = lambda k,v:((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
        return {k:process(k,v) for k,v in zip(qargs,line.strip().split(','))}
    qargs =['index','gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']+ qargs
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    gpu_info_list = [parse(line,qargs) for line in results]
    for gpu_info in gpu_info_list:
        gpu_info['mem_used'] = gpu_info['memory.total'] - gpu_info['memory.free']
        gpu_info['mem_used_pct'] = gpu_info['mem_used'] / gpu_info['memory.total']
    return gpu_info_list


def query_empty_gpu_list():
    gpu_info_list = query_gpu()
    empty_gpu_list = []
    for gpu_info in gpu_info_list:
        if gpu_info['mem_used'] < 20:
        # if gpu_info['mem_used_pct'] < 0.5:
            empty_gpu_list.append(gpu_info['index'])
    return empty_gpu_list


def adapt_channels(model_cfg):
    if model_cfg['model_name'] == 'mobilenet_v1':
        stage_cout_mults = model_cfg['stage_cout_mults']
        assert len(stage_cout_mults) == 14

        width_mults = [
                  stage_cout_mults[0],

            None, stage_cout_mults[1],

            None, stage_cout_mults[2],
            None, stage_cout_mults[3],

            None, stage_cout_mults[4],
            None, stage_cout_mults[5],

            None, stage_cout_mults[6],
            None, stage_cout_mults[7],
            None, stage_cout_mults[8],
            None, stage_cout_mults[9],
            None, stage_cout_mults[10],
            None, stage_cout_mults[11],

            None, stage_cout_mults[12],
            None, stage_cout_mults[13],
        ]
        return width_mults

    elif model_cfg['model_name'] == 'mobilenet_v2':
        stage_cout_mults = model_cfg['stage_cout_mults']
        block_cmid_mults = model_cfg['block_cmid_mults']
        assert len(stage_cout_mults) == 9
        assert len(block_cmid_mults) == 16
        width_mults = [
                                        stage_cout_mults[0],


                                 None, stage_cout_mults[1],

            block_cmid_mults[0], None, stage_cout_mults[2],
            block_cmid_mults[1], None, stage_cout_mults[2],
            
            block_cmid_mults[2], None, stage_cout_mults[3],
            block_cmid_mults[3], None, stage_cout_mults[3],
            block_cmid_mults[4], None, stage_cout_mults[3],

            block_cmid_mults[5], None, stage_cout_mults[4],
            block_cmid_mults[6], None, stage_cout_mults[4],
            block_cmid_mults[7], None, stage_cout_mults[4],
            block_cmid_mults[8], None, stage_cout_mults[4],

            block_cmid_mults[9], None, stage_cout_mults[5],
            block_cmid_mults[10], None, stage_cout_mults[5],
            block_cmid_mults[11], None, stage_cout_mults[5],

            block_cmid_mults[12], None, stage_cout_mults[6],
            block_cmid_mults[13], None, stage_cout_mults[6],
            block_cmid_mults[14], None, stage_cout_mults[6],

            block_cmid_mults[15], None, stage_cout_mults[7],


                                        stage_cout_mults[8],
        ]
        return width_mults

    else:
        print('NotImplementedError')
        raise NotImplementedError
    
################################################################



def _calc_conv2d_flops(input_size, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, verbose=False):
    assert kernel_size % 2 == 1, 'kernel_size should be odd.'
    output_size = int((input_size + padding * 2 - kernel_size) / stride) + 1
    flops = (output_size**2 * (out_channels / groups)) * (kernel_size**2 * in_channels)

    if verbose:
        print(f'Conv2d({in_channels}, {out_channels}, k={kernel_size}, s=({stride}, {stride}), pad=({padding}, {padding}), group={groups}, b={bias})\t\t\t\t{flops:,}')
    return flops, output_size, out_channels


def _calc_avgpool_flops(input_size, in_channels, output_size, verbose=False):
    flops = input_size**2 * in_channels
    out_channels = in_channels
    if verbose:
        print(f'AvgPool2d(output_size=({output_size}, {output_size}))\t\t\t\t{flops:,}')
    return flops, output_size, out_channels


def _calc_fc_flops(in_channels, out_channels, verbose=False):
    flops = in_channels*out_channels
    if verbose:
        print(f'Linear(in={in_channels}, out={out_channels})\t\t\t\t{flops:,}')
    return flops

def _calc_mbv1_block_flops(input_size, cin, cout, stride, cur_layer_id, width_mults, verbose=False):
    f_cin = make_divisible(cin * width_mults[cur_layer_id-1])
    total_flops = 0

    flops, input_size, cin = _calc_conv2d_flops(input_size, f_cin, f_cin, kernel_size=3, stride=stride, padding=1, groups=f_cin, bias=False, verbose=verbose)
    total_flops += flops
    cur_layer_id += 1

    f_cout = make_divisible(cout * width_mults[cur_layer_id])
    flops, input_size, cin = _calc_conv2d_flops(input_size, f_cin, f_cout, kernel_size=1, stride=1, padding=0, bias=False, verbose=verbose)
    total_flops += flops

    return total_flops, input_size, f_cout


def _calc_mbv2_block_flops(input_size, cin, cout, stride, expand_ratio, cur_layer_id, width_mults, verbose=False):
    f_cin = make_divisible(cin * width_mults[cur_layer_id-1])

    total_flops = 0
    if expand_ratio != 1:
        f_expand = make_divisible(cin * expand_ratio * width_mults[cur_layer_id])
        flops, input_size, f_cin = _calc_conv2d_flops(input_size, f_cin, f_expand, kernel_size=1, stride=1, padding=0, bias=False, verbose=verbose)
        total_flops += flops
        cur_layer_id += 1

    flops, input_size, f_cin = _calc_conv2d_flops(input_size, f_cin, f_cin, kernel_size=3, stride=stride, padding=1, groups=f_cin, bias=False, verbose=verbose)
    total_flops += flops
    cur_layer_id += 1

    f_cout = make_divisible(cout * width_mults[cur_layer_id])
    flops, input_size, f_cin = _calc_conv2d_flops(input_size, f_cin, f_cout, kernel_size=1, stride=1, padding=0, bias=False, verbose=verbose)
    total_flops += flops

    # return total_flops, input_size, f_cin
    return total_flops, input_size, f_cout
    





def calc_subnet_flops(model_cfg, verbose=False):
    if model_cfg['model_name'] == 'mobilenet_v1':
        input_size = model_cfg['resolution']
        width_mults = adapt_channels(model_cfg)
        cur_layer_id = -1
        total_flops = 0

        # head
        cur_layer_id += 1
        cout = make_divisible(32 * width_mults[cur_layer_id])
        flops, input_size, cin = _calc_conv2d_flops(input_size, 3, cout, kernel_size=3, stride=2, padding=1, bias=False, verbose=verbose)
        total_flops += flops

        # blocks
        cur_layer_id += 1; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=32, cout=64, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=64, cout=128, stride=2, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=128, cout=128, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=128, cout=256, stride=2, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=256, cout=256, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=256, cout=512, stride=2, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=512, cout=1024, stride=2, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=1024, cout=1024, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        
        # pool
        flops, input_size, cin = _calc_avgpool_flops(input_size, cin, output_size=1, verbose=verbose)
        total_flops += flops
        
        # classifier
        cin = input_size**2 * cin
        flops = _calc_fc_flops(cin, 1000, verbose=verbose)
        total_flops += flops

        if verbose:
            print(f'Total flops: {total_flops/1e6:.2f}M')

        return total_flops / 1e6

    elif model_cfg['model_name'] == 'mobilenet_v2':
        
        input_size = model_cfg['resolution']
        width_mults = adapt_channels(model_cfg)
        cur_layer_id = -1
        total_flops = 0

        # head
        cur_layer_id += 1
        cout = make_divisible(32 * width_mults[cur_layer_id])
        flops, input_size, cin = _calc_conv2d_flops(input_size, 3, cout, kernel_size=3, stride=2, padding=1, bias=False, verbose=verbose)
        total_flops += flops

        # blocks
        cur_layer_id += 1; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=32, cout=16, stride=1, expand_ratio=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops

        cur_layer_id += 2; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=16, cout=24, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=24, cout=24, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops

        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=24, cout=32, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=32, cout=32, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=32, cout=32, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops

        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=32, cout=64, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=64, cout=64, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=64, cout=64, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=64, cout=64, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops

        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=64, cout=96, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=96, cout=96, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=96, cout=96, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops

        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=96, cout=160, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=160, cout=160, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=160, cout=160, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops

        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=160, cout=320, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops

        # tail
        cur_layer_id += 3
        flops, input_size, cin = _calc_conv2d_flops(input_size, cin, 1280, kernel_size=1, stride=1, padding=0, bias=False, verbose=verbose)
        total_flops += flops
        
        # pool
        flops, input_size, cin = _calc_avgpool_flops(input_size, cin, output_size=1, verbose=verbose)
        total_flops += flops

        # classifier
        cin = input_size**2 * cin
        flops = _calc_fc_flops(cin, 1000, verbose=verbose)
        total_flops += flops

        if verbose:
            print(f'Total flops: {total_flops:.2f}M')
        
        return total_flops / 1e6

    else:
        print('NotImplementedError')
        raise NotImplementedError

def set_active_subnet(model, model_cfg):
    '''
    model_cfg = {    # mbv2
        'model_name': 'mobilenet_v2',
        'resolution' = 224,
        'stage_cout_mults' = [],  # len=1 +7 +1 = 9
        'block_cmid_mults' = [],  # len=1*(2-1) + (2+3+4+3+3+1)*(3-1)= 33
        'infer_metric': 301,
        'infer_metric_target': 300,
    },
    model_cfg = {    # mbv1
        'model_name': 'mobilenet_v1',
        'resolution' = 224,
        'stage_cout_mults' = [],  # len=1 +1+2+2+6+2= 14
        'infer_metric': 568,
        'infer_metric_target': 570,

    }
    '''
    width_mults = adapt_channels(model_cfg)

    model.apply(lambda m: setattr(m, 'width_mults', width_mults))
    model.apply(lambda m: setattr(m, 'infer_metric_target', model_cfg['infer_metric_target']))
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.active_resolution = model_cfg['resolution']
    else:
        model.active_resolution = model_cfg['resolution']




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


