"""config utilities for yml file."""
import os
import sys
import yaml

import argparse

parser = argparse.ArgumentParser(description='MutualNet')

parser.add_argument('--dataset', type=str, default='imagenet1k', choices=['imagenet1k', 'cifar100'])
parser.add_argument('--data_transforms', type=str, default=None, choices=['imagenet1k_mobile', 'imagenet1k_basic'])
parser.add_argument('--dataset_dir', type=str, default=None)
parser.add_argument('--data_loader_workers', type=int, default=4)
parser.add_argument('--n_classes', type=int, default=1000)
parser.add_argument('--val_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--nesterov', default=True, type=bool)
parser.add_argument('--lr_scheduler', default='cosine', type=str, choices=['cosine'])
parser.add_argument('--profiling', default=None, type=str)
parser.add_argument('--random_seed', default=1995, type=int)
parser.add_argument('--model_name', type=str, default='mobilenet_v2', choices=['mobilenet_v1', 'mobilenet_v2'])
parser.add_argument('--reset_parameters', type=bool, default=True)
parser.add_argument('--job_dir', default='/media/disk1/cbh/EXP/tmp', type=str)
parser.add_argument('--gpu_num', default=None, type=int)
parser.add_argument('--lr', default=0.5, type=float)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--test_only', default=False, type=bool)
parser.add_argument('--pretrained', default=None, type=str)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--sampler_method', default='None', type=str, choices=['bestup', 'worstup', 'None'])
parser.add_argument('--infer_metric_type', type=str, default='flops', choices=['flops', 'latency'])
parser.add_argument('--sampler_metric_target_map_path', default=None, type=str)
parser.add_argument('--sampler_num_sample', type=int, default=None)
parser.add_argument('--sample_type', type=str, default=None, choices=['max_randc_min', 'max_randc'])
parser.add_argument('--kd_type', default='max_kd', type=str, choices=['max_kd', 'hierachical_kd', 'None'])
parser.add_argument('--num_subnet_training', type=int, default=2)
parser.add_argument('--gen_map_num', type=int, default=None)
parser.add_argument('--pool_size', default=50, type=int)
parser.add_argument('--converge_to_pool', type=str, default='exp_rank', choices=['False', 'linear_uniform', 'sin_uniform', 'exp_uniform', 'linear_rank', 'sin_rank', 'exp_rank'])
parser.add_argument('--valid_all_freq', default=50, type=int)
parser.add_argument('--supernet_p_range', default=[1.0, 1e-2], nargs='+', type=float)
parser.add_argument('--pool_softmax_t_range', default=[1.0, 1e-2], nargs='+', type=float)
parser.add_argument('--metric_momentum', default=0.9, type=float)
parser.add_argument('--valid_last_epoch', default=5, type=int)
parser.add_argument('--valid_topk', default=5, type=int)
parser.add_argument('--resolution_range', type=int, default=[224, 128], nargs='+')
parser.add_argument('--resolution_step', type=int, default=8)
parser.add_argument('--width_mult_range', type=float, default=[0.7, 1.0], nargs='+')
parser.add_argument('--infer_metric_target_range', type=int, default=[10, 570], nargs='+')
parser.add_argument('--infer_metric_target_step', type=int, default=10)
parser.add_argument('--data_backend', type=str, default='pytorch', choices=['dali_cpu', 'dali_gpu', 'pytorch'])
parser.add_argument('--wandb_project', type=str, default='aaai_22_munet')
parser.add_argument('--lut_dir', type=str, default='/userhome/cbh/PROJECT/21.03/.DATA/MutualNet_fix/lut')
parser.add_argument('--debug_epoch', type=int, default=None)
run_args = parser.parse_args()

args = run_args

"""@nni.variable(nni.choice(0.2, 0.3, 0.4, 0.5), name=args.lr)"""
args.lr = args.lr

"""@nni.variable(nni.choice(1e-2), name=args.supernet_p_range[1])"""
args.supernet_p_range[1] = args.supernet_p_range[1]
args.pool_softmax_t_range[1] = args.pool_softmax_t_range[1]

"""@nni.variable(nni.choice(1024, 2048), name=args.batch_size)"""
args.batch_size = args.batch_size


# best: lr=0.3, t_end=1e-2, bs=1024


