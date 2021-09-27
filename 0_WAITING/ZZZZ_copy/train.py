import importlib
import os
import time
import random
import operator

import torch
import torch.cuda
import torch.nn.functional as F
import numpy as np
import datetime
from scipy.special import softmax
import torch.multiprocessing as mp

import ComputePostBN
from utils.setlogger import get_logger
from utils.model_profiling import model_profiling
from utils.config import args
from utils.datasets import get_imagenet_pytorch_train_loader, get_imagenet_pytorch_val_loader
import utils.comm as comm
from utils.comm import AverageMeter, round_metric
from make_lut import LatencyPredictor, BlockCfg

from utils.subnet_sampler import ArchSampler, SubnetGenerator


import wandb

print(args)


# set log files
saved_path = os.path.join(args.job_dir, '{}-{}'.format(args.dataset, args.model_name))
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
logger = get_logger(os.path.join(saved_path, '{}_div1optimizer.log'.format('test' if args.test_only else 'train')))

metric_target_step = args.infer_metric_target_step
metric_target_offset = args.infer_metric_target_range[0] % metric_target_step
subnet_generator = SubnetGenerator(args.model_name, args.resolution_range, args.resolution_step, args.width_mult_range, metric_target_step, metric_target_offset)


if args.infer_metric_type == 'flops':
    calc_infer_metric = comm.calc_subnet_flops
else:
    latency_predictor = LatencyPredictor(lut_dir=args.lut_dir)
    calc_infer_metric = latency_predictor.predict_subnet_latency

def gen_map_worker(worker_id, worker_num, args, workers_ret, base_seed):
    random.seed(base_seed + worker_id)
    infer_target_list = list(range(args.infer_metric_target_range[0]+metric_target_step, args.infer_metric_target_range[1]-metric_target_step+1, metric_target_step))
    model_cfgs = []

    for i in range(args.gen_map_num // worker_num):
        # print(infer_target_list)
        # exit()
        model_cfg = subnet_generator.sample_subnet_within_list(infer_target_list)
        # model_cfg = subnet_generator.sample_subnet()
        model_cfgs.append(model_cfg)
    workers_ret[worker_id] = model_cfgs


def set_random_seed():
    """set random seed"""
    if hasattr(args, 'random_seed'):
        seed = args.random_seed
    else:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_model():
    """get model"""
    model_lib = importlib.import_module('models.'+args.model_name)
    model = model_lib.Model(args.n_classes, input_size=max(args.resolution_range))
    return model

def get_optimizer(model):
    """get optimizer"""
    # all depthwise convolution (N, 1, x, x) has no weight decay
    # weight decay only on normal conv and fc
    if args.dataset == 'imagenet1k':
        model_params = []
        for params in model.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:  # normal conv
                weight_decay = args.weight_decay
            elif len(ps) == 2:  # fc
                weight_decay = args.weight_decay
            else:
                weight_decay = 0
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': args.lr, 'momentum': args.momentum,
                    'nesterov': args.nesterov}
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum, nesterov=args.nesterov,
                                    weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, len_train_loader, model, criterion, optimizer, lr_scheduler, arch_pools, best_archs, arch_sampler):

    t_start = time.time()
    model.train()

    if args.converge_to_pool != 'False':
        if 'linear' in args.converge_to_pool:
            supernet_sample_rate = comm.linear(max(args.supernet_p_range), min(args.supernet_p_range), epoch, args.epochs)
        elif 'sin' in args.converge_to_pool:
            supernet_sample_rate = comm.sin(max(args.supernet_p_range), min(args.supernet_p_range), epoch, args.epochs)
        elif 'exp' in args.converge_to_pool:
            supernet_sample_rate = comm.exp(max(args.supernet_p_range), min(args.supernet_p_range), epoch, args.epochs)
        else:
            raise NotImplementedError

        t = comm.exp(max(args.pool_softmax_t_range), min(args.pool_softmax_t_range), epoch, args.epochs)
    else:
        supernet_sample_rate = 1.0
    print(f'==debug==: supernet_sample_rate: {supernet_sample_rate}')



    for batch_idx, (inputs, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        # do max_net
        max_net = subnet_generator.sample_subnet(max_net=True)
        comm.set_active_subnet(model, max_net)
        max_output = model(inputs)
        loss = criterion(max_output, labels)
        loss.backward()
        max_output_detach = max_output.detach()


        # do other widths and resolution
        # 有放回抽样k个training_infer_metric_target, 加入training_infer_metric_target_list
        training_infer_metric_target_list = random.choices(arch_pools['infer_metric_target_list'], k=args.num_subnet_training-1)
        training_model_cfgs_list = []
        random_flag = random.random()
        for infer_metric_target in training_infer_metric_target_list:
            # 从超网中采样1个 target infer_metric 的子网
            if random_flag < supernet_sample_rate or len(arch_pools['pools'][infer_metric_target]) < args.pool_size:
                if args.sampler_method == 'None':
                    args.sampler_num_sample = 1

                candidate_model_cfgs = arch_sampler.sample_model_cfgs_according_to_prob(
                    infer_metric_target, n_samples=args.sampler_num_sample
                )
                my_pred_accs = []
                for model_cfg in candidate_model_cfgs:
                    comm.set_active_subnet(model, model_cfg)

                    with torch.no_grad():
                        performance_metric_tensor = -1.0 * criterion(model(inputs), labels)
                        my_pred_accs.append(performance_metric_tensor)

                if args.sampler_method == 'bestup':
                    idx, _ = max(enumerate(my_pred_accs), key=operator.itemgetter(1))
                elif args.sampler_method == 'worstup':
                    idx, _ = min(enumerate(my_pred_accs), key=operator.itemgetter(1))
                elif args.sampler_method == 'None':
                    idx = 0
                else:
                    raise NotImplementedError
                performance_metric_tensor = my_pred_accs[idx]
                candidate_model_cfg = candidate_model_cfgs[idx]
                candidate_model_cfg_str = str(candidate_model_cfg)
            # 从模型池中采样1个target infer_metric的子网
            else:
                if 'uniform' in args.converge_to_pool:
                    candidate_model_cfg_str = random.choice(list(arch_pools['pools'][infer_metric_target].keys()))
                elif 'rank' in args.converge_to_pool:
                    _prob = list(arch_pools['pools'][infer_metric_target].values())
                    _prob = np.array(_prob)

                    prob = softmax(_prob/t)
                    print(f'==debug==: epoch: {epoch}, t: {t}, max_prob: {max(prob)}')
                    [candidate_model_cfg_str] = random.choices(list(arch_pools['pools'][infer_metric_target].keys()), weights=prob)
                else:
                    raise NotImplementedError
                candidate_model_cfg = eval(candidate_model_cfg_str)
                comm.set_active_subnet(model, candidate_model_cfg)

                with torch.no_grad():
                    performance_metric_tensor = -1.0 * criterion(model(inputs), labels)

            assert infer_metric_target == candidate_model_cfg['infer_metric_target']

            training_model_cfgs_list.append(candidate_model_cfg)
            if candidate_model_cfg_str not in arch_pools['pools'][infer_metric_target]:
                arch_pools['pools'][infer_metric_target][candidate_model_cfg_str] = performance_metric_tensor.item()  # performance_metric: higher is bertter
            else:
                arch_pools['pools'][infer_metric_target][candidate_model_cfg_str] = arch_pools['pools'][infer_metric_target][candidate_model_cfg_str]*args.metric_momentum + performance_metric_tensor.item()*(1-args.metric_momentum)

            if len(arch_pools['pools'][infer_metric_target]) > arch_pools['max_size']:
                min_value_key = min(arch_pools['pools'][infer_metric_target].keys(), key=(lambda k: arch_pools['pools'][infer_metric_target][k]))
                print(f"infer_metric_target: {infer_metric_target}, push: {performance_metric_tensor.item()}, pop: {arch_pools['pools'][infer_metric_target][min_value_key]}")
                arch_pools['pools'][infer_metric_target].pop(min_value_key)

        for arch_id in range(args.num_subnet_training):   # range(start, stop) [start, stop)=[start, stop-1]
            if args.sample_type == 'max_randc_min' and arch_id == args.num_subnet_training-1:
                # do min_net
                min_net = subnet_generator.sample_subnet(min_net=True)
                comm.set_active_subnet(model, min_net)
            else:
                # do middle_net
                comm.set_active_subnet(model, training_model_cfgs_list[arch_id])
            output = model(inputs)

            if args.kd_type == 'max_kd':
                loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1), F.softmax(max_output_detach, dim=1))
            else:
                raise NotImplementedError

            loss.backward()

        optimizer.step()
        lr_scheduler.step()


        # print training log
        if not os.path.exists(args.job_dir):
            print('args.job_dir not exist...')
            raise NotImplementedError

        if batch_idx % args.print_freq == 0 or batch_idx == len_train_loader-1:
            with torch.no_grad():
                indices = torch.max(max_output, dim=1)[1]
                acc = (indices == labels).sum().cpu().numpy() / indices.size()[0]
                logger.info('TRAIN {:.1f}s LR:{:.4f} {}x Epoch:{}/{} Iter:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(
                    time.time() - t_start, optimizer.param_groups[0]['lr'], str(max(args.width_mult_range)), epoch,
                    args.epochs, batch_idx, len_train_loader, loss, acc)
                )
                t_start = time.time()


def validate(epoch, val_loader, len_val_loader, model, criterion, postloader, subnets_to_be_evaluated):
    t_start = time.time()
    model.eval()
    wandb_log_dict = {}
    infer_metric_list = []
    acc_list = []
    # print(subnets_to_be_evaluated)
    # exit()
    with torch.no_grad():
        for model_cfg in subnets_to_be_evaluated:
            comm.set_active_subnet(model, subnets_to_be_evaluated[model_cfg])
            model = ComputePostBN.ComputeBN(model, postloader)

            loss, acc, cnt = 0, 0, 0
            for _, (inputs, labels) in enumerate(val_loader):
                # inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                output = model(inputs)
                loss += criterion(output, labels).cpu().numpy() * labels.size()[0]
                indices = torch.max(output, dim=1)[1]
                acc += (indices == labels).sum().cpu().numpy()
                cnt += labels.size()[0]

            logger.info(f"VAL:{time.time() - t_start:.1f}s, id:{model_cfg}, infer_metric:{subnets_to_be_evaluated[model_cfg]['infer_metric']}, Epoch:{epoch}/{args.epochs}, Loss:{loss/cnt:.4f}, Acc:{acc/cnt:.4f}")
            t_start = time.time()

            infer_metric_list.append(subnets_to_be_evaluated[model_cfg]['infer_metric'])
            acc_list.append(acc/cnt)

            arch_key = f"{round_metric(subnets_to_be_evaluated[model_cfg]['infer_metric'], metric_target_step, metric_target_offset)}_acc1"
            
            if arch_key in wandb_log_dict:
                wandb_log_dict[arch_key] = max(acc/cnt, wandb_log_dict[arch_key])
            else:
                wandb_log_dict[arch_key] = acc/cnt
    logger.info(f'== Epoch:{epoch}/{args.epochs} ==')
    logger.info(f'infer_metric_list = {infer_metric_list}')
    logger.info(f'acc_list = {acc_list}')

    if 'wandb' in globals():
        logger.info(f'==wandb.log== {wandb_log_dict}')
        wandb.log(wandb_log_dict)


def train_val_test(arch_pools, best_archs, arch_sampler):
    """train and val"""
    # seed
    set_random_seed()

    # model
    model = get_model()
    # print(model)
    # exit()


    model_wrapper = torch.nn.DataParallel(model).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    # train_loader, val_loader = get_dataset()
    if args.dataset == 'imagenet1k':
        if args.data_backend == 'pytorch':
            train_loader, len_train_loader  = get_imagenet_pytorch_train_loader()
            val_loader, len_val_loader = get_imagenet_pytorch_val_loader()
        else:
            raise NotImplementedError
    elif args.dataset == 'cifar10':
        raise NotImplementedError
    else:
        raise NotImplementedError


    # check pretrained
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        # update keys from external models
        if type(checkpoint) == dict and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        new_keys = list(model_wrapper.state_dict().keys())
        old_keys = list(checkpoint.keys())
        new_keys = [key for key in new_keys if 'running' not in key]
        new_keys = [key for key in new_keys if 'tracked' not in key]
        old_keys = [key for key in old_keys if 'running' not in key]
        old_keys = [key for key in old_keys if 'tracked' not in key]
        if not args.test_only:
            old_keys = old_keys[:-2]
            new_keys = new_keys[:-2]

        new_checkpoint = {}
        for key_new, key_old in zip(new_keys, old_keys):
            new_checkpoint[key_new] = checkpoint[key_old]
        model_wrapper.load_state_dict(new_checkpoint, strict=False)
        print('Loaded model {}.'.format(args.pretrained))
    optimizer = get_optimizer(model_wrapper)
    # check resume training
    if args.resume:
        checkpoint = torch.load(args.resume)
        model_wrapper.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len_train_loader*args.epochs)
        lr_scheduler.last_epoch = last_epoch

        arch_pools = checkpoint['arch_pools']
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['np_random_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['torch_cuda_rng_state'])

        print('Loaded checkpoint {} at epoch {}.'.format(args.resume, last_epoch))
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len_train_loader*args.epochs)
        # last_epoch = lr_scheduler.last_epoch - 1
        last_epoch = -1
        # print model and do profiling
        if args.profiling:
            max_net = subnet_generator.sample_subnet(max_net=True)
            model_profiling(model, max_net, use_cuda=True, print_=True, verbose=False)
            # print(comm.calc_subnet_flops(max_net, verbose=False))
            print(calc_infer_metric(max_net, verbose=False))
            # exit()

            min_net = subnet_generator.sample_subnet(min_net=True)
            print(min_net)
            model_profiling(model, min_net, use_cuda=True, print_=True, verbose=False)
            # print(comm.calc_subnet_flops(min_net, verbose=False))
            print(calc_infer_metric(min_net, verbose=False))
            # exit()

            model_cfg = subnet_generator.sample_subnet()
            model_profiling(model, model_cfg, use_cuda=True, print_=True, verbose=False)
            # print(comm.calc_subnet_flops(model_cfg, verbose=False))
            print(calc_infer_metric(model_cfg, verbose=False))
            # exit()


    if args.test_only:
        logger.info('Start testing.')
        subnets_to_be_evaluated = None
        validate(last_epoch, val_loader, model_wrapper, criterion, train_loader, subnets_to_be_evaluated)
        return

    logger.info('Start training.')
    epoch_time = AverageMeter('Epoch time:', ':.2f')
    start = time.time()
    best_acc1 = 0.
    for epoch in range(last_epoch + 1, args.epochs):
        # train
        train(epoch, train_loader, len_train_loader, model_wrapper, criterion, optimizer, lr_scheduler, arch_pools, best_archs, arch_sampler)

        subnets_to_be_evaluated = {}
        subnets_to_be_evaluated["max_net"] = subnet_generator.sample_subnet(max_net=True)
        for infer_metric_target in arch_pools['infer_metric_target_list']:
            logger.info(f"=== {infer_metric_target}, {len(arch_pools['pools'][infer_metric_target])}")
            pool_list = list(arch_pools['pools'][infer_metric_target].items())
            pool_list.sort(key=operator.itemgetter(1), reverse=True)
            logger.info(f"=> {pool_list[:args.valid_topk]}")
            if ((epoch+1) % args.valid_all_freq == 0 or (epoch+1) > (args.epochs - args.valid_last_epoch)) and len(pool_list) >= args.valid_topk:
                subnets_to_be_evaluated[f'{infer_metric_target}'] = eval(pool_list[0][0])
                for i in range(1, args.valid_topk):
                    subnets_to_be_evaluated[f'{infer_metric_target}[{i}]'] = eval(pool_list[i][0])

        if (epoch+1) % args.valid_all_freq == 0 or (epoch+1) > (args.epochs - args.valid_last_epoch):
            subnets_to_be_evaluated['min_net'] = subnet_generator.sample_subnet(min_net=True)

        # val
        validate(epoch, val_loader, len_val_loader, model_wrapper, criterion, train_loader, subnets_to_be_evaluated)
        # torch.save(
        #     {
        #         'model': model_wrapper.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'last_epoch': epoch,
        #         'arch_pools': arch_pools,
        #         'random_state' : random.getstate(),
        #         'np_random_state': np.random.get_state(),
        #         'torch_rng_state' : torch.get_rng_state(),
        #         'torch_cuda_rng_state': torch.cuda.get_rng_state(),
        #     },
        #     os.path.join(saved_path, 'checkpoint.pt'))
        epoch_time.update(time.time() - start)
        start = time.time()
        total_cost = datetime.timedelta(seconds=int(epoch_time.avg * args.epochs))
        finish_time = time.time() + epoch_time.avg * (args.epochs-1 - epoch)
        finish_dt = datetime.datetime.fromtimestamp(finish_time).strftime("%m/%d %H:%M:%S")
        logger.info(f"========= {epoch_time}s, Total cost: {total_cost}s, Will finish at: {finish_dt}. =========")

        if args.debug_epoch and epoch > args.debug_epoch:
            exit()
    return


def main():
    if 'wandb' in globals():
        wandb.init(project=args.wandb_project, name=args.job_dir.split('/')[-2]+'/'+args.job_dir.split('/')[-1], config=args, dir=args.job_dir, resume=True)


    if args.gpu_num:
        while True:
            if not os.path.exists(args.job_dir):
                print('args.job_dir not exist...')
                raise NotImplementedError
            empty_gpu_list = comm.query_empty_gpu_list()
            logger.info(f'{datetime.datetime.now().strftime("%x-%X")}: {empty_gpu_list}')
            if len(empty_gpu_list) >= args.gpu_num:
                break
            time.sleep(360)
        visible_gpu_ids_str = ','.join(str(i) for i in empty_gpu_list[:args.gpu_num])
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpu_ids_str
        logger.info(f'CUDA_VISIBLE_DEVICES={visible_gpu_ids_str}')


    ### gen infer_metric table.map
    if (not args.test_only) and args.gen_map_num:
        from torch.multiprocessing import Pool
        # worker_num = min(mp.cpu_count(), 64)
        worker_num = 64
        p = Pool(worker_num)
        manager = mp.Manager()
        workers_ret = manager.dict()
        start = time.time()
        for i in range(worker_num):
            p.apply_async(gen_map_worker, args=(i, worker_num, args, workers_ret, args.random_seed))
            # gen_map_worker(i, worker_num, args, workers_ret, args.random_seed)

        print('Waiting for all sub worker done...')
        p.close()
        p.join()
        logger.info(f'Gen infer_metric map done, cost: {time.time()-start:.2f}s')

        start = time.time()
        with open(args.sampler_metric_target_map_path, 'w') as f:
            for worker_i in range(len(workers_ret)):
                for model_cfg in workers_ret[worker_i]:
                    f.write(f'{model_cfg}\n')
        logger.info(f'Write map file done, cost: {time.time()-start:.2f}s')

    #build model_cfg sampler
    start = time.time()
    arch_sampler = ArchSampler(args.model_name, args.sampler_metric_target_map_path, metric_target_step, metric_target_offset)
    logger.info(f'=> min infer_metric target: {arch_sampler.min_infer_metric_target}, max_infer_metric: {arch_sampler.max_infer_metric_target}, build arch_sampler cost:{time.time()-start:.2f}s')
    # exit()
    # else:
    #     arch_sampler = None

    arch_pools = {}
    arch_pools['max_size'] = args.pool_size
    arch_pools['infer_metric_target_list'] = sorted(arch_sampler.prob_map['infer_metric'].keys(), reverse=True)

    
    arch_pools['pools'] = {}
    for infer_metric_target in arch_pools['infer_metric_target_list']:
        arch_pools['pools'][infer_metric_target] = {}

    best_archs = {}
    arch_ids = ['max_net', 'min_net']
    if arch_pools:
        arch_ids = arch_ids + [i for i in arch_pools['infer_metric_target_list']]
    for arch_id in arch_ids:
        best_archs[f'{arch_id}'] = {'best_acc1': 0.0, 'best_acc5': 0.0, 'infer_metric': None}

    """train and eval model"""
    train_val_test(arch_pools, best_archs, arch_sampler)


if __name__ == "__main__":
    main()