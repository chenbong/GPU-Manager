# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random, time

from .comm import calc_subnet_flops, make_divisible, round_metric
from make_lut import LatencyPredictor

from .config import args

if args.infer_metric_type == 'flops':
    calc_infer_metric = calc_subnet_flops
else:
    latency_predictor = LatencyPredictor(lut_dir=args.lut_dir)
    calc_infer_metric = latency_predictor.predict_subnet_latency


def int2list(val, repeat_time=1):
    if isinstance(val, list):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]

def count_helper(v, infer_metric_target, m):
    if infer_metric_target not in m:
        m[infer_metric_target] = {}

    if v not in m[infer_metric_target]:
        m[infer_metric_target][v] = 0
    m[infer_metric_target][v] += 1 


def convert_count_to_prob(m):
    if isinstance(m[list(m.keys())[0]], dict):
        for k in m:
            convert_count_to_prob(m[k])
    else:
        t = sum(m.values())
        for k in m:
            m[k] = 1.0 * m[k] / t

def sample_helper(infer_metric_target, m):
    keys = list(m[infer_metric_target].keys())
    probs = list(m[infer_metric_target].values())
    return random.choices(keys, weights=probs)[0]   # random.choiceslist





class ArchSampler():
    def __init__(self, model_name, infer_metric_map_path, metric_target_step, metric_target_offset):
        super(ArchSampler, self).__init__()
        self.model_name = model_name
        self.metric_target_step = metric_target_step
        self.metric_target_offset = metric_target_offset

        if self.model_name == 'mobilenet_v2':
            self.cout_mults_key_list = ['stage_cout_mults', 'block_cmid_mults']
        elif self.model_name == 'mobilenet_v1':
            self.cout_mults_key_list = ['stage_cout_mults']
        
        with open(infer_metric_map_path, 'r') as fp:
            self.prob_map = self.build_trasition_prob_matrix(fp, self.metric_target_step, self.metric_target_offset)

        self.min_infer_metric_target = min(list(self.prob_map['infer_metric'].keys()))
        self.max_infer_metric_target = max(list(self.prob_map['infer_metric'].keys()))

    def build_trasition_prob_matrix(self, file_handler, metric_target_step, metric_target_offset):
        '''
        prob_map = {
            'infer_metric':{
                70: num,
                80: num,
                ...,
                300: num,
            },
            'resolution':{

            },
            'stage_cout_mults':{

            },
            'block_cmid_mults':{    # for mobilenet_v2

            },
        }
        '''
        prob_map = {}
        for k in ['infer_metric', 'resolution'] + self.cout_mults_key_list:
            prob_map[k] = {}

        cc = 0      # 
        for line in file_handler:
            vals = eval(line.strip())
            infer_metric_target = round_metric(vals['infer_metric'], metric_target_step, metric_target_offset)
            prob_map['infer_metric'][infer_metric_target] = prob_map['infer_metric'].get(infer_metric_target, 0) + 1

            r = vals['resolution']
            count_helper(r, infer_metric_target, prob_map['resolution'])
            
            for k in self.cout_mults_key_list:
                for idx, v in enumerate(vals[k]):
                    if idx not in prob_map[k]:
                        prob_map[k][idx] = {}
                    count_helper(v, infer_metric_target, prob_map[k][idx])

            cc += 1

        # convert count to probability
        for k in ['infer_metric', 'resolution'] + self.cout_mults_key_list:
            convert_count_to_prob(prob_map[k])
        prob_map['n_observations'] = cc

        return prob_map
    
    def sample_one_target(self, uniform=False):
        f_vals = list(self.prob_map['infer_metric'].keys())
        f_probs = list(self.prob_map['infer_metric'].values())

        if uniform:
            return random.choice(f_vals)
        else:
            return random.choices(f_vals, weights=f_probs)[0]

    def sample_model_cfgs_according_to_prob(self, infer_metric_target, n_samples=1, max_trials=1000):
        model_cfgs = []
        while len(model_cfgs) < n_samples:
            # for _trial in range(max_trials+1):
            while True:
                model_cfg = {}
                model_cfg['model_name'] = self.model_name
                model_cfg['resolution'] = sample_helper(infer_metric_target, self.prob_map['resolution'])

                for k in self.cout_mults_key_list:
                    model_cfg[k] = []
                    for idx in sorted(list(self.prob_map[k].keys())):
                        model_cfg[k].append(sample_helper(infer_metric_target, self.prob_map[k][idx]))

                # model_cfg['infer_metric'] = calc_subnet_flops(model_cfg)
                model_cfg['infer_metric'] = calc_infer_metric(model_cfg)

                model_cfg['infer_metric_target'] = round_metric(model_cfg['infer_metric'], self.metric_target_step, self.metric_target_offset)

                if model_cfg['infer_metric_target'] == infer_metric_target:
                    break
            #accepte the sample anyway
            model_cfgs.append(model_cfg)
        return model_cfgs

class SubnetGenerator():
    def __init__(self, model_name, resolution_range, resolution_step, width_range, metric_target_step, metric_target_offset):
        super(SubnetGenerator, self).__init__()
        self.model_name = model_name
        self.metric_target_step = metric_target_step
        self.metric_target_offset = metric_target_offset

        self.len_mults = {}
        if model_name == 'mobilenet_v2':
            self.len_mults['stage_cout_mults'] = 9      # len=1 +7 +1 = 9
            self.len_mults['block_cmid_mults'] = 16     # len=
        elif model_name == 'mobilenet_v1':
            self.len_mults['stage_cout_mults'] = 14     # len=1 +1+2+2+6+2= 14
        else:
            raise NotImplementedError

        self.resolution_range = resolution_range
        self.resolution_step = resolution_step
        self.width_range = width_range

        if self.model_name == 'mobilenet_v1':
            self.t_sum = 1.0
            self.alpha = 0.65
        elif self.model_name == 'mobilenet_v2':
            self.t_sum = 1.0
            self.alpha = 0.65
        else:
            raise NotImplementedError

    def sample_subnet(self, min_net=False, max_net=False):
    # def sample_subnet(self, min_net=False, max_net=False, t_sum=None, alpha=None):
        def _sample_val_from_range(range_start, range_stop, sample_min, sample_max, divisor=None):
            assert range_start < range_stop
            if sample_min:
                return range_start
            elif sample_max:
                return range_stop
            else:
                # rand_v = random.uniform(range_start, range_stop)
                rand_v = range_start + random.betavariate(self.alpha * self.t_sum, (1 - self.alpha) * self.t_sum) * (range_stop - range_start)
                # rand_v = range_start + random.betavariate(alpha * t_sum, (1 - alpha) * t_sum) * (range_stop - range_start)
                if divisor:
                    return make_divisible(rand_v, divisor=divisor, min_value=range_start)
                else:
                    return float('{:.3f}'.format(rand_v))

        def _sample_val_from_list(candidates_list, sample_min, sample_max):
            if sample_min:
                return min(candidates_list)
            elif sample_max:
                return max(candidates_list)
            else:
                return random.choice(candidates_list)

        model_cfg = {}
        model_cfg['model_name'] = self.model_name
        model_cfg['resolution'] = _sample_val_from_range(min(self.resolution_range), max(self.resolution_range), min_net, max_net, divisor=self.resolution_step)

        for k in self.len_mults.keys():
            model_cfg[k] = []
            for _ in range(self.len_mults[k]):
                model_cfg[k].append(_sample_val_from_range(self.width_range[0], self.width_range[1], min_net, max_net))

        # model_cfg['infer_metric'] = calc_subnet_flops(model_cfg)
        model_cfg['infer_metric'] = calc_infer_metric(model_cfg)
        model_cfg['infer_metric_target'] = round_metric(model_cfg['infer_metric'], self.metric_target_step, self.metric_target_offset)
        
        return model_cfg

    def sample_subnet_within_range(self, min_metric, max_metric):
        while True:
            model_cfg = self.sample_subnet() 
            if model_cfg['infer_metric'] >= min_metric and model_cfg['infer_metric'] <= max_metric:
                return model_cfg

    def sample_subnet_within_list(self, infer_metric_target_list):
        while True:
            model_cfg = self.sample_subnet()
            if model_cfg['infer_metric_target'] in infer_metric_target_list:
                return model_cfg


