import torch
import torch.nn as nn

from utils.comm import make_divisible
from utils.config import args



class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, layer_id=None):
        super(USConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.width_mults = None
        self.layer_id = layer_id

        self.f_groups = None

    def forward(self, inputs):
        in_channels = inputs.shape[1]
        if self.groups == 1:
            self.f_groups = 1
            if self.layer_id is None:
                out_channels = self.out_channels
            else:
                out_channels = make_divisible(self.out_channels * self.width_mults[self.layer_id])
        else:
            self.f_groups = in_channels
            out_channels = in_channels

        weight = self.weight[:out_channels, :in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.f_groups)

        return y


class USLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, layer_id=None):
        super(USLinear, self).__init__(in_features, out_features, bias=bias)
        self.width_mults = None
        self.layer_id = layer_id

    def forward(self, inputs):
        in_features = inputs.shape[1]
        if self.layer_id is None:
            out_features = self.out_features
        else:
            out_features = make_divisible(self.out_features_basic * self.width_mults[self.layer_id])

        weight = self.weight[:out_features, :in_features]
        if self.bias is not None:
            bias = self.bias[:out_features]
        else:
            bias = self.bias
        y = nn.functional.linear(inputs, weight, bias)
        return y

class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(USBatchNorm2d, self).__init__(num_features, affine=True, track_running_stats=False)
        self.width_mults = None
        self.infer_metric_target = None

        self.infer_target_list = list(range(args.infer_metric_target_range[0], args.infer_metric_target_range[1]+1, args.infer_metric_target_step))

        self.bn = nn.ModuleList([
            nn.BatchNorm2d(self.num_features, affine=False) for _ in self.infer_target_list
            # nn.BatchNorm2d(self.num_features, affine=False) for _ in range(1)
        ])


    def forward(self, inputs):
        num_features = inputs.size(1)

        # debug
        # y = nn.functional.batch_norm(
        #         inputs,
        #         self.bn[0].running_mean[:num_features],
        #         self.bn[0].running_var[:num_features],
        #         self.weight[:num_features],
        #         self.bias[:num_features],
        #         self.training,
        #         self.momentum,
        #         self.eps)
        # return y

        if self.infer_metric_target >= args.infer_metric_target_range[0] and self.infer_metric_target <= args.infer_metric_target_range[1]:
            idx = self.infer_target_list.index(self.infer_metric_target)
            y = nn.functional.batch_norm(
                inputs,
                self.bn[idx].running_mean[:num_features],
                self.bn[idx].running_var[:num_features],
                self.weight[:num_features],
                self.bias[:num_features],
                self.training,
                self.momentum,
                self.eps)
        else:
            y = nn.functional.batch_norm(
                inputs,
                self.running_mean,
                self.running_var,
                self.weight[:num_features],
                self.bias[:num_features],
                self.training,
                self.momentum,
                self.eps)
        return y
