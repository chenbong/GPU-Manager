import math
import torch.nn as nn

from utils.comm import make_divisible, adapt_channels



####################################################################
class CompInvertedResidual(nn.Module):
    def __init__(self, cin, cout, cmid, stride):
        super(CompInvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.residual_connection = stride == 1 and cin == cout

        layers = []
        if cin != cmid:
            layers += [
                nn.Conv2d(cin, cmid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(cmid),
                nn.ReLU6(inplace=True),
            ]

        # depthwise + project back
        layers += [
                nn.Conv2d(cmid, cmid, 3, stride, 1, groups=cmid, bias=False),
                nn.BatchNorm2d(cmid),
                nn.ReLU6(inplace=True),
        ]
        layers += [
                nn.Conv2d(cmid, cout, 1, 1, 0, bias=False),
                nn.BatchNorm2d(cout),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res

####################################################################
class _CompInvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand_ratio, cur_layer_id, width_mults):
        super(_CompInvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.residual_connection = stride == 1 and cin == cout

        layers = []
        expand = cin * expand_ratio
        real_cin = make_divisible(cin*width_mults[cur_layer_id-1])
        if expand_ratio != 1:
            real_expand = make_divisible(expand*width_mults[cur_layer_id])
        else:
            real_expand = expand

        # expand
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(real_cin, real_expand, 1, 1, 0, bias=False),
                nn.BatchNorm2d(real_expand),
                nn.ReLU6(inplace=True),
            ]
            cur_layer_id += 1

        # depthwise
        layers += [
                nn.Conv2d(real_expand, real_expand, 3, stride, 1, groups=expand, bias=False),
                nn.BatchNorm2d(real_expand),
                nn.ReLU6(inplace=True),
        ]
        cur_layer_id += 1

        # pointwise
        real_cout = make_divisible(cout*width_mults[cur_layer_id])
        layers += [
                nn.Conv2d(real_expand, real_cout, 1, 1, 0, bias=False),
                nn.BatchNorm2d(real_cout),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class CompModel(nn.Module):
    def __init__(self, n_classes=1000, width_mults=None):
        super(CompModel, self).__init__()
        if width_mults is None:
            model_cfg = {}
            model_cfg['model_name'] = 'mobilenet_v2'
            model_cfg['stage_cout_mults'] = [1.0]*9
            model_cfg['block_cmid_mults'] = [1.0]*16
            width_mults = adapt_channels(model_cfg)

        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.features = []
        cur_layer_id = -1

        # features_head
        cur_layer_id +=1
        real_cout = make_divisible(32*width_mults[cur_layer_id])
        self.features += [
            nn.Sequential(
                nn.Conv2d(3, real_cout, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(real_cout),
                nn.ReLU6(inplace=True),
            )
        ]

        # features_blocks
        cur_layer_id += 1; self.features += [_CompInvertedResidual(cin=32, cout=16, stride=1, expand_ratio=1, cur_layer_id=cur_layer_id, width_mults=width_mults), ]

        cur_layer_id += 2; self.features += [_CompInvertedResidual(cin=16, cout=24, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]
        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=24, cout=24, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]

        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=24, cout=32, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]
        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=32, cout=32, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]
        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=32, cout=32, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]

        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=32, cout=64, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]
        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=64, cout=64, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]
        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=64, cout=64, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]
        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=64, cout=64, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]

        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=64, cout=96, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]
        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=96, cout=96, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]
        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=96, cout=96, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]

        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=96, cout=160, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]
        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=160, cout=160, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]
        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=160, cout=160, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]

        cur_layer_id += 3; self.features += [_CompInvertedResidual(cin=160, cout=320, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults), ]

        # features_tail
        cur_layer_id += 3
        real_cin = make_divisible(320*width_mults[cur_layer_id-1])
        real_cout = make_divisible(1280*width_mults[cur_layer_id])
        self.features += [
            nn.Sequential(
                nn.Conv2d(real_cin, real_cout, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(real_cout),
                nn.ReLU6(inplace=True),
            )
        ]
        self.features = nn.Sequential(*self.features)

        # pool
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        # classifier
        cur_layer_id += 1
        real_cin = make_divisible(1280*width_mults[cur_layer_id-1])
        self.classifier = nn.Sequential(nn.Linear(real_cin, n_classes))
        
        self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
