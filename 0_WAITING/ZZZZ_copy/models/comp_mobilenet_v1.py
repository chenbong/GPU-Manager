import math
import torch.nn as nn

from utils.comm import make_divisible, adapt_channels

######################################################
def comp_conv3x3_dw_pw(cin, cout, stride, cur_layer_id=None, width_mults=None):
    layers = []

    #* depth wise
    real_cin = make_divisible(cin*width_mults[cur_layer_id-1]) if cur_layer_id else cin
    layers += [
        nn.Conv2d(real_cin, real_cin, kernel_size=3, stride=stride, padding=1, groups=real_cin, bias=False),
        nn.BatchNorm2d(real_cin),
        nn.ReLU6(inplace=True),
    ]
    cur_layer_id = cur_layer_id + 1 if cur_layer_id else None

    #* point wise
    real_cout = make_divisible(cout*width_mults[cur_layer_id]) if cur_layer_id else cout
    layers += [
        nn.Conv2d(real_cin, real_cout, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(real_cout),
        nn.ReLU6(inplace=True),
    ]
    return nn.Sequential(*layers)


class CompModel(nn.Module):
    def __init__(self, num_class=1000, width_mults=None):
        super(CompModel, self).__init__()
        if width_mults is None:
            model_cfg = {}
            model_cfg['model_name'] = 'mobilenet_v1'
            model_cfg['stage_cout_mults'] = [1.0]*14
            width_mults = adapt_channels(model_cfg)

        self.features = []
        cur_layer_id = -1

        # head
        cur_layer_id += 1
        real_cout = make_divisible(32*width_mults[cur_layer_id])
        self.features += [
            nn.Sequential(
                nn.Conv2d(3, real_cout, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(real_cout),
                nn.ReLU6(inplace=True),
            )
        ]

        # features blocks
        cur_layer_id += 1; self.features += [comp_conv3x3_dw_pw(cin=32, cout=64, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults),]

        cur_layer_id += 2; self.features += [comp_conv3x3_dw_pw(cin=64, cout=128, stride=2, cur_layer_id=cur_layer_id, width_mults=width_mults),]
        cur_layer_id += 2; self.features += [comp_conv3x3_dw_pw(cin=128, cout=128, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults),]

        cur_layer_id += 2; self.features += [comp_conv3x3_dw_pw(cin=128, cout=256, stride=2, cur_layer_id=cur_layer_id, width_mults=width_mults),]
        cur_layer_id += 2; self.features += [comp_conv3x3_dw_pw(cin=256, cout=256, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults),]

        cur_layer_id += 2; self.features += [comp_conv3x3_dw_pw(cin=256, cout=512, stride=2, cur_layer_id=cur_layer_id, width_mults=width_mults),]
        cur_layer_id += 2; self.features += [comp_conv3x3_dw_pw(cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults),]
        cur_layer_id += 2; self.features += [comp_conv3x3_dw_pw(cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults),]
        cur_layer_id += 2; self.features += [comp_conv3x3_dw_pw(cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults),]
        cur_layer_id += 2; self.features += [comp_conv3x3_dw_pw(cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults),]
        cur_layer_id += 2; self.features += [comp_conv3x3_dw_pw(cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults),]

        cur_layer_id += 2; self.features += [comp_conv3x3_dw_pw(cin=512, cout=1024, stride=2, cur_layer_id=cur_layer_id, width_mults=width_mults),]
        cur_layer_id += 2; self.features += [comp_conv3x3_dw_pw(cin=1024, cout=1024, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults),]

        self.features = nn.Sequential(*self.features)

        # pool
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        
        # classifier
        cur_layer_id += 2
        real_cin = make_divisible(1024*width_mults[cur_layer_id-1])
        self.classifier = nn.Sequential(nn.Linear(real_cin, num_class))

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