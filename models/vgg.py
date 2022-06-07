import torch.nn as nn
from torchvision import ops


# vgg_cfgs = {
#    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# modified vgg11 config
mvgg11 = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512]
svgg = [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512]

# max pooling arguments
ms = [2, 2, [(2, 2), (2, 1), (0, 1)], [(2, 2), (2, 1), (0, 1)]]
# batch norm
bn = [2, 4, 6]
svgg_bn = [2, 4]
# deform conv
dc = [0, 1, 2, 3, 4, 5, 6]
svgg_dc = [0, 1, 2, 3, 4]
# kernel size
ks = [3, 3, 3, 3, 3, 3, 2]
svgg_ks = [3, 3, 3, 3, 2]
# padding size
ps = [1, 1, 1, 1, 1, 1, 0]
svgg_ps = [1, 1, 1, 1, 0]
# stride size
ss = [1, 1, 1, 1, 1, 1, 1]
svgg_ss = [1, 1, 1, 1, 1]


class DeformConv2d(nn.Module):
    def __init__(self, n_in, n_out, ks, ss, ps):
        super().__init__()
        self.conv2d_offset = nn.Conv2d(n_in, 2 * ks * ks, ks, ss, ps)
        self.deformconv2d = ops.DeformConv2d(n_in, n_out, ks, ss, ps)
    def forward(self, x):
        return self.deformconv2d(x, self.conv2d_offset(x))


def vgg(cfg, dc, bn, ms, ks, ss, ps, conv=nn.Conv2d, batch_norm=False):
    layers = []
    in_channels = 3
    i_maxp = 0
    i_conv = 0
    for v in cfg:
        if v == 'M':
            if isinstance(ms[i_maxp], list) and len(ms[i_maxp]) == 3:
                layers += [nn.MaxPool2d(ms[i_maxp][0], ms[i_maxp][1], ms[i_maxp][2])]
            else:
                layers += [nn.MaxPool2d(ms[i_maxp])]
            i_maxp += 1
        else:
            if i_conv in dc:
                conv2d = DeformConv2d(in_channels, v, ks[i_conv], ss[i_conv], ps[i_conv])
            else:
                conv2d = conv(in_channels, v, ks[i_conv], ss[i_conv], ps[i_conv])
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            elif i_conv in bn:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            i_conv += 1
    return nn.Sequential(*layers)
