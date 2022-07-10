import torch.nn as nn
from typing import Union, List, cast, Type
from collections import OrderedDict

from torch import Tensor
from torchvision import models
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1

class VGGBNPretrain(nn.Module):
    def __init__(self, output_index = None):
        super().__init__()
        self.features = _make_vgg_layers(vgg_cfgs['A'], True)
        self.splits=vgg_split['A']
        self._initialize_weights()
        self.output_index = output_index

    def forward(self, x):
        x = self.features[self.splits[0][0]:self.splits[0][1]](x) # 1
        x = self.features[self.splits[1][0]:self.splits[1][1]](x) # 1/2
        x = self.features[self.splits[2][0]:self.splits[2][1]](x) # 1/4
        x0 = self.features[self.splits[3][0]:self.splits[3][1]](x) # 1/8
        x1 = self.features[self.splits[4][0]:self.splits[4][1]](x0) # 1/16
        x2 = self.features[-1](x1) # 1/32
        if self.output_index is None:
            return x0, x1, x2
        elif isinstance(self.output_index, int):
            return [x0,x1,x2][self.output_index]
        elif isinstance(self.output_index, list):
            return [[x0,x1,x2][index] for index in self.output_index]
        else:
            raise NotImplementedError

    def _initialize_weights(self):
        pretrain_model = models.vgg11_bn(True)
        state_dict = pretrain_model.state_dict()
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            if k.startswith('features'):
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict)

class VGGBNPretrainV2(VGGBNPretrain):
    def __init__(self, output_index=None):
        super().__init__(output_index)
        self.output_index=output_index

    def forward(self, x):
        x = self.features[self.splits[0][0]:self.splits[0][1]](x) # 1
        if self.output_index == 0: return x
        x = self.features[self.splits[1][0]:self.splits[1][1]](x) # 1/2
        if self.output_index == 1: return x
        x = self.features[self.splits[2][0]:self.splits[2][1]](x) # 1/4
        if self.output_index == 2: return x
        x = self.features[self.splits[3][0]:self.splits[3][1]](x) # 1/8
        if self.output_index == 3: return x
        x = self.features[self.splits[4][0]:self.splits[4][1]](x) # 1/16
        if self.output_index == 4: return x
        x = self.features[-1](x)
        if self.output_index == 5: return x

class VGGBNPretrainV3(VGGBNPretrain):
    def __init__(self, output_index=None):
        super().__init__(output_index)
        self.output_index=output_index

    def forward(self, x):
        x0 = self.features[self.splits[0][0]:self.splits[0][1]](x) # 1
        x1 = self.features[self.splits[1][0]:self.splits[1][1]](x0) # 1/2
        x2 = self.features[self.splits[2][0]:self.splits[2][1]](x1) # 1/4
        x3 = self.features[self.splits[3][0]:self.splits[3][1]](x2) # 1/8
        x4 = self.features[self.splits[4][0]:self.splits[4][1]](x3) # 1/16
        return x2, x3, x4

class VGGBNPretrainV4(VGGBNPretrain):
    def __init__(self, output_index=None):
        super().__init__(output_index)
        self.output_index=output_index

    def forward(self, x):
        x0 = self.features[self.splits[0][0]:self.splits[0][1]](x) # 1
        x1 = self.features[self.splits[1][0]:self.splits[1][1]](x0) # 1/2
        x2 = self.features[self.splits[2][0]:self.splits[2][1]](x1) # 1/4
        x3 = self.features[self.splits[3][0]:self.splits[3][1]](x2) # 1/8
        return x0, x1, x2, x3

def _make_vgg_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

vgg_cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

vgg_split={
    'A': [(0,3), (3,7), (7,14), (14,21), (21,27)]
}

class ResNet18Pretrain(nn.Module):
    def __init__(self):
        super().__init__()
        replace_stride_with_dilation=None
        block=BasicBlock
        layers=[2,2,2,2]

        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _init_pretrain(self):
        pretrain_model = models.resnet18(pretrained=True)
        state_dict = pretrain_model.state_dict()
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            if k.startswith('conv1') or k.startswith('bn1') or k.startswith('layer'):
                new_state_dict[k]=v
        self.load_state_dict(new_state_dict)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)