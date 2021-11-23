# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import time

import onnx
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.onnx

"""
This script converts pre-trained CenterPose model from Pytorch to ONNX to use with TensorRT or
Triton infernce.
"""

BN_MOMENTUM = 0.1


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CenterPoseNetwork(nn.Module):
    """CenterPoseNetwork class: definition of the dope network model."""

    def __init__(self, block, layers, heads, head_conv, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads

        super(CenterPoseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
        )

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, num_output,
                              kernel_size=1, stride=1, padding=0))
            else:
                fc = nn.Conv2d(
                    in_channels=256,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


def get_node_output_shape(node):
    return [x.dim_value for x in node.type.tensor_type.shape.dim]


def main(args):
    """Convert pre-trained PyTorch Centerpose model to ONNX for TensorRT to use."""
    model_loading_start_time = time.time()
    print('Loading torch model {}'.format(args.input))
    # TODO: Turn the below into parameters:
    block_class = Bottleneck
    layers = [3, 4, 23, 3]
    heads = {'hm': 1, 'wh': 2, 'hps': 16, 'reg': 2,
             'hm_hp': 8, 'hp_offset': 2, 'scale': 3}
    head_conv = 64
    net = CenterPoseNetwork(block_class, layers, heads, head_conv)
    net.load_state_dict(torch.load(args.input)['state_dict'], strict=True)
    print('Model loaded in {0:.2f} seconds'.format(
        time.time() - model_loading_start_time))

    x = Variable(torch.randn(1, 3, args.row, args.col))
    # Export the model
    torch.onnx.export(net, x, args.output, input_names=['input'],
                      output_names=['hm', 'wh', 'hps', 'reg', 'hm_hp', 'hp_offset', 'scale'])
    print('Saved ONNX model to {}'.format(args.output))

    # Validate and log onnx model information
    model = onnx.load(args.output)
    net_output = [(node.name, get_node_output_shape(node))
                  for node in model.graph.output]

    input_initializer = [node.name for node in model.graph.initializer]
    net_feed_input = [(node.name, get_node_output_shape(node)) for node in model.graph.input
                      if node.name not in input_initializer]

    print('\n=== onnx model info ===')
    print('Inputs:')
    print('\n'.join(map(str, net_feed_input)))
    print('Outputs:')
    print('\n'.join(map(str, net_output)))

    onnx.checker.check_model(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert CenterPose from PyTorch to ONNX')
    parser.add_argument(
        '--input', required=True, help='Absolute path to PyTorch model')
    parser.add_argument(
        '--output', required=True, help='Absolute path to output ONNX model')
    parser.add_argument('--row', default=512, type=int,
                        help='Input image rows')
    parser.add_argument('--col', default=512, type=int,
                        help='Input image columns')
    args = parser.parse_args()
    main(args)
