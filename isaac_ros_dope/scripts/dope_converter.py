#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time

import onnx
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.onnx
import torchvision.models as models

"""
This code is taken from the official DOPE Github repository:
# https://github.com/NVlabs/Deep_Object_Pose/blob/master/src/dope/inference/detector.py
# For the DopeNetwork model

This script converts pre-trained DOPE model to given format for TensorRT or Triton
infernce. It works with any pre-trained model provided on the official DOPE Github repository,
or trained using the training script
https://github.com/NVlabs/Deep_Object_Pose/blob/master/scripts/train.py in the repository.
"""


class DopeNetwork(nn.Module):
    """DopeNetwork class: definition of the dope network model."""

    def __init__(self, numBeliefMap=9, numAffinity=16):
        super(DopeNetwork, self).__init__()

        vgg_full = models.vgg19(pretrained=False).features
        self.vgg = nn.Sequential()
        for i_layer in range(24):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        # Add some layers
        i_layer = 23
        self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer + 1), nn.ReLU(inplace=True))
        self.vgg.add_module(
            str(i_layer + 2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer + 3), nn.ReLU(inplace=True))

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        self.m1_2 = DopeNetwork.create_stage(128, numBeliefMap, True)
        self.m2_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity, numBeliefMap, False)
        self.m3_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity, numBeliefMap, False)
        self.m4_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity, numBeliefMap, False)
        self.m5_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity, numBeliefMap, False)
        self.m6_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity, numBeliefMap, False)

        # print('---Affinity----------------------------------------------')
        # _1 are the affinity map stages
        self.m1_1 = DopeNetwork.create_stage(128, numAffinity, True)
        self.m2_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity, numAffinity, False)
        self.m3_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity, numAffinity, False)
        self.m4_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity, numAffinity, False)
        self.m5_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity, numAffinity, False)
        self.m6_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity, numAffinity, False)

    def forward(self, x):
        """Run inference on the neural network."""
        out1 = self.vgg(x)

        out1_2 = self.m1_2(out1)
        out1_1 = self.m1_1(out1)

        out2 = torch.cat([out1_2, out1_1, out1], 1)
        out2_2 = self.m2_2(out2)
        out2_1 = self.m2_1(out2)

        out3 = torch.cat([out2_2, out2_1, out1], 1)
        out3_2 = self.m3_2(out3)
        out3_1 = self.m3_1(out3)

        out4 = torch.cat([out3_2, out3_1, out1], 1)
        out4_2 = self.m4_2(out4)
        out4_1 = self.m4_1(out4)

        out5 = torch.cat([out4_2, out4_1, out1], 1)
        out5_2 = self.m5_2(out5)
        out5_1 = self.m5_1(out5)

        out6 = torch.cat([out5_2, out5_1, out1], 1)
        out6_2 = self.m6_2(out6)
        out6_1 = self.m6_1(out6)

        return torch.cat([out6_2, out6_1], 1)

    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        """Create the neural network layers for a single stage."""
        model = nn.Sequential()
        mid_channels = 128
        if first:
            padding = 1
            kernel = 3
            count = 6
            final_channels = 512
        else:
            padding = 3
            kernel = 7
            count = 10
            final_channels = mid_channels

        # First convolution
        model.add_module('0',
                         nn.Conv2d(
                             in_channels,
                             mid_channels,
                             kernel_size=kernel,
                             stride=1,
                             padding=padding))

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            model.add_module(
                str(i),
                nn.Conv2d(
                    mid_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding))
            i += 1

        # Penultimate convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        i += 1

        # Last convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
        i += 1

        return model


def get_node_output_shape(node):
    return [x.dim_value for x in node.type.tensor_type.shape.dim]


def save_pytorch(net, x, output_file):
    jit_net = torch.jit.script(net.module, (x, ))
    torch.jit.save(jit_net, output_file)


def save_onnx(net, x, output_file, input_name, output_name):
    torch.onnx.export(
        net.module, x, output_file,
        input_names=[input_name],
        output_names=[output_name],
        dynamo=True  # Use modern exporter for static shapes
    )

    # Validate and log onnx model information
    model = onnx.load(output_file)
    net_output = [(node.name, get_node_output_shape(node)) for node in model.graph.output]

    input_initializer = [node.name for node in model.graph.initializer]
    net_feed_input = [(node.name, get_node_output_shape(node)) for node in model.graph.input
                      if node.name not in input_initializer]

    print('\n=== onnx model info ===')
    print('Inputs:')
    print('\n'.join(map(str, net_feed_input)))
    print('Outputs:')
    print('\n'.join(map(str, net_output)))

    onnx.checker.check_model(model)


def main(args):
    """
    Convert pre-trained Pytorch DOPE model to given format for TensorRT and/or Triton to use.

    This is tested for the pre-train weights linked from the DOPE github page
    https://drive.google.com/drive/folders/1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg
    """
    model_loading_start_time = time.time()
    print('Loading torch model {}'.format(args.input))
    net = torch.nn.DataParallel(DopeNetwork(), [0]).cuda()
    net.load_state_dict(torch.load(args.input), strict=True)
    print('Model loaded in {0:.2f} seconds'.format(time.time() - model_loading_start_time))

    x = Variable(torch.randn(1, 3, args.row, args.col)).cuda()
    # Export the model
    output_file = args.output
    file_format = 'pt' if args.format == 'pytorch' else args.format
    if not output_file:
        path, filename = os.path.split(args.input)
        output_file = '{0}/{1}.{2}'.format(path, os.path.splitext(filename)[0], file_format)
    if args.format == 'pytorch':
        save_pytorch(net, x, output_file)
    else:
        save_onnx(net, x, output_file, args.input_name, args.output_name)
    print('Saved output model to {}'.format(output_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Dope model format')
    parser.add_argument(
        '--format', required=True, help='Target model format', choices=['onnx', 'pytorch'])
    parser.add_argument(
        '--input', required=True, help='Absolute path to PyTorch model')
    parser.add_argument(
        '--output', default='', help='Absolute path to output model')
    parser.add_argument(
        '--input_name', default='input', help='Input tensor name (ONNX model)')
    parser.add_argument(
        '--output_name', default='output', help='Output tensor name (ONNX model)')
    parser.add_argument('--row', default=480, type=int, help='Input image rows')
    parser.add_argument('--col', default=640, type=int, help='Input image columns')
    args = parser.parse_args()
    main(args)
