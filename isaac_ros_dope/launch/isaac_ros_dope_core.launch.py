# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from typing import Any, Dict

from ament_index_python.packages import get_package_share_directory
from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode


class IsaacROSDopeLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        # Tensor RT parameters
        model_file_path = LaunchConfiguration('model_file_path')
        engine_file_path = LaunchConfiguration('engine_file_path')
        input_tensor_names = LaunchConfiguration('input_tensor_names')
        input_binding_names = LaunchConfiguration('input_binding_names')
        input_tensor_formats = LaunchConfiguration('input_tensor_formats')
        output_tensor_names = LaunchConfiguration('output_tensor_names')
        output_binding_names = LaunchConfiguration('output_binding_names')
        output_tensor_formats = LaunchConfiguration('output_tensor_formats')
        tensorrt_verbose = LaunchConfiguration('tensorrt_verbose')
        force_engine_update = LaunchConfiguration('force_engine_update')

        # DOPE Decoder parameters
        object_name = LaunchConfiguration('object_name')
        enable_tf_publishing = LaunchConfiguration('enable_tf_publishing')
        map_peak_threshold = LaunchConfiguration('map_peak_threshold')

        return {
            'dope_inference_node': ComposableNode(
                name='dope_inference',
                package='isaac_ros_tensor_rt',
                plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
                parameters=[{
                    'model_file_path': model_file_path,
                    'engine_file_path': engine_file_path,
                    'input_tensor_names': input_tensor_names,
                    'input_binding_names': input_binding_names,
                    'input_tensor_formats': input_tensor_formats,
                    'output_tensor_names': output_tensor_names,
                    'output_binding_names': output_binding_names,
                    'output_tensor_formats': output_tensor_formats,
                    'verbose': tensorrt_verbose,
                    'force_engine_update': force_engine_update
                }]
            ),
            'dope_decoder_node': ComposableNode(
                name='dope_decoder',
                package='isaac_ros_dope',
                plugin='nvidia::isaac_ros::dope::DopeDecoderNode',
                parameters=[{
                    'object_name': object_name,
                    'enable_tf_publishing': enable_tf_publishing,
                    'map_peak_threshold': map_peak_threshold,
                }],
                remappings=[('belief_map_array', 'tensor_sub'),
                            ('dope/detections', 'detections'),
                            ('camera_info', '/dope_encoder/crop/camera_info')]
            )
        }

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) -> \
            Dict[str, launch.actions.OpaqueFunction]:

        # DNN Image Encoder parameters
        network_image_width = LaunchConfiguration('network_image_width')
        network_image_height = LaunchConfiguration('network_image_height')
        encoder_image_mean = LaunchConfiguration('encoder_image_mean')
        encoder_image_stddev = LaunchConfiguration('encoder_image_stddev')

        encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')

        return {
            'network_image_width': DeclareLaunchArgument(
                'network_image_width',
                default_value='1280',
                description='The input image width that the network expects'),
            'network_image_height': DeclareLaunchArgument(
                'network_image_height',
                default_value='720',
                description='The input image height that the network expects'),
            'encoder_image_mean': DeclareLaunchArgument(
                'encoder_image_mean',
                default_value='[0.485, 0.456, 0.406]',
                description='The mean for image normalization'),
            'encoder_image_stddev': DeclareLaunchArgument(
                'encoder_image_stddev',
                default_value='[0.229, 0.224, 0.225]',
                description='The standard deviation for image normalization'),
            'model_file_path': DeclareLaunchArgument(
                'model_file_path',
                default_value='',
                description='The absolute file path to the ONNX file'),
            'engine_file_path': DeclareLaunchArgument(
                'engine_file_path',
                default_value='',
                description='The absolute file path to the TensorRT engine file'),
            'input_tensor_names': DeclareLaunchArgument(
                'input_tensor_names',
                default_value='["input_tensor"]',
                description='A list of tensor names to bound to the specified input binding names'
            ),
            'input_binding_names': DeclareLaunchArgument(
                'input_binding_names',
                default_value='["input"]',
                description='A list of input tensor binding names (specified by model)'),
            'input_tensor_formats': DeclareLaunchArgument(
                'input_tensor_formats',
                default_value='["nitros_tensor_list_nchw_rgb_f32"]',
                description='The nitros format of the input tensors'),
            'output_tensor_names': DeclareLaunchArgument(
                'output_tensor_names',
                default_value='["output"]',
                description='A list of tensor names to bound to the specified output binding names'
            ),
            'output_binding_names': DeclareLaunchArgument(
                'output_binding_names',
                default_value='["output"]',
                description='A  list of output tensor binding names (specified by model)'),
            'output_tensor_formats': DeclareLaunchArgument(
                'output_tensor_formats',
                default_value='["nitros_tensor_list_nhwc_rgb_f32"]',
                description='The nitros format of the output tensors'),
            'tensorrt_verbose': DeclareLaunchArgument(
                'tensorrt_verbose',
                default_value='False',
                description='Whether TensorRT should verbosely log or not'),
            'object_name': DeclareLaunchArgument(
                'object_name',
                default_value='Ketchup',
                description='The object class that the DOPE network is detecting'),
            'map_peak_threshold': DeclareLaunchArgument(
                'map_peak_threshold',
                default_value='0.1',
                description='The minimum value of a peak in a DOPE belief map'),
            'force_engine_update': DeclareLaunchArgument(
                'force_engine_update',
                default_value='False',
                description='Whether TensorRT should update the TensorRT engine file or not'),
            'enable_tf_publishing': DeclareLaunchArgument(
                'enable_tf_publishing',
                default_value='False',
                description='Whether Dope Decoder will broadcast poses to the TF tree or not'),
            'dope_encoder_launch': IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
                ),
                launch_arguments={
                    'input_image_width': str(interface_specs['camera_resolution']['width']),
                    'input_image_height': str(interface_specs['camera_resolution']['height']),
                    'network_image_width': network_image_width,
                    'network_image_height': network_image_height,
                    'image_mean': encoder_image_mean,
                    'image_stddev': encoder_image_stddev,
                    'attach_to_shared_component_container': 'True',
                    'component_container_name': '/isaac_ros_examples/container',
                    'dnn_image_encoder_namespace': 'dope_encoder',
                    'image_input_topic': '/image_rect',
                    'camera_info_input_topic': '/camera_info_rect',
                    'tensor_output_topic': '/tensor_pub',
                    'keep_aspect_ratio': 'False'
                }.items(),
            ),
        }
