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


class IsaacROSCenterPoseLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        # Triton parameters
        model_name = LaunchConfiguration('model_name')
        model_repository_paths = LaunchConfiguration('model_repository_paths')
        max_batch_size = LaunchConfiguration('max_batch_size')
        input_tensor_names = LaunchConfiguration('input_tensor_names')
        input_binding_names = LaunchConfiguration('input_binding_names')
        input_tensor_formats = LaunchConfiguration('input_tensor_formats')
        output_tensor_names = LaunchConfiguration('output_tensor_names')
        output_binding_names = LaunchConfiguration('output_binding_names')
        output_tensor_formats = LaunchConfiguration('output_tensor_formats')

        # Centerpose Decoder parameters
        output_field_size = LaunchConfiguration('output_field_size')
        cuboid_scaling_factor = LaunchConfiguration('cuboid_scaling_factor')
        score_threshold = LaunchConfiguration('score_threshold')
        object_name = LaunchConfiguration('object_name')

        return {
            'centerpose_inference_node': ComposableNode(
                name='centerpose_inference',
                package='isaac_ros_triton',
                plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
                parameters=[{
                    'model_name': model_name,
                    'model_repository_paths': model_repository_paths,
                    'max_batch_size': max_batch_size,
                    'input_tensor_names': input_tensor_names,
                    'input_binding_names': input_binding_names,
                    'input_tensor_formats': input_tensor_formats,
                    'output_tensor_names': output_tensor_names,
                    'output_binding_names': output_binding_names,
                    'output_tensor_formats': output_tensor_formats,
                }]
            ),
            'centerpose_decoder_node': ComposableNode(
                name='centerpose_decoder_node',
                package='isaac_ros_centerpose',
                plugin='nvidia::isaac_ros::centerpose::CenterPoseDecoderNode',
                parameters=[{
                    'output_field_size': output_field_size,
                    'cuboid_scaling_factor': cuboid_scaling_factor,
                    'score_threshold': score_threshold,
                    'object_name': object_name,
                }],
                remappings=[
                    ('camera_info', 'camera_info_rect'),
                ],
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

        tensor_names = [
            'bboxes',
            'scores',
            'kps',
            'clses',
            'obj_scale',
            'kps_displacement_mean',
            'kps_heatmap_mean',
        ]
        tensor_names_str = '['
        for tensor_name in tensor_names:
            tensor_names_str += f'"{tensor_name}"'
            if tensor_name != tensor_names[-1]:
                tensor_names_str += ','
        tensor_names_str += ']'

        return {
            'network_image_width': DeclareLaunchArgument(
                'network_image_width',
                default_value='512',
                description='The input image width that the network expects'),
            'network_image_height': DeclareLaunchArgument(
                'network_image_height',
                default_value='512',
                description='The input image height that the network expects'),
            'encoder_image_mean': DeclareLaunchArgument(
                'encoder_image_mean',
                default_value='[0.408, 0.447, 0.47]',
                description='The mean for image normalization'),
            'encoder_image_stddev': DeclareLaunchArgument(
                'encoder_image_stddev',
                default_value='[0.289, 0.274, 0.278]',
                description='The standard deviation for image normalization'),
            'model_name': DeclareLaunchArgument(
                'model_name',
                default_value='',
                description='The name of the model'),
            'model_repository_paths': DeclareLaunchArgument(
                'model_repository_paths',
                default_value='',
                description='The absolute path to the repository of models'),
            'max_batch_size': DeclareLaunchArgument(
                'max_batch_size',
                default_value='0',
                description='The maximum allowed batch size of the model'),
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
                default_value=tensor_names_str,
                description='A list of tensor names to bound to the specified output binding names'
            ),
            'output_binding_names': DeclareLaunchArgument(
                'output_binding_names',
                default_value=tensor_names_str,
                description='A  list of output tensor binding names (specified by model)'),
            'output_tensor_formats': DeclareLaunchArgument(
                'output_tensor_formats',
                default_value='["nitros_tensor_list_nhwc_rgb_f32"]',
                description='The nitros format of the output tensors'),
            'output_field_size': DeclareLaunchArgument(
                'output_field_size',
                default_value='[128, 128]',
                description='The size of the 2D keypoint decoding from the network output'),
            'cuboid_scaling_factor': DeclareLaunchArgument(
                'cuboid_scaling_factor',
                default_value='1.0',
                description='Scales the cuboid used for calculating detected objects size'),
            'score_threshold': DeclareLaunchArgument(
                'score_threshold',
                default_value='0.3',
                description='The threshold for scores values to discard.'),
            'object_name': DeclareLaunchArgument(
                'object_name',
                default_value='Ketchup',
                description='The object class that the DOPE network is detecting'),
            'centerpose_encoder_launch': IncludeLaunchDescription(
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
                    'dnn_image_encoder_namespace': 'centerpose_encoder',
                    'image_input_topic': '/image_rect',
                    'camera_info_input_topic': '/camera_info_rect',
                    'tensor_output_topic': '/tensor_pub',
                    'keep_aspect_ratio': 'False'
                }.items(),
            ),
        }
