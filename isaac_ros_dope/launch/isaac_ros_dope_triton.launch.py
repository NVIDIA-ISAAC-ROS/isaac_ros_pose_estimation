# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for DOPE encoder->Triton->DOPE decoder."""
    launch_args = [
        DeclareLaunchArgument(
            'input_image_width',
            default_value='640',
            description='The input image width'),
        DeclareLaunchArgument(
            'input_image_height',
            default_value='480',
            description='The input image height'),
        DeclareLaunchArgument(
            'network_image_width',
            default_value='640',
            description='The input image width that the network expects'),
        DeclareLaunchArgument(
            'network_image_height',
            default_value='480',
            description='The input image height that the network expects'),
        DeclareLaunchArgument(
            'encoder_image_mean',
            default_value='[0.5, 0.5, 0.5]',
            description='The mean for image normalization'),
        DeclareLaunchArgument(
            'encoder_image_stddev',
            default_value='[0.5, 0.5, 0.5]',
            description='The standard deviation for image normalization'),
        DeclareLaunchArgument(
            'model_name',
            default_value='',
            description='The name of the model'),
        DeclareLaunchArgument(
            'model_repository_paths',
            default_value='',
            description='The absolute path to the repository of models'),
        DeclareLaunchArgument(
            'max_batch_size',
            default_value='0',
            description='The maximum allowed batch size of the model'),
        DeclareLaunchArgument(
            'input_tensor_names',
            default_value='["input_tensor"]',
            description='A list of tensor names to bound to the specified input binding names'),
        DeclareLaunchArgument(
            'input_binding_names',
            default_value='["input_1"]',
            description='A list of input tensor binding names (specified by model)'),
        DeclareLaunchArgument(
            'input_tensor_formats',
            default_value='["nitros_tensor_list_nchw_rgb_f32"]',
            description='The nitros format of the input tensors'),
        DeclareLaunchArgument(
            'output_tensor_names',
            default_value='["output_tensor"]',
            description='A list of tensor names to bound to the specified output binding names'),
        DeclareLaunchArgument(
            'output_binding_names',
            default_value='["softmax_1"]',
            description='A  list of output tensor binding names (specified by model)'),
        DeclareLaunchArgument(
            'output_tensor_formats',
            default_value='["nitros_tensor_list_nhwc_rgb_f32"]',
            description='The nitros format of the output tensors'),
        DeclareLaunchArgument(
            'object_name',
            default_value='Ketchup',
            description='The object class that the DOPE network is detecting'),
    ]

    # DNN Image Encoder parameters
    network_image_width = LaunchConfiguration('network_image_width')
    network_image_height = LaunchConfiguration('network_image_height')
    encoder_image_mean = LaunchConfiguration('encoder_image_mean')
    encoder_image_stddev = LaunchConfiguration('encoder_image_stddev')

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

    # DOPE Decoder parameters
    object_name = LaunchConfiguration('object_name')

    image_resize_node = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        name='image_resize',
        parameters=[{
                'output_width': network_image_width,
                'output_height': network_image_width
        }],
    )

    dope_encoder_node = ComposableNode(
        name='dope_encoder',
        package='isaac_ros_dnn_image_encoder',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        parameters=[{
            'input_image_width': network_image_width,
            'input_image_height': network_image_height,
            'network_image_width': network_image_width,
            'network_image_height': network_image_height,
            'image_mean': encoder_image_mean,
            'image_stddev': encoder_image_stddev,
        }],
        remappings=[
            ('image', 'resize/image'),
            ('encoded_tensor', 'tensor_pub')
        ])

    dope_inference_node = ComposableNode(
        name='dope_inference',
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
        }])

    dope_decoder_node = ComposableNode(
        name='dope_decoder',
        package='isaac_ros_dope',
        plugin='nvidia::isaac_ros::dope::DopeDecoderNode',
        parameters=[{
            'object_name': object_name,
        }],
        remappings=[('belief_map_array', 'tensor_sub'),
                    ('dope/pose_array', 'poses')])

    container = ComposableNodeContainer(
        name='dope_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            image_resize_node, dope_encoder_node, dope_inference_node, dope_decoder_node
        ],
        output='screen',
    )

    final_launch_description = launch_args + [container]
    return launch.LaunchDescription(final_launch_description)
