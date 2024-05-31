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

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for Centerpose."""
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

    launch_args = [
        DeclareLaunchArgument(
            'input_image_width',
            default_value='600',
            description='The input image width'),
        DeclareLaunchArgument(
            'input_image_height',
            default_value='800',
            description='The input image height'),
        DeclareLaunchArgument(
            'network_image_width',
            default_value='512',
            description='The input image width that the network expects'),
        DeclareLaunchArgument(
            'network_image_height',
            default_value='512',
            description='The input image height that the network expects'),
        DeclareLaunchArgument(
            'encoder_image_mean',
            default_value='[0.408, 0.447, 0.47]',
            description='The mean for image normalization'),
        DeclareLaunchArgument(
            'encoder_image_stddev',
            default_value='[0.289, 0.274, 0.278]',
            description='The standard deviation for image normalization'),
        DeclareLaunchArgument(
            'model_file_path',
            default_value='',
            description='The absolute file path to the ONNX file'),
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='',
            description='The absolute file path to the TensorRT engine file'),
        DeclareLaunchArgument(
            'input_tensor_names',
            default_value='["input_tensor"]',
            description='A list of tensor names to bound to the specified input binding names'),
        DeclareLaunchArgument(
            'input_binding_names',
            default_value='["input"]',
            description='A list of input tensor binding names (specified by model)'),
        DeclareLaunchArgument(
            'input_tensor_formats',
            default_value='["nitros_tensor_list_nchw_rgb_f32"]',
            description='The nitros format of the input tensors'),
        DeclareLaunchArgument(
            'output_tensor_names',
            default_value=tensor_names_str,
            description='A list of tensor names to bound to the specified output binding names'),
        DeclareLaunchArgument(
            'output_binding_names',
            default_value=tensor_names_str,
            description='A  list of output tensor binding names (specified by model)'),
        DeclareLaunchArgument(
            'output_tensor_formats',
            default_value='["nitros_tensor_list_nhwc_rgb_f32"]',
            description='The nitros format of the output tensors'),
        DeclareLaunchArgument(
            'tensorrt_verbose',
            default_value='False',
            description='Whether TensorRT should verbosely log or not'),
        DeclareLaunchArgument(
            'force_engine_update',
            default_value='False',
            description='Whether TensorRT should update the TensorRT engine file or not'),
        DeclareLaunchArgument(
            'output_field_size',
            default_value='[128, 128]',
            description='Represents the size of the 2D keypoint decoding from the network output'),
        DeclareLaunchArgument(
            'cuboid_scaling_factor',
            default_value='1.0',
            description='Scales the cuboid used for calculating the size of the objects detected'),
        DeclareLaunchArgument(
            'score_threshold',
            default_value='0.3',
            description='The threshold for scores values to discard.'),
        DeclareLaunchArgument(
            'object_name',
            default_value='shoe',
            description='The name of the category instance that is being detected',
        ),
        DeclareLaunchArgument(
            'show_axes',
            default_value='True',
            description='Whether to show axes or not for visualization',
        ),
        DeclareLaunchArgument(
            'bounding_box_color',
            default_value='0x000000ff',
            description='The color of the bounding box for visualization',
        ),
    ]

    # DNN Image Encoder parameters
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    network_image_width = LaunchConfiguration('network_image_width')
    network_image_height = LaunchConfiguration('network_image_height')
    encoder_image_mean = LaunchConfiguration('encoder_image_mean')
    encoder_image_stddev = LaunchConfiguration('encoder_image_stddev')

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

    # Centerpose Decoder parameters
    output_field_size = LaunchConfiguration('output_field_size')
    cuboid_scaling_factor = LaunchConfiguration('cuboid_scaling_factor')
    score_threshold = LaunchConfiguration('score_threshold')
    object_name = LaunchConfiguration('object_name')

    # Centerpose visualization parameters
    show_axes = LaunchConfiguration('show_axes')
    bounding_box_color = LaunchConfiguration('bounding_box_color')

    centerpose_inference_node = ComposableNode(
        name='centerpose_inference',
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
        }])

    centerpose_decoder_node = ComposableNode(
        name='centerpose_decoder_node',
        package='isaac_ros_centerpose',
        plugin='nvidia::isaac_ros::centerpose::CenterPoseDecoderNode',
        parameters=[{
            'output_field_size': output_field_size,
            'cuboid_scaling_factor': cuboid_scaling_factor,
            'score_threshold': score_threshold,
            'object_name': object_name,
        }],
    )

    centerpose_visualizer_node = ComposableNode(
        name='centerpose_visualizer_node',
        package='isaac_ros_centerpose',
        plugin='nvidia::isaac_ros::centerpose::CenterPoseVisualizerNode',
        parameters=[{
            'show_axes': show_axes,
            'bounding_box_color': bounding_box_color,
        }],
    )

    rclcpp_container = ComposableNodeContainer(
        name='centerpose_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            centerpose_inference_node, centerpose_decoder_node, centerpose_visualizer_node],
        output='screen',
    )

    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    centerpose_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': input_image_width,
            'input_image_height': input_image_height,
            'network_image_width': network_image_width,
            'network_image_height': network_image_height,
            'image_mean': encoder_image_mean,
            'image_stddev': encoder_image_stddev,
            'attach_to_shared_component_container': 'True',
            'component_container_name': 'centerpose_container',
            'dnn_image_encoder_namespace': 'centerpose_encoder',
            'image_input_topic': '/image',
            'camera_info_input_topic': '/camera_info',
            'tensor_output_topic': '/tensor_pub',
        }.items(),
    )

    final_launch_container = launch_args + [rclcpp_container, centerpose_encoder_launch]
    return LaunchDescription(final_launch_container)
