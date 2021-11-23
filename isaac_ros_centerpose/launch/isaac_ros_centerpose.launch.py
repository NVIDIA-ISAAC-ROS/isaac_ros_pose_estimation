# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for Centerpose."""
    model_name = 'centerpose_shoe'
    launch_dir_path = os.path.dirname(os.path.realpath(__file__))
    model_dir_path = launch_dir_path + '/../test/models'
    config = os.path.join(
        get_package_share_directory('isaac_ros_centerpose'),
        'config',
        'decoder_params.yaml'
    )

    centerpose_encoder_node = ComposableNode(
        name='centerpose_encoder',
        package='isaac_ros_dnn_encoders',
        plugin='isaac_ros::dnn_inference::DnnImageEncoderNode',
        parameters=[{
            'network_image_width': 512,
                'network_image_height': 512,
                'network_image_encoding': 'rgb8',
                'network_normalization_type': 'image_normalization',
                'image_mean': [0.408, 0.447, 0.47],
                'image_stddev': [0.289, 0.274, 0.278]
        }],
        remappings=[('encoded_tensor', 'tensor_pub')])

    centerpose_inference_node = ComposableNode(
        name='centerpose_inference',
        package='isaac_ros_triton',
        plugin='isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': model_name,
            'model_repository_paths': [model_dir_path],
            'max_batch_size': 0,
            'input_tensor_names': ['input'],
            'input_binding_names': ['input'],
            'output_tensor_names': ['hm', 'wh', 'hps', 'reg', 'hm_hp', 'hp_offset', 'scale'],
            'output_binding_names': ['hm', 'wh', 'hps', 'reg', 'hm_hp', 'hp_offset', 'scale']
        }])

    rclcpp_container = ComposableNodeContainer(
        name='centerpose_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            centerpose_encoder_node, centerpose_inference_node],
        output='screen',
    )

    return LaunchDescription([
        rclcpp_container,
        Node(name='centerpose_decoder_node', package='isaac_ros_centerpose',
             executable='CenterPoseDecoder', parameters=[config], output='screen')])
