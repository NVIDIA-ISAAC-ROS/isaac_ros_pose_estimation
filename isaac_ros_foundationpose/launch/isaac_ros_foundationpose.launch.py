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

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

MESH_FILE_NAME = '/tmp/textured_simple.obj'
TEXTURE_MAP_NAME = '/tmp/texture_map.png'
REFINE_MODEL_NAME = '/tmp/refine_model.onnx'
REFINE_ENGINE_NAME = '/tmp/refine_trt_engine.plan'
SCORE_MODEL_NAME = '/tmp/score_model.onnx'
SCORE_ENGINE_NAME = '/tmp/score_trt_engine.plan'


def generate_launch_description():
    """Generate launch description for testing relevant nodes."""
    rviz_config_path = os.path.join(
        get_package_share_directory('isaac_ros_foundationpose'),
        'rviz', 'foundationpose.rviz')

    launch_args = [
        DeclareLaunchArgument(
            'mesh_file_path',
            default_value=MESH_FILE_NAME,
            description='The absolute file path to the mesh file'),

        DeclareLaunchArgument(
            'texture_path',
            default_value=TEXTURE_MAP_NAME,
            description='The absolute file path to the texture map'),

        DeclareLaunchArgument(
            'refine_model_file_path',
            default_value=REFINE_MODEL_NAME,
            description='The absolute file path to the refine model'),

        DeclareLaunchArgument(
            'refine_engine_file_path',
            default_value=REFINE_ENGINE_NAME,
            description='The absolute file path to the refine trt engine'),

        DeclareLaunchArgument(
            'score_model_file_path',
            default_value=SCORE_MODEL_NAME,
            description='The absolute file path to the score model'),

        DeclareLaunchArgument(
            'score_engine_file_path',
            default_value=SCORE_ENGINE_NAME,
            description='The absolute file path to the score trt engine'),

        DeclareLaunchArgument(
            'mask_height',
            default_value='480',
            description='The height of the mask generated from the bounding box'),

        DeclareLaunchArgument(
            'mask_width',
            default_value='640',
            description='The width of the mask generated from the bounding box'),

        DeclareLaunchArgument(
            'launch_bbox_to_mask',
            default_value='False',
            description='Flag to enable bounding box to mask converter'),

        DeclareLaunchArgument(
            'launch_rviz',
            default_value='False',
            description='Flag to enable Rviz2 launch'),

    ]

    mesh_file_path = LaunchConfiguration('mesh_file_path')
    texture_path = LaunchConfiguration('texture_path')
    refine_model_file_path = LaunchConfiguration('refine_model_file_path')
    refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
    score_model_file_path = LaunchConfiguration('score_model_file_path')
    score_engine_file_path = LaunchConfiguration('score_engine_file_path')
    mask_height = LaunchConfiguration('mask_height')
    mask_width = LaunchConfiguration('mask_width')
    launch_rviz = LaunchConfiguration('launch_rviz')
    launch_bbox_to_mask = LaunchConfiguration('launch_bbox_to_mask')

    foundationpose_node = ComposableNode(
        name='foundationpose',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[{
            'mesh_file_path': mesh_file_path,
            'texture_path': texture_path,

            'refine_model_file_path': refine_model_file_path,
            'refine_engine_file_path': refine_engine_file_path,
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names': ['input1', 'input2'],
            'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],

            'score_model_file_path': score_model_file_path,
            'score_engine_file_path': score_engine_file_path,
            'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'score_input_binding_names': ['input1', 'input2'],
            'score_output_tensor_names': ['output_tensor'],
            'score_output_binding_names': ['output1'],
        }],
        remappings=[
            ('pose_estimation/depth_image', 'depth_registered/image_rect'),
            ('pose_estimation/image', 'rgb/image_rect_color'),
            ('pose_estimation/camera_info', 'rgb/camera_info'),
            ('pose_estimation/segmentation', 'segmentation'),
            ('pose_estimation/output', 'output')])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        condition=IfCondition(launch_rviz))

    detection2_d_array_filter_node = ComposableNode(
        name='detection2_d_array_filter',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Detection2DArrayFilter',
        remappings=[('detection2_d_array', 'detections_output')]
    )
    detection2_d_to_mask_node = ComposableNode(
        name='detection2_d_to_mask',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
        parameters=[{
            'mask_width': mask_width,
            'mask_height': mask_height
        }],
        remappings=[('segmentation', 'rt_detr_segmentation')]
    )

    foundationpose_bbox_container = ComposableNodeContainer(
        name='foundationpose_container',
        namespace='foundationpose_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            foundationpose_node,
            detection2_d_array_filter_node,
            detection2_d_to_mask_node],
        output='screen',
        condition=IfCondition(launch_bbox_to_mask)
    )

    foundationpose_container = ComposableNodeContainer(
        name='foundationpose_container',
        namespace='foundationpose_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[foundationpose_node],
        output='screen',
        condition=UnlessCondition(launch_bbox_to_mask)
    )

    return launch.LaunchDescription(launch_args + [foundationpose_container,
                                                   foundationpose_bbox_container,
                                                   rviz_node])
