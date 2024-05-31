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

from typing import Any, Dict

from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode


class IsaacROSCenterPoseVisualizerLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        # Centerpose visualization parameters
        show_axes = LaunchConfiguration('show_axes')
        bounding_box_color = LaunchConfiguration('bounding_box_color')

        return {
            'centerpose_visualizer_node': ComposableNode(
                name='centerpose_visualizer',
                package='isaac_ros_centerpose',
                plugin='nvidia::isaac_ros::centerpose::CenterPoseVisualizerNode',
                parameters=[{
                    'show_axes': show_axes,
                    'bounding_box_color': bounding_box_color,
                }],
                remappings=[
                    ('image', 'image_rect'),
                    ('camera_info', 'camera_info_rect')
                ],
            ),
        }

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) -> \
            Dict[str, launch.actions.OpaqueFunction]:

        return {
            'show_axes': DeclareLaunchArgument(
                'show_axes',
                default_value='True',
                description='Whether to show axes or not for visualization'),
            'bounding_box_color': DeclareLaunchArgument(
                'bounding_box_color',
                default_value='0x000000ff',
                description='The color of the bounding box for visualization'),
        }
