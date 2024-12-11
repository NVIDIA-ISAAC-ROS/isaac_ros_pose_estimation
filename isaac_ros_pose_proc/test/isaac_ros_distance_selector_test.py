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

import time

from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import pytest
import rclpy


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    composable_nodes = [
        ComposableNode(
            package='isaac_ros_pose_proc',
            plugin='nvidia::isaac_ros::pose_proc::DistanceSelectorNode',
            name='distance_selector_node',
            namespace=IsaacROSDistanceSelectorTest.generate_namespace()
        )
    ]

    distance_selector_container = ComposableNodeContainer(
        name='distance_selector_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=composable_nodes,
        namespace=IsaacROSDistanceSelectorTest.generate_namespace(),
        output='screen'
    )

    return IsaacROSDistanceSelectorTest.generate_test_description([distance_selector_container])


class IsaacROSDistanceSelectorTest(IsaacROSBaseTest):

    def test_distance_selector(self):
        """
        Test distance selector to select a desired pose from a given pose array.

        The selected pose has the smallest euclidean distance to a given input pose.
        """
        received_messages = {}

        self.generate_namespace_lookup(['pose_array_input', 'pose_input', 'pose_output'])

        pose_array_pub = self.node.create_publisher(
            PoseArray, self.namespaces['pose_array_input'], self.DEFAULT_QOS)

        pose_pub = self.node.create_publisher(
            PoseStamped, self.namespaces['pose_input'], self.DEFAULT_QOS)

        subs = self.create_logging_subscribers([('pose_output', PoseStamped)], received_messages)

        try:
            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 5
            end_time = time.time() + TIMEOUT
            done = False

            # Input PoseArray
            POSE_ARRAY_IN = PoseArray(poses=[
                Pose(position=Point(x=0.0, y=0.0, z=0.0)),
                Pose(position=Point(x=1.0, y=1.0, z=1.0)),
                Pose(position=Point(x=-1.0, y=-1.0, z=-1.0))
            ])

            # Input PoseStamped
            POSE_IN = PoseStamped(pose=Pose(
                position=Point(x=2.0, y=2.0, z=2.0)
            ))

            # Expected output PoseStamped
            POSE_OUT = PoseStamped(pose=Pose(
                position=Point(x=1.0, y=1.0, z=1.0)
            ))

            # Wait until received expected pose
            while time.time() < end_time:
                pose_array_pub.publish(POSE_ARRAY_IN)
                pose_pub.publish(POSE_IN)
                rclpy.spin_once(self.node, timeout_sec=(0.1))
                if 'pose_output' in received_messages:
                    done = True
                    break

            self.assertTrue(done, 'Appropriate output not received')
            pose = received_messages['pose_output']
            self.assertEqual(pose, POSE_OUT)

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(pose_pub)
