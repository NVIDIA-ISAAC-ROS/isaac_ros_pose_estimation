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

import math
import time

from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import pytest
import rclpy

NUM_SAMPLES = 2


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    composable_nodes = [
        ComposableNode(
            package='isaac_ros_pose_proc',
            plugin='nvidia::isaac_ros::pose_proc::AveragingFilterNode',
            name='averaging_filter_node',
            namespace=IsaacROSAveragingFilterTest.generate_namespace(),
            parameters=[{
                'num_samples': NUM_SAMPLES
            }]
        )
    ]

    averaging_container = ComposableNodeContainer(
        name='averaging_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=composable_nodes,
        namespace=IsaacROSAveragingFilterTest.generate_namespace(),
        output='screen'
    )

    return IsaacROSAveragingFilterTest.generate_test_description([averaging_container])


class IsaacROSAveragingFilterTest(IsaacROSBaseTest):

    def test_averaging_filter(self):
        """
        Test averaging filter to stabilize noisy poses.

        Given two poses, the filter should output a single averaged pose.
        The quaternion is averaged by dividing the sum of each vector
        component by the norm of the summed quaternion.
        """
        received_messages = {}

        self.generate_namespace_lookup(['pose_input', 'pose_output'])

        pose_pub = self.node.create_publisher(
            PoseStamped, self.namespaces['pose_input'], self.DEFAULT_QOS)

        subs = self.create_logging_subscribers(
            [('pose_output', PoseStamped)], received_messages, accept_multiple_messages=True)

        try:
            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 5
            end_time = time.time() + TIMEOUT
            done = False

            # Input poses
            POSES_IN = [
                PoseStamped(pose=Pose(
                    position=Point(x=1.0, y=2.0, z=3.0),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                )),
                PoseStamped(pose=Pose(
                    position=Point(x=2.0, y=4.0, z=6.0),
                    orientation=Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)
                ))
            ]

            # Expected output poses
            POSES_OUT = [
                PoseStamped(pose=Pose(
                    position=Point(x=1.5, y=3.0, z=4.5),
                    orientation=Quaternion(
                        x=math.sqrt(2)/2, y=0.0, z=0.0, w=math.sqrt(2)/2))
                )
            ]

            # Wait until received expected poses
            while time.time() < end_time:
                for pose in POSES_IN:
                    pose_pub.publish(pose)
                rclpy.spin_once(self.node, timeout_sec=(0.1))
                if 'pose_output' in received_messages and \
                        len(received_messages['pose_output']) >= len(POSES_OUT):
                    done = True
                    break

            self.assertTrue(done, 'Appropriate output not received')
            received_poses = received_messages['pose_output']
            self.assertEqual(len(received_poses), len(POSES_OUT))
            for received, expected in zip(received_poses, POSES_OUT):
                self.assertAlmostEqual(received.pose.position.x,
                                       expected.pose.position.x, places=7)
                self.assertAlmostEqual(received.pose.position.y,
                                       expected.pose.position.y, places=7)
                self.assertAlmostEqual(received.pose.position.z,
                                       expected.pose.position.z, places=7)
                self.assertAlmostEqual(received.pose.orientation.x,
                                       expected.pose.orientation.x, places=7)
                self.assertAlmostEqual(received.pose.orientation.y,
                                       expected.pose.orientation.y, places=7)
                self.assertAlmostEqual(received.pose.orientation.z,
                                       expected.pose.orientation.z, places=7)
                self.assertAlmostEqual(received.pose.orientation.w,
                                       expected.pose.orientation.w, places=7)

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(pose_pub)
