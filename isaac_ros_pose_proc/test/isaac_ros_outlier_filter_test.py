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

from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import pytest
import rclpy


NUM_SAMPLES = 5
DISTANCE_THRESHOLD = 0.5
ANGLE_THRESHOLD = 30.0
TIMEOUT = 5.0


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    composable_nodes = [
        ComposableNode(
            package='isaac_ros_pose_proc',
            plugin='nvidia::isaac_ros::pose_proc::OutlierFilterNode',
            name='outlier_filter_node',
            namespace=IsaacROSOutlierFilterTest.generate_namespace(),
            parameters=[{
                'num_samples': NUM_SAMPLES,
                'distance_threshold': DISTANCE_THRESHOLD,
                'angle_threshold': ANGLE_THRESHOLD,
                'pose_stale_time_threshold': TIMEOUT
            }]
        )
    ]

    outlier_filter_container = ComposableNodeContainer(
        name='outlier_filter_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=composable_nodes,
        namespace=IsaacROSOutlierFilterTest.generate_namespace(),
        output='screen'
    )

    return IsaacROSOutlierFilterTest.generate_test_description([outlier_filter_container])


class IsaacROSOutlierFilterTest(IsaacROSBaseTest):

    def test_outlier_filter(self):
        """
        Test filtering poses with large position or orientation deviations.

        The first pose received is always published. All subsequent poses with euclidean
        distance to the first pose less than DISTANCE_THRESHOLD and shortest path angle difference
        less than ANGLE_THRESHOLD are published.
        """
        received_messages = {}

        self.generate_namespace_lookup(['pose_input', 'pose_output'])

        pose_pub = self.node.create_publisher(
            PoseStamped, self.namespaces['pose_input'], self.DEFAULT_QOS)

        subs = self.create_logging_subscribers(
            [('pose_output', PoseStamped)], received_messages, accept_multiple_messages=True)

        try:
            # Wait at most TIMEOUT seconds for subscriber to respond
            end_time = time.time() + TIMEOUT
            done = False

            # Input poses
            POSES_IN = [
                PoseStamped(pose=Pose(
                    position=Point(x=1.2, y=2.1, z=3.3),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                )),
                PoseStamped(pose=Pose(
                    position=Point(x=1.0, y=2.0, z=3.0),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                )),
                PoseStamped(pose=Pose(
                    position=Point(x=1.3, y=2.4, z=3.1),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                )),
                PoseStamped(pose=Pose(
                    position=Point(x=1.0, y=2.0, z=3.0),
                    orientation=Quaternion(x=0.0, y=0.284, z=0.0, w=0.959)
                )),
                PoseStamped(pose=Pose(
                    position=Point(x=1.1, y=2.2, z=3.1),
                    orientation=Quaternion(x=0.131, y=0.0, z=0.0, w=0.991)
                ))
            ]

            # Expected output poses
            POSES_OUT = [
                PoseStamped(pose=Pose(
                    position=Point(x=1.2, y=2.1, z=3.3),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                )),
                PoseStamped(pose=Pose(
                    position=Point(x=1.0, y=2.0, z=3.0),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                )),
                PoseStamped(pose=Pose(
                    position=Point(x=1.1, y=2.2, z=3.1),
                    orientation=Quaternion(x=0.131, y=0.0, z=0.0, w=0.991)
                )),
            ]

            # Wait until received expected poses
            while time.time() < end_time:
                # Publish poses
                for pose in POSES_IN:
                    pose_pub.publish(pose)
                rclpy.spin_once(self.node, timeout_sec=(0.1))
                if 'pose_output' in received_messages and \
                        len(received_messages['pose_output']) >= len(POSES_OUT):
                    done = True
                    break

            self.assertTrue(done, 'Appropriate output not received')
            poses = received_messages['pose_output']
            self.assertEqual(len(poses), len(POSES_OUT))
            self.assertEqual(poses, POSES_OUT)

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(pose_pub)
