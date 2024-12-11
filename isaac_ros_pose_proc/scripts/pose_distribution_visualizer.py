#!/usr/bin/env python3

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

from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node


class PoseDistributionVisualizer(Node):

    def __init__(self):
        super().__init__('pose_distribution_visualizer')

        self._pose_sub = self.create_subscription(
            PoseStamped,
            'pose_input',
            self.pose_callback,
            10
        )

        self._pose_history = {
            'position': {
                'x': [],
                'y': [],
                'z': []
            },
            'orientation': {
                'x': [],
                'y': [],
                'z': [],
                'w': [],
            }
        }

        self.get_logger().info('Pose visualization starting')

    def visualize_data(self):
        for key, data_dict in self._pose_history.items():
            fig, ax = plt.subplots(1, len(data_dict.values()))
            for i, (label, value) in enumerate(data_dict.items()):
                ax[i].hist(value)
                ax[i].set_title(label)
            fig.suptitle(key)
            plt.show()

    def pose_callback(self, msg):
        self._pose_history['position']['x'].append(msg.pose.position.x)
        self._pose_history['position']['y'].append(msg.pose.position.y)
        self._pose_history['position']['z'].append(msg.pose.position.z)
        self._pose_history['orientation']['x'].append(msg.pose.orientation.x)
        self._pose_history['orientation']['y'].append(msg.pose.orientation.y)
        self._pose_history['orientation']['z'].append(msg.pose.orientation.z)
        self._pose_history['orientation']['w'].append(msg.pose.orientation.w)


def main(args=None):
    try:
        rclpy.init(args=args)
        node = PoseDistributionVisualizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.visualize_data()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
