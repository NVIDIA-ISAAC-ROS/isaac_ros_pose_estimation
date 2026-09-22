# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Fixture-backed launch test for the Isaac ROS CenterPose decoder."""

import json
import os
import pathlib
import time

from isaac_ros_tensor_msgs.msg import TensorList
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo
from tensor_msgs.msg import ExperimentalTensor
from vision_msgs.msg import Detection3DArray


TENSOR_NAMES = [
    'bboxes',
    'scores',
    'kps',
    'clses',
    'obj_scale',
    'kps_displacement_mean',
    'kps_heatmap_mean',
]
FLOAT_DTYPE_CODE = 2
TIMEOUT_SEC = 10
POSITION_TOL = 1e-4
SIZE_TOL = 1e-5
QUATERNION_DOT_TOL = 1e-4


@pytest.mark.rostest
def generate_test_description():
    """Launch the decoder node under test."""
    centerpose_decoder_node = ComposableNode(
        name='centerpose_decoder_node',
        package='isaac_ros_centerpose',
        plugin='nvidia::isaac_ros::centerpose::CenterPoseDecoderNode',
        namespace=IsaacROSCenterPoseDecoderFixtureTest.generate_namespace(),
        parameters=[{
            'output_field_size': [128, 128],
            'cuboid_scaling_factor': 1.0,
            'score_threshold': 0.3,
            'object_name': 'shoe',
        }],
    )

    rclcpp_container = ComposableNodeContainer(
        name='rclcpp_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[centerpose_decoder_node],
        output='screen',
    )

    return IsaacROSCenterPoseDecoderFixtureTest.generate_test_description(
        [rclcpp_container])


class IsaacROSCenterPoseDecoderFixtureTest(IsaacROSBaseTest):
    """Validate CenterPose decoder output against a fixed tensor fixture."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    @staticmethod
    def _load_tensor_list(test_folder):
        with open(test_folder / 'tensor_data.json', encoding='utf-8') as tensor_file:
            tensor_json = json.load(tensor_file)

        tensor_list = TensorList()
        for tensor_name in TENSOR_NAMES:
            tensor_data = np.asarray(tensor_json[tensor_name], dtype=np.float32)

            tensor = ExperimentalTensor()
            tensor.dtype_code = FLOAT_DTYPE_CODE
            tensor.dtype_bits = 32
            tensor.dtype_lanes = 1
            tensor.shape = list(tensor_data.shape)
            tensor.strides = []
            tensor.byte_offset = 0
            tensor.data = tensor_data.tobytes()
            tensor_list.names.append(tensor_name)
            tensor_list.tensors.append(tensor)

        return tensor_list

    @staticmethod
    def _load_ground_truth(test_folder):
        with open(test_folder / 'ground_truth.json', encoding='utf-8') as ground_truth_file:
            return json.load(ground_truth_file)['objects']

    @staticmethod
    def _position_array(detection):
        position = detection.bbox.center.position
        return np.asarray([position.x, position.y, position.z], dtype=np.float64)

    @staticmethod
    def _size_array(detection):
        size = detection.bbox.size
        return np.asarray([size.x, size.y, size.z], dtype=np.float64)

    @staticmethod
    def _quaternion_array(detection):
        orientation = detection.bbox.center.orientation
        return np.asarray(
            [orientation.x, orientation.y, orientation.z, orientation.w],
            dtype=np.float64)

    def _assert_detection_matches_ground_truth(self, detection, expected):
        actual_position = self._position_array(detection)
        expected_position = np.asarray(expected['location'], dtype=np.float64)
        np.testing.assert_allclose(
            actual_position, expected_position, atol=POSITION_TOL, rtol=0.0)

        actual_size = self._size_array(detection)
        expected_size = np.asarray(expected['relative_scale'], dtype=np.float64)
        np.testing.assert_allclose(
            actual_size, expected_size, atol=SIZE_TOL, rtol=0.0)

        actual_quaternion = self._quaternion_array(detection)
        expected_quaternion = np.asarray(expected['quaternion_xyzw'], dtype=np.float64)
        actual_quaternion /= np.linalg.norm(actual_quaternion)
        expected_quaternion /= np.linalg.norm(expected_quaternion)
        self.assertLessEqual(
            1.0 - abs(float(np.dot(actual_quaternion, expected_quaternion))),
            QUATERNION_DOT_TOL)

        self.assertEqual(1, len(detection.results))
        self.assertEqual('0', detection.results[0].hypothesis.class_id)
        self.assertGreaterEqual(detection.results[0].hypothesis.score, 0.3)
        self.assertEqual(detection.bbox.center, detection.results[0].pose.pose)

    def test_centerpose_decoder_fixture(self):
        """Publish tensor fixture data and assert the decoder result."""
        received_messages = {}
        self.generate_namespace_lookup(['tensor_sub', 'camera_info', 'centerpose/detections'])

        tensor_pub = self.node.create_publisher(
            TensorList, self.namespaces['tensor_sub'], self.DEFAULT_QOS)
        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'], self.DEFAULT_QOS)
        subs = self.create_logging_subscribers(
            [('centerpose/detections', Detection3DArray)], received_messages)

        try:
            test_folder = self.filepath / 'test_cases' / 'shoe'
            tensor_list = self._load_tensor_list(test_folder)
            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')
            ground_truth = self._load_ground_truth(test_folder)

            end_time = time.time() + TIMEOUT_SEC
            while time.time() < end_time:
                timestamp = self.node.get_clock().now().to_msg()
                tensor_list.header.stamp = timestamp
                tensor_list.header.frame_id = camera_info.header.frame_id
                camera_info.header.stamp = timestamp

                tensor_pub.publish(tensor_list)
                camera_info_pub.publish(camera_info)
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if 'centerpose/detections' in received_messages:
                    break

            self.assertIn(
                'centerpose/detections', received_messages,
                'Timed out waiting for CenterPose decoder output')

            detection_array = received_messages['centerpose/detections']
            self.assertEqual(len(ground_truth), len(detection_array.detections))

            unmatched_ground_truth = list(ground_truth)
            for detection in detection_array.detections:
                actual_position = self._position_array(detection)
                closest_idx = min(
                    range(len(unmatched_ground_truth)),
                    key=lambda i: np.linalg.norm(
                        actual_position -
                        np.asarray(unmatched_ground_truth[i]['location'], dtype=np.float64)))
                expected = unmatched_ground_truth.pop(closest_idx)
                self._assert_detection_matches_ground_truth(detection, expected)

        finally:
            for sub in subs:
                self.assertTrue(self.node.destroy_subscription(sub))
            self.assertTrue(self.node.destroy_publisher(tensor_pub))
            self.assertTrue(self.node.destroy_publisher(camera_info_pub))
