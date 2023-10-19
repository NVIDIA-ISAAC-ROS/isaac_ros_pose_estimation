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

"""
Basic Proof-Of-Life test for the Isaac ROS Deep Object Pose Estimation (DOPE) Node.

This test checks that an image can be encoder into a tensor, run through
TensorRT using a DOPE model, and the inference decoded into a series of poses.
"""
import os
import pathlib
import time

from geometry_msgs.msg import PoseArray
from isaac_ros_test import IsaacROSBaseTest, JSONConversion

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import pytest
import rclpy

from sensor_msgs.msg import Image

MODEL_FILE_NAME = 'dope_ketchup_pol.onnx'

MODEL_GENERATION_TIMEOUT_SEC = 300
INIT_WAIT_SEC = 10
MODEL_PATH = '/tmp/dope_trt_engine.plan'


@pytest.mark.rostest
def generate_test_description():
    dope_encoder_node = ComposableNode(
        name='dope_encoder',
        package='isaac_ros_dnn_image_encoder',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        namespace=IsaacROSDopePOLTest.generate_namespace(),
        parameters=[{
            'network_image_width': 640,
            'network_image_height': 480,
            'input_image_width': 852,
            'input_image_height': 480
        }],
        remappings=[('encoded_tensor', 'tensor_pub')])

    dope_inference_node = ComposableNode(
        name='dope_inference',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        namespace=IsaacROSDopePOLTest.generate_namespace(),
        parameters=[{
            'model_file_path': os.path.dirname(__file__) +
                '/../../test/models/' + MODEL_FILE_NAME,
            'engine_file_path': MODEL_PATH,
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'output_tensor_names': ['output'],
            'output_binding_names': ['output'],
            'output_tensor_formats': ['nitros_tensor_list_nhwc_rgb_f32'],
            'verbose': False,
            'force_engine_update': False,
        }])

    dope_decoder_node = ComposableNode(
        name='dope_decoder',
        package='isaac_ros_dope',
        plugin='nvidia::isaac_ros::dope::DopeDecoderNode',
        namespace=IsaacROSDopePOLTest.generate_namespace(),
        parameters=[{
            'object_name': 'Ketchup',
            'frame_id': 'map'
        }],
        remappings=[('belief_map_array', 'tensor_sub'),
                    ('dope/pose_array', 'poses')])

    container = ComposableNodeContainer(
        name='dope_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[dope_encoder_node, dope_inference_node, dope_decoder_node],
        output='screen',
    )

    return IsaacROSDopePOLTest.generate_test_description([container])


class IsaacROSDopePOLTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    def test_pol(self):
        self.node._logger.info(f'Generating model (timeout={MODEL_GENERATION_TIMEOUT_SEC}s)')
        start_time = time.time()
        wait_cycles = 1
        while not os.path.isfile(MODEL_PATH):
            time_diff = time.time() - start_time
            if time_diff > MODEL_GENERATION_TIMEOUT_SEC:
                self.fail('Model generation timed out')
            if time_diff > wait_cycles*10:
                self.node._logger.info(
                    f'Waiting for model generation to finish... ({int(time_diff)}s passed)')
                wait_cycles += 1
            time.sleep(1)

        # Wait for TensorRT engine
        time.sleep(INIT_WAIT_SEC)

        self.node._logger.info(
            f'Model generation was finished (took {(time.time() - start_time)}s)')

        received_messages = {}

        self.generate_namespace_lookup(['image', 'poses'])

        image_pub = self.node.create_publisher(
            Image, self.namespaces['image'], self.DEFAULT_QOS)

        # The current DOPE decoder outputs PoseArray
        subs = self.create_logging_subscribers(
            [('poses', PoseArray)], received_messages)

        try:
            json_file = self.filepath / 'test_cases/pose_estimation_0/image.json'
            image = JSONConversion.load_image_from_json(json_file)
            image.header.stamp = self.node.get_clock().now().to_msg()

            TIMEOUT = 60
            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                image_pub.publish(image)
                rclpy.spin_once(self.node, timeout_sec=(0.1))
                if 'poses' in received_messages:
                    done = True
                    break
            self.assertTrue(done, 'Timeout. Appropriate output not received')
            self.node._logger.info('A message was successfully received')

            received_poses = received_messages['poses'].poses
            self.node._logger.info(f'Poses received: {received_poses}')
            self.assertGreaterEqual(len(received_poses), 1, 'Did not receive at least one pose')

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
