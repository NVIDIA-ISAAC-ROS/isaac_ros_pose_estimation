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

"""
Basic Proof-Of-Life test for the Isaac ROS FoundationPose 6D Pose Estimation Node.

This node takes RGBD image and segmentation mask into a tensor, run through
TensorRT using a FoundationPose models, and the inference decoded into a series of poses.
"""
import os
import pathlib
import sys
import time

import cv2
from cv_bridge import CvBridge
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection3DArray

# Add current directory to Python path for imports.
sys.path.append(os.path.dirname(__file__))
from generate_engines import generate_foundationpose_engines, get_engines  # noqa: E402, I100


MODEL_GENERATION_TIMEOUT_SEC = 900
INIT_WAIT_SEC = 10
DELTA = 0.05

MESH_FILE_NAME = 'textured_simple.obj'


@pytest.mark.rostest
def generate_test_description():
    # Generate engine files.
    generate_foundationpose_engines()

    # Get engines info from utils.
    engines = get_engines()

    foundationpose_node = ComposableNode(
        name='foundationpose',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        namespace=IsaacROSFoundationPosePOLTest.generate_namespace(),
        parameters=[{
            'mesh_file_path': os.path.dirname(__file__) +
                '/test_cases/foundationpose/' + MESH_FILE_NAME,
            'refine_iterations': 1,

            'refine_model_file_path':  os.path.dirname(__file__) +
                '/../../test/models/' + engines['refine_model_name'],
            'refine_engine_file_path': engines['refine_engine_path'],
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names': ['input1', 'input2'],
            'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],

            'score_model_file_path':  os.path.dirname(__file__) +
                '/../../test/models/' + engines['score_model_name'],
            'score_engine_file_path': engines['score_engine_path'],
            'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'score_input_binding_names': ['input1', 'input2'],
            'score_output_tensor_names': ['output_tensor'],
            'score_output_binding_names': ['output1'],
        }])

    container = ComposableNodeContainer(
        name='foundationpose_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[foundationpose_node],
        output='screen',
    )

    return IsaacROSFoundationPosePOLTest.generate_test_description([container])


class IsaacROSFoundationPosePOLTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case()
    def test_pol(self, test_folder):
        # Get engines info from utils
        engines = get_engines()

        self.node._logger.info(
            f'Generating model (timeout={MODEL_GENERATION_TIMEOUT_SEC}s)')
        start_time = time.time()
        wait_cycles = 1
        while (not os.path.isfile(engines['score_engine_path']) or
               not os.path.isfile(engines['refine_engine_path'])):
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

        received_messages = {}

        self.generate_namespace_lookup(['pose_estimation/depth_image',
                                        'pose_estimation/image',
                                        'pose_estimation/camera_info',
                                        'pose_estimation/segmentation',
                                        'pose_estimation/output'])

        subs = self.create_logging_subscribers(
            [('pose_estimation/output', Detection3DArray)], received_messages)

        depth_pub = self.node.create_publisher(
            Image, self.namespaces['pose_estimation/depth_image'], self.DEFAULT_QOS
        )
        rgb_image_pub = self.node.create_publisher(
            Image, self.namespaces['pose_estimation/image'], self.DEFAULT_QOS
        )
        mask_pub = self.node.create_publisher(
            Image, self.namespaces['pose_estimation/segmentation'], self.DEFAULT_QOS
        )
        cam_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['pose_estimation/camera_info'], self.DEFAULT_QOS
        )

        try:
            depth_path = str(test_folder) + '/mustard_depth.png'
            mask = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            depth_array = np.array(mask, dtype=np.float32)
            # Apply the same principle with the demo
            depth_array = depth_array / 1e3
            depth_image = CvBridge().cv2_to_imgmsg(depth_array, '32FC1')

            bgr_path = str(test_folder) + '/mustard_rgb.png'
            bgr = cv2.imread(
                bgr_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_image = CvBridge().cv2_to_imgmsg(rgb, 'rgb8')

            segmentation_path = str(test_folder) + '/mustard_mask.png'
            raw_segmentation = cv2.imread(
                segmentation_path, cv2.IMREAD_GRAYSCALE)
            mask = CvBridge().cv2_to_imgmsg(raw_segmentation, 'mono8')

            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')

            TIMEOUT = 60
            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                time_now_msg = self.node.get_clock().now().to_msg()
                depth_image.header.stamp = time_now_msg
                rgb_image.header.stamp = time_now_msg
                mask.header.stamp = time_now_msg
                camera_info.header.stamp = time_now_msg
                rgb_image.header.frame_id = 'tf_camera'
                camera_info.header.frame_id = 'tf_camera'

                rgb_image_pub.publish(rgb_image)
                depth_pub.publish(depth_image)
                mask_pub.publish(mask)
                cam_info_pub.publish(camera_info)
                rclpy.spin_once(self.node, timeout_sec=(10))
                if 'pose_estimation/output' in received_messages:
                    done = True
                    break

            self.assertTrue(done, 'Timeout. Appropriate output not received')
            self.node._logger.info('A message was successfully received')

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(rgb_image_pub)
            self.node.destroy_publisher(depth_pub)
            self.node.destroy_publisher(cam_info_pub)
            self.node.destroy_publisher(mask_pub)
