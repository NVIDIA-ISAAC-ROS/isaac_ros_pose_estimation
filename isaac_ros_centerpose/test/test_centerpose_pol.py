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
Basic Proof-Of-Life test for the Isaac ROS CenterPose Node.

This test checks that an image can be encoder into a tensor, run through
Triton using a CenterPose model, and the inference decoded into a series of poses.
"""
import errno
import os
import pathlib
import time

from ament_index_python.packages import get_package_share_directory
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import pytest
import rclpy

from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection3DArray

MODEL_GENERATION_TIMEOUT_SEC = 600
INIT_WAIT_SEC = 10
MODEL_PATH = '/tmp/centerpose_trt_engine.plan'


@pytest.mark.rostest
def generate_test_description():
    try:
        os.remove(MODEL_PATH)
    except OSError as e:
        if e.errno != errno.ENOENT:
            print(f'File exists but error deleting {MODEL_PATH}')

    launch_dir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    model_dir_path = launch_dir_path / 'models'
    model_file_name = 'centerpose_shoe.onnx'

    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    centerpose_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': '600',
            'input_image_height': '800',
            'network_image_width': '512',
            'network_image_height': '512',
            'image_mean': '[0.408, 0.447, 0.47]',
            'image_stddev': '[0.289, 0.274, 0.278]',
            'attach_to_shared_component_container': 'True',
            'component_container_name': 'rclcpp_container',
            'dnn_image_encoder_namespace': IsaacROSCenterPosePOLTest.generate_namespace(),
            'tensor_output_topic': 'tensor_pub',
        }.items(),
    )

    centerpose_inference_node = ComposableNode(
        name='centerpose_inference',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        namespace=IsaacROSCenterPosePOLTest.generate_namespace(),
        parameters=[{
            'model_file_path': str(model_dir_path / model_file_name),
            'engine_file_path': MODEL_PATH,
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'output_tensor_names': ['bboxes', 'scores', 'kps', 'clses',
                                    'obj_scale', 'kps_displacement_mean',
                                    'kps_heatmap_mean'],
            'output_binding_names': ['bboxes', 'scores', 'kps', 'clses',
                                     'obj_scale', 'kps_displacement_mean',
                                     'kps_heatmap_mean'],
            'output_tensor_formats': ['nitros_tensor_list_nhwc_rgb_f32'],
            'verbose': False,
            'force_engine_update': False,
            'max_workspace_size': 512*1024*1024,  # 512MB
        }],
    )

    centerpose_decoder_node = ComposableNode(
        name='centerpose_decoder_node',
        package='isaac_ros_centerpose',
        plugin='nvidia::isaac_ros::centerpose::CenterPoseDecoderNode',
        namespace=IsaacROSCenterPosePOLTest.generate_namespace(),
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
        composable_node_descriptions=[
            centerpose_inference_node, centerpose_decoder_node,
        ],
        output='screen',
    )

    return IsaacROSCenterPosePOLTest.generate_test_description(
        [rclcpp_container, centerpose_encoder_launch])


class IsaacROSCenterPosePOLTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    def test_pol(self):

        self.node._logger.info(
            f'Generating model (timeout={MODEL_GENERATION_TIMEOUT_SEC}s)')
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

        self.generate_namespace_lookup(['image', 'camera_info', 'centerpose/detections'])

        image_pub = self.node.create_publisher(
            Image, self.namespaces['image'], self.DEFAULT_QOS)
        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'], self.DEFAULT_QOS)

        # The current CenterPose decoder outputs Detection3DArray
        subs = self.create_logging_subscribers(
            [('centerpose/detections', Detection3DArray)], received_messages)

        try:
            json_file = self.filepath / 'test_cases/shoe/image.json'
            image = JSONConversion.load_image_from_json(json_file)
            info_file = self.filepath / 'test_cases/shoe/camera_info.json'
            camera_info = JSONConversion.load_camera_info_from_json(info_file)

            TIMEOUT = 60
            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                camera_info_pub.publish(camera_info)
                image_pub.publish(image)
                rclpy.spin_once(self.node, timeout_sec=(0.1))
                if 'centerpose/detections' in received_messages:
                    done = True
                    break
            self.assertTrue(done, 'Timeout. Appropriate output not received')
            self.node._logger.info('A message was successfully received')

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
