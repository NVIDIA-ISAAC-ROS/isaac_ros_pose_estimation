# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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

_TEST_CASE_NAMESPACE = 'dope_node_test'


@pytest.mark.rostest
def generate_test_description():
    dope_encoder_node = ComposableNode(
        name='dope_encoder',
        package='isaac_ros_dnn_encoders',
        plugin='isaac_ros::dnn_inference::DnnImageEncoderNode',
        namespace=IsaacROSDopePOLTest.generate_namespace(),
        parameters=[{
            'network_image_width': 640,
            'network_image_height': 480,
            'network_image_encoding': 'rgb8',
            'network_normalization_type': 'positive_negative'
        }],
        remappings=[('encoded_tensor', 'tensor_pub')])

    dope_inference_node = ComposableNode(
        name='dope_inference',
        package='isaac_ros_tensor_rt',
        plugin='isaac_ros::dnn_inference::TensorRTNode',
        namespace=IsaacROSDopePOLTest.generate_namespace(),
        parameters=[{
            'model_file_path': os.path.dirname(__file__) +
                '/../../test/models/' + MODEL_FILE_NAME,
            'engine_file_path': '/tmp/trt_engine.plan',
            'input_tensor_names': ['input'],
            'input_binding_names': ['input'],
            'output_tensor_names': ['output'],
            'output_binding_names': ['output'],
            'verbose': False,
            'force_engine_update': True
        }])

    dope_decoder_node = ComposableNode(
        name='dope_decoder',
        package='isaac_ros_dope',
        plugin='isaac_ros::dope::DopeDecoderNode',
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
        executable='component_container',
        composable_node_descriptions=[dope_encoder_node, dope_inference_node, dope_decoder_node],
        output='screen',
    )

    return IsaacROSDopePOLTest.generate_test_description([container])


class IsaacROSDopePOLTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    def test_pol(self):
        TIMEOUT = 300
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

            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                image_pub.publish(image)
                rclpy.spin_once(self.node, timeout_sec=(0.1))
                if 'poses' in received_messages:
                    done = True
                    break
            self.assertTrue(done, 'Appropriate output not received')

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
