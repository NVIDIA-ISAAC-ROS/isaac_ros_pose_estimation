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

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

# Number of Realsense messages to be dropped in 1 second
HAWK_EXPECT_FREQ = 28
# Expected number of Realsense messages in 1 second
INPUT_IMAGES_DROP_FREQ = 30

RT_DETR_MODEL_INPUT_SIZE = 640  # RT-DETR models expect 640x640 encoded image size
RT_DETR_MODEL_NUM_CHANNELS = 3  # RT-DETR models expect 3 image channels

REALSENSE_IMAGE_WIDTH = 1280
REALSENSE_IMAGE_HEIGHT = 720

VISUALIZATION_DOWNSCALING_FACTOR = 10

REALSENSE_TO_RT_DETR_RATIO = REALSENSE_IMAGE_WIDTH / RT_DETR_MODEL_INPUT_SIZE

REFINE_MODEL_PATH = '/tmp/refine_model.onnx'
REFINE_ENGINE_PATH = '/tmp/refine_trt_engine.plan'
SCORE_MODEL_PATH = '/tmp/score_model.onnx'
SCORE_ENGINE_PATH = '/tmp/score_trt_engine.plan'


def generate_launch_description():
    """Generate launch description for testing relevant nodes."""
    rviz_config_path = os.path.join(
        get_package_share_directory('isaac_ros_foundationpose'),
        'rviz', 'foundationpose_realsense.rviz')

    launch_args = [
        DeclareLaunchArgument(
            'hawk_expect_freq',
            default_value=str(HAWK_EXPECT_FREQ),
            description='Number of Realsense messages to be dropped in 1 second'),

        DeclareLaunchArgument(
            'input_images_drop_freq',
            default_value=str(INPUT_IMAGES_DROP_FREQ),
            description='Expected number of Realsense messages in 1 second'),

        DeclareLaunchArgument(
            'mesh_file_path',
            default_value='',
            description='The absolute file path to the mesh file'),

        DeclareLaunchArgument(
            'texture_path',
            default_value='',
            description='The absolute file path to the texture map'),

        DeclareLaunchArgument(
            'refine_model_file_path',
            default_value=REFINE_MODEL_PATH,
            description='The absolute file path to the refine model'),

        DeclareLaunchArgument(
            'refine_engine_file_path',
            default_value=REFINE_ENGINE_PATH,
            description='The absolute file path to the refine trt engine'),

        DeclareLaunchArgument(
            'score_model_file_path',
            default_value=SCORE_MODEL_PATH,
            description='The absolute file path to the score model'),

        DeclareLaunchArgument(
            'score_engine_file_path',
            default_value=SCORE_ENGINE_PATH,
            description='The absolute file path to the score trt engine'),

        DeclareLaunchArgument(
            'rt_detr_model_file_path',
            default_value='',
            description='The absolute file path to the RT-DETR ONNX file'),

        DeclareLaunchArgument(
            'rt_detr_engine_file_path',
            default_value='',
            description='The absolute file path to the RT-DETR TensorRT engine file'),

        DeclareLaunchArgument(
            'launch_rviz',
            default_value='False',
            description='Flag to enable Rviz2 launch'),

        DeclareLaunchArgument(
            'container_name',
            default_value='foundationpose_container',
            description='Name for ComposableNodeContainer'),
    ]

    hawk_expect_freq = LaunchConfiguration('hawk_expect_freq')
    input_images_drop_freq = LaunchConfiguration('input_images_drop_freq')
    mesh_file_path = LaunchConfiguration('mesh_file_path')
    texture_path = LaunchConfiguration('texture_path')
    refine_model_file_path = LaunchConfiguration('refine_model_file_path')
    refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
    score_model_file_path = LaunchConfiguration('score_model_file_path')
    score_engine_file_path = LaunchConfiguration('score_engine_file_path')
    rt_detr_model_file_path = LaunchConfiguration('rt_detr_model_file_path')
    rt_detr_engine_file_path = LaunchConfiguration('rt_detr_engine_file_path')
    launch_rviz = LaunchConfiguration('launch_rviz')
    container_name = LaunchConfiguration('container_name')

    # RealSense
    realsense_config_file_path = os.path.join(
        get_package_share_directory('isaac_ros_foundationpose'),
        'config', 'realsense.yaml'
    )

    realsense_node = ComposableNode(
        package='realsense2_camera',
        plugin='realsense2_camera::RealSenseNodeFactory',
        parameters=[realsense_config_file_path]
    )

    # Drops hawk_expect_freq out of input_images_drop_freq RealSense messages
    drop_node = ComposableNode(
        name='drop_node',
        package='isaac_ros_nitros_topic_tools',
        plugin='nvidia::isaac_ros::nitros::NitrosCameraDropNode',
        parameters=[{
            'X': hawk_expect_freq,
            'Y': input_images_drop_freq,
            'mode': 'mono+depth',
            'depth_format_string': 'nitros_image_mono16'
        }],
        remappings=[
            ('image_1', '/color/image_raw'),
            ('camera_info_1', '/color/camera_info'),
            ('depth_1', '/aligned_depth_to_color/image_raw'),
            ('image_1_drop', 'rgb/image_rect_color'),
            ('camera_info_1_drop', 'rgb/camera_info'),
            ('depth_1_drop', 'depth_uint16'),
        ]
    )

    # Realsense depth is in uint16 and millimeters. Convert to float32 and meters
    convert_metric_node = ComposableNode(
        package='isaac_ros_depth_image_proc',
        plugin='nvidia::isaac_ros::depth_image_proc::ConvertMetricNode',
        remappings=[
            ('image_raw', 'depth_uint16'),
            ('image', 'depth_image')
        ]
    )

    # Resize and pad RealSense images to RT-DETR model input image size
    # Resize from REALSENSE_IMAGE_WIDTH x REALSENSE_IMAGE_HEIGHT to
    # REALSENSE_IMAGE_WIDTH/REALSENSE_TO_RT_DETR_RATIO x
    #   REALSENSE_IMAGE_HEIGHT/REALSENSE_TO_RT_DETR_RATIO
    # output height constraint is not used since keep_aspect_ratio is True
    resize_left_rt_detr_node = ComposableNode(
        name='resize_left_rt_detr_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': REALSENSE_IMAGE_WIDTH,
            'input_height': REALSENSE_IMAGE_HEIGHT,
            'output_width': RT_DETR_MODEL_INPUT_SIZE,
            'output_height': RT_DETR_MODEL_INPUT_SIZE,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }],
        remappings=[
            ('image', 'rgb/image_rect_color'),
            ('camera_info', 'rgb/camera_info'),
            ('resize/image', 'color_image_resized'),
            ('resize/camera_info', 'camera_info_resized')
        ]
    )
    # Pad the image from REALSENSE_IMAGE_WIDTH/REALSENSE_TO_RT_DETR_RATIO x
    #   REALSENSE_IMAGE_HEIGHT/REALSENSE_TO_RT_DETR_RATIO
    # to RT_DETR_MODEL_INPUT_WIDTH x RT_DETR_MODEL_INPUT_HEIGHT
    pad_node = ComposableNode(
        name='pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': RT_DETR_MODEL_INPUT_SIZE,
            'output_image_height': RT_DETR_MODEL_INPUT_SIZE,
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[(
            'image', 'color_image_resized'
        )]
    )

    # Convert image to tensor and reshape
    image_to_tensor_node = ComposableNode(
        name='image_to_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'image',
        }],
        remappings=[
            ('image', 'padded_image'),
            ('tensor', 'normalized_tensor'),
        ]
    )
    interleave_to_planar_node = ComposableNode(
        name='interleaved_to_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [RT_DETR_MODEL_INPUT_SIZE,
                                   RT_DETR_MODEL_INPUT_SIZE,
                                   RT_DETR_MODEL_NUM_CHANNELS]
        }],
        remappings=[
            ('interleaved_tensor', 'normalized_tensor')
        ]
    )
    reshape_node = ComposableNode(
        name='reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [RT_DETR_MODEL_NUM_CHANNELS,
                                   RT_DETR_MODEL_INPUT_SIZE,
                                   RT_DETR_MODEL_INPUT_SIZE],
            'output_tensor_shape': [1, RT_DETR_MODEL_NUM_CHANNELS,
                                    RT_DETR_MODEL_INPUT_SIZE,
                                    RT_DETR_MODEL_INPUT_SIZE]
        }],
        remappings=[
            ('tensor', 'planar_tensor')
        ],
    )
    rtdetr_preprocessor_node = ComposableNode(
        name='rtdetr_preprocessor',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
        remappings=[
            ('encoded_tensor', 'reshaped_tensor')
        ]
    )

    # RT-DETR objection detection pipeline
    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': rt_detr_model_file_path,
            'engine_file_path': rt_detr_engine_file_path,
            'output_binding_names': ['labels', 'boxes', 'scores'],
            'output_tensor_names': ['labels', 'boxes', 'scores'],
            'input_tensor_names': ['images', 'orig_target_sizes'],
            'input_binding_names': ['images', 'orig_target_sizes'],
            'force_engine_update': False
        }]
    )
    rtdetr_decoder_node = ComposableNode(
        name='rtdetr_decoder',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
    )

    # Create a binary segmentation mask from a Detection2DArray published by RT-DETR.
    # The segmentation mask is of size int(REALSENSE_IMAGE_WIDTH/REALSENSE_TO_RT_DETR_RATIO) x
    # int(REALSENSE_IMAGE_HEIGHT/REALSENSE_TO_RT_DETR_RATIO)
    detection2_d_to_mask_node = ComposableNode(
        name='detection2_d_to_mask',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
        parameters=[{
            'mask_width': int(REALSENSE_IMAGE_WIDTH/REALSENSE_TO_RT_DETR_RATIO),
            'mask_height': int(REALSENSE_IMAGE_HEIGHT/REALSENSE_TO_RT_DETR_RATIO)}],
        remappings=[('detection2_d_array', 'detections_output'),
                    ('segmentation', 'rt_detr_segmentation')])

    # Resize segmentation mask to ESS model image size so it can be used by FoundationPose
    # FoundationPose requires depth, rgb image and segmentation mask to be of the same size
    # Resize from int(REALSENSE_IMAGE_WIDTH/REALSENSE_TO_RT_DETR_RATIO) x
    # int(REALSENSE_IMAGE_HEIGHT/REALSENSE_TO_RT_DETR_RATIO) to
    # ESS_MODEL_IMAGE_WIDTH x ESS_MODEL_IMAGE_HEIGHT
    # output height constraint is used since keep_aspect_ratio is False
    # and the image is padded
    resize_mask_node = ComposableNode(
        name='resize_mask_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': int(REALSENSE_IMAGE_WIDTH/REALSENSE_TO_RT_DETR_RATIO),
            'input_height': int(REALSENSE_IMAGE_HEIGHT/REALSENSE_TO_RT_DETR_RATIO),
            'output_width': REALSENSE_IMAGE_WIDTH,
            'output_height': REALSENSE_IMAGE_HEIGHT,
            'keep_aspect_ratio': False,
            'disable_padding': False
        }],
        remappings=[
            ('image', 'rt_detr_segmentation'),
            ('camera_info', 'camera_info_resized'),
            ('resize/image', 'segmentation'),
            ('resize/camera_info', 'camera_info_segmentation')
        ]
    )

    resize_left_viz = ComposableNode(
        name='resize_left_viz',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': REALSENSE_IMAGE_WIDTH,
            'input_height': REALSENSE_IMAGE_HEIGHT,
            'output_width': int(REALSENSE_IMAGE_WIDTH/VISUALIZATION_DOWNSCALING_FACTOR),
            'output_height': int(REALSENSE_IMAGE_HEIGHT/VISUALIZATION_DOWNSCALING_FACTOR),
            'keep_aspect_ratio': False,
            'encoding_desired': 'rgb8',
            'disable_padding': False
        }],
        remappings=[
            ('image', 'rgb/image_rect_color'),
            ('camera_info', 'rgb/camera_info'),
            ('resize/image', 'rgb/image_rect_color_viz'),
            ('resize/camera_info', 'rgb/camera_info_viz')
        ]
    )

    foundationpose_node = ComposableNode(
        name='foundationpose_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[{
            'mesh_file_path': mesh_file_path,
            'texture_path': texture_path,

            'refine_model_file_path': refine_model_file_path,
            'refine_engine_file_path': refine_engine_file_path,
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names': ['input1', 'input2'],
            'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],

            'score_model_file_path': score_model_file_path,
            'score_engine_file_path': score_engine_file_path,
            'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'score_input_binding_names': ['input1', 'input2'],
            'score_output_tensor_names': ['output_tensor'],
            'score_output_binding_names': ['output1'],
        }],
        remappings=[
            ('pose_estimation/depth_image', 'depth_image'),
            ('pose_estimation/image', 'rgb/image_rect_color'),
            ('pose_estimation/camera_info', 'rgb/camera_info'),
            ('pose_estimation/segmentation', 'segmentation'),
            ('pose_estimation/output', 'output')])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        condition=IfCondition(launch_rviz))

    foundationpose_container = ComposableNodeContainer(
        name=container_name,
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            realsense_node,
            drop_node,
            convert_metric_node,
            resize_left_rt_detr_node,
            pad_node,
            image_to_tensor_node,
            interleave_to_planar_node,
            reshape_node,
            rtdetr_preprocessor_node,
            tensor_rt_node,
            rtdetr_decoder_node,
            detection2_d_to_mask_node,
            resize_mask_node,
            foundationpose_node,
            resize_left_viz
        ],
        output='screen'
    )

    return launch.LaunchDescription(launch_args + [foundationpose_container,
                                                   rviz_node])
