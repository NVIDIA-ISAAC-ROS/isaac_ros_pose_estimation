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

# Number of Hawk messages to be dropped in 1 second
HAWK_EXPECT_FREQ = 15
# Expected number of Hawk messages in 1 second
INPUT_IMAGES_DROP_FREQ = 30

RT_DETR_MODEL_INPUT_WIDTH = 640
RT_DETR_MODEL_INPUT_HEIGHT = 640
RT_DETR_MODEL_NUM_CHANNELS = 3  # RT-DETR models expect 3 image channels

ESS_MODEL_IMAGE_WIDTH = 480
ESS_MODEL_IMAGE_HEIGHT = 288

HAWK_IMAGE_WIDTH = 1920
HAWK_IMAGE_HEIGHT = 1200

VISUALIZATION_DOWNSCALING_FACTOR = 10

HAWK_TO_RT_DETR_RATIO = HAWK_IMAGE_WIDTH / RT_DETR_MODEL_INPUT_WIDTH

REFINE_MODEL_PATH = '/tmp/refine_model.onnx'
REFINE_ENGINE_PATH = '/tmp/refine_trt_engine.plan'
SCORE_MODEL_PATH = '/tmp/score_model.onnx'
SCORE_ENGINE_PATH = '/tmp/score_trt_engine.plan'


def generate_launch_description():
    """Generate launch description for testing relevant nodes."""
    rviz_config_path = os.path.join(
        get_package_share_directory('isaac_ros_foundationpose'),
        'rviz', 'foundationpose_hawk.rviz')

    launch_args = [
        DeclareLaunchArgument(
            'hawk_expect_freq',
            default_value=str(HAWK_EXPECT_FREQ),
            description='Number of Hawk messages to be dropped in 1 second'),

        DeclareLaunchArgument(
            'input_images_drop_freq',
            default_value=str(INPUT_IMAGES_DROP_FREQ),
            description='Expected number of Hawk messages in 1 second'),

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
            description='The absolute file path to the RT-DETR  TensorRT engine file'),

        DeclareLaunchArgument(
            'ess_depth_engine_file_path',
            default_value='',
            description='The absolute path to the ESS engine plan.'),

        DeclareLaunchArgument(
            'ess_depth_threshold',
            default_value='0.9',
            description='Threshold value ranges between 0.0 and 1.0 '
                        'for filtering disparity with confidence.'),

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
    ess_depth_engine_file_path = LaunchConfiguration(
        'ess_depth_engine_file_path')
    ess_depth_threshold = LaunchConfiguration('ess_depth_threshold')
    launch_rviz = LaunchConfiguration('launch_rviz')
    container_name = LaunchConfiguration('container_name')

    # Launch hawk
    # Publishes left and right images at HAWK_IMAGE_WIDTH x HAWK_IMAGE_HEIGHT resolution
    correlated_timestamp_driver_node = ComposableNode(
        package='isaac_ros_correlated_timestamp_driver',
        plugin='nvidia::isaac_ros::correlated_timestamp_driver::CorrelatedTimestampDriverNode',
        name='correlated_timestamp_driver',
    )
    hawk_node = ComposableNode(
        name='hawk_node',
        package='isaac_ros_hawk',
        plugin='nvidia::isaac_ros::hawk::HawkNode',
        parameters=[{'module_id': 5,
                     'input_qos': 'SENSOR_DATA',
                     'output_qos': 'SENSOR_DATA',
                     'enable_diagnostics': True,
                     'topics_list': ['left/image_raw'],
                     'expected_fps_list': [30.0],
                     'jitter_tolerance_us': 30000}],
        remappings=[
            ('/hawk_front/correlated_timestamp', '/correlated_timestamp')
        ]
    )

    # Drops hawk_expect_freq out of input_images_drop_freq Hawk messages
    drop_node = ComposableNode(
        name='drop_node',
        package='isaac_ros_nitros_topic_tools',
        plugin='nvidia::isaac_ros::nitros::NitrosCameraDropNode',
        parameters=[{
            'input_qos': 'SENSOR_DATA',
            'output_qos': 'SENSOR_DATA',
            'X': hawk_expect_freq,
            'Y': input_images_drop_freq,
            'mode': 'stereo',
            'sync_queue_size': 100
        }],
        remappings=[
            ('image_1', 'left/image_raw'),
            ('camera_info_1', 'left/camera_info'),
            ('image_1_drop', 'left/image_raw_drop'),
            ('camera_info_1_drop', 'left/camera_info_drop'),
            ('image_2', 'right/image_raw'),
            ('camera_info_2', 'right/camera_info'),
            ('image_2_drop', 'right/image_raw_drop'),
            ('camera_info_2_drop', 'right/camera_info_drop'),
        ]
    )

    # Rectify left and right images to be used by ESS(depth estimation)
    # and RT-DETR(object detection)
    left_rectify_node = ComposableNode(
        name='left_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': HAWK_IMAGE_WIDTH,
            'output_height': HAWK_IMAGE_HEIGHT,
            'input_qos': 'SENSOR_DATA',
            'output_qos': 'DEFAULT'
        }],
        remappings=[
            ('image_raw', 'left/image_raw_drop'),
            ('camera_info', 'left/camera_info_drop'),
            ('image_rect', 'left/image_rect'),
            ('camera_info_rect', 'left/camera_info_rect')
        ]
    )
    right_rectify_node = ComposableNode(
        name='right_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': HAWK_IMAGE_WIDTH,
            'output_height': HAWK_IMAGE_HEIGHT,
            'input_qos': 'SENSOR_DATA',
            'output_qos': 'DEFAULT'
        }],
        remappings=[
            ('image_raw', 'right/image_raw_drop'),
            ('camera_info', 'right/camera_info_drop'),
            ('image_rect', 'right/image_rect'),
            ('camera_info_rect', 'right/camera_info_rect')
        ]
    )

    # Resize and pad hawk images to RT-DETR model input image size
    # Resize from HAWK_IMAGE_WIDTH x HAWK_IMAGE_HEIGHT to
    # HAWK_IMAGE_WIDTH/HAWK_TO_RT_DETR_RATIO x HAWK_IMAGE_HEIGHT/HAWK_TO_RT_DETR_RATIO
    # output height constraint is not used since keep_aspect_ratio is True
    resize_left_rt_detr_node = ComposableNode(
        name='resize_left_rt_detr_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': HAWK_IMAGE_WIDTH,
            'input_height': HAWK_IMAGE_HEIGHT,
            'output_width': RT_DETR_MODEL_INPUT_WIDTH,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }],
        remappings=[
            ('image', 'left/image_rect'),
            ('camera_info', 'left/camera_info_rect')
        ]
    )
    # Pad the image from HAWK_IMAGE_WIDTH/HAWK_TO_RT_DETR_RATIO x
    #   HAWK_IMAGE_HEIGHT/HAWK_TO_RT_DETR_RATIO
    # to RT_DETR_MODEL_INPUT_WIDTH x RT_DETR_MODEL_INPUT_HEIGHT
    pad_node = ComposableNode(
        name='pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': RT_DETR_MODEL_INPUT_WIDTH,
            'output_image_height': RT_DETR_MODEL_INPUT_HEIGHT,
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[(
            'image', 'resize/image'
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
            'input_tensor_shape': [RT_DETR_MODEL_INPUT_WIDTH,
                                   RT_DETR_MODEL_INPUT_HEIGHT,
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
                                   RT_DETR_MODEL_INPUT_WIDTH,
                                   RT_DETR_MODEL_INPUT_HEIGHT],
            'output_tensor_shape': [1, RT_DETR_MODEL_NUM_CHANNELS,
                                    RT_DETR_MODEL_INPUT_WIDTH,
                                    RT_DETR_MODEL_INPUT_HEIGHT]
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
    # The segmentation mask is of size int(HAWK_IMAGE_WIDTH/HAWK_TO_RT_DETR_RATIO) x
    # int(HAWK_IMAGE_HEIGHT/HAWK_TO_RT_DETR_RATIO)
    detection2_d_array_filter_node = ComposableNode(
        name='detection2_d_array_filter',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Detection2DArrayFilter',
        remappings=[('detection2_d_array', 'detections_output')]
    )
    detection2_d_to_mask_node = ComposableNode(
        name='detection2_d_to_mask',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
        parameters=[{
            'mask_width': int(HAWK_IMAGE_WIDTH/HAWK_TO_RT_DETR_RATIO),
            'mask_height': int(HAWK_IMAGE_HEIGHT/HAWK_TO_RT_DETR_RATIO)
        }],
        remappings=[('segmentation', 'rt_detr_segmentation')]
    )

    # Resize segmentation mask to ESS model image size so it can be used by FoundationPose
    # FoundationPose requires depth, rgb image and segmentation mask to be of the same size
    # Resize from int(HAWK_IMAGE_WIDTH/HAWK_TO_RT_DETR_RATIO) x
    # int(HAWK_IMAGE_HEIGHT/HAWK_TO_RT_DETR_RATIO) to
    # ESS_MODEL_IMAGE_WIDTH x ESS_MODEL_IMAGE_HEIGHT
    # output height constraint is used since keep_aspect_ratio is False
    # and the image is padded
    resize_mask_node = ComposableNode(
        name='resize_mask_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': int(HAWK_IMAGE_WIDTH/HAWK_TO_RT_DETR_RATIO),
            'input_height': int(HAWK_IMAGE_HEIGHT/HAWK_TO_RT_DETR_RATIO),
            'output_width': ESS_MODEL_IMAGE_WIDTH,
            'output_height': ESS_MODEL_IMAGE_HEIGHT,
            'keep_aspect_ratio': False,
            'disable_padding': False
        }],
        remappings=[
            ('image', 'rt_detr_segmentation'),
            ('camera_info', 'resize/camera_info'),
            ('resize/image', 'segmentation'),
            ('resize/camera_info', 'camera_info_segmentation')
        ]
    )

    # Resize hawk images to ESS model image size so it can be used by FoundationPose
    # FoundationPose requires depth, rgb image and segmentation mask to be of the same size
    resize_left_ess_size = ComposableNode(
        name='resize_left_ess_size',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': HAWK_IMAGE_WIDTH,
            'input_height': HAWK_IMAGE_HEIGHT,
            'output_width': ESS_MODEL_IMAGE_WIDTH,
            'output_height': ESS_MODEL_IMAGE_HEIGHT,
            'keep_aspect_ratio': False,
            'encoding_desired': 'rgb8',
            'disable_padding': False
        }],
        remappings=[
            ('image', 'left/image_rect'),
            ('camera_info', 'left/camera_info_rect'),
            ('resize/image', 'rgb/image_rect_color'),
            ('resize/camera_info', 'rgb/camera_info')
        ]
    )

    # Launch ESS(depth estimation)
    ess_node = ComposableNode(
        name='ess_node',
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
        parameters=[{'engine_file_path': ess_depth_engine_file_path,
                     'threshold': ess_depth_threshold
                     }],
        remappings=[
            ('left/image_rect', 'left/image_rect'),
            ('right/image_rect', 'right/image_rect'),
            ('left/camera_info', 'left/camera_info_rect'),
            ('right/camera_info', 'right/camera_info_rect')
        ]
    )
    disparity_to_depth_node = ComposableNode(
        name='disparity_to_depth',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode',
        remappings=[(
            'depth', 'depth_image'
        )]
    )

    # Launch isaac_ros_foundationpose
    selector_node = ComposableNode(
        name='selector_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Selector',
        parameters=[{
            'reset_period': 10000
        }],
        remappings=[
            ('depth_image', 'depth_image'),
            ('image', 'rgb/image_rect_color'),
            ('camera_info', 'rgb/camera_info'),
            ('segmentation', 'segmentation')])

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
        }])

    foundationpose_tracking_node = ComposableNode(
        name='foundationpose_tracking_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode',
        parameters=[{
            'mesh_file_path': mesh_file_path,
            'texture_path': texture_path,

            'refine_model_file_path': refine_model_file_path,
            'refine_engine_file_path': refine_engine_file_path,
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names': ['input1', 'input2'],
            'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],
        }])

    resize_left_viz = ComposableNode(
        name='resize_left_viz',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': HAWK_IMAGE_WIDTH,
            'input_height': HAWK_IMAGE_HEIGHT,
            'output_width': int(HAWK_IMAGE_WIDTH/VISUALIZATION_DOWNSCALING_FACTOR),
            'output_height': int(HAWK_IMAGE_HEIGHT/VISUALIZATION_DOWNSCALING_FACTOR),
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
            correlated_timestamp_driver_node,
            hawk_node,
            drop_node,
            left_rectify_node,
            right_rectify_node,
            resize_left_rt_detr_node,
            pad_node,
            image_to_tensor_node,
            interleave_to_planar_node,
            reshape_node,
            rtdetr_preprocessor_node,
            tensor_rt_node,
            rtdetr_decoder_node,
            detection2_d_array_filter_node,
            detection2_d_to_mask_node,
            resize_mask_node,
            resize_left_ess_size,
            ess_node,
            disparity_to_depth_node,
            selector_node,
            foundationpose_tracking_node,
            foundationpose_node,
            resize_left_viz
        ],
        output='screen'
    )

    return launch.LaunchDescription(launch_args + [foundationpose_container,
                                                   rviz_node])
