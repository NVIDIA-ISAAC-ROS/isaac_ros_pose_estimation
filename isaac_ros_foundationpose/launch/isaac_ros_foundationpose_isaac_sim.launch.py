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
import platform

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


RT_DETR_MODEL_INPUT_SIZE = 640  # RT-DETR models expect 640x640 encoded image size
RT_DETR_MODEL_NUM_CHANNELS = 3  # RT-DETR models expect 3 image channels

ESS_MODEL_IMAGE_WIDTH = 480
ESS_MODEL_IMAGE_HEIGHT = 288

SIM_IMAGE_WIDTH = 1920
SIM_IMAGE_HEIGHT = 1200

SIM_TO_RT_DETR_RATIO = SIM_IMAGE_WIDTH / RT_DETR_MODEL_INPUT_SIZE

ISAAC_ROS_ASSETS_PATH = os.path.join(os.environ['ISAAC_ROS_WS'], 'isaac_ros_assets')
ISAAC_ROS_MODELS_PATH = os.path.join(ISAAC_ROS_ASSETS_PATH, 'models')
ISAAC_ROS_FP_MESHES_PATH = os.path.join(ISAAC_ROS_ASSETS_PATH,
                                        'isaac_ros_foundationpose')
STEREO_DISPARITY_MODELS_PATH = os.path.join(ISAAC_ROS_MODELS_PATH,
                                            'dnn_stereo_disparity',
                                            'dnn_stereo_disparity_v4.1.0_onnx_trt10.16')
SYNTHETICA_DETR_MODELS_PATH = os.path.join(ISAAC_ROS_MODELS_PATH, 'synthetica_detr')
FOUDNATIONPOSE_MODELS_PATH = os.path.join(ISAAC_ROS_MODELS_PATH, 'foundationpose')
REFINE_ENGINE_PATH = os.path.join(FOUDNATIONPOSE_MODELS_PATH, 'refine_trt_engine.plan')
SCORE_ENGINE_PATH = os.path.join(FOUDNATIONPOSE_MODELS_PATH, 'score_trt_engine.plan')
ESS_ENGINE_PATH = os.path.join(STEREO_DISPARITY_MODELS_PATH, 'light_ess.engine')
ESS_PLUGIN_PATH = os.path.join(
    STEREO_DISPARITY_MODELS_PATH, 'plugins', platform.machine(), 'ess_plugins.so')
RTDETR_ENGINE_PATH = os.path.join(SYNTHETICA_DETR_MODELS_PATH, 'sdetr_grasp.plan')
MESH_OBJ_PATH = os.path.join(ISAAC_ROS_FP_MESHES_PATH,
                             'Mac_and_cheese_0_1', 'Mac_and_cheese_0_1.obj')
MESH_TEX_PATH = os.path.join(ISAAC_ROS_FP_MESHES_PATH,
                             'Mac_and_cheese_0_1', 'materials', 'textures', 'baked_mesh_tex0.png')


def generate_launch_description():
    """Generate launch description for testing relevant nodes."""
    rviz_config_path = os.path.join(
        get_package_share_directory('isaac_ros_foundationpose'),
        'rviz', 'foundationpose_isaac_sim.rviz')

    launch_args = [
        DeclareLaunchArgument(
            'mesh_file_path',
            default_value=MESH_OBJ_PATH,
            description='The absolute file path to the mesh file'),

        DeclareLaunchArgument(
            'texture_path',
            default_value=MESH_TEX_PATH,
            description='The absolute file path to the texture map'),

        DeclareLaunchArgument(
            'refine_engine_file_path',
            default_value=REFINE_ENGINE_PATH,
            description='The absolute file path to the refine trt engine'),

        DeclareLaunchArgument(
            'score_engine_file_path',
            default_value=SCORE_ENGINE_PATH,
            description='The absolute file path to the score trt engine'),

        DeclareLaunchArgument(
            'rt_detr_engine_file_path',
            default_value=RTDETR_ENGINE_PATH,
            description='The absolute file path to the RT-DETR TensorRT engine file'),

        DeclareLaunchArgument(
            'ess_depth_engine_file_path',
            default_value=ESS_ENGINE_PATH,
            description='The absolute path to the ESS engine plan.'),

        DeclareLaunchArgument(
            'ess_plugin_file_path',
            default_value=ESS_PLUGIN_PATH,
            description='The absolute path to the ESS TensorRT plugin library.'),

        DeclareLaunchArgument(
            'ess_depth_threshold',
            default_value='0.4',
            description='Threshold value ranges between 0.0 and 1.0 '
                        'for filtering disparity with confidence.'),

        DeclareLaunchArgument(
            'launch_rviz',
            default_value='True',
            description='Flag to enable Rviz2 launch'),

        DeclareLaunchArgument(
            'container_name',
            default_value='foundationpose_container',
            description='Name for ComposableNodeContainer'),
    ]

    mesh_file_path = LaunchConfiguration('mesh_file_path')
    texture_path = LaunchConfiguration('texture_path')
    refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
    score_engine_file_path = LaunchConfiguration('score_engine_file_path')
    rt_detr_engine_file_path = LaunchConfiguration('rt_detr_engine_file_path')
    ess_depth_engine_file_path = LaunchConfiguration('ess_depth_engine_file_path')
    ess_plugin_file_path = LaunchConfiguration('ess_plugin_file_path')
    ess_depth_threshold = LaunchConfiguration('ess_depth_threshold')
    launch_rviz = LaunchConfiguration('launch_rviz')
    container_name = LaunchConfiguration('container_name')

    # Resize and pad Isaac Sim images to RT-DETR model input image size
    resize_left_rt_detr_node = ComposableNode(
        name='resize_left_rt_detr_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': SIM_IMAGE_WIDTH,
            'input_height': SIM_IMAGE_HEIGHT,
            'output_width': RT_DETR_MODEL_INPUT_SIZE,
            'output_height': RT_DETR_MODEL_INPUT_SIZE,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }],
        remappings=[
            ('image', 'front_stereo_camera/left/image_rect_color'),
            ('camera_info', 'front_stereo_camera/left/camera_info')
        ]
    )
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

    # Convert Detection2DArray from RT-DETR to a binary segmentation mask
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
            'mask_width': int(SIM_IMAGE_WIDTH/SIM_TO_RT_DETR_RATIO),
            'mask_height': int(SIM_IMAGE_HEIGHT/SIM_TO_RT_DETR_RATIO)
        }],
        remappings=[('segmentation', 'rt_detr_segmentation')]
    )

    # Resize segmentation mask to ESS model image size so it can be used by FoundationPose
    # FoundationPose requires depth, rgb image and segmentation mask to be of the same size
    resize_mask_node = ComposableNode(
        name='resize_mask_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': int(SIM_IMAGE_WIDTH/SIM_TO_RT_DETR_RATIO),
            'input_height': int(SIM_IMAGE_HEIGHT/SIM_TO_RT_DETR_RATIO),
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

    # Resize Isaac Sim images to ESS model image size so it can be used by FoundationPose
    # FoundationPose requires depth, rgb image and segmentation mask to be of the same size
    # Note: The resized rgb camera model is named "camera_info" because that is the
    # name expected by the Rviz2
    resize_left_ess_size = ComposableNode(
        name='resize_left_ess_size',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': SIM_IMAGE_WIDTH,
            'input_height': SIM_IMAGE_HEIGHT,
            'output_width': ESS_MODEL_IMAGE_WIDTH,
            'output_height': ESS_MODEL_IMAGE_HEIGHT,
            'keep_aspect_ratio': False,
            'encoding_desired': 'rgb8',
            'disable_padding': False
        }],
        remappings=[
            ('image', 'front_stereo_camera/left/image_rect_color'),
            ('camera_info', 'front_stereo_camera/left/camera_info'),
            ('resize/image', 'front_stereo_camera/left/image_rect_color_resized'),
            ('resize/camera_info', 'front_stereo_camera/left/camera_info_resized')
        ]
    )
    resize_right_ess_size = ComposableNode(
        name='resize_right_ess_size',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': SIM_IMAGE_WIDTH,
            'input_height': SIM_IMAGE_HEIGHT,
            'output_width': ESS_MODEL_IMAGE_WIDTH,
            'output_height': ESS_MODEL_IMAGE_HEIGHT,
            'keep_aspect_ratio': False,
            'encoding_desired': 'rgb8',
            'disable_padding': False
        }],
        remappings=[
            ('image', 'front_stereo_camera/right/image_rect_color'),
            ('camera_info', 'front_stereo_camera/right/camera_info'),
            ('resize/image', 'front_stereo_camera/right/image_rect_color_resized'),
            ('resize/camera_info', 'front_stereo_camera/right/camera_info_resized')
        ]
    )

    # Launch ESS (depth estimation) - decomposed pipeline
    ess_left_normalize_node = ComposableNode(
        name='ess_left_normalize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        parameters=[{
            'mean': [127.5, 127.5, 127.5],
            'stddev': [127.5, 127.5, 127.5],
        }],
        remappings=[
            ('image', 'front_stereo_camera/left/image_rect_color_resized'),
            ('normalized_image', 'ess_left/image_normalize')
        ]
    )
    ess_left_tensor_node = ComposableNode(
        name='ess_left_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'left_image',
        }],
        remappings=[
            ('image', 'ess_left/image_normalize'),
            ('tensor', 'ess_left/tensor'),
        ]
    )
    ess_left_planar_node = ComposableNode(
        name='ess_left_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [ESS_MODEL_IMAGE_HEIGHT, ESS_MODEL_IMAGE_WIDTH, 3],
            'output_tensor_name': 'left_image'
        }],
        remappings=[
            ('interleaved_tensor', 'ess_left/tensor'),
            ('planar_tensor', 'ess_left/tensor_planar')
        ]
    )
    ess_left_reshape_node = ComposableNode(
        name='ess_left_reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'left_image',
            'input_tensor_shape': [3, ESS_MODEL_IMAGE_HEIGHT, ESS_MODEL_IMAGE_WIDTH],
            'output_tensor_shape': [1, 3, ESS_MODEL_IMAGE_HEIGHT, ESS_MODEL_IMAGE_WIDTH]
        }],
        remappings=[
            ('tensor', 'ess_left/tensor_planar'),
            ('reshaped_tensor', 'ess_left/tensor_reshape')
        ]
    )

    ess_right_normalize_node = ComposableNode(
        name='ess_right_normalize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        parameters=[{
            'mean': [127.5, 127.5, 127.5],
            'stddev': [127.5, 127.5, 127.5],
        }],
        remappings=[
            ('image', 'front_stereo_camera/right/image_rect_color_resized'),
            ('normalized_image', 'ess_right/image_normalize')
        ]
    )
    ess_right_tensor_node = ComposableNode(
        name='ess_right_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'right_image',
        }],
        remappings=[
            ('image', 'ess_right/image_normalize'),
            ('tensor', 'ess_right/tensor'),
        ]
    )
    ess_right_planar_node = ComposableNode(
        name='ess_right_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [ESS_MODEL_IMAGE_HEIGHT, ESS_MODEL_IMAGE_WIDTH, 3],
            'output_tensor_name': 'right_image'
        }],
        remappings=[
            ('interleaved_tensor', 'ess_right/tensor'),
            ('planar_tensor', 'ess_right/tensor_planar')
        ]
    )
    ess_right_reshape_node = ComposableNode(
        name='ess_right_reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'right_image',
            'input_tensor_shape': [3, ESS_MODEL_IMAGE_HEIGHT, ESS_MODEL_IMAGE_WIDTH],
            'output_tensor_shape': [1, 3, ESS_MODEL_IMAGE_HEIGHT, ESS_MODEL_IMAGE_WIDTH]
        }],
        remappings=[
            ('tensor', 'ess_right/tensor_planar'),
            ('reshaped_tensor', 'ess_right/tensor_reshape')
        ]
    )

    ess_tensor_pair_sync_node = ComposableNode(
        name='ess_tensor_pair_sync_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::TensorPairSyncNode',
        parameters=[{
            'input_tensor1_name': 'left_image',
            'input_tensor2_name': 'right_image',
            'output_tensor1_name': 'input_left',
            'output_tensor2_name': 'input_right'
        }],
        remappings=[
            ('tensor1', 'ess_left/tensor_reshape'),
            ('tensor2', 'ess_right/tensor_reshape'),
            ('tensor_pub', 'ess/tensor_pub'),
        ]
    )

    ess_tensor_rt_node = ComposableNode(
        name='ess_tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': ess_depth_engine_file_path,
            'input_tensor_names': ['input_left', 'input_right'],
            'input_binding_names': ['input_left', 'input_right'],
            'output_tensor_names': ['output_left', 'output_conf'],
            'output_binding_names': ['output_left', 'output_conf'],
            'verbose': False,
            'force_engine_update': False,
            'custom_plugin_lib': ess_plugin_file_path,
        }],
        remappings=[
            ('tensor_pub', 'ess/tensor_pub'),
            ('tensor_sub', 'ess/tensor_sub'),
        ]
    )

    ess_decoder_node = ComposableNode(
        name='ess_decoder',
        package='isaac_ros_dnn_stereo_decoder',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::DNNStereoDecoderNode',
        parameters=[{
            'disparity_tensor_name': 'output_left',
            'confidence_tensor_name': 'output_conf',
            'confidence_threshold': ess_depth_threshold,
            'cache_camera_info': True,
        }],
        remappings=[
            ('tensor_sub', 'ess/tensor_sub'),
            ('right/camera_info', 'front_stereo_camera/right/camera_info_resized')
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
    foundationpose_node = ComposableNode(
        name='foundationpose_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[{
            'mesh_file_path': mesh_file_path,
            'texture_path': texture_path,
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
        }],
        remappings=[
            ('pose_estimation/depth_image', 'depth_image'),
            ('pose_estimation/image', 'front_stereo_camera/left/image_rect_color_resized'),
            ('pose_estimation/camera_info', 'front_stereo_camera/left/camera_info_resized'),
            ('pose_estimation/segmentation', 'segmentation'),
            ('pose_estimation/output', 'output')]
        )

    refine_trt_node = ComposableNode(
        name='refine_trt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': refine_engine_file_path,
            'input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'input_binding_names': ['input1', 'input2'],
            'output_tensor_names': ['output_tensor1', 'output_tensor2'],
            'output_binding_names': ['output1', 'output2'],
            'force_engine_update': False,
            'verbose': False,
            'max_batch_size': 42,
        }],
        remappings=[
            ('tensor_pub', 'refine/tensor_pub'),
            ('tensor_sub', 'refine/tensor_sub'),
        ])

    score_trt_node = ComposableNode(
        name='score_trt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': score_engine_file_path,
            'input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'input_binding_names': ['input1', 'input2'],
            'output_tensor_names': ['output_tensor'],
            'output_binding_names': ['output1'],
            'force_engine_update': False,
            'verbose': False,
            'max_batch_size': 252,
        }],
        remappings=[
            ('tensor_pub', 'score/tensor_pub'),
            ('tensor_sub', 'score/tensor_sub'),
        ])

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
            resize_right_ess_size,
            ess_left_normalize_node,
            ess_left_tensor_node,
            ess_left_planar_node,
            ess_left_reshape_node,
            ess_right_normalize_node,
            ess_right_tensor_node,
            ess_right_planar_node,
            ess_right_reshape_node,
            ess_tensor_pair_sync_node,
            ess_tensor_rt_node,
            ess_decoder_node,
            disparity_to_depth_node,
            refine_trt_node,
            score_trt_node,
            foundationpose_node
        ],
        output='screen'
    )

    return launch.LaunchDescription(launch_args + [foundationpose_container,
                                                   rviz_node])
