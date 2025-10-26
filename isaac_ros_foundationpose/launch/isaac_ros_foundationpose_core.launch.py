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

from typing import Any, Dict

from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

# Expected number of input messages in 1 second
INPUT_IMAGES_EXPECT_FREQ = 30
# Number of input messages to be dropped in 1 second
INPUT_IMAGES_DROP_FREQ = 28

# RT-DETR models expect 640x640 encoded image size
RT_DETR_MODEL_INPUT_SIZE = 640
# RT-DETR models expect 3 image channels
RT_DETR_MODEL_NUM_CHANNELS = 3

VISUALIZATION_DOWNSCALING_FACTOR = 10

REFINE_ENGINE_PATH = '/tmp/refine_trt_engine.plan'
SCORE_ENGINE_PATH = '/tmp/score_trt_engine.plan'


class IsaacROSFoundationPoseLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:

        # Drop node parameters
        input_images_expect_freq = LaunchConfiguration('input_images_expect_freq')
        input_images_drop_freq = LaunchConfiguration('input_images_drop_freq')
        # FoundationPose parameters
        mesh_file_path = LaunchConfiguration('mesh_file_path')
        refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
        score_engine_file_path = LaunchConfiguration('score_engine_file_path')
        # RT-DETR parameters
        rt_detr_engine_file_path = LaunchConfiguration('rt_detr_engine_file_path')
        input_width = interface_specs['camera_resolution']['width']
        input_height = interface_specs['camera_resolution']['height']
        input_to_RT_DETR_ratio = input_width / RT_DETR_MODEL_INPUT_SIZE
        return {
            # Drops input_images_expect_freq out of input_images_drop_freq input messages
            'drop_node':  ComposableNode(
                name='drop_node',
                package='isaac_ros_nitros_topic_tools',
                plugin='nvidia::isaac_ros::nitros::NitrosCameraDropNode',
                parameters=[{
                    'X': input_images_drop_freq,
                    'Y': input_images_expect_freq,
                    'mode': 'mono+depth',
                    'depth_format_string': 'nitros_image_mono16'
                }],
                remappings=[
                    ('image_1', 'image_rect'),
                    ('camera_info_1', 'camera_info_rect'),
                    ('depth_1', 'depth'),
                    ('image_1_drop', 'rgb/image_rect_color'),
                    ('camera_info_1_drop', 'rgb/camera_info'),
                    ('depth_1_drop', 'depth_image'),
                ]
            ),

            # Resize and pad input images to RT-DETR model input image size
            # Resize from IMAGE_WIDTH x IMAGE_HEIGHT to
            # IMAGE_WIDTH/input_TO_RT_DETR_RATIO x IMAGE_HEIGHT/input_TO_RT_DETR_RATIO
            # output height constraint is not used since keep_aspect_ratio is True
            'resize_left_rt_detr_node': ComposableNode(
                name='resize_left_rt_detr_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                parameters=[{
                    'input_width': input_width,
                    'input_height': input_height,
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
            ),
            # Pad the image from IMAGE_WIDTH/input_TO_RT_DETR_RATIO x
            # IMAGE_HEIGHT/input_TO_RT_DETR_RATIO
            # to RT_DETR_MODEL_INPUT_WIDTH x RT_DETR_MODEL_INPUT_HEIGHT
            'pad_node': ComposableNode(
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
            ),

            # Convert image to tensor and reshape
            'image_to_tensor_node': ComposableNode(
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
            ),

            'interleave_to_planar_node': ComposableNode(
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
            ),

            'reshape_node': ComposableNode(
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
            ),

            'rtdetr_preprocessor_node': ComposableNode(
                name='rtdetr_preprocessor',
                package='isaac_ros_rtdetr',
                plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
                remappings=[
                    ('encoded_tensor', 'reshaped_tensor')
                ]
            ),

            # RT-DETR objection detection pipeline
            'tensor_rt_node': ComposableNode(
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
            ),
            'rtdetr_decoder_node': ComposableNode(
                name='rtdetr_decoder',
                package='isaac_ros_rtdetr',
                plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
            ),

            # Create a binary segmentation mask from a Detection2DArray published by RT-DETR.
            # The segmentation mask is of size
            # int(IMAGE_WIDTH/input_to_RT_DETR_ratio) x int(IMAGE_HEIGHT/input_to_RT_DETR_ratio)
            'detection2_d_array_filter_node': ComposableNode(
                name='detection2_d_array_filter',
                package='isaac_ros_foundationpose',
                plugin='nvidia::isaac_ros::foundationpose::Detection2DArrayFilter',
                remappings=[('detection2_d_array', 'detections_output')]
            ),
            'detection2_d_to_mask_node': ComposableNode(
                name='detection2_d_to_mask',
                package='isaac_ros_foundationpose',
                plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
                parameters=[{
                    'mask_width': int(input_width/input_to_RT_DETR_ratio),
                    'mask_height': int(input_height/input_to_RT_DETR_ratio)
                }],
                remappings=[('segmentation', 'rt_detr_segmentation')]
            ),

            # Resize segmentation mask to ESS model image size so it can be used by FoundationPose
            # FoundationPose requires depth, rgb image and segmentation mask to be of the same size
            # Resize from int(IMAGE_WIDTH/input_to_RT_DETR_ratio) x
            # int(IMAGE_HEIGHT/input_to_RT_DETR_ratio)
            # to ESS_MODEL_IMAGE_WIDTH x ESS_MODEL_IMAGE_HEIGHT
            # output height constraint is used since keep_aspect_ratio is False
            # and the image is padded
            'resize_mask_node': ComposableNode(
                name='resize_mask_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                parameters=[{
                    'input_width': int(input_width/input_to_RT_DETR_ratio),
                    'input_height': int(input_height/input_to_RT_DETR_ratio),
                    'output_width': input_width,
                    'output_height': input_height,
                    'keep_aspect_ratio': False,
                    'disable_padding': False
                }],
                remappings=[
                    ('image', 'rt_detr_segmentation'),
                    ('camera_info', 'camera_info_resized'),
                    ('resize/image', 'segmentation'),
                    ('resize/camera_info', 'camera_info_segmentation')
                ]
            ),

            'resize_left_viz': ComposableNode(
                name='resize_left_viz',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                parameters=[{
                    'input_width': input_width,
                    'input_height': input_height,
                    'output_width': int(input_width/VISUALIZATION_DOWNSCALING_FACTOR) * 2,
                    'output_height': int(input_height/VISUALIZATION_DOWNSCALING_FACTOR) * 2,
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
            ),

            'foundationpose_node': ComposableNode(
                name='foundationpose_node',
                package='isaac_ros_foundationpose',
                plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
                parameters=[{
                    'mesh_file_path': mesh_file_path,

                    'refine_engine_file_path': refine_engine_file_path,
                    'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
                    'refine_input_binding_names': ['input1', 'input2'],
                    'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
                    'refine_output_binding_names': ['output1', 'output2'],

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
                    ('pose_estimation/output', 'output')]
            ),
        }

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) -> \
            Dict[str, launch.actions.OpaqueFunction]:

        return {
            'input_images_expect_freq': DeclareLaunchArgument(
                'input_images_expect_freq',
                default_value=str(INPUT_IMAGES_EXPECT_FREQ),
                description='Expected number of input messages in 1 second'),

            'input_images_drop_freq': DeclareLaunchArgument(
                'input_images_drop_freq',
                default_value=str(INPUT_IMAGES_DROP_FREQ),
                description='Number of input messages to be dropped in 1 second'),

            'mesh_file_path': DeclareLaunchArgument(
                'mesh_file_path',
                default_value='',
                description='The absolute file path to the mesh file'),

            'refine_engine_file_path': DeclareLaunchArgument(
                'refine_engine_file_path',
                default_value=REFINE_ENGINE_PATH,
                description='The absolute file path to the refine trt engine'),

            'score_engine_file_path': DeclareLaunchArgument(
                'score_engine_file_path',
                default_value=SCORE_ENGINE_PATH,
                description='The absolute file path to the score trt engine'),

            'rt_detr_model_file_path': DeclareLaunchArgument(
                'rt_detr_model_file_path',
                default_value='',
                description='The absolute file path to the RT-DETR ONNX file'),

            'rt_detr_engine_file_path': DeclareLaunchArgument(
                'rt_detr_engine_file_path',
                default_value='',
                description='The absolute file path to the RT-DETR TensorRT engine file'),
        }


def generate_launch_description():
    foundationpose_container = ComposableNodeContainer(
        package='rclcpp_components',
        name='foundationpose_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=IsaacROSFoundationPoseLaunchFragment
        .get_composable_nodes().values(),
        output='screen'
    )

    return launch.LaunchDescription(
        [foundationpose_container] +
        IsaacROSFoundationPoseLaunchFragment.get_launch_actions().values())
