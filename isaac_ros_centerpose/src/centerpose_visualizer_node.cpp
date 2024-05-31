// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_centerpose/centerpose_visualizer_node.hpp"

#include "isaac_ros_nitros_detection3_d_array_type/nitros_detection3_d_array.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"

#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace centerpose
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char VIDEO_INPUT_COMPONENT_KEY[] = "centerpose_visualizer/video_buffer_input";
constexpr char VIDEO_INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_bgr8";
constexpr char VIDEO_INPUT_TOPIC_NAME[] = "image";

constexpr char DETECTION_INPUT_COMPONENT_KEY[] = "centerpose_visualizer/detections_input";
constexpr char DETECTION_INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_detection3_d_array";
constexpr char DETECTION_INPUT_TOPIC_NAME[] = "centerpose/detections";

constexpr char CAMERA_INFO_INPUT_COMPONENT_KEY[] = "centerpose_visualizer/camera_model_input";
constexpr char CAMERA_INFO_INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_camera_info";
constexpr char CAMERA_INFO_INPUT_TOPIC_NAME[] = "camera_info";

constexpr char OUTPUT_COMPONENT_KEY[] = "sink/sink";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_bgr8";
constexpr char OUTPUT_TOPIC_NAME[] = "centerpose/image_visualized";

constexpr char APP_YAML_FILENAME[] = "config/centerpose_visualizer_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_centerpose";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"gxf_isaac_centerpose", "gxf/lib/libgxf_isaac_centerpose.so"},
};

const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_centerpose_visualizer",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {VIDEO_INPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = VIDEO_INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = VIDEO_INPUT_TOPIC_NAME,
    }
  },
  {DETECTION_INPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = DETECTION_INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = DETECTION_INPUT_TOPIC_NAME,
    }
  },
  {CAMERA_INFO_INPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = CAMERA_INFO_INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = CAMERA_INFO_INPUT_TOPIC_NAME,
    }
  },
  {OUTPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = OUTPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME,
      .frame_id_source_key = VIDEO_INPUT_TOPIC_NAME,
    }
  }
};
#pragma GCC diagnostic pop

CenterPoseVisualizerNode::CenterPoseVisualizerNode(rclcpp::NodeOptions options)
:  nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  show_axes_{declare_parameter<bool>("show_axes", true)},
  bounding_box_color_{declare_parameter<int64_t>(
      "bounding_box_color",
      0x000000ff)}
{
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosDetection3DArray>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();

  startNitrosNode();
}

CenterPoseVisualizerNode::~CenterPoseVisualizerNode() = default;

void CenterPoseVisualizerNode::postLoadGraphCallback()
{
  RCLCPP_DEBUG(get_logger(), "Inside postLoadGraphCallback of CenterPose");
  getNitrosContext().setParameterBool(
    "centerpose_visualizer", "nvidia::isaac::centerpose::CenterPoseVisualizer", "show_axes",
    show_axes_);
  getNitrosContext().setParameterInt32(
    "centerpose_visualizer", "nvidia::isaac::centerpose::CenterPoseVisualizer",
    "bounding_box_color",
    static_cast<int32_t>(bounding_box_color_));
}

}  // namespace centerpose
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::centerpose::CenterPoseVisualizerNode)
