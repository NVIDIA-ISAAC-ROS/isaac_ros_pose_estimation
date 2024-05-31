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

#include "isaac_ros_centerpose/centerpose_decoder_node.hpp"

#include "isaac_ros_nitros_detection3_d_array_type/nitros_detection3_d_array.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"

#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace centerpose
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_COMPONENT_KEY[] = "centerpose_decoder/input";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nchw_rgb_f32";
constexpr char INPUT_TOPIC_NAME[] = "tensor_sub";

constexpr char CAMERA_INFO_INPUT_COMPONENT_KEY[] = "centerpose_decoder/camera_model_input";
constexpr char CAMERA_INFO_INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_camera_info";
constexpr char CAMERA_INFO_INPUT_TOPIC_NAME[] = "camera_info";

constexpr char OUTPUT_COMPONENT_KEY[] = "sink/sink";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_detection3_d_array";
constexpr char OUTPUT_TOPIC_NAME[] = "centerpose/detections";

constexpr char APP_YAML_FILENAME[] = "config/centerpose_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_centerpose";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"gxf_isaac_centerpose", "gxf/lib/libgxf_isaac_centerpose.so"},
};

const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_centerpose_decoder",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_TOPIC_NAME,
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
      .frame_id_source_key = INPUT_COMPONENT_KEY,
    }
  }
};
#pragma GCC diagnostic pop

CenterPoseDecoderNode::CenterPoseDecoderNode(rclcpp::NodeOptions options)
:  nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  output_field_size_{declare_parameter<std::vector<int64_t>>(
      "output_field_size",
      std::vector<int64_t>({}))},
  cuboid_scaling_factor_{declare_parameter<double>("cuboid_scaling_factor", 0.0)},
  score_threshold_{declare_parameter<double>("score_threshold", 1.0)},
  object_name_{declare_parameter<std::string>("object_name", "")}
{
  if (output_field_size_.empty() || output_field_size_.size() != 2) {
    throw std::invalid_argument("Error: received invalid output field size");
  }

  if (cuboid_scaling_factor_ <= 0.0) {
    throw std::invalid_argument(
            "Error: received a less than or equal to zero cuboid scaling factor");
  }

  if (score_threshold_ >= 1.0) {
    throw std::invalid_argument(
            "Error: received score threshold greater or equal to 1.0");
  }

  if (object_name_.empty()) {
    RCLCPP_WARN(get_logger(), "Received empty object name. Defaulting to unknown.");
    object_name_ = "unknown";
  }
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosDetection3DArray>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();

  startNitrosNode();
}

CenterPoseDecoderNode::~CenterPoseDecoderNode() = default;

void CenterPoseDecoderNode::postLoadGraphCallback()
{
  RCLCPP_DEBUG(get_logger(), "Inside postLoadGraphCallback of CenterPose");

  std::vector<int32_t> output_field_size{
    static_cast<int32_t>(output_field_size_[0]),
    static_cast<int32_t>(output_field_size_[1])
  };
  getNitrosContext().setParameter1DInt32Vector(
    "centerpose_decoder", "nvidia::isaac::centerpose::CenterPosePostProcessor", "output_field_size",
    output_field_size);

  getNitrosContext().setParameterFloat32(
    "centerpose_decoder", "nvidia::isaac::centerpose::CenterPosePostProcessor",
    "cuboid_scaling_factor",
    static_cast<float>(cuboid_scaling_factor_));

  getNitrosContext().setParameterFloat32(
    "centerpose_decoder", "nvidia::isaac::centerpose::CenterPosePostProcessor",
    "score_threshold",
    static_cast<float>(score_threshold_));

  getNitrosContext().setParameterStr(
    "centerpose_decoder_to_isaac", "nvidia::isaac::centerpose::CenterPoseDetectionToIsaac",
    "object_name", object_name_);
}

}  // namespace centerpose
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::centerpose::CenterPoseDecoderNode)
